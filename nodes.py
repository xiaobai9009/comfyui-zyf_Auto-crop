import torch
import numpy as np
from PIL import Image
import logging
import math
from typing import Optional, Tuple

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .models import BoundingBox, CropRegion, DetectionResult
from .detection_manager import DetectionManager, ModelLoadError
from .crop_calculator import CropCalculator
from .image_processor import ImageProcessor

# 配置日志
logger = logging.getLogger(__name__)


# 预设宽高比映射 (宽:高) -> 是否为竖向(高>宽)
_PRESET_RATIO_MAP = {
    "1:1": (1, 1),
    "3:2": (3, 2),
    "4:3": (4, 3),
    "16:9": (16, 9),
    "2:3": (2, 3),
    "3:4": (3, 4),
    "9:16": (9, 16),
}


def _resolve_aspect_ratio_value(宽高比: str, 自定义宽: int, 自定义高: int):
    """
    解析宽高比字符串为 (宽, 高) 元组，失败返回 None。
    """
    if 宽高比 == "原始":
        return None
    if 宽高比 == "自定义":
        if 自定义宽 <= 0 or 自定义高 <= 0:
            return None
        return (自定义宽, 自定义高)
    if 宽高比 in _PRESET_RATIO_MAP:
        return _PRESET_RATIO_MAP[宽高比]
    return None


def _auto_swap_aspect_ratio(
    宽高比: str,
    自定义宽: int,
    自定义高: int,
    自动宽高比: bool,
    img_width: int,
    img_height: int
) -> Tuple[str, int, int]:
    """
    根据图像方向自动反转宽高比。

    启用自动宽高比且宽高比不为"原始"时：
    - 竖图(h>w) + 横版比例(rw>rh,如16:9) → 比例反转为rh:rw(9:16)
    - 横图(w>h) + 竖版比例(rw<rh,如2:3) → 比例反转为rh:rw(3:2)
    - 1:1、方图或方向已匹配 → 保持不变
    - 自定义比例：按 自定义宽/高 判断方向

    返回: (新宽高比, 新自定义宽, 新自定义高)
    """
    if not 自动宽高比 or 宽高比 == "原始":
        return 宽高比, 自定义宽, 自定义高
    if img_width <= 0 or img_height <= 0:
        return 宽高比, 自定义宽, 自定义高

    parsed = _resolve_aspect_ratio_value(宽高比, 自定义宽, 自定义高)
    if parsed is None:
        return 宽高比, 自定义宽, 自定义高
    rw, rh = parsed

    # 方形比例(1:1)或方形图像,不反转
    if rw == rh or img_width == img_height:
        return 宽高比, 自定义宽, 自定义高

    is_image_portrait = img_height > img_width
    is_ratio_portrait = rw < rh  # 宽<高 的比例为竖向

    need_swap = (is_image_portrait and not is_ratio_portrait) or \
                (not is_image_portrait and is_ratio_portrait)
    if not need_swap:
        return 宽高比, 自定义宽, 自定义高

    swapped_rw, swapped_rh = rh, rw

    if 宽高比 == "自定义":
        return "自定义", swapped_rw, swapped_rh

    # 查找反转换后的预设
    for k, v in _PRESET_RATIO_MAP.items():
        if v == (swapped_rw, swapped_rh):
            return k, 自定义宽, 自定义高
    # 没找到对应预设(理论上不应发生),保持原值
    return 宽高比, 自定义宽, 自定义高


class AutoCropBorderNode:
    """
    自动检测并裁剪图像四周的纯色边框
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "容差值": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "颜色容差值，用于判断边框是否为纯色。值越大，容忍的颜色差异越大，越容易裁剪掉边框"
                }),
                "最小裁剪尺寸": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "裁剪后图像的最小尺寸（宽度或高度），防止过度裁剪导致图像太小"
                }),
            },
            "optional": {
                "遮罩": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "遮罩")
    FUNCTION = "crop_border"
    CATEGORY = "图像/变换"
    
    def crop_border(self, 图像, 容差值=10, 最小裁剪尺寸=10, 遮罩=None):
        """
        裁剪图像边框
        
        Args:
            图像: ComfyUI 图像张量 (B, H, W, C)
            遮罩: 输入遮罩张量 (B, H, W)，可选
            容差值: 颜色容差值，用于判断是否为纯色
            最小裁剪尺寸: 最小裁剪尺寸，防止裁剪过度
        """
        # 转换为 numpy 数组处理
        batch_size = 图像.shape[0]
        result_images = []
        result_masks = []
        crop_regions = []  # 记录每张图像的裁剪区域
        
        for i in range(batch_size):
            # 获取单张图像 (H, W, C)
            img_tensor = 图像[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            orig_h, orig_w = img_np.shape[:2]
            
            # 裁剪边框，返回裁剪后的图像和裁剪区域
            cropped, crop_region = self._crop_single_image_with_region(img_np, 容差值, 最小裁剪尺寸)
            
            # 转回张量
            cropped_tensor = torch.from_numpy(cropped.astype(np.float32) / 255.0)
            result_images.append(cropped_tensor)
            
            # 处理遮罩
            if 遮罩 is not None and i < 遮罩.shape[0]:
                # 裁剪输入的遮罩
                mask_np = 遮罩[i].cpu().numpy()
                top, bottom, left, right = crop_region
                cropped_mask = mask_np[top:bottom, left:right]
                cropped_mask_tensor = torch.from_numpy(cropped_mask)
            else:
                # 创建全白遮罩
                h, w = cropped.shape[:2]
                cropped_mask_tensor = torch.ones((h, w), dtype=torch.float32)
            
            result_masks.append(cropped_mask_tensor)
        
        # 合并批次
        result = torch.stack(result_images, dim=0)
        mask = torch.stack(result_masks, dim=0)
        
        return (result, mask)
    
    def _crop_single_image_with_region(self, img_np, tolerance, min_crop_size):
        """
        裁剪单张图像的纯色边框，返回裁剪后的图像和裁剪区域
        使用更智能的边界检测算法
        
        返回:
            cropped: 裁剪后的图像
            region: (top, bottom, left, right) 裁剪区域
        """
        h, w = img_np.shape[:2]
        
        # 获取四个角的颜色作为边框参考色
        corner_colors = [
            img_np[0, 0],           # 左上
            img_np[0, w-1],         # 右上
            img_np[h-1, 0],         # 左下
            img_np[h-1, w-1]        # 右下
        ]
        
        # 检测上边界
        top = 0
        for y in range(h):
            if not self._is_border_row(img_np, y, corner_colors, tolerance):
                top = y
                break
        
        # 检测下边界
        bottom = h
        for y in range(h - 1, -1, -1):
            if not self._is_border_row(img_np, y, corner_colors, tolerance):
                bottom = y + 1
                break
        
        # 检测左边界
        left = 0
        for x in range(w):
            if not self._is_border_col(img_np, x, corner_colors, tolerance):
                left = x
                break
        
        # 检测右边界
        right = w
        for x in range(w - 1, -1, -1):
            if not self._is_border_col(img_np, x, corner_colors, tolerance):
                right = x + 1
                break
        
        # 确保裁剪后的尺寸不小于最小值
        crop_h = bottom - top
        crop_w = right - left
        
        if crop_h < min_crop_size or crop_w < min_crop_size:
            return img_np, (0, h, 0, w)
        
        # 裁剪图像
        cropped = img_np[top:bottom, left:right]
        
        return cropped, (top, bottom, left, right)
    
    def _is_border_row(self, img_np, y, corner_colors, tolerance):
        """
        检查某一行是否为边框色
        与四个角的颜色进行比较，使用更严格的判断标准
        """
        row = img_np[y]
        
        # 检查这一行是否与任一角的颜色相似
        for corner_color in corner_colors:
            diff = np.abs(row.astype(np.int16) - corner_color.astype(np.int16))
            max_diff = np.max(diff, axis=1)  # 每个像素的最大通道差异
            
            # 使用更严格的判断：要求所有像素都相似
            similar_pixels = np.sum(max_diff <= tolerance)
            if similar_pixels == len(row):  # 100%的像素相似
                return True
            
            # 或者至少99%的像素相似（允许极少数噪点）
            if similar_pixels >= len(row) * 0.99:
                return True
        
        return False
    
    def _is_border_col(self, img_np, x, corner_colors, tolerance):
        """
        检查某一列是否为边框色
        与四个角的颜色进行比较，使用更严格的判断标准
        """
        col = img_np[:, x]
        
        # 检查这一列是否与任一角的颜色相似
        for corner_color in corner_colors:
            diff = np.abs(col.astype(np.int16) - corner_color.astype(np.int16))
            max_diff = np.max(diff, axis=1)  # 每个像素的最大通道差异
            
            # 使用更严格的判断：要求所有像素都相似
            similar_pixels = np.sum(max_diff <= tolerance)
            if similar_pixels == len(col):  # 100%的像素相似
                return True
            
            # 或者至少99%的像素相似（允许极少数噪点）
            if similar_pixels >= len(col) * 0.99:
                return True
        
        return False


class PersonCropNode:
    """
    按人物主体比例缩放节点。
    采用"人脸优先、身体次之"的智能裁剪策略。
    """
    
    # 宽高比预设选项
    ASPECT_RATIO_OPTIONS = ["原始", "自定义", "1:1", "3:2", "4:3", "16:9", "2:3", "3:4", "9:16"]
    
    # 缩放模式选项
    SCALE_MODE_OPTIONS = ["适应", "裁剪", "填充"]
    
    # 采样方法选项
    SAMPLING_OPTIONS = ["lanczos", "nearest", "bilinear", "bicubic"]
    
    # 倍数取整选项
    MULTIPLE_OPTIONS = ["8", "16", "32", "64", "128", "256", "512", "无"]
    
    # 缩放到边选项
    SCALE_TO_SIDE_OPTIONS = ["无", "最长边", "最短边", "宽度", "高度", "总像素(千像素)"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
            },
            "optional": {
                "自动宽高比": ("BOOLEAN", {
                    "default": False,
                    "label_on": "启用",
                    "label_off": "关闭",
                    "tooltip": "开启后,当宽高比不是'原始'时,根据输入图像的横竖方向自动反转比例。例如:选2:3遇横图自动变3:2;选16:9遇竖图自动变9:16;自定义比例同样按宽高方向自动转换"
                }),
                "遮罩": ("MASK",),
                "宽高比": (cls.ASPECT_RATIO_OPTIONS, {
                    "default": "原始",
                    "tooltip": "目标宽高比。选择'原始'保持检测到的人物主体比例,或选择预设比例(如3:4适合竖屏,16:9适合横屏)"
                }),
                "比例宽": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "自定义宽高比的宽度值（仅当宽高比选择'自定义'时生效）"
                }),
                "比例高": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "自定义宽高比的高度值（仅当宽高比选择'自定义'时生效）"
                }),
                "缩放模式": (cls.SCALE_MODE_OPTIONS, {
                    "default": "适应",
                    "tooltip": "缩放模式：'适应'保持宽高比等比缩放，'裁剪'填充目标尺寸并裁剪多余部分，'填充'直接拉伸"
                }),
                "采样方法": (cls.SAMPLING_OPTIONS, {
                    "default": "lanczos",
                    "tooltip": "图像缩放时的采样算法。lanczos质量最高但速度较慢，nearest速度最快但质量较低"
                }),
                "倍数取整": (cls.MULTIPLE_OPTIONS, {
                    "default": "16",
                    "tooltip": "将输出尺寸向上取整到指定倍数（如16的倍数），有助于某些AI模型的处理。选择'无'则不取整"
                }),
                "缩放到边": (cls.SCALE_TO_SIDE_OPTIONS, {
                    "default": "无",
                    "tooltip": "指定缩放目标：'最长边'将长边缩放到目标长度，'最短边'将短边缩放到目标长度，'宽度'/'高度'指定具体边"
                }),
                "缩放到长度": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "配合'缩放到边'使用，指定目标边的像素长度（或总像素的千像素数）"
                }),
                "背景颜色": ("STRING", {
                    "default": "#000000",
                    "tooltip": "背景填充颜色，使用十六进制颜色代码（如#000000为黑色，#FFFFFF为白色）"
                }),
                "人物索引": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "当检测到多个人物时，选择要裁剪的人物（0为所有人物的中心区域，1为最大人物，2为第二大，以此类推）。0的含义就是尽可能展现图片中的所有人"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "遮罩")
    FUNCTION = "crop_person"
    CATEGORY = "图像/变换"
    
    def __init__(self):
        """初始化节点，创建检测管理器和裁剪计算器实例"""
        self.detection_manager = None
        self.crop_calculator = CropCalculator(min_face_visibility=0.67)
        self.image_processor = ImageProcessor()
    
    def _parse_aspect_ratio(self, 宽高比: str, 比例宽: int, 比例高: int) -> Optional[Tuple[int, int]]:
        """
        解析宽高比设置
        
        参数:
            宽高比: 预设选项
            比例宽: 自定义宽度比例
            比例高: 自定义高度比例
            
        返回:
            宽高比元组或 None
        """
        if 宽高比 == "原始":
            return None
        elif 宽高比 == "自定义":
            return (比例宽, 比例高)
        elif 宽高比 == "1:1":
            return (1, 1)
        elif 宽高比 == "3:2":
            return (3, 2)
        elif 宽高比 == "4:3":
            return (4, 3)
        elif 宽高比 == "16:9":
            return (16, 9)
        elif 宽高比 == "2:3":
            return (2, 3)
        elif 宽高比 == "3:4":
            return (3, 4)
        elif 宽高比 == "9:16":
            return (9, 16)
        return None
    
    def _parse_hex_color(self, hex_color: str) -> Tuple[int, int, int]:
        """
        解析十六进制颜色值
        
        参数:
            hex_color: 十六进制颜色字符串 (如 #000000)
            
        返回:
            RGB 元组
        """
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (0, 0, 0)
    
    def _apply_scale_and_multiple(
        self,
        image_np: np.ndarray,
        缩放到边: str,
        缩放到长度: int,
        倍数取整: str,
        采样方法: str,
        缩放模式: str,
        aspect_ratio: Optional[Tuple[int, int]],
        背景颜色: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        应用缩放和倍数取整
        
        参数:
            image_np: 输入图像
            缩放到边: 缩放目标边
            缩放到长度: 目标长度
            倍数取整: 倍数取整值
            采样方法: 采样方法
            缩放模式: 缩放模式
            aspect_ratio: 宽高比
            背景颜色: 背景颜色
            
        返回:
            处理后的图像
        """
        h, w = image_np.shape[:2]
        current_ratio = w / h
        
        # 采样方法映射
        resample_map = {
            "lanczos": Image.LANCZOS,
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC
        }
        resample = resample_map.get(采样方法, Image.LANCZOS)
        
        # 确定目标宽高比
        if aspect_ratio is not None:
            target_ratio = aspect_ratio[0] / aspect_ratio[1]
        else:
            target_ratio = current_ratio
        
        # 计算目标尺寸
        new_w, new_h = w, h
        
        if 缩放到边 != "无":
            # 根据"缩放到边"设置计算目标尺寸
            if 缩放到边 == "最长边":
                # 最长边设置为目标长度
                if target_ratio >= 1.0:
                    # 横向图像，宽度是长边
                    new_w = 缩放到长度
                    new_h = int(new_w / target_ratio)
                else:
                    # 纵向图像，高度是长边
                    new_h = 缩放到长度
                    new_w = int(new_h * target_ratio)
            elif 缩放到边 == "最短边":
                # 最短边设置为目标长度
                if target_ratio >= 1.0:
                    # 横向图像，高度是短边
                    new_h = 缩放到长度
                    new_w = int(new_h * target_ratio)
                else:
                    # 纵向图像，宽度是短边
                    new_w = 缩放到长度
                    new_h = int(new_w / target_ratio)
            elif 缩放到边 == "宽度":
                new_w = 缩放到长度
                new_h = int(new_w / target_ratio)
            elif 缩放到边 == "高度":
                new_h = 缩放到长度
                new_w = int(new_h * target_ratio)
            elif 缩放到边 == "总像素(千像素)":
                target_pixels = 缩放到长度 * 1000
                # 根据宽高比计算尺寸
                # w * h = target_pixels
                # w / h = target_ratio
                # => w = sqrt(target_pixels * target_ratio)
                # => h = sqrt(target_pixels / target_ratio)
                new_w = int((target_pixels * target_ratio) ** 0.5)
                new_h = int((target_pixels / target_ratio) ** 0.5)
        
        # 倍数取整（向上取整，保持宽高比）
        if 倍数取整 != "无":
            multiple = int(倍数取整)
            
            # 向上取整到倍数
            import math
            new_w = math.ceil(new_w / multiple) * multiple
            new_h = math.ceil(new_h / multiple) * multiple
            
            # 如果指定了宽高比，调整以尽可能接近目标宽高比
            if aspect_ratio is not None:
                actual_ratio = new_w / new_h
                # 如果宽高比偏差较大，调整一个维度
                if abs(actual_ratio - target_ratio) > 0.05:
                    if actual_ratio > target_ratio:
                        # 实际比例太宽，增加高度
                        new_h_adjusted = int(new_w / target_ratio)
                        new_h_adjusted = math.ceil(new_h_adjusted / multiple) * multiple
                        new_h = new_h_adjusted
                    else:
                        # 实际比例太窄，增加宽度
                        new_w_adjusted = int(new_h * target_ratio)
                        new_w_adjusted = math.ceil(new_w_adjusted / multiple) * multiple
                        new_w = new_w_adjusted
        
        # 转换为 PIL 图像进行处理
        pil_image = Image.fromarray(image_np)
        bg_rgb = self._parse_hex_color(背景颜色)
        
        # 根据缩放模式执行
        if 缩放模式 == "填充":
            result = pil_image.resize((new_w, new_h), resample)
            return np.array(result), None
        
        if 缩放模式 == "裁剪":
            # 先中心裁剪到目标宽高比，再缩放到目标尺寸
            cur_ratio = current_ratio
            if abs(cur_ratio - target_ratio) < 1e-6:
                cropped = pil_image
            else:
                if cur_ratio > target_ratio:
                    crop_w = int(h * target_ratio)
                    left = max(0, (w - crop_w) // 2)
                    box = (left, 0, left + crop_w, h)
                else:
                    crop_h = int(w / target_ratio)
                    top = max(0, (h - crop_h) // 2)
                    box = (0, top, w, top + crop_h)
                cropped = pil_image.crop(box)
            result = cropped.resize((new_w, new_h), resample)
            return np.array(result), None
        
        # 适应：letterbox 等比缩放并居中填充到目标尺寸
        scale = min(new_w / w, new_h / h)
        resized_w = max(1, int(round(w * scale)))
        resized_h = max(1, int(round(h * scale)))
        resized = pil_image.resize((resized_w, resized_h), resample)
        canvas = Image.new("RGB", (new_w, new_h), bg_rgb)
        x_off = (new_w - resized_w) // 2
        y_off = (new_h - resized_h) // 2
        canvas.paste(resized, (x_off, y_off))
        return np.array(canvas), None

    def _apply_scale_to_mask(
        self,
        mask_np: np.ndarray,
        缩放到边: str,
        缩放到长度: int,
        倍数取整: str,
        缩放模式: str,
        aspect_ratio: Optional[Tuple[int, int]],
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        h, w = mask_np.shape[:2]
        new_w, new_h = target_size
        current_ratio = w / h
        if aspect_ratio is not None:
            target_ratio = aspect_ratio[0] / aspect_ratio[1]
        else:
            target_ratio = current_ratio
        resample = Image.NEAREST
        pil_mask = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        if 缩放模式 == "填充":
            result = pil_mask.resize((new_w, new_h), resample)
            return np.array(result).astype(np.float32) / 255.0
        if 缩放模式 == "裁剪":
            if abs(current_ratio - target_ratio) < 1e-6:
                cropped = pil_mask
            else:
                if current_ratio > target_ratio:
                    crop_w = int(h * target_ratio)
                    left = max(0, (w - crop_w) // 2)
                    box = (left, 0, left + crop_w, h)
                else:
                    crop_h = int(w / target_ratio)
                    top = max(0, (h - crop_h) // 2)
                    box = (0, top, w, top + crop_h)
                cropped = pil_mask.crop(box)
            result = cropped.resize((new_w, new_h), resample)
            return np.array(result).astype(np.float32) / 255.0
        # 适应：letterbox
        scale = min(new_w / w, new_h / h)
        resized_w = max(1, int(round(w * scale)))
        resized_h = max(1, int(round(h * scale)))
        resized = pil_mask.resize((resized_w, resized_h), resample)
        canvas = Image.new("L", (new_w, new_h), color=0)
        x_off = (new_w - resized_w) // 2
        y_off = (new_h - resized_h) // 2
        canvas.paste(resized, (x_off, y_off))
        return np.array(canvas).astype(np.float32) / 255.0
    
    def crop_person(
        self,
        图像: torch.Tensor,
        遮罩: Optional[torch.Tensor] = None,
        宽高比: str = "原始",
        比例宽: int = 1,
        比例高: int = 1,
        缩放模式: str = "适应",
        采样方法: str = "lanczos",
        倍数取整: str = "16",
        缩放到边: str = "无",
        缩放到长度: int = 1024,
        背景颜色: str = "#000000",
        人物索引: int = 0,
        自动宽高比: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        按人物主体比例缩放主函数。
        
        参数:
            图像: ComfyUI 图像张量 (B, H, W, C)
            遮罩: 输入遮罩（可选）
            宽高比: 宽高比预设选项
            比例宽: 自定义宽度比例
            比例高: 自定义高度比例
            缩放模式: 缩放模式（适应/裁剪/填充）
            采样方法: 采样方法
            倍数取整: 倍数取整值
            缩放到边: 缩放目标边
            缩放到长度: 目标长度
            背景颜色: 背景颜色
            人物索引: 人物索引（按面积排序）
            自动宽高比: 启用后按图像方向自动反转非"原始"的比例
            
        返回:
            裁剪后的图像张量元组
        """
        try:
            # 懒加载检测管理器
            if self.detection_manager is None:
                self.detection_manager = DetectionManager()
                self.detection_manager.set_device("cpu")
                try:
                    self.detection_manager.load_models()
                except ModelLoadError as e:
                    logger.error(f"模型加载失败: {e}")
                    logger.info("由于模型加载失败，返回原图")
                    return (图像,)
            
            # 获取批次大小
            batch_size = 图像.shape[0]
            
            # 处理批次中的每张图像
            processed_images = []
            processed_masks = []
            crop_regions = []
            max_width = 0
            max_height = 0
            
            for i in range(batch_size):
                # 提取单张图像
                single_image_tensor = 图像[i:i+1]
                # 获取单图尺寸用于自动宽高比判断
                try:
                    _probe_np = self.image_processor.tensor_to_numpy(single_image_tensor)
                    _probe_h, _probe_w = _probe_np.shape[:2]
                except Exception:
                    _probe_h, _probe_w = 0, 0
                # 自动宽高比:按图像方向调整(竖图/横图)比例
                eff_宽高比, eff_比例宽, eff_比例高 = _auto_swap_aspect_ratio(
                    宽高比, 比例宽, 比例高, 自动宽高比, _probe_w, _probe_h
                )
                # 解析宽高比
                aspect_ratio = self._parse_aspect_ratio(eff_宽高比, eff_比例宽, eff_比例高)
                
                try:
                    # 处理单张图像，获取裁剪后的图像和裁剪区域
                    cropped, crop_region = self._process_single_image(
                        single_image_tensor,
                        aspect_ratio,
                        人物索引
                    )
                    
                    # 应用缩放和倍数取整
                    cropped_np = self.image_processor.tensor_to_numpy(cropped)
                    processed_np, fallback_mask = self._apply_scale_and_multiple(
                        cropped_np,
                        缩放到边,
                        缩放到长度,
                        倍数取整,
                        采样方法,
                        缩放模式,
                        aspect_ratio,
                        背景颜色
                    )
                    cropped = self.image_processor.numpy_to_tensor(processed_np)
                    # 缓存内容区域遮罩（用于无输入遮罩时）
                    if 'content_region_masks' not in locals():
                        content_region_masks = []
                    if fallback_mask is None:
                        _, h, w, _ = cropped.shape
                        fallback_mask = np.zeros((h, w), dtype=np.float32)
                    content_region_masks.append(torch.from_numpy(fallback_mask))
                    
                    # 记录最大尺寸
                    _, h, w, _ = cropped.shape
                    max_width = max(max_width, w)
                    max_height = max(max_height, h)
                    
                    processed_images.append(cropped)
                    crop_regions.append(crop_region)
                    
                except Exception as e:
                    logger.error(f"处理批次中第 {i} 张图像时出错: {e}")
                    logger.info(f"将原图 {i} 包含在输出批次中")
                    
                    # 失败时包含原图
                    _, h, w, _ = single_image_tensor.shape
                    max_width = max(max_width, w)
                    max_height = max(max_height, h)
                    
                    processed_images.append(single_image_tensor)
                    crop_regions.append(None)
            
            # 处理遮罩
            for i in range(batch_size):
                crop_region = crop_regions[i]
                
                if 遮罩 is not None and i < 遮罩.shape[0] and crop_region is not None:
                    # 裁剪输入的遮罩
                    mask_np = 遮罩[i].cpu().numpy()
                    orig_h, orig_w = mask_np.shape
                    
                    # 裁剪遮罩
                    cropped_mask = mask_np[crop_region.top:crop_region.bottom, 
                                          crop_region.left:crop_region.right]
                    
                    # 应用与图像相同的缩放
                    _, h, w, _ = processed_images[i].shape
                    if cropped_mask.shape != (h, w):
                        # 缩放遮罩到与图像相同的尺寸
                        cropped_mask = self._apply_scale_to_mask(
                            cropped_mask,
                            缩放到边,
                            缩放到长度,
                            倍数取整,
                            缩放模式,
                            aspect_ratio,
                            (w, h)
                        )
                    
                    processed_masks.append(torch.from_numpy(cropped_mask))
                else:
                    _, h, w, _ = processed_images[i].shape
                    if 遮罩 is not None and i < (遮罩.shape[0] if hasattr(遮罩, "shape") else 0):
                        # 有输入遮罩但没有裁剪区域（回退路径）：按相同规则缩放/letterbox
                        mask_np = 遮罩[i].cpu().numpy()
                        scaled_mask = self._apply_scale_to_mask(
                            mask_np,
                            缩放到边,
                            缩放到长度,
                            倍数取整,
                            缩放模式,
                            aspect_ratio,
                            (w, h)
                        )
                        processed_masks.append(torch.from_numpy(scaled_mask))
                    else:
                        # 无输入遮罩：返回内容区域为 1 的遮罩（适应模式），或全 1（裁剪/填充）
                        if 'content_region_masks' in locals() and i < len(content_region_masks):
                            processed_masks.append(content_region_masks[i])
                        else:
                            processed_masks.append(torch.zeros((h, w), dtype=torch.float32))
            
            # 统一批次中所有图像的尺寸
            bg_color = self._parse_hex_color(背景颜色)
            
            if batch_size > 1:
                unified_images = []
                unified_masks = []
                for idx, img_tensor in enumerate(processed_images):
                    # 如果尺寸不匹配，需要调整
                    _, h, w, _ = img_tensor.shape
                    mask_tensor = processed_masks[idx]
                    
                    if h != max_height or w != max_width:
                        # 转换为 numpy 进行调整
                        img_np = self.image_processor.tensor_to_numpy(img_tensor)
                        mask_np = mask_tensor.cpu().numpy()
                        
                        # 创建统一尺寸的画布（使用背景颜色）
                        canvas = np.full((max_height, max_width, 3), bg_color, dtype=np.uint8)
                        mask_canvas = np.zeros((max_height, max_width), dtype=np.float32)
                        
                        # 将图像和遮罩居中放置
                        y_offset = (max_height - h) // 2
                        x_offset = (max_width - w) // 2
                        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img_np
                        mask_canvas[y_offset:y_offset+h, x_offset:x_offset+w] = mask_np
                        
                        # 转回张量
                        img_tensor = self.image_processor.numpy_to_tensor(canvas)
                        mask_tensor = torch.from_numpy(mask_canvas)
                    
                    unified_images.append(img_tensor)
                    unified_masks.append(mask_tensor)
                
                # 合并批次
                result_image = torch.cat(unified_images, dim=0)
                result_mask = torch.stack(unified_masks, dim=0)
            else:
                result_image = processed_images[0]
                result_mask = processed_masks[0].unsqueeze(0)
            
            return (result_image, result_mask)
            
        except Exception as e:
            logger.error(f"crop_person 发生意外错误: {e}")
            logger.info("由于意外错误，返回原图")
            # 返回原图和对应的遮罩
            _, h, w, _ = 图像.shape
            mask = torch.ones((图像.shape[0], h, w), dtype=torch.float32)
            return (图像, mask)
    
    def _process_single_image(
        self,
        image_tensor: torch.Tensor,
        aspect_ratio: Optional[Tuple[int, int]],
        person_index: int
    ) -> Tuple[torch.Tensor, Optional[CropRegion]]:
        """
        处理单张图像的裁剪。
        
        参数:
            image_tensor: 单张图像张量 (1, H, W, C)
            aspect_ratio: 目标宽高比
            person_index: 人物索引
            
        返回:
            裁剪后的图像张量 (1, H', W', C) 和裁剪区域
        """
        # 转换为 numpy 数组
        image_np = self.image_processor.tensor_to_numpy(image_tensor)
        img_height, img_width = image_np.shape[:2]
        image_area = img_width * img_height
        
        # 如果 aspect_ratio 为 None（选择"原始"），使用原图的宽高比
        if aspect_ratio is None:
            # 计算原图的宽高比，简化为最简分数
            from math import gcd
            ratio_gcd = gcd(img_width, img_height)
            aspect_ratio = (img_width // ratio_gcd, img_height // ratio_gcd)
            logger.info(f"使用原图宽高比: {aspect_ratio[0]}:{aspect_ratio[1]}")
        
        # 执行检测
        detection_result = self.detection_manager.detect(image_np)
        
        # 获取主要人脸和身体
        primary_face = detection_result.get_primary_face(person_index)
        primary_body = detection_result.get_primary_body(person_index)
        
        # 情况1: 无检测结果 - 返回原图
        if primary_face is None and primary_body is None:
            logger.info("未检测到人脸或身体，返回原图")
            return image_tensor, None
        
        # 计算裁剪区域（不使用边距，直接最大化人物主体显示）
        crop_region = self.crop_calculator.calculate_crop_region(
            image_size=(img_width, img_height),
            face_box=primary_face,
            body_box=primary_body,
            padding=0,  # 不使用边距
            aspect_ratio=aspect_ratio
        )
        
        # 情况3: 无法满足可见度要求 - 返回原图
        if crop_region is None:
            logger.info("无法满足可见度要求，返回原图")
            return image_tensor, None
        
        # 执行裁剪
        cropped_np = self.image_processor.crop_image(image_np, crop_region)
        
        # 转回张量
        cropped_tensor = self.image_processor.numpy_to_tensor(cropped_np)
        
        return cropped_tensor, crop_region


class PersonCropAdvancedNode(PersonCropNode):
    @classmethod
    def INPUT_TYPES(cls):
        dm = DetectionManager()
        face_opts = dm.list_face_model_options()
        body_opts = dm.list_body_model_options()
        def pick_default(options, prefix):
            for o in options:
                if o.startswith(prefix):
                    return o
            return options[0] if options else prefix
        default_face = pick_default(face_opts, "face_yolov8m.pt")
        default_body = pick_default(body_opts, "person_yolov8m-seg.pt")
        base = super().INPUT_TYPES()
        base["optional"] = {
            **base["optional"],
            "设备": (["cpu", "cuda"], {
                "default": "cpu",
                "tooltip": "强制推理设备，默认CPU以降低显存占用；选择cuda可加速"
            }),
            "脸部模型": (face_opts, {
                "default": default_face,
                "tooltip": "选择人脸检测模型（显示大小为本地或提示值），缺失将自动下载"
            }),
            "人物模型": (body_opts, {
                "default": default_body,
                "tooltip": "选择人物模型（分割/检测），缺失将自动下载"
            }),
        }
        return base
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "遮罩")
    FUNCTION = "crop_person_adv"
    CATEGORY = "图像/变换"
    
    def __init__(self):
        super().__init__()
        self._selected_face = None
        self._selected_body = None
    
    def _extract_filename(self, label: str) -> str:
        if "(" in label:
            return label.split("(", 1)[0].strip()
        return label.strip()
    
    def crop_person_adv(
        self,
        图像: torch.Tensor,
        遮罩: Optional[torch.Tensor] = None,
        宽高比: str = "原始",
        比例宽: int = 1,
        比例高: int = 1,
        缩放模式: str = "适应",
        采样方法: str = "lanczos",
        倍数取整: str = "16",
        缩放到边: str = "无",
        缩放到长度: int = 1024,
        背景颜色: str = "#000000",
        人物索引: int = 0,
        自动宽高比: bool = False,
        设备: str = "cpu",
        脸部模型: str = "face_yolov8s.pt",
        人物模型: str = "person_yolov8s-seg.pt",
    ):
        face_filename = self._extract_filename(脸部模型)
        body_filename = self._extract_filename(人物模型)
        if self.detection_manager is None:
            self.detection_manager = DetectionManager()
            self.detection_manager.set_device(设备)
            self.detection_manager.set_selected_models(face_filename=face_filename, body_filename=body_filename)
            try:
                self.detection_manager.load_models()
            except ModelLoadError:
                return (图像,)
        else:
            need_reload = (
                face_filename != self._selected_face or
                body_filename != self._selected_body
            )
            self.detection_manager.set_device(设备)
            if need_reload:
                self.detection_manager.set_selected_models(face_filename=face_filename, body_filename=body_filename)
                try:
                    self.detection_manager.load_models()
                except ModelLoadError:
                    return (图像,)
        self._selected_face = face_filename
        self._selected_body = body_filename
        result = super().crop_person(
            图像=图像,
            遮罩=遮罩,
            宽高比=宽高比,
            比例宽=比例宽,
            比例高=比例高,
            缩放模式=缩放模式,
            采样方法=采样方法,
            倍数取整=倍数取整,
            缩放到边=缩放到边,
            缩放到长度=缩放到长度,
            背景颜色=背景颜色,
            人物索引=人物索引,
            自动宽高比=自动宽高比,
        )
        if self.detection_manager is not None:
            try:
                self.detection_manager.to_cpu()
            except Exception:
                pass
        return result


class HalfBodyCropNode:
    """
    半身照裁剪节点。
    对远景照片进行智能分析，以人脸为中心裁剪生成标准半身照（胸部或臀部以上区域）。
    当检测到输入图像已为半身照或头像照片时，自动略过裁剪处理。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "裁剪范围": (["头像特写", "胸部以上", "腰部以上", "臀部以上", "膝盖以上", "全身照"], {
                    "default": "臀部以上",
                    "tooltip": "裁剪范围：'头像特写'(头部80-90%)、'胸部以上'(上半身60-70%)、'腰部以上'、'臀部以上'(标准半身)、'膝盖以上'、'全身照'(完整人体40-50%)"
                }),
                "裁剪比例": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "智能裁剪比例：0.0=自动(使用裁剪范围预设)，0.4-0.5=全身照，0.6-0.7=半身照，0.8-0.9=头像特写。值越大裁剪越紧凑"
                }),
                "自动宽高比": ("BOOLEAN", {
                    "default": False,
                    "label_on": "启用",
                    "label_off": "关闭",
                    "tooltip": "开启后,当宽高比不是'原始'时,根据输入图像的横竖方向自动反转比例。例如:选2:3遇横图自动变3:2;选16:9遇竖图自动变9:16;自定义比例同样按宽高方向自动转换"
                }),
                "宽高比": (["原始", "自定义", "1:1", "3:2", "4:3", "16:9", "2:3", "3:4", "9:16"], {
                    "default": "原始",
                    "tooltip": "宽高比：'原始'=保持当前裁剪方式不变；选择比例时对裁剪范围后的图片进行比例裁剪（缩小）以达到比例要求，尽量不把人脸裁剪掉"
                }),
                "自定义宽": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "自定义宽（仅在宽高比选择'自定义'时生效）"
                }),
                "自定义高": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "自定义高（仅在宽高比选择'自定义'时生效）"
                }),
                "缩放模式": (["裁剪", "拉伸", "填充"], {
                    "default": "裁剪",
                    "tooltip": "缩放模式：'裁剪'等比缩放后裁剪到目标尺寸，'拉伸'等比缩放后letterbox填充到目标尺寸，'填充'直接拉伸到目标尺寸。具体行为取决于'缩放到边'和'缩放到长度'"
                }),
                "采样方法": (["lanczos", "nearest", "bilinear", "bicubic"], {
                    "default": "lanczos",
                    "tooltip": "图像缩放时的采样算法。lanczos质量最高但速度较慢，nearest速度最快但质量较低"
                }),
                "倍数取整": (["8", "16", "32", "64", "128", "256", "512", "无"], {
                    "default": "16",
                    "tooltip": "将输出尺寸向上取整到指定倍数（如16的倍数），有助于某些AI模型的处理。选择'无'则不取整"
                }),
                "缩放到边": (["无", "最长边", "最短边", "宽度", "高度", "总像素(千像素)"], {
                    "default": "最长边",
                    "tooltip": "指定缩放目标：'最长边'将长边缩放到目标长度，'最短边'将短边缩放到目标长度，'宽度'/'高度'指定具体边"
                }),
                "缩放到长度": ("INT", {
                    "default": 1344,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "配合'缩放到边'使用，指定目标边的像素长度（或总像素的千像素数）"
                }),
                "背景颜色": ("STRING", {
                    "default": "#000000",
                    "tooltip": "背景填充颜色，使用十六进制颜色代码（如#000000为黑色，#FFFFFF为白色）。仅在'填充'模式下生效"
                }),
                "人物索引": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "当检测到多个人物时，选择要裁剪的人物（0为所有人物的中心区域，1为最大人物，2为第二大，以此类推）。0的含义就是尽可能展现图片中的所有人"
                }),
                "边距": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 500,
                    "step": 10,
                    "tooltip": "头部周围的额外边距（像素），确保头部不被裁剪"
                }),
            },
            "optional": {
                "遮罩": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "遮罩")
    FUNCTION = "crop_half_body"
    CATEGORY = "图像/变换"
    
    def __init__(self):
        """初始化节点，创建检测管理器实例"""
        self.detection_manager = None
    
    def _calculate_half_body_region(
        self,
        image_np: np.ndarray,
        head_box,
        face_box,
        body_box,
        crop_range: str,
        crop_ratio: float,
        padding: int
    ) -> Optional[CropRegion]:
        """
        计算半身照裁剪区域。
        
        使用统一的人体比例参考系统，以人脸为基准（避免头发长短影响）。
        
        参数:
            image_np: 图像 numpy 数组
            head_box: 头部边界框（仅用于顶部边距参考）
            face_box: 人脸边界框（主要计算基准）
            body_box: 人体边界框
            crop_range: 裁剪范围（头像特写/胸部以上/腰部以上/臀部以上/膝盖以上/全身照）
            crop_ratio: 智能裁剪比例（0.0=自动，0.4-0.5=全身，0.6-0.7=半身，0.8-0.9=头像）
            padding: 边距
            
        返回:
            裁剪区域
        """
        img_height, img_width = image_np.shape[:2]
        
        # 必须有人脸才能进行精确裁剪
        if face_box is None:
            logger.warning("未检测到人脸，无法进行精确裁剪")
            # 备用：使用头部框
            if head_box is None:
                return None
            primary_box = head_box
        else:
            primary_box = face_box
        
        # === 统一的人体比例计算系统 ===
        # 以人脸为基准，根据人体标准比例计算各部位位置
        # 标准人体比例（从下巴到脚底约6-6.5个脸长）：
        # - 下巴到胸部：约0.5个脸长
        # - 下巴到腰部：约1.5个脸长
        # - 下巴到臀部：约2.5个脸长
        # - 下巴到膝盖：约4个脸长
        # - 下巴到脚底：约6个脸长
        
        face_height = primary_box.height
        chin_y = primary_box.y2  # 下巴位置（人脸框底部）
        face_center_x = primary_box.center_x
        
        # 根据裁剪范围确定底部位置（统一使用下巴为基准）
        crop_range_bottom_map = {
            "头像特写": chin_y + int(face_height * 0.3),  # 下巴以下一点点
            "胸部以上": chin_y + int(face_height * 0.8),  # 下巴+0.8脸长≈胸部
            "腰部以上": chin_y + int(face_height * 1.5),  # 下巴+1.5脸长≈腰部
            "臀部以上": chin_y + int(face_height * 2.5),  # 下巴+2.5脸长≈臀部
            "膝盖以上": chin_y + int(face_height * 4.0),  # 下巴+4脸长≈膝盖
        }
        # 全身照特殊处理：使用身体检测框
        # 不在map中定义，下面单独处理
        
        # === 全身照特殊处理：基于身体检测框 ===
        if crop_range == "全身照" and body_box is not None:
            # 使用身体框确定顶部和底部（包含完整人体）
            top_y = max(0, body_box.y1 - padding)
            bottom_y = min(body_box.y2 + padding, img_height)
            # 不再直接返回，继续执行宽度计算逻辑（按原图比例裁剪周边多余部分）
        # 无人身体检测框时，回退到比例计算（下面会处理）
        
        # 如果用户指定了裁剪比例，使用比例覆盖预设
        if crop_ratio > 0.0:
            # crop_ratio表示人体在画面中的占比
            # 0.8-0.9 = 头像特写（头部占画面80-90%）
            # 0.6-0.7 = 半身照（上半身占画面60-70%）
            # 0.4-0.5 = 全身照（完整人体占画面40-50%）
            
            # 根据比例计算需要的画面高度
            if body_box is not None:
                # 有身体检测，使用实际身体高度
                body_height = body_box.height
                # 画面高度 = 身体高度 / 比例
                target_frame_height = int(body_height / crop_ratio)
                bottom_y = chin_y + int(target_frame_height * 0.7)  # 下巴以下占70%
            else:
                # 无身体检测，使用人脸比例估算
                # 人脸约占画面的(1 - crop_ratio)部分
                face_portion = 1.0 - crop_ratio
                if face_portion > 0.1:  # 避免除零
                    target_frame_height = int(face_height / face_portion)
                    bottom_y = chin_y + int(target_frame_height * 0.7)
                else:
                    # 比例过大，使用全身照预设
                    bottom_y = chin_y + int(face_height * 6.0)
        else:
            # 使用预设裁剪范围
            if crop_range == "全身照":
                # 全身照无比例且无身体检测框时的回退
                bottom_y = chin_y + int(face_height * 6.0)
            else:
                bottom_y = crop_range_bottom_map.get(crop_range, crop_range_bottom_map["臀部以上"])
        
        # 如果有身体检测框，进行验证和修正
        if body_box is not None:
            # 确保裁剪底部不超过身体底部
            bottom_y = min(bottom_y, body_box.y2)
            # 确保裁剪底部不低于图像底部
            bottom_y = min(bottom_y, img_height)
        else:
            # 无身体检测，确保不超出图像边界
            bottom_y = min(bottom_y, img_height)
        
        # === 计算顶部位置 ===
        # 优先使用头部框确定顶部（包含头发），如果没人头框则从人脸向上推算
        if head_box is not None:
            # 头部检测框已包含整个头部和头发，只需添加边距
            top_y = max(0, head_box.y1 - padding)
        else:
            # 仅有人脸框，从人脸顶部向上推算整个头部（约1.3倍脸高）
            face_top_to_head_top = int(face_height * 0.3)  # 额头到头顶约0.3脸高
            top_y = max(0, primary_box.y1 - face_top_to_head_top - padding)
        
        # === 计算宽度（尽量保持原图比例）===
        # 高度由裁剪范围决定，宽度按原图宽高比计算，裁剪掉周边多余部分
        crop_height = bottom_y - top_y
        if crop_height <= 0:
            logger.warning("裁剪高度无效")
            return None
        
        original_ratio = img_width / img_height
        crop_width = int(crop_height * original_ratio)
        
        # 确保宽度至少能包含人脸（保护人脸不被裁剪）
        if face_box is not None:
            min_face_width = int(face_box.width * 1.5)
            crop_width = max(crop_width, min_face_width)
        
        # 确保宽度不超过图像宽度
        crop_width = min(crop_width, img_width)
        
        # 计算左右边界（以人脸中心为基准）
        left_x = int(face_center_x - crop_width / 2)
        right_x = left_x + crop_width
        
        # 调整以确保不超出图像边界
        if left_x < 0:
            left_x = 0
            right_x = min(crop_width, img_width)
        if right_x > img_width:
            right_x = img_width
            left_x = max(0, right_x - crop_width)
        
        # 创建裁剪区域
        region = CropRegion(
            left=left_x,
            top=top_y,
            right=right_x,
            bottom=bottom_y
        )
        
        # 确保区域有效
        if region.width <= 0 or region.height <= 0:
            logger.warning("计算出的裁剪区域无效")
            return None
        
        return region
    
    def _adjust_region_aspect_ratio(
        self,
        region: CropRegion,
        image_width: int,
        image_height: int,
        aspect_ratio: str,
        custom_w: int = 3,
        custom_h: int = 2,
        face_box = None
    ) -> CropRegion:
        """
        调整裁剪区域以达到指定的宽高比。
        
        通过对裁剪区域进行比例裁剪（缩小）来达到目标宽高比，
        不再向四周扩展原图画面。始终尽量保护人脸不被裁剪掉。
        
        参数:
            region: 原始裁剪区域
            image_width: 原图宽度
            image_height: 原图高度
            aspect_ratio: 宽高比字符串
            custom_w: 自定义宽
            custom_h: 自定义高
            face_box: 人脸框（用于保护人脸不被裁剪）
            
        返回:
            调整后的裁剪区域
        """
        if aspect_ratio == "原始":
            return region
        
        # 解析宽高比
        ratio_map = {
            "1:1": (1, 1),
            "3:2": (3, 2),
            "4:3": (4, 3),
            "16:9": (16, 9),
            "2:3": (2, 3),
            "3:4": (3, 4),
            "9:16": (9, 16),
        }
        
        if aspect_ratio == "自定义":
            if custom_w <= 0 or custom_h <= 0:
                return region
            target_ratio = custom_w / custom_h
        elif aspect_ratio in ratio_map:
            rw, rh = ratio_map[aspect_ratio]
            target_ratio = rw / rh
        else:
            return region
        
        # 计算当前裁剪区域的实际宽高比
        crop_w = region.width
        crop_h = region.height
        if crop_w <= 0 or crop_h <= 0:
            return region
        current_ratio = crop_w / crop_h
        
        # 如果比例已经匹配（或非常接近），直接返回
        if abs(current_ratio - target_ratio) < 0.01:
            return region
        
        new_left = region.left
        new_right = region.right
        new_top = region.top
        new_bottom = region.bottom
        
        # 人脸在裁剪区域内的位置（用于保护人脸不被裁剪）
        face_left = face_right = face_top = face_bottom = None
        if face_box is not None:
            face_left = face_box.x1
            face_right = face_box.x2
            face_top = face_box.y1
            face_bottom = face_box.y2
        
        if current_ratio < target_ratio:
            # 当前比例偏窄（高大于宽），需要减小高度来增大比例
            # 目标高度 = 当前宽度 / 目标比例
            target_height = int(round(crop_w / target_ratio))
            if target_height >= crop_h:
                return region
            height_to_crop = crop_h - target_height
            
            # 保护人脸：如果目标高度小于人脸高度，最多只裁剪到人脸高度
            if face_box is not None:
                face_height = face_bottom - face_top
                if target_height < face_height:
                    target_height = face_height
                    height_to_crop = crop_h - target_height
            
            if height_to_crop > 0:
                # 上下可裁剪量（保护人脸：不裁进人脸区域）
                if face_box is not None:
                    top_can_crop = max(0, face_top - region.top)
                    bottom_can_crop = max(0, region.bottom - face_bottom)
                else:
                    top_can_crop = crop_h // 2
                    bottom_can_crop = crop_h // 2
                
                # 均分裁剪量，不超过各侧可裁剪量
                top_crop = height_to_crop // 2
                bottom_crop = height_to_crop - top_crop
                
                actual_top_crop = min(top_crop, top_can_crop)
                remaining = height_to_crop - actual_top_crop
                actual_bottom_crop = min(remaining, bottom_can_crop)
                
                # 剩余部分继续向另一侧裁剪
                remaining = height_to_crop - actual_top_crop - actual_bottom_crop
                if remaining > 0:
                    if actual_top_crop < top_can_crop:
                        extra = min(remaining, top_can_crop - actual_top_crop)
                        actual_top_crop += extra
                        remaining -= extra
                    if remaining > 0 and actual_bottom_crop < bottom_can_crop:
                        extra = min(remaining, bottom_can_crop - actual_bottom_crop)
                        actual_bottom_crop += extra
                        remaining -= extra
                
                new_top = region.top + actual_top_crop
                new_bottom = region.bottom - actual_bottom_crop
        else:
            # 当前比例偏宽（宽大于高），需要减小宽度来减小比例
            # 目标宽度 = 当前高度 * 目标比例
            target_width = int(round(crop_h * target_ratio))
            if target_width >= crop_w:
                return region
            width_to_crop = crop_w - target_width
            
            # 保护人脸：如果目标宽度小于人脸宽度，最多只裁剪到人脸宽度
            if face_box is not None:
                face_width = face_right - face_left
                if target_width < face_width:
                    target_width = face_width
                    width_to_crop = crop_w - target_width
            
            if width_to_crop > 0:
                # 左右可裁剪量（保护人脸：不裁进人脸区域）
                if face_box is not None:
                    left_can_crop = max(0, face_left - region.left)
                    right_can_crop = max(0, region.right - face_right)
                else:
                    left_can_crop = crop_w // 2
                    right_can_crop = crop_w // 2
                
                # 均分裁剪量，不超过各侧可裁剪量
                left_crop = width_to_crop // 2
                right_crop = width_to_crop - left_crop
                
                actual_left_crop = min(left_crop, left_can_crop)
                remaining = width_to_crop - actual_left_crop
                actual_right_crop = min(remaining, right_can_crop)
                
                # 剩余部分继续向另一侧裁剪
                remaining = width_to_crop - actual_left_crop - actual_right_crop
                if remaining > 0:
                    if actual_left_crop < left_can_crop:
                        extra = min(remaining, left_can_crop - actual_left_crop)
                        actual_left_crop += extra
                        remaining -= extra
                    if remaining > 0 and actual_right_crop < right_can_crop:
                        extra = min(remaining, right_can_crop - actual_right_crop)
                        actual_right_crop += extra
                        remaining -= extra
                
                new_left = region.left + actual_left_crop
                new_right = region.right - actual_right_crop
        
        # 边界保护
        new_left = max(0, new_left)
        new_right = min(image_width, new_right)
        new_top = max(0, new_top)
        new_bottom = min(image_height, new_bottom)
        
        return CropRegion(
            left=new_left,
            top=new_top,
            right=new_right,
            bottom=new_bottom
        )
    
    def _parse_hex_color(self, hex_color: str) -> Tuple[int, int, int]:
        """解析十六进制颜色代码"""
        hex_color = hex_color.strip()
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        if len(hex_color) == 3:
            hex_color = ''.join(c * 2 for c in hex_color)
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b)
        except (ValueError, IndexError):
            return (0, 0, 0)
    
    def _apply_post_scaling(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        缩放模式: str,
        采样方法: str,
        倍数取整: str,
        缩放到边: str,
        缩放到长度: int,
        背景颜色: str,
        宽高比: str = "原始",
        自定义宽: int = 3,
        自定义高: int = 2,
        人脸位置: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对裁剪后的图像应用缩放后处理。
        
        参数:
            image_np: 裁剪后的图像 numpy 数组
            mask_np: 裁剪后的遮罩 numpy 数组
            缩放模式: 缩放模式（裁剪/拉伸/填充）
            采样方法: 采样方法
            倍数取整: 倍数取整
            缩放到边: 缩放目标边
            缩放到长度: 目标长度
            背景颜色: 背景颜色
            宽高比: 宽高比
            自定义宽: 自定义宽
            自定义高: 自定义高
            人脸位置: 人脸在裁剪后图像中的位置 (x1, y1, x2, y2)，用于保护人脸
            
        返回:
            (处理后的图像, 处理后的遮罩)
        """
        h, w = image_np.shape[:2]
        
        # 解析采样方法
        resample_map = {
            "lanczos": Image.LANCZOS,
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC
        }
        resample = resample_map.get(采样方法, Image.LANCZOS)
        
        # 计算目标尺寸
        # 第一步：先按"宽高比"确定目标宽高比（如果指定了）
        target_aspect = None
        target_rw, target_rh = None, None
        if 宽高比 != "原始":
            ratio_map = {
                "1:1": (1, 1),
                "3:2": (3, 2),
                "4:3": (4, 3),
                "16:9": (16, 9),
                "2:3": (2, 3),
                "3:4": (3, 4),
                "9:16": (9, 16),
            }
            if 宽高比 == "自定义" and 自定义宽 > 0 and 自定义高 > 0:
                target_rw, target_rh = 自定义宽, 自定义高
            elif 宽高比 in ratio_map:
                target_rw, target_rh = ratio_map[宽高比]
            if target_rw is not None and target_rh is not None:
                target_aspect = target_rw / target_rh
        
        # 第二步：根据"缩放到边"和"宽高比"综合计算 new_w 和 new_h
        if 缩放到边 == "无":
            # 不缩放：直接使用裁剪后图像的尺寸
            # 但如果指定了宽高比，需要调整尺寸以满足宽高比
            if target_aspect is not None:
                # 保持面积接近，按当前长边等比调整
                if w >= h:
                    # 当前偏宽
                    new_w = w
                    new_h = int(new_w / target_aspect)
                else:
                    # 当前偏高
                    new_h = h
                    new_w = int(new_h * target_aspect)
            else:
                # 无宽高比要求，无缩放要求，直接返回原图
                return image_np, mask_np
        elif 缩放到边 == "最长边":
            # 最长边 = 缩放到长度
            if target_aspect is not None:
                # 有宽高比要求：长边按宽高比决定
                if target_rw >= target_rh:
                    # 横向比例（如16:9），new_w=缩放到长度
                    new_w = 缩放到长度
                    new_h = int(new_w * target_rh / target_rw)
                else:
                    # 纵向比例（如9:16），new_h=缩放到长度
                    new_h = 缩放到长度
                    new_w = int(new_h * target_rw / target_rh)
            else:
                # 无宽高比要求：按原图比例
                if w >= h:
                    new_w = 缩放到长度
                    new_h = int(new_w * h / w)
                else:
                    new_h = 缩放到长度
                    new_w = int(new_h * w / h)
        elif 缩放到边 == "最短边":
            # 最短边 = 缩放到长度
            if target_aspect is not None:
                if target_rw <= target_rh:
                    new_w = 缩放到长度
                    new_h = int(new_w * target_rh / target_rw)
                else:
                    new_h = 缩放到长度
                    new_w = int(new_h * target_rw / target_rh)
            else:
                if w <= h:
                    new_w = 缩放到长度
                    new_h = int(new_w * h / w)
                else:
                    new_h = 缩放到长度
                    new_w = int(new_h * w / h)
        elif 缩放到边 == "宽度":
            new_w = 缩放到长度
            if target_aspect is not None:
                new_h = int(new_w * target_rh / target_rw)
            else:
                new_h = int(new_w * h / w)
        elif 缩放到边 == "高度":
            new_h = 缩放到长度
            if target_aspect is not None:
                new_w = int(new_h * target_rw / target_rh)
            else:
                new_w = int(new_h * w / h)
        elif 缩放到边 == "总像素(千像素)":
            target_pixels = 缩放到长度 * 1000
            if target_aspect is not None:
                new_w = int((target_pixels * target_aspect) ** 0.5)
                new_h = int(target_pixels / new_w)
            else:
                new_w = int((target_pixels * w / h) ** 0.5)
                new_h = int((target_pixels * h / w) ** 0.5)
        else:
            return image_np, mask_np
        
        # 倍数取整
        if 倍数取整 != "无":
            multiple = int(倍数取整)
            new_w = math.ceil(new_w / multiple) * multiple
            new_h = math.ceil(new_h / multiple) * multiple
            new_w = max(new_w, multiple)
            new_h = max(new_h, multiple)
        
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        # 应用缩放
        pil_image = Image.fromarray(image_np)
        pil_mask = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        bg_rgb = self._parse_hex_color(背景颜色)
        
        if 缩放模式 == "填充":
            resized_img = pil_image.resize((new_w, new_h), resample)
            resized_mask = pil_mask.resize((new_w, new_h), Image.NEAREST)
            return np.array(resized_img), np.array(resized_mask).astype(np.float32) / 255.0
        
        if 缩放模式 == "裁剪":
            # 关键修复：以人脸为中心进行缩放和裁剪，保护人脸
            # 1. 计算等比缩放比例（保证覆盖目标尺寸）
            scale = max(new_w / w, new_h / h)
            resized_w = max(1, int(round(w * scale)))
            resized_h = max(1, int(round(h * scale)))
            temp_img = pil_image.resize((resized_w, resized_h), resample)
            temp_mask = pil_mask.resize((resized_w, resized_h), Image.NEAREST)
            
            # 2. 计算裁剪起点：优先以人脸中心为基准
            if 人脸位置 is not None:
                fx1, fy1, fx2, fy2 = 人脸位置
                face_cx = (fx1 + fx2) / 2
                face_cy = (fy1 + fy2) / 2
                # 缩放后人脸中心
                scaled_face_cx = face_cx * scale
                scaled_face_cy = face_cy * scale
                # 计算裁剪框左上和右下
                left = int(scaled_face_cx - new_w / 2)
                top = int(scaled_face_cy - new_h / 2)
                # 边界保护
                left = max(0, min(left, resized_w - new_w))
                top = max(0, min(top, resized_h - new_h))
            else:
                # 无人脸位置时，中心裁剪
                left = (resized_w - new_w) // 2
                top = (resized_h - new_h) // 2
            
            result_img = temp_img.crop((left, top, left + new_w, top + new_h))
            result_mask = temp_mask.crop((left, top, left + new_w, top + new_h))
            return np.array(result_img), np.array(result_mask).astype(np.float32) / 255.0
        
        if 缩放模式 == "拉伸":
            scale = min(new_w / w, new_h / h)
            resized_w = max(1, int(round(w * scale)))
            resized_h = max(1, int(round(h * scale)))
            temp_img = pil_image.resize((resized_w, resized_h), resample)
            temp_mask = pil_mask.resize((resized_w, resized_h), Image.NEAREST)
            canvas_img = Image.new("RGB", (new_w, new_h), bg_rgb)
            canvas_mask = Image.new("L", (new_w, new_h), color=0)
            x_off = (new_w - resized_w) // 2
            y_off = (new_h - resized_h) // 2
            canvas_img.paste(temp_img, (x_off, y_off))
            canvas_mask.paste(temp_mask, (x_off, y_off))
            return np.array(canvas_img), np.array(canvas_mask).astype(np.float32) / 255.0
        
        return image_np, mask_np
    
    def crop_half_body(
        self,
        图像: torch.Tensor,
        裁剪范围: str = "臀部以上",
        裁剪比例: float = 0.0,
        自动宽高比: bool = False,
        宽高比: str = "原始",
        自定义宽: int = 3,
        自定义高: int = 2,
        缩放模式: str = "裁剪",
        采样方法: str = "lanczos",
        倍数取整: str = "16",
        缩放到边: str = "最长边",
        缩放到长度: int = 1344,
        背景颜色: str = "#000000",
        人物索引: int = 0,
        边距: int = 50,
        遮罩: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        半身照裁剪主函数。
        
        参数:
            图像: ComfyUI 图像张量 (B, H, W, C)
            裁剪范围: 头像特写/胸部以上/腰部以上/臀部以上/膝盖以上/全身照
            裁剪比例: 智能裁剪比例（0.0=自动使用预设，0.4-0.5=全身，0.6-0.7=半身，0.8-0.9=头像）
            自动宽高比: 启用后按图像方向自动反转非"原始"的比例
            宽高比: 宽高比（原始/自定义/1:1/3:2/4:3/16:9/2:3/3:4/9:16）
            自定义宽: 自定义宽比（仅在宽高比=自定义时生效）
            自定义高: 自定义高比（仅在宽高比=自定义时生效）
            缩放模式: 缩放模式（裁剪/拉伸/填充）
            采样方法: 采样方法（lanczos/nearest/bilinear/bicubic）
            倍数取整: 倍数取整（8/16/32/64/128/256/512/无）
            缩放到边: 缩放目标边（无/最长边/最短边/宽度/高度/总像素(千像素)）
            缩放到长度: 缩放目标长度
            背景颜色: 背景颜色（仅在'填充'模式下生效）
            人物索引: 人物索引（按面积排序）
            边距: 头部/人脸周围的额外边距
            遮罩: 输入遮罩（可选）
            
        返回:
            裁剪后的图像张量元组
        """
        try:
            # 懒加载检测管理器
            if self.detection_manager is None:
                self.detection_manager = DetectionManager()
                self.detection_manager.set_device("cpu")
                try:
                    self.detection_manager.load_models()
                except ModelLoadError as e:
                    logger.error(f"模型加载失败: {e}")
                    logger.info("由于模型加载失败，返回原图")
                    return (图像,)
            
            # 获取批次大小
            batch_size = 图像.shape[0]
            
            # 处理批次中的每张图像
            result_images = []
            result_masks = []
            
            for i in range(batch_size):
                # 提取单张图像
                img_tensor = 图像[i]
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                img_height, img_width = img_np.shape[:2]
                
                # 自动宽高比:按本图方向调整(竖图/横图)比例
                eff_宽高比, eff_自定义宽, eff_自定义高 = _auto_swap_aspect_ratio(
                    宽高比, 自定义宽, 自定义高, 自动宽高比, img_width, img_height
                )
                
                # 执行头部检测
                head_boxes = self.detection_manager.detect_heads(img_np)
                
                # 执行人脸检测
                face_boxes = self.detection_manager.detect_faces(img_np)
                
                # 执行人体检测
                body_boxes = self.detection_manager.detect_bodies(img_np)
                
                # 按人物索引选择
                # 人物索引说明：
                #   0: 所有人物的合成边界框（尽可能展现图片中的所有人）
                #   1: 面积最大的人物
                #   2: 面积第二大的人物
                #   ...
                # 先合并 face 和 body 为 DetectionResult 统一处理（0=合成框）
                from .models import DetectionResult
                detection_result = DetectionResult(faces=face_boxes, bodies=body_boxes)
                primary_face = detection_result.get_primary_face(人物索引)
                primary_body = detection_result.get_primary_body(人物索引)
                
                # 头部仍然单独处理（因为 DetectionResult 不包含 head_boxes）
                if head_boxes:
                    sorted_heads = sorted(head_boxes, key=lambda h: h.area, reverse=True)
                    if 人物索引 == 0:
                        # 0 索引：合成所有头部边界框
                        hx1 = min(h.x1 for h in head_boxes)
                        hy1 = min(h.y1 for h in head_boxes)
                        hx2 = max(h.x2 for h in head_boxes)
                        hy2 = max(h.y2 for h in head_boxes)
                        avg_conf = sum(h.confidence for h in head_boxes) / len(head_boxes)
                        from .models import BoundingBox
                        primary_head = BoundingBox(x1=hx1, y1=hy1, x2=hx2, y2=hy2, confidence=avg_conf)
                    elif 人物索引 - 1 < len(sorted_heads):
                        primary_head = sorted_heads[人物索引 - 1]
                    else:
                        primary_head = sorted_heads[0]
                else:
                    primary_head = None
                
                # 如果没有检测到头部或人脸，返回原图
                if primary_head is None and primary_face is None:
                    logger.info("未检测到头部或人脸，返回原图")
                    result_images.append(img_tensor)
                    
                    if 遮罩 is not None and i < 遮罩.shape[0]:
                        result_masks.append(遮罩[i])
                    else:
                        result_masks.append(torch.ones((img_height, img_width), dtype=torch.float32))
                    continue
                
                # 计算半身照裁剪区域
                crop_region = self._calculate_half_body_region(
                    img_np, primary_head, primary_face, primary_body, 裁剪范围, 裁剪比例, 边距
                )
                
                if crop_region is None:
                    logger.info("无法计算有效的裁剪区域，返回原图")
                    result_images.append(img_tensor)
                    
                    if 遮罩 is not None and i < 遮罩.shape[0]:
                        result_masks.append(遮罩[i])
                    else:
                        result_masks.append(torch.ones((img_height, img_width), dtype=torch.float32))
                    continue
                
                # 根据宽高比调整裁剪区域
                if eff_宽高比 != "原始":
                    crop_region = self._adjust_region_aspect_ratio(
                        crop_region, img_width, img_height, eff_宽高比, eff_自定义宽, eff_自定义高, primary_face
                    )
                    if crop_region is None or crop_region.width <= 0 or crop_region.height <= 0:
                        logger.warning("宽高比调整后区域无效，使用原裁剪区域")
                        crop_region = self._calculate_half_body_region(
                            img_np, primary_head, primary_face, primary_body, 裁剪范围, 裁剪比例, 边距
                        )
                
                # 执行裁剪
                cropped_np = img_np[crop_region.top:crop_region.bottom, 
                                   crop_region.left:crop_region.right]
                
                # 处理遮罩
                if 遮罩 is not None and i < 遮罩.shape[0]:
                    mask_np = 遮罩[i].cpu().numpy()
                    cropped_mask_np = mask_np[crop_region.top:crop_region.bottom, 
                                              crop_region.left:crop_region.right]
                else:
                    crop_h, crop_w = cropped_np.shape[:2]
                    cropped_mask_np = np.ones((crop_h, crop_w), dtype=np.float32)
                
                # 应用缩放后处理
                # 触发条件：指定了宽高比（非"原始"） 或 需要缩放 或 缩放模式不是"裁剪"
                if eff_宽高比 != "原始" or 缩放到边 != "无" or 缩放模式 != "裁剪":
                    # 计算人脸在裁剪后图像中的相对位置
                    face_pos = None
                    if primary_face is not None:
                        face_x1 = primary_face.x1 - crop_region.left
                        face_y1 = primary_face.y1 - crop_region.top
                        face_x2 = primary_face.x2 - crop_region.left
                        face_y2 = primary_face.y2 - crop_region.top
                        face_pos = (face_x1, face_y1, face_x2, face_y2)
                    
                    cropped_np, cropped_mask_np = self._apply_post_scaling(
                        cropped_np, cropped_mask_np, 缩放模式, 采样方法, 倍数取整, 缩放到边, 缩放到长度, 背景颜色,
                        eff_宽高比, eff_自定义宽, eff_自定义高, face_pos
                    )
                
                # 转回张量
                cropped_tensor = torch.from_numpy(cropped_np.astype(np.float32) / 255.0)
                result_images.append(cropped_tensor)
                result_masks.append(torch.from_numpy(cropped_mask_np))
            
            # 合并批次
            result_image = torch.stack(result_images, dim=0)
            result_mask = torch.stack(result_masks, dim=0)
            
            return (result_image, result_mask)
            
        except Exception as e:
            logger.error(f"crop_half_body 发生意外错误: {e}")
            logger.info("由于意外错误，返回原图")
            _, h, w, _ = 图像.shape
            mask = torch.ones((图像.shape[0], h, w), dtype=torch.float32)
            return (图像, mask)


class HalfBodyCropPreviewNode:
    """
    半身照裁剪预览节点。
    生成裁剪区域可视化预览，帮助用户调整参数。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "裁剪范围": (["头像特写", "胸部以上", "腰部以上", "臀部以上", "膝盖以上", "全身照"], {
                    "default": "臀部以上",
                    "tooltip": "裁剪范围预览"
                }),
                "裁剪比例": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "智能裁剪比例预览"
                }),
                "宽高比": (["原始", "自定义", "1:1", "3:2", "4:3", "16:9", "2:3", "3:4", "9:16"], {
                    "default": "原始",
                    "tooltip": "宽高比预览"
                }),
                "自定义宽": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "自定义宽"
                }),
                "自定义高": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "自定义高"
                }),
                "边距": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 500,
                    "step": 10,
                    "tooltip": "边距预览"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("预览图像", "裁剪信息")
    FUNCTION = "generate_preview"
    CATEGORY = "图像/变换"
    
    def __init__(self):
        self.detection_manager = None
    
    def generate_preview(
        self,
        图像: torch.Tensor,
        裁剪范围: str = "臀部以上",
        裁剪比例: float = 0.0,
        宽高比: str = "原始",
        自定义宽: int = 3,
        自定义高: int = 2,
        边距: int = 50
    ) -> Tuple[torch.Tensor, str]:
        """
        生成裁剪区域可视化预览。
        
        返回:
            带裁剪框标注的预览图像和裁剪信息文本
        """
        try:
            if self.detection_manager is None:
                self.detection_manager = DetectionManager()
                self.detection_manager.set_device("cpu")
                try:
                    self.detection_manager.load_models()
                except ModelLoadError as e:
                    logger.error(f"模型加载失败: {e}")
                    return (图像, "模型加载失败")
            
            batch_size = 图像.shape[0]
            result_images = []
            info_texts = []
            
            for i in range(batch_size):
                img_tensor = 图像[i]
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8).copy()
                img_height, img_width = img_np.shape[:2]
                
                # 执行检测
                head_boxes = self.detection_manager.detect_heads(img_np)
                face_boxes = self.detection_manager.detect_faces(img_np)
                body_boxes = self.detection_manager.detect_bodies(img_np)
                
                primary_head = sorted(head_boxes, key=lambda h: h.area, reverse=True)[0] if head_boxes else None
                primary_face = sorted(face_boxes, key=lambda f: f.area, reverse=True)[0] if face_boxes else None
                primary_body = sorted(body_boxes, key=lambda b: b.area, reverse=True)[0] if body_boxes else None
                
                if primary_head is None and primary_face is None:
                    info_texts.append("未检测到头部或人脸")
                    result_images.append(img_tensor)
                    continue
                
                # 创建临时节点来计算裁剪区域
                temp_node = HalfBodyCropNode()
                temp_node.detection_manager = self.detection_manager
                
                crop_region = temp_node._calculate_half_body_region(
                    img_np, primary_head, primary_face, primary_body, 裁剪范围, 裁剪比例, 边距
                )
                
                if crop_region is None:
                    info_texts.append("无法计算裁剪区域")
                    result_images.append(img_tensor)
                    continue
                
                # 根据宽高比调整裁剪区域
                if 宽高比 != "原始":
                    crop_region = temp_node._adjust_region_aspect_ratio(
                        crop_region, img_width, img_height, 宽高比, 自定义宽, 自定义高, primary_face
                    )
                    if crop_region is None or crop_region.width <= 0 or crop_region.height <= 0:
                        crop_region = temp_node._calculate_half_body_region(
                            img_np, primary_head, primary_face, primary_body, 裁剪范围, 裁剪比例, 边距
                        )
                
                # 在图像上绘制裁剪框（红色）
                preview_img = img_np.copy()
                
                # 添加文字标注
                crop_w = crop_region.right - crop_region.left
                crop_h = crop_region.bottom - crop_region.top
                info_text = f"裁剪范围: {裁剪范围}\n"
                info_text += f"裁剪比例: {裁剪比例:.2f}\n"
                if 宽高比 != "原始":
                    info_text += f"宽高比: {宽高比}\n"
                info_text += f"裁剪尺寸: {crop_w}x{crop_h}\n"
                info_text += f"检测到: "
                if primary_head: info_text += "头部 "
                if primary_face: info_text += "人脸 "
                if primary_body: info_text += "身体"
                
                info_texts.append(info_text)
                
                if HAS_CV2:
                    cv2.rectangle(preview_img, 
                                 (crop_region.left, crop_region.top), 
                                 (crop_region.right, crop_region.bottom), 
                                 (255, 0, 0), 3)
                    
                    # 绘制检测框（绿色=头部，蓝色=身体）
                    if primary_head is not None:
                        cv2.rectangle(preview_img,
                                     (primary_head.x1, primary_head.y1),
                                     (primary_head.x2, primary_head.y2),
                                     (0, 255, 0), 2)
                    if primary_face is not None:
                        cv2.rectangle(preview_img,
                                     (primary_face.x1, primary_face.y1),
                                     (primary_face.x2, primary_face.y2),
                                     (0, 255, 255), 2)
                    if primary_body is not None:
                        cv2.rectangle(preview_img,
                                     (primary_body.x1, primary_body.y1),
                                     (primary_body.x2, primary_body.y2),
                                     (255, 0, 255), 2)
                    
                    # 在图像上添加文字
                    y_pos = crop_region.top - 10 if crop_region.top > 50 else crop_region.bottom + 30
                    cv2.putText(preview_img, f"{裁剪范围} ({crop_w}x{crop_h})", 
                               (crop_region.left + 5, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    # 如果没有cv2，使用numpy绘制简单的框
                    preview_img[crop_region.top:crop_region.top+3, crop_region.left:crop_region.right] = [255, 0, 0]
                    preview_img[crop_region.bottom-3:crop_region.bottom, crop_region.left:crop_region.right] = [255, 0, 0]
                    preview_img[crop_region.top:crop_region.bottom, crop_region.left:crop_region.left+3] = [255, 0, 0]
                    preview_img[crop_region.top:crop_region.bottom, crop_region.right-3:crop_region.right] = [255, 0, 0]
                
                # 转回张量
                preview_tensor = torch.from_numpy(preview_img.astype(np.float32) / 255.0)
                result_images.append(preview_tensor)
            
            result_image = torch.stack(result_images, dim=0)
            info_text = "\n---\n".join(info_texts)
            
            return (result_image, info_text)
            
        except Exception as e:
            logger.error(f"生成预览失败: {e}")
            return (图像, f"生成预览失败: {e}")


# 注册节点
NODE_CLASS_MAPPINGS = {
    "ZYFAutoCropBorder": AutoCropBorderNode,
    "ZYFPersonCrop": PersonCropNode,
    "ZYFPersonCropAdvanced": PersonCropAdvancedNode,
    "ZYFHalfBodyCrop": HalfBodyCropNode,
    "ZYFHalfBodyCropPreview": HalfBodyCropPreviewNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZYFAutoCropBorder": "ZYF自动裁剪边框",
    "ZYFPersonCrop": "ZYF(按人物主体)比例缩放",
    "ZYFPersonCropAdvanced": "ZYF(按人物主体)比例缩放(2)",
    "ZYFHalfBodyCrop": "ZYF(聚焦人体裁剪)",
    "ZYFHalfBodyCropPreview": "ZYF(聚焦人体裁剪)预览",
}
