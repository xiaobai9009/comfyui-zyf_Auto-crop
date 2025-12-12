import torch
import numpy as np
from PIL import Image
import logging
import math
from typing import Optional, Tuple

from .models import BoundingBox, CropRegion, DetectionResult
from .detection_manager import DetectionManager, ModelLoadError
from .crop_calculator import CropCalculator
from .image_processor import ImageProcessor

# 配置日志
logger = logging.getLogger(__name__)


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
                "遮罩": ("MASK",),
                "宽高比": (cls.ASPECT_RATIO_OPTIONS, {
                    "default": "原始",
                    "tooltip": "目标宽高比。选择'原始'保持检测到的人物主体比例，或选择预设比例（如3:4适合竖屏，16:9适合横屏）"
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
                    "tooltip": "当检测到多个人物时，选择要裁剪的人物（0为最大的人物，1为第二大，以此类推）"
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
        人物索引: int = 0
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
            
            # 解析宽高比
            aspect_ratio = self._parse_aspect_ratio(宽高比, 比例宽, 比例高)
            
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
        primary_body = detection_result.get_primary_body()
        
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
        设备: str = "cpu",
        脸部模型: str = "face_yolov8s.pt",
        人物模型: str = "person_yolov8s-seg.pt",
    ):
        face_filename = self._extract_filename(脸部模型)
        body_filename = self._extract_filename(人物模型)
        if self.detection_manager is None:
            self.detection_manager = DetectionManager()
            self.detection_manager.set_device(设备)
            self.detection_manager.set_selected_models(face_filename, body_filename)
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
                self.detection_manager.set_selected_models(face_filename, body_filename)
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
        )
        if self.detection_manager is not None:
            try:
                self.detection_manager.to_cpu()
            except Exception:
                pass
        return result

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ZYFAutoCropBorder": AutoCropBorderNode,
    "ZYFPersonCrop": PersonCropNode,
    "ZYFPersonCropAdvanced": PersonCropAdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZYFAutoCropBorder": "ZYF自动裁剪边框",
    "ZYFPersonCrop": "ZYF(按人物主体)比例缩放",
    "ZYFPersonCropAdvanced": "ZYF(按人物主体)比例缩放(2)",
}
