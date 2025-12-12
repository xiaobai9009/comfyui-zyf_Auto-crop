"""
检测管理器，负责加载和管理 YOLOv8 模型，执行人脸和人体检测。
"""

import os
import logging
from typing import List, Optional
import numpy as np
import requests
import torch
from pathlib import Path

from .models import BoundingBox, DetectionResult


# 配置日志
logger = logging.getLogger(__name__)

# 可选模型列表
FACE_MODEL_CANDIDATES = [
    "face_yolov8m.pt",
    "face_yolov8n.pt",
    "face_yolov8n_v2.pt",
    "face_yolov8s.pt",
]

BODY_MODEL_CANDIDATES = [
    "person_yolov8m-seg.pt",
    "person_yolov8n-seg.pt",
    "person_yolov8s-seg.pt",
]

# 默认下载源（优先国内镜像）
HF_BASE_MIRROR = "https://hf-mirror.com/ashllay/YOLO_Models/resolve/main/"
HF_BASE_OFFICIAL = "https://huggingface.co/ashllay/YOLO_Models/resolve/main/"
ENABLE_REMOTE_SIZE = False
SIZE_HINTS = {
    "face_yolov8m.pt": 50 * 1024 * 1024,
    "face_yolov8n.pt": 10 * 1024 * 1024,
    "face_yolov8n_v2.pt": 12 * 1024 * 1024,
    "face_yolov8s.pt": 20 * 1024 * 1024,
    "person_yolov8m-seg.pt": 50 * 1024 * 1024,
    "person_yolov8n-seg.pt": 20 * 1024 * 1024,
    "person_yolov8s-seg.pt": 30 * 1024 * 1024,
}


class ModelLoadError(Exception):
    """模型加载失败异常"""
    def __init__(self, model_path: str, reason: str):
        self.model_path = model_path
        self.reason = reason
        super().__init__(f"Failed to load model from {model_path}: {reason}")


class DetectionManager:
    """
    检测管理器类，负责 YOLOv8 模型的加载和推理。
    """
    
    def __init__(self, comfyui_root: Optional[str] = None):
        """
        初始化检测管理器。
        
        参数:
            comfyui_root: ComfyUI 根目录路径，如果为 None 则自动检测
        """
        self.comfyui_root = comfyui_root or self._find_comfyui_root()
        self.face_model = None
        self.body_model = None
        self.face_model_filename = "face_yolov8m.pt"
        self.body_model_filename = "person_yolov8m-seg.pt"
        self._device = "cpu"
        
        # 解析模型路径
        self.face_model_path = self._resolve_model_path(
            f"models/ultralytics/bbox/{self.face_model_filename}"
        )
        self.body_model_path = self._resolve_model_path(
            f"models/ultralytics/segm/{self.body_model_filename}"
        )
    
    def set_device(self, device: Optional[str] = None):
        if device == "cpu":
            self._device = "cpu"
        elif device == "cuda":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _find_comfyui_root(self) -> str:
        """
        自动查找 ComfyUI 根目录。
        从当前文件向上查找，直到找到包含 'models' 目录的路径。
        
        返回:
            ComfyUI 根目录路径
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 向上查找最多 5 层
        for _ in range(5):
            models_dir = os.path.join(current_dir, "models")
            if os.path.exists(models_dir) and os.path.isdir(models_dir):
                return current_dir
            
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # 已到达根目录
                break
            current_dir = parent_dir
        
        # 如果找不到，返回当前工作目录
        return os.getcwd()
    
    def _resolve_model_path(self, relative_path: str) -> str:
        """
        解析模型的完整路径。
        
        参数:
            relative_path: 相对于 ComfyUI 根目录的路径
            
        返回:
            模型文件的完整路径
        """
        return os.path.join(self.comfyui_root, relative_path)
    
    def set_selected_models(self, face_filename: Optional[str] = None, body_filename: Optional[str] = None):
        """
        设置待加载的人脸和人体模型文件名，并更新目标路径。
        """
        if face_filename:
            self.face_model_filename = face_filename
        if body_filename:
            self.body_model_filename = body_filename
        self.face_model_path = self._resolve_model_path(f"models/ultralytics/bbox/{self.face_model_filename}")
        self.body_model_path = self._resolve_model_path(f"models/ultralytics/segm/{self.body_model_filename}")

    def _download_model_dynamic(self, filename: str, target_path: str, model_type: str) -> bool:
        """
        从 Hugging Face 下载模型文件。
        优先使用国内镜像 hf-mirror.com，失败则使用 huggingface.co
        
        参数:
            filename: 要下载的文件名
            target_path: 本地保存路径
            model_type: 模型类型 ("face" 或 "body")，用于日志
            
        返回:
            下载成功返回 True，否则返回 False
        """
        # 确保目录存在
        model_dir = os.path.dirname(target_path)
        os.makedirs(model_dir, exist_ok=True)
        
        # 构造候选远程路径（有的仓库可能放在根或子目录）
        candidate_remote_paths = [
            filename,
            f"bbox/{filename}",
            f"segm/{filename}",
        ]
        urls = []
        for p in candidate_remote_paths:
            urls.append(HF_BASE_MIRROR + p)
            urls.append(HF_BASE_OFFICIAL + p)
        
        for url in urls:
            source_name = "hf-mirror.com" if "hf-mirror.com" in url else "huggingface.co"
            try:
                logger.info(f"正在从 {source_name} 下载 {filename}...")
                logger.info(f"下载地址: {url}")
                
                # 下载文件
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                # 获取文件大小
                total_size = int(response.headers.get('content-length', 0))
                
                # 写入文件
                with open(target_path, 'wb') as f:
                    if total_size == 0:
                        f.write(response.content)
                    else:
                        downloaded = 0
                        chunk_size = 8192
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                # 显示下载进度
                                progress = (downloaded / total_size) * 100
                                if downloaded % (chunk_size * 100) == 0:  # 每 800KB 显示一次
                                    logger.info(f"下载进度: {progress:.1f}% ({downloaded}/{total_size} bytes)")
                
                logger.info(f"模型下载成功: {target_path}")
                return True
                
            except requests.exceptions.Timeout:
                logger.warning(f"从 {source_name} 下载超时")
            except requests.exceptions.RequestException as e:
                logger.warning(f"从 {source_name} 下载失败: {str(e)}")
            except Exception as e:
                logger.warning(f"下载过程中发生错误: {str(e)}")
        
        logger.error(f"所有下载源均失败，无法下载 {filename}")
        return False

    
    def load_models(self) -> bool:
        """
        加载人脸和人体检测模型。
        
        返回:
            如果模型加载成功返回 True，否则返回 False
            
        异常:
            ModelLoadError: 当模型文件不存在或加载失败时抛出
        """
        try:
            # 尝试导入 ultralytics
            try:
                from ultralytics import YOLO
            except ImportError:
                error_msg = (
                    "ultralytics library not found. "
                    "Please install it with: pip install ultralytics"
                )
                logger.error(error_msg)
                raise ModelLoadError("ultralytics", error_msg)
            
            # 检查并下载人脸模型
            if not os.path.exists(self.face_model_path):
                logger.info(f"人脸检测模型未找到: {self.face_model_path}")
                logger.info("尝试自动下载模型...")
                if not self._download_model_dynamic(self.face_model_filename, self.face_model_path, "face"):
                    error_msg = (
                        f"人脸检测模型下载失败。\n"
                        f"请手动下载模型并放置到: {self.face_model_path}"
                    )
                    logger.error(error_msg)
                    raise ModelLoadError(self.face_model_path, "下载失败")
            
            # 检查并下载身体模型
            if not os.path.exists(self.body_model_path):
                logger.info(f"身体检测模型未找到: {self.body_model_path}")
                logger.info("尝试自动下载模型...")
                if not self._download_model_dynamic(self.body_model_filename, self.body_model_path, "body"):
                    error_msg = (
                        f"身体检测模型下载失败。\n"
                        f"请手动下载模型并放置到: {self.body_model_path}"
                    )
                    logger.error(error_msg)
                    raise ModelLoadError(self.body_model_path, "下载失败")
            
            # 加载人脸检测模型
            try:
                logger.info(f"Loading face detection model from: {self.face_model_path}")
                self.face_model = YOLO(self.face_model_path)
                logger.info("Face detection model loaded successfully")
            except Exception as e:
                error_msg = f"Failed to load face model: {str(e)}"
                logger.error(error_msg)
                raise ModelLoadError(self.face_model_path, error_msg)
            
            # 加载人体分割模型
            try:
                logger.info(f"Loading body detection model from: {self.body_model_path}")
                self.body_model = YOLO(self.body_model_path)
                logger.info("Body detection model loaded successfully")
            except Exception as e:
                error_msg = f"Failed to load body model: {str(e)}"
                logger.error(error_msg)
                raise ModelLoadError(self.body_model_path, error_msg)
            
            return True
            
        except ModelLoadError:
            # 重新抛出 ModelLoadError
            raise
        except Exception as e:
            # 捕获其他未预期的异常
            error_msg = f"Unexpected error during model loading: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError("unknown", error_msg)

    def _format_size(self, size_bytes: int) -> str:
        try:
            if size_bytes is None or size_bytes <= 0:
                return ""
            gb = size_bytes / (1024 ** 3)
            mb = size_bytes / (1024 ** 2)
            if gb >= 1:
                return f"{gb:.2f}GB"
            return f"{mb:.0f}MB"
        except Exception:
            return ""

    def _get_local_file_size(self, path: str) -> Optional[int]:
        try:
            if os.path.exists(path):
                return os.path.getsize(path)
        except Exception:
            return None
        return None

    def _fetch_remote_size(self, filename: str) -> Optional[int]:
        if not ENABLE_REMOTE_SIZE:
            return None
        candidate_remote_paths = [
            filename,
            f"bbox/{filename}",
            f"segm/{filename}",
        ]
        for p in candidate_remote_paths:
            for base in (HF_BASE_MIRROR, HF_BASE_OFFICIAL):
                url = base + p
                try:
                    resp = requests.head(url, timeout=10)
                    if resp.status_code == 200:
                        v = resp.headers.get("content-length")
                        if v is not None:
                            return int(v)
                except Exception:
                    pass
        return None

    def list_face_model_options(self) -> List[str]:
        options = []
        for name in FACE_MODEL_CANDIDATES:
            path = self._resolve_model_path(f"models/ultralytics/bbox/{name}")
            size = self._get_local_file_size(path)
            if size is None:
                size = SIZE_HINTS.get(name)
            label = name
            size_str = self._format_size(size)
            if size_str:
                label = f"{name} ({size_str})"
            options.append(label)
        return options

    def list_body_model_options(self) -> List[str]:
        options = []
        for name in BODY_MODEL_CANDIDATES:
            path = self._resolve_model_path(f"models/ultralytics/segm/{name}")
            size = self._get_local_file_size(path)
            if size is None:
                size = SIZE_HINTS.get(name)
            label = name
            size_str = self._format_size(size)
            if size_str:
                label = f"{name} ({size_str})"
            options.append(label)
        return options
    
    def to_cpu(self):
        try:
            if self.face_model is not None and hasattr(self.face_model, "model"):
                self.face_model.model.to("cpu")
            if self.body_model is not None and hasattr(self.body_model, "model"):
                self.body_model.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    
    def _ensure_model_device(self, yolo_model, desired_device: str):
        try:
            if yolo_model is None:
                return
            if not hasattr(yolo_model, "model"):
                return
            current_device = next(yolo_model.model.parameters()).device
            target = torch.device(desired_device if desired_device in ("cpu", "cuda") else "cpu")
            if current_device.type != target.type:
                yolo_model.model.to(target)
            # 同步 YOLO 对象上的设备标记与预测器
            try:
                yolo_model.device = target
            except Exception:
                pass
            try:
                if hasattr(yolo_model, "predictor") and yolo_model.predictor is not None:
                    if hasattr(yolo_model.predictor, "model") and yolo_model.predictor.model is not None:
                        yolo_model.predictor.model.to(target)
                    yolo_model.predictor.device = target
                    if hasattr(yolo_model.predictor, "args") and hasattr(yolo_model.predictor.args, "device"):
                        yolo_model.predictor.args.device = str(target)
            except Exception:
                pass
        except Exception:
            pass
    
    def prepare_for_inference(self):
        self._ensure_model_device(self.face_model, self._device)
        self._ensure_model_device(self.body_model, self._device)
    
    def detect_faces(self, image: np.ndarray) -> List[BoundingBox]:
        """
        在输入图像上运行人脸检测模型。
        
        参数:
            image: 输入图像，numpy 数组格式 (H, W, C)，值范围 [0, 255]
            
        返回:
            检测到的人脸边界框列表
            
        异常:
            RuntimeError: 当模型未加载时抛出
        """
        if self.face_model is None:
            error_msg = "Face detection model not loaded. Call load_models() first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            self._ensure_model_device(self.face_model, self._device)
            with torch.no_grad():
                results = self.face_model(image, verbose=False, device=self._device)
            
            # 转换结果为 BoundingBox 列表
            face_boxes = []
            
            if len(results) > 0:
                result = results[0]  # 获取第一个结果（单张图像）
                
                # 检查是否有检测结果
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    # 遍历每个检测框
                    for box in boxes:
                        # 获取边界框坐标 (x1, y1, x2, y2)
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # 获取置信度
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # 创建 BoundingBox 对象
                        face_box = BoundingBox(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            confidence=confidence
                        )
                        face_boxes.append(face_box)
            
            logger.info(f"Detected {len(face_boxes)} face(s)")
            return face_boxes
            
        except Exception as e:
            error_msg = f"Error during face detection: {str(e)}"
            logger.error(error_msg)
            # 返回空列表而不是抛出异常，以实现优雅降级
            return []

    
    def detect_bodies(self, image: np.ndarray) -> List[BoundingBox]:
        """
        在输入图像上运行人体分割模型，从分割结果中提取边界框。
        
        参数:
            image: 输入图像，numpy 数组格式 (H, W, C)，值范围 [0, 255]
            
        返回:
            检测到的人体边界框列表
            
        异常:
            RuntimeError: 当模型未加载时抛出
        """
        if self.body_model is None:
            error_msg = "Body detection model not loaded. Call load_models() first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            self._ensure_model_device(self.body_model, self._device)
            with torch.no_grad():
                results = self.body_model(image, verbose=False, device=self._device)
            
            # 转换结果为 BoundingBox 列表
            body_boxes = []
            
            if len(results) > 0:
                result = results[0]  # 获取第一个结果（单张图像）
                
                # 检查是否有检测结果
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    # 遍历每个检测框
                    for box in boxes:
                        try:
                            cls_id = int(box.cls[0].cpu().numpy())
                            name = None
                            if hasattr(result, "names") and isinstance(result.names, dict):
                                name = result.names.get(cls_id)
                            if name is not None and name != "person":
                                continue
                        except Exception:
                            pass
                        # 获取边界框坐标 (x1, y1, x2, y2)
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # 获取置信度
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # 创建 BoundingBox 对象
                        body_box = BoundingBox(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            confidence=confidence
                        )
                        body_boxes.append(body_box)
            
            logger.info(f"Detected {len(body_boxes)} body/bodies")
            return body_boxes
            
        except Exception as e:
            error_msg = f"Error during body detection: {str(e)}"
            logger.error(error_msg)
            # 返回空列表而不是抛出异常，以实现优雅降级
            return []
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        在图像上同时运行人脸和人体检测。
        
        参数:
            image: 输入图像，numpy 数组格式 (H, W, C)，值范围 [0, 255]
            
        返回:
            包含人脸和人体检测结果的 DetectionResult 对象
        """
        faces = self.detect_faces(image)
        bodies = self.detect_bodies(image)
        
        return DetectionResult(faces=faces, bodies=bodies)
