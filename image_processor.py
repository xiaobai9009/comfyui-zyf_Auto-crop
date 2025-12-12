"""
图像处理器，负责张量转换和裁剪操作。
"""

import numpy as np
import torch
from typing import Union

from .models import CropRegion


class ImageProcessor:
    """
    图像处理器类，处理 ComfyUI 张量格式与 numpy 数组之间的转换，
    以及图像裁剪操作。
    """
    
    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        将 ComfyUI 张量格式转换为 numpy 数组。
        
        ComfyUI 张量格式: (B, H, W, C)，值范围 [0.0, 1.0]
        输出 numpy 格式: (H, W, C)，值范围 [0, 255]，dtype=uint8
        
        参数:
            tensor: ComfyUI 图像张量，形状为 (B, H, W, C)
            
        返回:
            numpy 数组，形状为 (H, W, C)，如果批次大小 > 1，返回第一张图像
            
        注意:
            - 如果输入是批次 (B > 1)，只返回第一张图像
            - 自动将值范围从 [0.0, 1.0] 转换为 [0, 255]
        """
        # 确保输入是 torch.Tensor
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        
        # 检查张量维度
        if tensor.ndim != 4:
            raise ValueError(
                f"Expected 4D tensor (B, H, W, C), got shape {tensor.shape}"
            )
        
        # 获取批次大小
        batch_size = tensor.shape[0]
        
        # 如果是批次，取第一张图像
        if batch_size > 1:
            image_tensor = tensor[0]
        else:
            image_tensor = tensor[0]
        
        # 转换为 numpy 数组
        # 从 GPU 移到 CPU（如果在 GPU 上）
        if image_tensor.is_cuda:
            image_tensor = image_tensor.cpu()
        
        # 转换为 numpy
        image_np = image_tensor.numpy()
        
        # 将值范围从 [0.0, 1.0] 转换为 [0, 255]
        image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)
        
        return image_np
    
    @staticmethod
    def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
        """
        将 numpy 数组转换回 ComfyUI 张量格式。
        
        输入 numpy 格式: (H, W, C)，值范围 [0, 255]
        输出 ComfyUI 张量格式: (1, H, W, C)，值范围 [0.0, 1.0]
        
        参数:
            array: numpy 数组，形状为 (H, W, C)
            
        返回:
            ComfyUI 图像张量，形状为 (1, H, W, C)
            
        注意:
            - 自动添加批次维度 (B=1)
            - 自动将值范围从 [0, 255] 转换为 [0.0, 1.0]
        """
        # 确保输入是 numpy 数组
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(array)}")
        
        # 检查数组维度
        if array.ndim != 3:
            raise ValueError(
                f"Expected 3D array (H, W, C), got shape {array.shape}"
            )
        
        # 确保数据类型正确
        if array.dtype != np.uint8:
            # 如果不是 uint8，尝试转换
            array = array.astype(np.uint8)
        
        # 将值范围从 [0, 255] 转换为 [0.0, 1.0]
        array_float = array.astype(np.float32) / 255.0
        
        # 转换为 torch 张量
        tensor = torch.from_numpy(array_float)
        
        # 添加批次维度: (H, W, C) -> (1, H, W, C)
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    @staticmethod
    def crop_image(image: np.ndarray, region: CropRegion) -> np.ndarray:
        """
        使用 CropRegion 坐标裁剪 numpy 数组图像。
        
        参数:
            image: 输入图像，numpy 数组，形状为 (H, W, C)
            region: 裁剪区域，定义裁剪边界
            
        返回:
            裁剪后的图像，numpy 数组，形状为 (H', W', C)
            
        注意:
            - 裁剪区域坐标应该在图像边界内
            - 如果裁剪区域超出边界，会自动限制
        """
        # 确保输入是 numpy 数组
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
        
        # 检查数组维度
        if image.ndim != 3:
            raise ValueError(
                f"Expected 3D array (H, W, C), got shape {image.shape}"
            )
        
        # 获取图像尺寸
        img_height, img_width = image.shape[:2]
        
        # 限制裁剪区域在图像边界内
        clamped_region = region.clamp_to_image(img_width, img_height)
        
        # 执行裁剪
        # numpy 数组索引: [top:bottom, left:right, :]
        cropped = image[
            clamped_region.top:clamped_region.bottom,
            clamped_region.left:clamped_region.right,
            :
        ]
        
        return cropped
