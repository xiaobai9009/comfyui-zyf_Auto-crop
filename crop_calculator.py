"""
人脸优先裁剪计算器。
实现智能裁剪算法，确保人脸完整可见，同时最大化身体显示。
"""

from typing import Optional, Tuple
from .models import BoundingBox, CropRegion


class CropCalculator:
    """
    裁剪计算器类，实现人脸优先的智能裁剪算法。
    """
    
    def __init__(self, min_face_visibility: float = 0.67):
        """
        初始化裁剪计算器。
        
        参数:
            min_face_visibility: 最小人脸可见度阈值，默认 0.67 (三分之二)
        """
        self.min_face_visibility = min_face_visibility
    
    def calculate_face_visibility(
        self,
        face_box: BoundingBox,
        crop_region: CropRegion
    ) -> float:
        """
        计算人脸在裁剪区域内的可见度比例。
        
        参数:
            face_box: 人脸边界框
            crop_region: 裁剪区域
            
        返回:
            可见度比例 (交集面积 / 人脸面积)，范围 [0.0, 1.0]
        """
        # 将裁剪区域转换为边界框以计算交集
        crop_box = crop_region.to_bounding_box()
        
        # 计算交集面积
        intersection = face_box.intersection_area(crop_box)
        
        # 计算可见度比例
        if face_box.area == 0:
            return 0.0
        
        return intersection / face_box.area
    
    def apply_padding(
        self,
        region: CropRegion,
        padding: int
    ) -> CropRegion:
        """
        在裁剪区域的所有边上应用边距。
        
        参数:
            region: 原始裁剪区域
            padding: 边距像素数
            
        返回:
            应用边距后的新裁剪区域
        """
        return CropRegion(
            left=region.left - padding,
            top=region.top - padding,
            right=region.right + padding,
            bottom=region.bottom + padding
        )
    
    def adjust_to_aspect_ratio(
        self,
        region: CropRegion,
        target_ratio: float,
        face_box: BoundingBox,
        image_size: Tuple[int, int],
        strict: bool = True,
        body_box: Optional[BoundingBox] = None
    ) -> Optional[CropRegion]:
        """
        调整裁剪区域以匹配目标宽高比，同时保持人脸包含并最大化人物主体显示。
        
        参数:
            region: 原始裁剪区域
            target_ratio: 目标宽高比 (宽度/高度)
            face_box: 人脸边界框（必须保持包含）
            image_size: 图像尺寸 (宽度, 高度)
            strict: 是否严格遵守宽高比（True时如果无法满足则返回None）
            body_box: 身体边界框（可选，用于最大化人物主体显示）
            
        返回:
            调整后的裁剪区域，如果strict=True且无法满足则返回None
        """
        img_width, img_height = image_size
        current_ratio = region.aspect_ratio
        
        # 如果当前比例已经接近目标比例，直接返回
        if abs(current_ratio - target_ratio) < 0.01:
            return region
        
        # 计算人脸中心和区域中心
        face_center_x = face_box.center_x
        face_center_y = face_box.center_y
        region_center_x = region.center_x
        region_center_y = region.center_y
        
        # 策略：找到最大的裁剪区域，同时满足宽高比和人脸可见度要求
        best_region = None
        best_area = 0
        best_visibility = 0.0
        
        # 尝试不同的中心点位置（以人脸为中心，逐渐扩展到区域中心）
        for center_weight in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
            # 插值计算中心点（从人脸中心到区域中心）
            center_x = int(face_center_x * center_weight + region_center_x * (1 - center_weight))
            center_y = int(face_center_y * center_weight + region_center_y * (1 - center_weight))
            
            # 尝试不同的尺寸（从小到大，找到最大可行的）
            # 计算可能的最大尺寸
            max_width = img_width
            max_height = img_height
            
            # 根据宽高比，尝试不同的尺寸
            if target_ratio >= 1.0:
                # 横向图像，高度是限制因素
                for height in range(max_height, 100, -50):
                    width = int(height * target_ratio)
                    if width > max_width:
                        continue
                    
                    # 计算裁剪区域
                    left = center_x - width // 2
                    top = center_y - height // 2
                    right = left + width
                    bottom = top + height
                    
                    # 调整以适应图像边界
                    if left < 0:
                        left = 0
                        right = width
                    if right > img_width:
                        right = img_width
                        left = right - width
                    if top < 0:
                        top = 0
                        bottom = height
                    if bottom > img_height:
                        bottom = img_height
                        top = bottom - height
                    
                    # 检查是否超出边界
                    if left < 0 or right > img_width or top < 0 or bottom > img_height:
                        continue
                    
                    # 验证宽高比
                    actual_width = right - left
                    actual_height = bottom - top
                    actual_ratio = actual_width / actual_height if actual_height > 0 else 0
                    if abs(actual_ratio - target_ratio) > 0.01:
                        continue
                    
                    # 创建候选区域
                    candidate = CropRegion(left, top, right, bottom)
                    
                    # 计算人脸可见度
                    visibility = self.calculate_face_visibility(face_box, candidate)
                    
                    # 如果可见度满足要求
                    if visibility >= self.min_face_visibility:
                        area = candidate.area
                        # 选择面积最大的区域
                        if area > best_area:
                            best_area = area
                            best_region = candidate
                            best_visibility = visibility
                        break  # 找到这个中心点的最大区域，继续下一个中心点
            else:
                # 纵向图像，宽度是限制因素
                for width in range(max_width, 100, -50):
                    height = int(width / target_ratio)
                    if height > max_height:
                        continue
                    
                    # 计算裁剪区域
                    left = center_x - width // 2
                    top = center_y - height // 2
                    right = left + width
                    bottom = top + height
                    
                    # 调整以适应图像边界
                    if left < 0:
                        left = 0
                        right = width
                    if right > img_width:
                        right = img_width
                        left = right - width
                    if top < 0:
                        top = 0
                        bottom = height
                    if bottom > img_height:
                        bottom = img_height
                        top = bottom - height
                    
                    # 检查是否超出边界
                    if left < 0 or right > img_width or top < 0 or bottom > img_height:
                        continue
                    
                    # 验证宽高比
                    actual_width = right - left
                    actual_height = bottom - top
                    actual_ratio = actual_width / actual_height if actual_height > 0 else 0
                    if abs(actual_ratio - target_ratio) > 0.01:
                        continue
                    
                    # 创建候选区域
                    candidate = CropRegion(left, top, right, bottom)
                    
                    # 计算人脸可见度
                    visibility = self.calculate_face_visibility(face_box, candidate)
                    
                    # 如果可见度满足要求
                    if visibility >= self.min_face_visibility:
                        area = candidate.area
                        # 选择面积最大的区域
                        if area > best_area:
                            best_area = area
                            best_region = candidate
                            best_visibility = visibility
                        break  # 找到这个中心点的最大区域，继续下一个中心点
        
        # 如果是严格模式，检查是否满足最小可见度要求
        if strict and best_visibility < self.min_face_visibility:
            return None
        
        return best_region
    
    def maximize_body_visibility(
        self,
        crop_region: CropRegion,
        face_box: BoundingBox,
        body_box: BoundingBox,
        image_size: Tuple[int, int],
        aspect_ratio: Optional[float] = None
    ) -> CropRegion:
        """
        在保持人脸完整的前提下，最大化身体可见度。
        当无法包含全身时，优先包含上半身。
        
        参数:
            crop_region: 当前裁剪区域
            face_box: 人脸边界框
            body_box: 身体边界框
            image_size: 图像尺寸 (宽度, 高度)
            aspect_ratio: 目标宽高比（如果指定）
            
        返回:
            优化后的裁剪区域
        """
        img_width, img_height = image_size
        
        # 如果已经完全包含身体，直接返回
        if crop_region.contains_box(body_box):
            return crop_region
        
        # 尝试扩展裁剪区域以包含更多身体
        new_left = min(crop_region.left, body_box.x1)
        new_top = min(crop_region.top, body_box.y1)
        new_right = max(crop_region.right, body_box.x2)
        new_bottom = max(crop_region.bottom, body_box.y2)
        
        # 如果指定了宽高比，需要保持比例
        if aspect_ratio is not None:
            expanded = CropRegion(new_left, new_top, new_right, new_bottom)
            expanded = self.adjust_to_aspect_ratio(
                expanded, aspect_ratio, face_box, image_size
            )
            
            # 检查扩展后是否仍然包含人脸
            if expanded.contains_box(face_box):
                return expanded.clamp_to_image(img_width, img_height)
        
        # 如果无法包含全身，优先包含上半身
        # 上半身定义为从人脸底部到身体中点
        body_mid_y = body_box.y1 + body_box.height // 2
        
        # 尝试包含上半身
        upper_body_bottom = max(face_box.y2, body_mid_y)
        new_bottom = min(new_bottom, upper_body_bottom)
        
        # 如果指定了宽高比，调整以保持比例
        if aspect_ratio is not None:
            new_height = new_bottom - new_top
            new_width = int(new_height * aspect_ratio)
            
            # 以人脸为中心调整宽度
            face_center_x = face_box.center_x
            new_left = int(face_center_x - new_width / 2)
            new_right = new_left + new_width
            
            # 确保人脸包含
            if new_left > face_box.x1:
                new_left = face_box.x1
                new_right = new_left + new_width
            if new_right < face_box.x2:
                new_right = face_box.x2
                new_left = new_right - new_width
        
        result = CropRegion(new_left, new_top, new_right, new_bottom)
        return result.clamp_to_image(img_width, img_height)
    
    def calculate_crop_region(
        self,
        image_size: Tuple[int, int],
        face_box: Optional[BoundingBox],
        body_box: Optional[BoundingBox],
        padding: int,
        aspect_ratio: Optional[Tuple[int, int]] = None
    ) -> Optional[CropRegion]:
        """
        计算最优裁剪区域。
        实现人脸优先策略：确保人脸完整可见，同时最大化身体显示。
        
        参数:
            image_size: 图像尺寸 (宽度, 高度)
            face_box: 人脸边界框（可选）
            body_box: 身体边界框（可选）
            padding: 边距像素数
            aspect_ratio: 目标宽高比 (宽度, 高度)，可选
            
        返回:
            计算得到的裁剪区域，如果无法满足可见度要求则返回 None
        """
        img_width, img_height = image_size
        
        # 情况1: 无人脸无身体 - 返回 None（调用者应返回原图）
        if face_box is None and body_box is None:
            return None
        
        # 情况2: 仅有身体无人脸 - 以身体中心裁剪
        if face_box is None and body_box is not None:
            return self._calculate_body_only_crop(
                body_box, image_size, padding, aspect_ratio
            )
        
        # 情况3: 有人脸（可能有身体）- 人脸优先策略
        if face_box is not None:
            return self._calculate_face_priority_crop(
                face_box, body_box, image_size, padding, aspect_ratio
            )
        
        return None
    
    def _calculate_body_only_crop(
        self,
        body_box: BoundingBox,
        image_size: Tuple[int, int],
        padding: int,
        aspect_ratio: Optional[Tuple[int, int]]
    ) -> CropRegion:
        """
        计算仅有身体时的裁剪区域（以身体中心为基准）。
        
        参数:
            body_box: 身体边界框
            image_size: 图像尺寸
            padding: 边距
            aspect_ratio: 目标宽高比
            
        返回:
            裁剪区域
        """
        img_width, img_height = image_size
        
        # 以身体为基础建立初始区域
        initial_region = CropRegion(
            left=body_box.x1,
            top=body_box.y1,
            right=body_box.x2,
            bottom=body_box.y2
        )
        
        # 应用边距
        region_with_padding = self.apply_padding(initial_region, padding)
        
        # 如果指定了宽高比，调整区域
        if aspect_ratio is not None:
            target_ratio = aspect_ratio[0] / aspect_ratio[1]
            
            # 计算需要的尺寸
            if region_with_padding.aspect_ratio < target_ratio:
                # 需要增加宽度
                new_width = int(region_with_padding.height * target_ratio)
                new_height = region_with_padding.height
            else:
                # 需要增加高度
                new_width = region_with_padding.width
                new_height = int(region_with_padding.width / target_ratio)
            
            # 以身体中心为基准
            center_x = body_box.center_x
            center_y = body_box.center_y
            
            region_with_padding = CropRegion(
                left=int(center_x - new_width / 2),
                top=int(center_y - new_height / 2),
                right=int(center_x + new_width / 2),
                bottom=int(center_y + new_height / 2)
            )
        
        # 限制在图像边界内
        return region_with_padding.clamp_to_image(img_width, img_height)
    
    def _calculate_face_priority_crop(
        self,
        face_box: BoundingBox,
        body_box: Optional[BoundingBox],
        image_size: Tuple[int, int],
        padding: int,
        aspect_ratio: Optional[Tuple[int, int]]
    ) -> Optional[CropRegion]:
        """
        计算人脸优先的裁剪区域。
        
        参数:
            face_box: 人脸边界框
            body_box: 身体边界框（可选）
            image_size: 图像尺寸
            padding: 边距
            aspect_ratio: 目标宽高比
            
        返回:
            裁剪区域，如果无法满足可见度要求则返回 None
        """
        img_width, img_height = image_size
        
        # Step 1: 如果有身体信息，以人脸和身体的并集为基础；否则只用人脸
        if body_box is not None:
            # 计算人脸和身体的并集区域
            initial_region = CropRegion(
                left=min(face_box.x1, body_box.x1),
                top=min(face_box.y1, body_box.y1),
                right=max(face_box.x2, body_box.x2),
                bottom=max(face_box.y2, body_box.y2)
            )
        else:
            # 只有人脸，以人脸为基础
            initial_region = CropRegion(
                left=face_box.x1,
                top=face_box.y1,
                right=face_box.x2,
                bottom=face_box.y2
            )
        
        # Step 2: 应用边距
        region_with_padding = self.apply_padding(initial_region, padding)
        
        # Step 3: 如果指定了宽高比，使用严格的宽高比裁剪
        if aspect_ratio is not None:
            target_ratio = aspect_ratio[0] / aspect_ratio[1]
            strict_region = self.adjust_to_aspect_ratio(
                region_with_padding, target_ratio, face_box, image_size, 
                strict=True, body_box=body_box
            )
            
            # 如果严格模式成功，返回结果
            if strict_region is not None:
                return strict_region
            
            # 如果严格模式失败（无法保证人脸三分之二可见），回退到灵活模式
            # 这种情况下不强制宽高比，优先保证人脸可见度
        
        # Step 4: 限制在图像边界内
        final_region = region_with_padding.clamp_to_image(img_width, img_height)
        
        return final_region
