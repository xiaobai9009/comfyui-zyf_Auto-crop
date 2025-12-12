"""
人物裁剪节点的数据模型。
包含 BoundingBox、CropRegion 和 DetectionResult 数据类。
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BoundingBox:
    """
    边界框数据类，用于表示人脸或人体检测结果。
    格式为 (x1, y1, x2, y2) 的坐标。
    """
    x1: int  # 左上角 x 坐标
    y1: int  # 左上角 y 坐标
    x2: int  # 右下角 x 坐标
    y2: int  # 右下角 y 坐标
    confidence: float  # 检测置信度

    @property
    def center_x(self) -> float:
        """返回边界框中心点的 x 坐标"""
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        """返回边界框中心点的 y 坐标"""
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> int:
        """返回边界框宽度"""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """返回边界框高度"""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """返回边界框面积"""
        return self.width * self.height

    def intersection_area(self, other: 'BoundingBox') -> int:
        """
        计算与另一个边界框的交集面积。
        
        参数:
            other: 另一个边界框
            
        返回:
            交集面积，如果没有交集则返回 0
        """
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x1 < x2 and y1 < y2:
            return (x2 - x1) * (y2 - y1)
        return 0


@dataclass
class CropRegion:
    """
    裁剪区域数据类，定义最终裁剪的边界坐标。
    """
    left: int    # 左边界
    top: int     # 上边界
    right: int   # 右边界
    bottom: int  # 下边界

    @property
    def width(self) -> int:
        """返回裁剪区域宽度"""
        return self.right - self.left

    @property
    def height(self) -> int:
        """返回裁剪区域高度"""
        return self.bottom - self.top

    @property
    def aspect_ratio(self) -> float:
        """返回裁剪区域宽高比"""
        return self.width / self.height if self.height > 0 else 0

    @property
    def center_x(self) -> float:
        """返回裁剪区域中心点的 x 坐标"""
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        """返回裁剪区域中心点的 y 坐标"""
        return (self.top + self.bottom) / 2

    @property
    def area(self) -> int:
        """返回裁剪区域面积"""
        return self.width * self.height

    def contains_box(self, box: BoundingBox) -> bool:
        """
        检查是否完全包含指定边界框。
        
        参数:
            box: 要检查的边界框
            
        返回:
            如果完全包含则返回 True，否则返回 False
        """
        return (self.left <= box.x1 and 
                self.right >= box.x2 and
                self.top <= box.y1 and 
                self.bottom >= box.y2)

    def to_bounding_box(self) -> BoundingBox:
        """
        转换为 BoundingBox 用于计算交集。
        
        返回:
            对应的 BoundingBox 对象
        """
        return BoundingBox(self.left, self.top, self.right, self.bottom, 1.0)

    def clamp_to_image(self, img_width: int, img_height: int) -> 'CropRegion':
        """
        将裁剪区域限制在图像边界内。
        
        参数:
            img_width: 图像宽度
            img_height: 图像高度
            
        返回:
            限制后的新 CropRegion 对象
        """
        return CropRegion(
            left=max(0, self.left),
            top=max(0, self.top),
            right=min(img_width, self.right),
            bottom=min(img_height, self.bottom)
        )


@dataclass
class DetectionResult:
    """
    检测结果数据类，包含人脸和人体检测结果列表。
    """
    faces: List[BoundingBox]   # 人脸边界框列表
    bodies: List[BoundingBox]  # 人体边界框列表

    def get_primary_face(self, index: int = 0) -> Optional[BoundingBox]:
        """
        获取主要人脸（按面积排序）。
        
        参数:
            index: 人脸索引，默认为 0（最大的人脸）
            
        返回:
            指定索引的人脸边界框，如果没有人脸则返回 None
        """
        if not self.faces:
            return None
        sorted_faces = sorted(self.faces, key=lambda f: f.area, reverse=True)
        return sorted_faces[min(index, len(sorted_faces) - 1)]

    def get_primary_body(self) -> Optional[BoundingBox]:
        """
        获取主要人体（按面积排序）。
        
        返回:
            最大的人体边界框，如果没有人体则返回 None
        """
        if not self.bodies:
            return None
        return max(self.bodies, key=lambda b: b.area)

    def is_closeup_shot(self, image_area: int) -> bool:
        """
        判断是否为特写镜头（人脸占比超过50%）。
        
        参数:
            image_area: 图像总面积
            
        返回:
            如果人脸面积超过图像面积的 50% 则返回 True
        """
        face = self.get_primary_face()
        if face is None:
            return False
        return face.area > image_area * 0.5
