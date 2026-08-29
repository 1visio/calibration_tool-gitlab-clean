from typing import Protocol

from .models import Frame, PointCloud, StripeProfile


class FrameSource(Protocol):
    def capture(self) -> Frame:
        """采集一帧灰度图及其元数据。"""


class StripeExtractor(Protocol):
    def extract(self, frame: Frame) -> StripeProfile:
        """从一帧图像输出逐列亚像素光条中心。"""


class ProfileReconstructor(Protocol):
    def reconstruct(self, profile: StripeProfile) -> PointCloud:
        """将二维光条中心转换为相机坐标系三维点。"""
