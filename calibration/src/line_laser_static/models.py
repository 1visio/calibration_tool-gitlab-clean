from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class FrameMetadata:
    frame_id: int
    timestamp_ns: int
    width: int
    height: int
    exposure_us: float
    gain_db: float
    pixel_format: str
    camera_model: str
    serial_number: str
    sdk_version: str
    offset_x_px: int = 0
    offset_y_px: int = 0
    full_width_px: int | None = None
    full_height_px: int | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("FrameMetadata.width/height 必须为正数")
        if self.offset_x_px < 0 or self.offset_y_px < 0:
            raise ValueError("相机 ROI 的 OffsetX/OffsetY 不能为负数")
        if self.full_width_px is not None:
            if self.full_width_px <= 0:
                raise ValueError("full_width_px 必须为正数")
            if self.offset_x_px + self.width > self.full_width_px:
                raise ValueError("相机 ROI 在 x 方向超出完整图像")
        if self.full_height_px is not None:
            if self.full_height_px <= 0:
                raise ValueError("full_height_px 必须为正数")
            if self.offset_y_px + self.height > self.full_height_px:
                raise ValueError("相机 ROI 在 y 方向超出完整图像")


@dataclass(frozen=True)
class Frame:
    image: NDArray[np.generic]
    metadata: FrameMetadata

    def __post_init__(self) -> None:
        if self.image.ndim != 2:
            raise ValueError("Frame.image 必须是二维灰度图")
        expected = (self.metadata.height, self.metadata.width)
        if self.image.shape != expected:
            raise ValueError(f"图像尺寸 {self.image.shape} 与元数据 {expected} 不一致")


@dataclass(frozen=True)
class StripeProfile:
    u_px: FloatArray
    v_px: FloatArray
    intensity: FloatArray
    confidence: FloatArray
    valid: BoolArray
    contrast: FloatArray | None = None
    snr: FloatArray | None = None
    fwhm_px: FloatArray | None = None
    saturated: BoolArray | None = None

    def __post_init__(self) -> None:
        _validate_vectors(
            "StripeProfile",
            self.u_px,
            self.v_px,
            self.intensity,
            self.confidence,
            self.valid,
        )
        for field_name, vector in (
            ("contrast", self.contrast),
            ("snr", self.snr),
            ("fwhm_px", self.fwhm_px),
            ("saturated", self.saturated),
        ):
            if vector is None:
                continue
            if vector.ndim != 1 or vector.size != self.u_px.size:
                raise ValueError(
                    f"StripeProfile.{field_name} 必须是一维且与 u_px 等长"
                )


@dataclass(frozen=True)
class PointCloud:
    x_mm: FloatArray
    y_mm: FloatArray
    z_mm: FloatArray
    intensity: FloatArray
    confidence: FloatArray
    valid: BoolArray

    def __post_init__(self) -> None:
        _validate_vectors(
            "PointCloud",
            self.x_mm,
            self.y_mm,
            self.z_mm,
            self.intensity,
            self.confidence,
            self.valid,
        )

    @property
    def size(self) -> int:
        return int(self.x_mm.size)


def _validate_vectors(name: str, *vectors: NDArray[np.generic]) -> None:
    if not vectors:
        raise ValueError(f"{name} 至少需要一个向量")
    if any(vector.ndim != 1 for vector in vectors):
        raise ValueError(f"{name} 的字段必须是一维向量")
    sizes = {vector.size for vector in vectors}
    if len(sizes) != 1:
        raise ValueError(f"{name} 的字段长度必须一致")
