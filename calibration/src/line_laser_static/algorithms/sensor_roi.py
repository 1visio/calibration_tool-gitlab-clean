from __future__ import annotations

from dataclasses import replace

from ..models import Frame, StripeProfile
from .mono import MonoStripeExtractor


class SensorRoiMonoStripeExtractor(MonoStripeExtractor):
    """处理相机硬件 ROI 输出，并返回完整传感器坐标。"""

    def __init__(
        self,
        window_radius: int = 5,
        background_percentile: float = 50.0,
        min_contrast: float = 30.0,
        min_snr: float = 10.0,
        noise_floor: float = 1.0,
        min_fwhm_px: float = 2.0,
        max_fwhm_px: float = 10.0,
        saturation_fraction: float = 0.98,
        reject_saturated: bool = True,
        sensor_max_value: float | None = None,
    ) -> None:
        super().__init__(
            window_radius=window_radius,
            background_percentile=background_percentile,
            min_contrast=min_contrast,
            min_snr=min_snr,
            noise_floor=noise_floor,
            min_fwhm_px=min_fwhm_px,
            max_fwhm_px=max_fwhm_px,
            saturation_fraction=saturation_fraction,
            reject_saturated=reject_saturated,
            sensor_max_value=sensor_max_value,
        )

    def extract(self, frame: Frame) -> StripeProfile:
        local_profile = super().extract(frame)
        return replace(
            local_profile,
            u_px=local_profile.u_px + float(frame.metadata.offset_x_px),
            v_px=local_profile.v_px + float(frame.metadata.offset_y_px),
        )
