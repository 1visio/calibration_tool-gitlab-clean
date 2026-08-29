from __future__ import annotations

from .algorithms import (
    CentroidStripeExtractor,
    MonoStripeExtractor,
    SensorRoiMonoStripeExtractor,
)
from .calibration import LaserCalibration, load_calibration
from .config import RunConfig
from .pipeline import StaticProfilePipeline
from .reconstruction import RayPlaneReconstructor
from .sources import DahengUsb3FrameSource, SyntheticFrameSource


def build_pipeline(config: RunConfig) -> tuple[StaticProfilePipeline, LaserCalibration]:
    if config.source.name == "synthetic":
        source = SyntheticFrameSource(**config.source.options)
    elif config.source.name == "daheng_usb3":
        source = DahengUsb3FrameSource(**config.source.options)
    else:
        raise ValueError(
            f"未知采集适配器 {config.source.name!r}；请在 bootstrap.py 接入 FrameSource"
        )

    if config.extraction.name == "centroid":
        extractor = CentroidStripeExtractor(**config.extraction.options)
    elif config.extraction.name == "mono":
        extractor = MonoStripeExtractor(**config.extraction.options)
    elif config.extraction.name == "mono_sensor_roi":
        extractor = SensorRoiMonoStripeExtractor(**config.extraction.options)
    else:
        raise ValueError(
            f"未知提取适配器 {config.extraction.name!r}；请在 bootstrap.py 接入 StripeExtractor"
        )

    calibration = load_calibration(config.calibration_file)
    if config.reconstruction.name == "ray_plane":
        reconstructor = RayPlaneReconstructor(calibration, **config.reconstruction.options)
    else:
        raise ValueError(
            f"未知重建适配器 {config.reconstruction.name!r}；"
            "请在 bootstrap.py 接入 ProfileReconstructor"
        )
    return StaticProfilePipeline(source, extractor, reconstructor), calibration
