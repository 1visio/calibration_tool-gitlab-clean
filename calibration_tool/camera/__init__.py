"""相机适配、帧质量分析和标定数据采集服务。"""

from .capture import preview_camera, run_capture_plan
from .config import (
    CameraChannelDefinition,
    CameraChannelRegistry,
    load_camera_channel_registry,
    load_camera_config,
    load_capture_plan,
)
from .factory import build_camera_provider
from .models import CameraConfig, CameraDeviceInfo, CapturedFrame
from .plan_builder import (
    CaptureRecipe,
    CaptureRecipeItem,
    CaptureSplit,
    build_capture_plan_from_recipe,
    capture_plan_summary,
    capture_plan_to_document,
    save_generated_capture_plan,
)

__all__ = [
    "CameraConfig",
    "CameraChannelDefinition",
    "CameraChannelRegistry",
    "CameraDeviceInfo",
    "CapturedFrame",
    "CaptureRecipe",
    "CaptureRecipeItem",
    "CaptureSplit",
    "build_camera_provider",
    "build_capture_plan_from_recipe",
    "capture_plan_summary",
    "capture_plan_to_document",
    "load_camera_config",
    "load_camera_channel_registry",
    "load_capture_plan",
    "preview_camera",
    "run_capture_plan",
    "save_generated_capture_plan",
]
