from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from ..laser import LaserConfig


PIXEL_FORMATS = {"Mono8", "Mono12"}
IMAGE_FORMATS = {"tif", "png"}
QUALITY_MODES = {"generic", "chessboard", "laser"}
CAMERA_RECONFIGURE_FIELDS = (
    "pixel_format",
    "offset_x",
    "offset_y",
    "width",
    "height",
    "timeout_ms",
)


@dataclass(frozen=True, slots=True)
class CameraDeviceInfo:
    model: str
    serial_number: str
    ip_address: str = ""
    transport: str = "GigE"

    @property
    def display_name(self) -> str:
        suffix = f" · {self.ip_address}" if self.ip_address else ""
        return f"{self.model} · SN {self.serial_number}{suffix}"


@dataclass(frozen=True, slots=True)
class CameraConfig:
    exposure_us: float = 1200.0
    gain_db: float = 0.0
    pixel_format: str = "Mono8"
    offset_x: int = 0
    offset_y: int = 0
    width: int = 2448
    height: int = 2048
    timeout_ms: int = 2000

    def __post_init__(self) -> None:
        if not math.isfinite(self.exposure_us) or self.exposure_us <= 0:
            raise ValueError("exposure_us 必须是有限正数")
        if not math.isfinite(self.gain_db):
            raise ValueError("gain_db 必须是有限数")
        if self.pixel_format not in PIXEL_FORMATS:
            raise ValueError(f"pixel_format 必须是 {sorted(PIXEL_FORMATS)} 之一")
        if min(self.offset_x, self.offset_y) < 0:
            raise ValueError("ROI 偏移不能为负数")
        if min(self.width, self.height, self.timeout_ms) <= 0:
            raise ValueError("ROI 尺寸和 timeout_ms 必须为正数")

    @property
    def sensor_max_value(self) -> float:
        return 255.0 if self.pixel_format == "Mono8" else 4095.0

    def updated(self, values: Mapping[str, Any]) -> "CameraConfig":
        allowed = set(asdict(self))
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"未知相机参数：{sorted(unknown)}")
        return replace(self, **dict(values))


def requires_camera_reconfigure(current: CameraConfig, target: CameraConfig) -> bool:
    """结构参数变化时需要停流重配；曝光/增益可走统一在线更新通道。"""

    return any(
        getattr(current, name) != getattr(target, name)
        for name in CAMERA_RECONFIGURE_FIELDS
    )


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    image: np.ndarray
    camera_frame_number: int
    camera_timestamp_ticks: int | None
    host_timestamp_ns: int
    host_monotonic_ns: int
    offset_x: int = 0
    offset_y: int = 0

    def __post_init__(self) -> None:
        if self.image.ndim != 2 or self.image.dtype not in (
            np.dtype(np.uint8),
            np.dtype(np.uint16),
        ):
            raise ValueError("CapturedFrame.image 必须是二维 uint8/uint16 灰度图")


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    max_saturation_fraction: float = 0.05
    max_dark_fraction: float = 0.95
    min_dynamic_range_u8: float = 20.0
    min_laser_coverage: float = 0.30
    require_chessboard_detection: bool = True
    max_chessboard_saturation_fraction: float = 0.005
    max_chessboard_highlight_fraction: float = 0.20
    max_chessboard_p995_u8: float = 245.0
    max_laser_saturation_fraction: float = 0.001
    max_laser_peak_saturation_fraction: float = 0.02
    max_laser_peak_near_saturation_fraction: float = 0.10
    max_laser_saturated_width_px: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "max_saturation_fraction",
            "max_dark_fraction",
            "min_laser_coverage",
            "max_chessboard_saturation_fraction",
            "max_chessboard_highlight_fraction",
            "max_laser_saturation_fraction",
            "max_laser_peak_saturation_fraction",
            "max_laser_peak_near_saturation_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0, 1]")
        if self.min_dynamic_range_u8 < 0:
            raise ValueError("min_dynamic_range_u8 不能为负数")
        if not 0.0 <= float(self.max_chessboard_p995_u8) <= 255.0:
            raise ValueError("max_chessboard_p995_u8 必须位于 [0, 255]")
        if self.max_laser_saturated_width_px < 0:
            raise ValueError("max_laser_saturated_width_px 不能为负数")


@dataclass(frozen=True, slots=True)
class FrameQuality:
    mean_dn: float
    p01_dn: float
    p50_dn: float
    p99_dn: float
    dynamic_range_u8: float
    saturation_fraction: float
    dark_fraction: float
    focus_laplacian: float
    laser_coverage: float | None = None
    laser_peak_saturation_fraction: float | None = None
    laser_peak_near_saturation_fraction: float | None = None
    laser_saturated_width_p95_px: float | None = None
    laser_fwhm_p50_px: float | None = None
    laser_fwhm_p95_px: float | None = None
    chessboard_highlight_fraction: float | None = None
    chessboard_p995_u8: float | None = None
    chessboard_detected: bool | None = None
    chessboard_pattern_used: tuple[int, int] | None = None
    chessboard_detection_method: str | None = None
    chessboard_hint: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.warnings


@dataclass(frozen=True, slots=True)
class CaptureTask:
    task_id: str
    frames: int
    filename_template: str
    config: CameraConfig
    pose_id: str = ""
    role: str = "generic"
    instruction: str = ""
    settle_frames: int = 5
    image_format: str = "tif"
    quality_mode: str = "generic"
    tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id or any(char in self.task_id for char in "\\/:*?\"<>|"):
            raise ValueError("task_id 不能为空且不能包含 Windows 非法文件名字符")
        if self.frames <= 0 or self.settle_frames < 0:
            raise ValueError("frames 必须为正数，settle_frames 不能为负数")
        if self.image_format not in IMAGE_FORMATS:
            raise ValueError(f"image_format 必须是 {sorted(IMAGE_FORMATS)} 之一")
        if self.config.pixel_format == "Mono12" and self.image_format != "tif":
            raise ValueError("Mono12 必须保存为 tif")
        if self.quality_mode not in QUALITY_MODES:
            raise ValueError(f"quality_mode 必须是 {sorted(QUALITY_MODES)} 之一")
        generated = [self.relative_path(index) for index in range(1, self.frames + 1)]
        expected_suffix = ".tif" if self.image_format == "tif" else ".png"
        if any(path.suffix.lower() != expected_suffix for path in generated):
            raise ValueError(
                f"任务 {self.task_id} 的 filename_template 必须生成 {expected_suffix} 文件"
            )
        if len({str(path).casefold() for path in generated}) != len(generated):
            raise ValueError(f"任务 {self.task_id} 的 filename_template 会生成重名文件")

    def relative_path(self, index: int) -> Path:
        suffix = ".tif" if self.image_format == "tif" else ".png"
        exposure = self.config.exposure_us
        exposure_text = str(int(exposure)) if float(exposure).is_integer() else f"{exposure:.12g}"
        values = {
            "task_id": self.task_id,
            "pose_id": self.pose_id,
            "role": self.role,
            "index": index,
            "index02": f"{index:02d}",
            "index04": f"{index:04d}",
            "index_suffix": "" if index == 1 else f"_{index:02d}",
            "exposure_us": exposure_text,
            "exposure_folder": f"{exposure_text}us",
            "suffix": suffix,
        }
        try:
            relative = Path(self.filename_template.format(**values))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"任务 {self.task_id} 的 filename_template 无效：{exc}") from exc
        if relative.is_absolute() or ".." in relative.parts or not relative.name:
            raise ValueError(f"任务 {self.task_id} 的输出路径必须位于数据集内部：{relative}")
        return relative


@dataclass(frozen=True, slots=True)
class CapturePlan:
    dataset_id: str
    output_dir: Path
    backend: str
    serial_number: str
    base_config: CameraConfig
    tasks: tuple[CaptureTask, ...]
    quality_thresholds: QualityThresholds = QualityThresholds()
    board_pattern: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    backend_options: dict[str, Any] = field(default_factory=dict)
    laser: LaserConfig = LaserConfig()

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id 不能为空")
        if self.backend not in {"mvs", "daheng", "synthetic"}:
            raise ValueError("backend 必须是 mvs、daheng 或 synthetic")
        if not self.tasks:
            raise ValueError("采集计划至少需要一个 task")
        if self.board_pattern is not None and min(self.board_pattern) <= 0:
            raise ValueError("board_pattern 必须包含两个正整数")
        if any(task.quality_mode == "chessboard" for task in self.tasks) and self.board_pattern is None:
            raise ValueError("chessboard 质量模式需要配置 board_pattern")
        task_ids = [task.task_id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("采集计划包含重复 task_id")
        paths = [task.relative_path(index) for task in self.tasks for index in range(1, task.frames + 1)]
        if len({str(path).casefold() for path in paths}) != len(paths):
            raise ValueError("不同采集任务会写入同一个文件")


@dataclass(frozen=True, slots=True)
class CaptureResult:
    output_dir: Path
    task_count: int
    frame_count: int
    quality_passed_frames: int
    quality_warning_frames: int


class CameraSession(Protocol):
    device: CameraDeviceInfo
    config: CameraConfig
    network_packet_size: int | None

    def configure(self, config: CameraConfig) -> CameraConfig: ...
    def update_exposure_gain(self, exposure_us: float, gain_db: float) -> CameraConfig: ...
    def start(self) -> None: ...
    def get_frame(self, timeout_ms: int | None = None) -> CapturedFrame: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class CameraProvider(Protocol):
    def list_devices(self) -> list[CameraDeviceInfo]: ...
    def open(self, serial_number: str, config: CameraConfig) -> CameraSession: ...


ProgressCallback = Callable[[dict[str, Any]], None]
