from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ..errors import ConfigError
from ..io_utils import canonical_mapping_hash, load_document, resolve_relative
from ..laser import default_laser_orientation, parse_laser_config
from .models import CameraConfig, CapturePlan, CaptureTask, QualityThresholds


@dataclass(frozen=True, slots=True)
class CameraChannelDefinition:
    """一个可切换相机通道及其推荐分析 workflow。"""

    name: str
    label: str
    camera_config: Path
    workflow_plan: Path | None = None


@dataclass(frozen=True, slots=True)
class CameraChannelRegistry:
    """相机通道注册表；各通道继续复用已有单相机 YAML。"""

    source: Path
    default_channel: str
    channels: tuple[CameraChannelDefinition, ...]

    def get(self, name: str | None = None) -> CameraChannelDefinition:
        selected = str(name or self.default_channel).strip()
        for channel in self.channels:
            if channel.name == selected:
                return channel
        available = ", ".join(channel.name for channel in self.channels)
        raise ConfigError(f"未知相机通道 {selected!r}；可用通道：{available}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} 必须是映射")
    return dict(value)


def _camera_config(value: Any, *, base: CameraConfig | None = None) -> CameraConfig:
    try:
        return (base or CameraConfig()).updated(_mapping(value, "camera"))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"相机参数无效：{exc}") from exc


def _thresholds(value: Any) -> QualityThresholds:
    try:
        return QualityThresholds(**_mapping(value, "quality"))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"质量阈值无效：{exc}") from exc


def _board_pattern(value: Any) -> tuple[int, int] | None:
    board = _mapping(value, "board")
    if not board:
        return None
    try:
        pattern = (int(board["pattern_cols"]), int(board["pattern_rows"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError("board 需要正整数 pattern_cols 和 pattern_rows") from exc
    if min(pattern) <= 0:
        raise ConfigError("board pattern_cols/pattern_rows 必须为正整数")
    return pattern


def load_camera_channel_registry(path: str | Path) -> CameraChannelRegistry:
    source = Path(path).expanduser().resolve()
    document = load_document(source)
    return _camera_channel_registry(source, document)


def _camera_channel_registry(
    source: Path,
    document: Mapping[str, Any],
) -> CameraChannelRegistry:
    channels_value = document.get("channels")
    if not isinstance(channels_value, Mapping) or not channels_value:
        raise ConfigError(f"相机通道注册表缺少非空 channels：{source}")

    channels: list[CameraChannelDefinition] = []
    allowed = {"label", "config", "workflow_plan"}
    for raw_name, raw_value in channels_value.items():
        name = str(raw_name).strip()
        if not name:
            raise ConfigError("相机通道名称不能为空")
        if isinstance(raw_value, (str, Path)):
            item = {"config": raw_value}
        else:
            item = _mapping(raw_value, f"channels.{name}")
        unknown = set(item) - allowed
        if unknown:
            raise ConfigError(f"channels.{name} 包含未知字段：{sorted(unknown)}")
        config_value = item.get("config")
        if not config_value:
            raise ConfigError(f"channels.{name} 缺少 config")
        label = str(item.get("label", name)).strip()
        if not label:
            raise ConfigError(f"channels.{name}.label 不能为空")
        workflow_value = item.get("workflow_plan")
        channels.append(
            CameraChannelDefinition(
                name=name,
                label=label,
                camera_config=resolve_relative(source, config_value),
                workflow_plan=(
                    resolve_relative(source, workflow_value) if workflow_value else None
                ),
            )
        )

    default_channel = str(document.get("default_channel", channels[0].name)).strip()
    registry = CameraChannelRegistry(source, default_channel, tuple(channels))
    registry.get(default_channel)
    return registry


def load_camera_config(
    path: str | Path,
    *,
    channel: str | None = None,
) -> dict[str, Any]:
    """加载单相机配置，或从统一通道注册表中选择一个配置。

    旧版单相机 YAML 保持兼容；注册表模式额外返回 ``channel``、
    ``channel_label``、``channel_registry``、``camera_config_source`` 和
    推荐的 ``workflow_plan``。
    """

    source = Path(path).expanduser().resolve()
    document = load_document(source)
    if "channels" in document:
        registry = _camera_channel_registry(source, document)
        selected = registry.get(channel)
        selected_document = load_document(selected.camera_config)
        if "channels" in selected_document:
            raise ConfigError(
                f"相机通道不能嵌套引用另一个通道注册表：{selected.camera_config}"
            )
        runtime = _camera_runtime(selected.camera_config, selected_document)
        runtime.update(
            source=source,
            camera_config_source=selected.camera_config,
            channel=selected.name,
            channel_label=selected.label,
            channel_registry=registry,
            workflow_plan=selected.workflow_plan,
        )
        return runtime
    if channel is not None and str(channel).strip():
        raise ConfigError(f"指定了相机通道 {channel!r}，但配置不是通道注册表：{source}")
    runtime = _camera_runtime(source, document)
    runtime.update(
        camera_config_source=source,
        channel=None,
        channel_label=runtime["backend"],
        channel_registry=None,
        workflow_plan=None,
    )
    return runtime


def _camera_runtime(source: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    backend = str(document.get("backend", "mvs"))
    if backend not in {"mvs", "daheng", "synthetic"}:
        raise ConfigError("backend 必须是 mvs、daheng 或 synthetic")
    calibration_src_value = document.get("calibration_src")
    calibration_src = (
        Path(__file__).resolve().parents[2] / "calibration" / "src"
        if calibration_src_value is None
        else resolve_relative(source, calibration_src_value)
    )
    return {
        "source": source,
        "backend": backend,
        "serial_number": str(document.get("serial_number", "")),
        "camera": _camera_config(document.get("camera")),
        "quality_thresholds": _thresholds(document.get("quality")),
        "board_pattern": _board_pattern(document.get("board")),
        "backend_options": _mapping(document.get("backend_options"), "backend_options"),
        "calibration_src": calibration_src,
        "laser": parse_laser_config(
            document.get("laser"),
            default_orientation=default_laser_orientation(backend),
        ),
    }


def load_capture_plan(path: str | Path) -> CapturePlan:
    source = Path(path).expanduser().resolve()
    document = load_document(source)
    base_config = _camera_config(document.get("camera"))
    backend = str(document.get("backend", "mvs"))
    if backend not in {"mvs", "daheng", "synthetic"}:
        raise ConfigError("backend 必须是 mvs、daheng 或 synthetic")
    tasks_value = document.get("tasks")
    if not isinstance(tasks_value, list) or not tasks_value:
        raise ConfigError("capture plan 的 tasks 必须是非空列表")
    tasks: list[CaptureTask] = []
    try:
        for index, raw in enumerate(tasks_value, start=1):
            item = _mapping(raw, f"tasks[{index}]")
            task_config = _camera_config(item.pop("camera", None), base=base_config)
            tasks.append(CaptureTask(config=task_config, **item))
        output_value = document.get("output_dir")
        if not output_value:
            raise ConfigError("capture plan 缺少 output_dir")
        return CapturePlan(
            dataset_id=str(document.get("dataset_id", "")),
            output_dir=resolve_relative(source, output_value),
            backend=backend,
            serial_number=str(document.get("serial_number", "")),
            base_config=base_config,
            tasks=tuple(tasks),
            quality_thresholds=_thresholds(document.get("quality")),
            board_pattern=_board_pattern(document.get("board")),
            metadata=_mapping(document.get("metadata"), "metadata"),
            backend_options=_mapping(document.get("backend_options"), "backend_options"),
            laser=parse_laser_config(
                document.get("laser"),
                default_orientation=default_laser_orientation(backend),
            ),
        )
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"采集计划无效：{exc}") from exc


def capture_plan_payload(plan: CapturePlan) -> dict[str, Any]:
    def camera(value: CameraConfig) -> dict[str, Any]:
        return asdict(value)

    return {
        "dataset_id": plan.dataset_id,
        "output_dir": str(plan.output_dir.expanduser().resolve()),
        "backend": plan.backend,
        "serial_number": plan.serial_number,
        "camera": camera(plan.base_config),
        "quality": asdict(plan.quality_thresholds),
        "board_pattern": list(plan.board_pattern) if plan.board_pattern else None,
        "metadata": plan.metadata,
        "backend_options": plan.backend_options,
        "laser": asdict(plan.laser),
        "tasks": [
            {
                "task_id": task.task_id,
                "frames": task.frames,
                "filename_template": task.filename_template,
                "camera": camera(task.config),
                "pose_id": task.pose_id,
                "role": task.role,
                "instruction": task.instruction,
                "settle_frames": task.settle_frames,
                "image_format": task.image_format,
                "quality_mode": task.quality_mode,
                "tags": task.tags,
            }
            for task in plan.tasks
        ],
    }


def capture_plan_hash(plan: CapturePlan) -> str:
    return canonical_mapping_hash(capture_plan_payload(plan))
