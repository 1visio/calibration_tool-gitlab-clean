"""纯逻辑的批量采集计划生成器。

本模块只负责把适合 GUI 编辑的采集配方转换为现有的
``CapturePlan``/``CaptureTask``。采集执行仍然由 ``run_capture_plan``
负责，laser_state 等操作提示只写入 task tags/instruction，不改变相机驱动
或标定算法。
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from ..io_utils import dump_yaml
from ..laser import LaserConfig, parse_laser_config
from .config import load_capture_plan
from .models import CameraConfig, CapturePlan, CaptureTask, QualityThresholds


LASER_STATES = frozenset({"on", "off", "unchanged"})
_QUALITY_ALIASES = {"nolaser": "generic", "chess": "chessboard"}
_PLAN_SUFFIXES = frozenset({".yaml", ".yml"})
_INVALID_COMPONENT_CHARS = set('\\/:*?"<>|{}')


@dataclass(frozen=True, slots=True)
class CaptureRecipeItem:
    """一个可勾选的图像类型/曝光项目。"""

    enabled: bool = True
    role: str = "generic"
    filename_prefix: str = "capture"
    exposure_us: float = 1200.0
    laser_state: str = "unchanged"
    quality_mode: str = "generic"
    frames: int = 1
    settle_frames: int = 5
    image_format: str = "tif"
    instruction_template: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled 必须是布尔值")
        _validate_component(self.role, "role")
        _validate_component(self.filename_prefix, "filename_prefix")
        _validate_positive_number(self.exposure_us, "exposure_us")
        if not isinstance(self.laser_state, str) or self.laser_state not in LASER_STATES:
            raise ValueError(f"laser_state 必须是 {sorted(LASER_STATES)} 之一")
        if not isinstance(self.quality_mode, str) or self.quality_mode not in {
            "generic",
            "chessboard",
            "chess",
            "laser",
            "nolaser",
        }:
            raise ValueError("quality_mode 必须是 generic、chessboard、chess、laser 或 nolaser")
        if isinstance(self.frames, bool) or not isinstance(self.frames, int) or self.frames <= 0:
            raise ValueError("frames 必须为正整数")
        if (
            isinstance(self.settle_frames, bool)
            or not isinstance(self.settle_frames, int)
            or self.settle_frames < 0
        ):
            raise ValueError("settle_frames 不能为负数")
        if not isinstance(self.image_format, str) or self.image_format not in {"tif", "png"}:
            raise ValueError("image_format 必须是 tif 或 png")
        if not isinstance(self.instruction_template, str):
            raise ValueError("instruction_template 必须是字符串")


@dataclass(frozen=True, slots=True)
class CaptureRecipe:
    """GUI 可编辑的批量采集配方。

    ``camera``、``backend_options`` 和 ``metadata`` 允许 Mapping，便于从配置
    编辑器读取；生成计划时会转换为现有模型使用的强类型对象。
    """

    dataset_id: str
    output_dir: str | Path
    plan_output_path: str | Path
    fit_group_count: int
    include_validation: bool = False
    validation_group_count: int = 0
    start_index: int = 1
    index_digits: int = 3
    camera: CameraConfig | Mapping[str, Any] = field(default_factory=CameraConfig)
    serial_number: str = ""
    backend: str = "mvs"
    backend_options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    items: Sequence[CaptureRecipeItem | Mapping[str, Any]] = field(default_factory=tuple)
    # 11x8 是项目现有示例的棋盘规格；实际项目可显式覆盖。
    board_pattern: tuple[int, int] | None = (11, 8)
    quality_thresholds: QualityThresholds | Mapping[str, Any] = field(
        default_factory=QualityThresholds
    )
    laser: LaserConfig | Mapping[str, Any] = field(default_factory=LaserConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id 不能为空")
        if (
            isinstance(self.fit_group_count, bool)
            or not isinstance(self.fit_group_count, int)
            or self.fit_group_count <= 0
        ):
            raise ValueError("fit_group_count 必须为正整数")
        if (
            isinstance(self.validation_group_count, bool)
            or not isinstance(self.validation_group_count, int)
            or self.validation_group_count < 0
        ):
            raise ValueError("validation_group_count 不能为负数")
        if not isinstance(self.include_validation, bool):
            raise ValueError("include_validation 必须是布尔值")
        if self.include_validation and self.validation_group_count <= 0:
            raise ValueError("启用验证集时 validation_group_count 必须大于 0")
        if (
            isinstance(self.start_index, bool)
            or not isinstance(self.start_index, int)
            or self.start_index < 0
        ):
            raise ValueError("start_index 不能为负数")
        if (
            isinstance(self.index_digits, bool)
            or not isinstance(self.index_digits, int)
            or self.index_digits <= 0
        ):
            raise ValueError("index_digits 必须为正整数")
        if not isinstance(self.items, Sequence) or isinstance(self.items, (str, bytes)):
            raise ValueError("items 必须是采集项目序列")
        converted = tuple(_coerce_recipe_item(item) for item in self.items)
        object.__setattr__(self, "items", converted)
        object.__setattr__(self, "board_pattern", _normalize_board_pattern(self.board_pattern))
        object.__setattr__(self, "laser", _coerce_laser_config(self.laser))


@dataclass(frozen=True, slots=True)
class CaptureSplit:
    """生成计划中的一个连续编号分组。"""

    name: str
    group_count: int
    start_index: int
    index_digits: int = 3

    def __post_init__(self) -> None:
        if self.name not in {"fit", "validation"}:
            raise ValueError("split name 必须是 fit 或 validation")
        if (
            isinstance(self.group_count, bool)
            or not isinstance(self.group_count, int)
            or self.group_count <= 0
        ):
            raise ValueError("split group_count 必须为正整数")
        if (
            isinstance(self.start_index, bool)
            or not isinstance(self.start_index, int)
            or self.start_index < 0
        ):
            raise ValueError("split start_index 不能为负数")
        if (
            isinstance(self.index_digits, bool)
            or not isinstance(self.index_digits, int)
            or self.index_digits <= 0
        ):
            raise ValueError("split index_digits 必须为正整数")

    @property
    def end_index(self) -> int:
        return self.start_index + self.group_count - 1

    def pose_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{index:0{self.index_digits}d}"
            for index in range(self.start_index, self.end_index + 1)
        )


def build_capture_plan_from_recipe(
    recipe: CaptureRecipe | Mapping[str, Any],
) -> CapturePlan:
    """将配方转换为现有 ``CapturePlan``，不创建输出数据集目录。"""

    recipe = _coerce_recipe(recipe)
    output_dir = _resolve_path(recipe.output_dir, "output_dir")
    if output_dir.exists() and output_dir.is_file():
        raise ValueError(f"output_dir 不能是文件：{output_dir}")
    plan_output_path = _resolve_plan_path(recipe.plan_output_path)
    _validate_plan_location(output_dir, plan_output_path)

    enabled_items = [item for item in recipe.items if item.enabled]
    if not enabled_items:
        raise ValueError("至少启用一种图像类型")

    base_config = _coerce_camera(recipe.camera)
    quality_thresholds = _coerce_quality_thresholds(recipe.quality_thresholds)
    board_pattern = _normalize_board_pattern(recipe.board_pattern)
    if any(_normalize_quality_mode(item.quality_mode) == "chessboard" for item in enabled_items):
        if board_pattern is None:
            raise ValueError("chessboard 质量模式需要 board_pattern")

    splits = [
        CaptureSplit(
            "fit",
            recipe.fit_group_count,
            recipe.start_index,
            recipe.index_digits,
        )
    ]
    if recipe.include_validation:
        splits.append(
            CaptureSplit(
                "validation",
                recipe.validation_group_count,
                recipe.start_index + recipe.fit_group_count,
                recipe.index_digits,
            )
        )

    tasks: list[CaptureTask] = []
    serial_number = "" if recipe.serial_number is None else str(recipe.serial_number)
    for split in splits:
        for numeric_index, pose_id in zip(
            range(split.start_index, split.end_index + 1), split.pose_ids(), strict=True
        ):
            for item_index, item in enumerate(recipe.items, start=1):
                if not item.enabled:
                    continue
                quality_mode = _normalize_quality_mode(item.quality_mode)
                task_id = f"{split.name}_{pose_id}_{item_index:02d}_{item.role}"
                task_config = base_config.updated({"exposure_us": float(item.exposure_us)})
                # 三联图共享 split 目录，文件名直接由配方前缀和姿态编号组成：
                # ``fit/chess 001.tif``、``fit/laser 001.tif``、``fit/nolaser 001.tif``。
                # 曝光已经保存在 task.camera/config 中，不再重复编码到目录层级。
                filename_template = (
                    f"{split.name}/{item.filename_prefix} "
                    "{pose_id}{index_suffix}{suffix}"
                )
                instruction = _render_instruction(
                    item.instruction_template,
                    split=split.name,
                    numeric_index=numeric_index,
                    pose_id=pose_id,
                    task_id=task_id,
                    item=item,
                )
                tags = {
                    "split": split.name,
                    "group_index": numeric_index,
                    "laser_state": item.laser_state,
                    "recipe_role": item.role,
                    "filename_prefix": item.filename_prefix,
                    "quality_mode_requested": item.quality_mode,
                }
                tasks.append(
                    CaptureTask(
                        task_id=task_id,
                        frames=item.frames,
                        filename_template=filename_template,
                        config=task_config,
                        pose_id=pose_id,
                        role=item.role,
                        instruction=instruction,
                        settle_frames=item.settle_frames,
                        image_format=item.image_format,
                        quality_mode=quality_mode,
                        tags=tags,
                    )
                )

    try:
        return CapturePlan(
            dataset_id=recipe.dataset_id.strip(),
            output_dir=output_dir,
            backend=str(recipe.backend),
            serial_number=serial_number,
            base_config=base_config,
            tasks=tuple(tasks),
            quality_thresholds=quality_thresholds,
            board_pattern=board_pattern,
            metadata=dict(recipe.metadata),
            backend_options=dict(recipe.backend_options),
            laser=recipe.laser,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"采集配方无法生成计划：{exc}") from exc


def capture_plan_to_document(
    plan: CapturePlan | CaptureRecipe | Mapping[str, Any],
) -> dict[str, Any]:
    """转换为 ``load_capture_plan`` 支持的 schema_version=1 文档。"""

    if not isinstance(plan, CapturePlan):
        plan = build_capture_plan_from_recipe(plan)
    document: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": plan.dataset_id,
        "output_dir": str(plan.output_dir.expanduser().resolve()),
        "backend": plan.backend,
        "serial_number": plan.serial_number,
        "camera": asdict(plan.base_config),
        "quality": asdict(plan.quality_thresholds),
        "metadata": dict(plan.metadata),
        "backend_options": dict(plan.backend_options),
        "laser": asdict(plan.laser),
        "tasks": [],
    }
    if plan.board_pattern is not None:
        document["board"] = {
            "pattern_cols": plan.board_pattern[0],
            "pattern_rows": plan.board_pattern[1],
        }
    document["tasks"] = [
        {
            "task_id": task.task_id,
            "frames": task.frames,
            "filename_template": task.filename_template,
            "camera": asdict(task.config),
            "pose_id": task.pose_id,
            "role": task.role,
            "instruction": task.instruction,
            "settle_frames": task.settle_frames,
            "image_format": task.image_format,
            "quality_mode": task.quality_mode,
            "tags": dict(task.tags),
        }
        for task in plan.tasks
    ]
    return document


def save_generated_capture_plan(
    plan_or_recipe: CapturePlan | CaptureRecipe | Mapping[str, Any],
    plan_output_path: str | Path | CapturePlan | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """原子保存计划，并在提交前用现有 loader 做 round-trip 校验。

    传入 ``CaptureRecipe`` 时可省略 ``plan_output_path``，此时使用配方中的
    ``plan_output_path``；传入 ``CapturePlan`` 时必须显式提供路径。
    """

    recipe: CaptureRecipe | None = None
    generated_plan: CapturePlan
    if isinstance(plan_or_recipe, CapturePlan):
        generated_plan = plan_or_recipe
        if isinstance(plan_output_path, CapturePlan):
            raise ValueError("CapturePlan 不能同时作为保存路径")
        if plan_output_path is None:
            raise ValueError("保存 CapturePlan 时必须提供 plan_output_path")
        target = _resolve_plan_path(plan_output_path)
    else:
        recipe = _coerce_recipe(plan_or_recipe)
        if isinstance(plan_output_path, CapturePlan):
            generated_plan = plan_output_path
            recipe_output = _resolve_path(recipe.output_dir, "output_dir")
            if (
                generated_plan.dataset_id != recipe.dataset_id.strip()
                or generated_plan.output_dir.resolve() != recipe_output
            ):
                raise ValueError("传入的 CapturePlan 与 CaptureRecipe 不匹配")
            target = _resolve_plan_path(recipe.plan_output_path)
        else:
            generated_plan = build_capture_plan_from_recipe(recipe)
            target = _resolve_plan_path(recipe.plan_output_path if plan_output_path is None else plan_output_path)

    _validate_plan_location(generated_plan.output_dir, target)
    if target.exists() and not overwrite:
        raise FileExistsError(f"采集计划已经存在，不会静默覆盖：{target}")

    document = capture_plan_to_document(generated_plan)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        # 复用项目现有 YAML 写入工具；最终提交仍通过同目录 os.replace 原子完成。
        dump_yaml(temporary_path, document)
        try:
            loaded = load_capture_plan(temporary_path)
        except ConfigError as exc:
            raise ValueError(f"生成的 YAML 无法被 load_capture_plan 重新加载：{exc}") from exc
        _assert_round_trip(generated_plan, loaded)
        if target.exists() and not overwrite:
            raise FileExistsError(f"采集计划已经存在，不会静默覆盖：{target}")
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


def capture_plan_summary(
    plan: CapturePlan | CaptureRecipe | Mapping[str, Any],
) -> dict[str, Any]:
    """返回适合 GUI 显示的计划统计和逐任务记录。"""

    if not isinstance(plan, CapturePlan):
        plan = build_capture_plan_from_recipe(plan)
    records: list[dict[str, Any]] = []
    split_poses: dict[str, set[str]] = {"fit": set(), "validation": set()}
    for task in plan.tasks:
        first_path = task.relative_path(1)
        split = str(task.tags.get("split", first_path.parts[0] if first_path.parts else ""))
        pose_id = task.pose_id
        if split in split_poses:
            split_poses[split].add(pose_id)
        records.append(
            {
                "split": split,
                "pose_id": pose_id,
                "task_id": task.task_id,
                "role": task.role,
                "exposure_us": task.config.exposure_us,
                "laser_state": task.tags.get("laser_state", "unchanged"),
                "quality_mode": task.quality_mode,
                "relative_output_path": first_path.as_posix(),
                "frames": task.frames,
            }
        )
    fit_count = len(split_poses["fit"])
    validation_count = len(split_poses["validation"])
    return {
        "fit_group_count": fit_count,
        "validation_group_count": validation_count,
        "task_count": len(plan.tasks),
        "image_count": sum(task.frames for task in plan.tasks),
        "tasks": records,
        "task_records": records,
    }


def _coerce_recipe_item(value: CaptureRecipeItem | Mapping[str, Any]) -> CaptureRecipeItem:
    if isinstance(value, CaptureRecipeItem):
        return value
    if isinstance(value, Mapping):
        try:
            return CaptureRecipeItem(**dict(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"采集项目无效：{exc}") from exc
    raise ValueError("采集项目必须是 CaptureRecipeItem 或映射")


def _coerce_recipe(value: CaptureRecipe | Mapping[str, Any]) -> CaptureRecipe:
    if isinstance(value, CaptureRecipe):
        return value
    if isinstance(value, Mapping):
        try:
            return CaptureRecipe(**dict(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"采集配方无效：{exc}") from exc
    raise ValueError("recipe 必须是 CaptureRecipe 或映射")


def _coerce_camera(value: CameraConfig | Mapping[str, Any]) -> CameraConfig:
    if isinstance(value, CameraConfig):
        return value
    if isinstance(value, Mapping):
        try:
            return CameraConfig(**dict(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"camera 配置无效：{exc}") from exc
    raise ValueError("camera 必须是 CameraConfig 或映射")


def _coerce_quality_thresholds(
    value: QualityThresholds | Mapping[str, Any],
) -> QualityThresholds:
    if isinstance(value, QualityThresholds):
        return value
    if isinstance(value, Mapping):
        try:
            return QualityThresholds(**dict(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"quality_thresholds 无效：{exc}") from exc
    raise ValueError("quality_thresholds 必须是 QualityThresholds 或映射")


def _coerce_laser_config(value: LaserConfig | Mapping[str, Any]) -> LaserConfig:
    if isinstance(value, LaserConfig):
        return value
    try:
        return parse_laser_config(value)
    except ConfigError as exc:
        raise ValueError(str(exc)) from exc


def _normalize_quality_mode(value: str) -> str:
    return _QUALITY_ALIASES.get(value, value)


def _normalize_board_pattern(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        try:
            value = (value["pattern_cols"], value["pattern_rows"])
        except (KeyError, TypeError) as exc:
            raise ValueError("board_pattern 需要两个正整数") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("board_pattern 需要两个正整数")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("board_pattern 需要两个正整数")
    try:
        pattern = (int(value[0]), int(value[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError("board_pattern 需要两个正整数") from exc
    if min(pattern) <= 0:
        raise ValueError("board_pattern 需要两个正整数")
    return pattern


def _validate_positive_number(value: Any, name: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限正数")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限正数") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} 必须是有限正数")


def _validate_component(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 不能为空")
    if value in {".", ".."} or any(char in _INVALID_COMPONENT_CHARS for char in value):
        raise ValueError(f"{name} 不能包含路径分隔符、路径逃逸或模板字符")


def _resolve_path(value: str | Path, name: str) -> Path:
    if isinstance(value, Path):
        raw = value
    elif isinstance(value, str) and value.strip():
        raw = Path(value.strip())
    else:
        raise ValueError(f"{name} 不能为空且必须是合法路径")
    try:
        return raw.expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{name} 不是合法路径：{value}") from exc


def _resolve_plan_path(value: str | Path | CapturePlan) -> Path:
    if isinstance(value, CapturePlan):
        raise ValueError("plan_output_path 必须是 YAML 文件路径")
    target = _resolve_path(value, "plan_output_path")
    if target.suffix.lower() not in _PLAN_SUFFIXES:
        raise ValueError("plan_output_path 必须以 .yaml 或 .yml 结尾")
    return target


def _validate_plan_location(output_dir: Path, plan_path: Path) -> None:
    try:
        plan_path.relative_to(output_dir)
    except ValueError:
        return
    raise ValueError("plan_output_path 必须位于数据集 output_dir 外部")


def _render_instruction(
    template: str,
    *,
    split: str,
    numeric_index: int,
    pose_id: str,
    task_id: str,
    item: CaptureRecipeItem,
) -> str:
    if not template:
        return ""
    exposure = float(item.exposure_us)
    exposure_text = str(int(exposure)) if exposure.is_integer() else f"{exposure:.12g}"
    values = {
        "split": split,
        "index": numeric_index,
        "index02": f"{numeric_index:02d}",
        "pose_id": pose_id,
        "task_id": task_id,
        "role": item.role,
        "filename_prefix": item.filename_prefix,
        "exposure_us": exposure_text,
        "exposure_folder": f"{exposure_text}us",
        "laser_state": item.laser_state,
        "quality_mode": item.quality_mode,
    }
    try:
        return template.format(**values)
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(f"instruction_template 无效：{exc}") from exc


def _assert_round_trip(expected: CapturePlan, actual: CapturePlan) -> None:
    if (
        expected.dataset_id != actual.dataset_id
        or expected.output_dir.resolve() != actual.output_dir.resolve()
        or expected.backend != actual.backend
        or expected.serial_number != actual.serial_number
        or expected.base_config != actual.base_config
        or expected.quality_thresholds != actual.quality_thresholds
        or expected.board_pattern != actual.board_pattern
        or expected.laser != actual.laser
        or _yaml_compare_value(expected.metadata) != _yaml_compare_value(actual.metadata)
        or _yaml_compare_value(expected.backend_options) != _yaml_compare_value(actual.backend_options)
        or len(expected.tasks) != len(actual.tasks)
    ):
        raise ValueError("生成的 YAML round-trip 后计划基础字段不一致")
    for expected_task, actual_task in zip(expected.tasks, actual.tasks, strict=True):
        if (
            expected_task.task_id != actual_task.task_id
            or expected_task.frames != actual_task.frames
            or expected_task.filename_template != actual_task.filename_template
            or expected_task.config != actual_task.config
            or expected_task.pose_id != actual_task.pose_id
            or expected_task.role != actual_task.role
            or expected_task.instruction != actual_task.instruction
            or expected_task.settle_frames != actual_task.settle_frames
            or expected_task.image_format != actual_task.image_format
            or expected_task.quality_mode != actual_task.quality_mode
            or _yaml_compare_value(expected_task.tags) != _yaml_compare_value(actual_task.tags)
        ):
            raise ValueError(f"生成的 YAML round-trip 后 task 不一致：{expected_task.task_id}")


def _yaml_compare_value(value: Any) -> Any:
    """将 YAML safe_dump/safe_load 的 tuple/list 差异归一化。"""

    if isinstance(value, Mapping):
        return {key: _yaml_compare_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_yaml_compare_value(item) for item in value]
    return value


__all__ = [
    "CaptureRecipeItem",
    "CaptureRecipe",
    "CaptureSplit",
    "LASER_STATES",
    "build_capture_plan_from_recipe",
    "capture_plan_to_document",
    "save_generated_capture_plan",
    "capture_plan_summary",
]
