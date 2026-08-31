"""Calibration Run 的领域模型以及 workflow report 适配。

``run_workflow`` 目前返回普通 ``dict``，本模块不改变它的行为，而是把该
report 转换为供后续 GUI 页面使用的稳定数据模型。YAML 的文件 I/O 放在
``calibration_run_io`` 中，避免模型依赖具体的持久化实现。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from .errors import ConfigError


CALIBRATION_RUN_SCHEMA_VERSION = 1
_GATE_STATUSES = ("pass", "warn", "fail")


def _copy_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field_name} 必须是映射")
    return deepcopy(dict(value))


def _coerce_path(
    value: str | Path,
    field_name: str,
    *,
    base_dir: Path | None = None,
) -> Path:
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"{field_name} 必须是路径")
    path = Path(value).expanduser()
    is_absolute = path.is_absolute() or PureWindowsPath(str(value)).is_absolute()
    if base_dir is not None and not is_absolute:
        path = base_dir / path
    # report 可能是在 Windows 生成、在 POSIX 环境审计。POSIX 的 Path 无法
    # 识别盘符路径，此时保留原字符串，不要把它错误拼到 report 目录下。
    if base_dir is not None and is_absolute and not path.is_absolute():
        return path
    return path.resolve() if base_dir is not None else path


def _coerce_optional_path(
    value: Any,
    field_name: str,
    *,
    base_dir: Path | None = None,
) -> Path | None:
    if value is None or value == "":
        return None
    return _coerce_path(value, field_name, base_dir=base_dir)


def _coerce_datetime(value: Any, field_name: str) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
            return datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ConfigError(f"{field_name} 必须是 ISO-8601 时间：{value!r}") from exc
    raise ConfigError(f"{field_name} 必须是 ISO-8601 时间")


def _first_value(document: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in document and document[name] is not None:
            return document[name]
    return default


def _as_serialized_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _as_serialized_path(value: Path | None) -> str | None:
    return str(value) if value is not None else None


def _path_parts_for_inference(path: Path) -> tuple[str, ...]:
    if not path.is_absolute():
        windows_path = PureWindowsPath(str(path))
        if windows_path.is_absolute():
            return windows_path.parts
    return path.parts


def _coerce_gates(value: Any, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [deepcopy(dict(value))]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{field_name} 必须是列表")
    # workflow 当前会忽略非映射 gate；沿用这一兼容行为，避免旧报告因外围
    # 诊断字段而无法读取。
    return [deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]


def _collect_stage_gates(stages: Sequence["CalibrationStage"]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for stage in stages:
        for raw_gate in stage.quality_gates:
            gate = deepcopy(raw_gate)
            gate_id = str(gate.get("id", "gate"))
            prefix = f"{stage.stage}."
            gate["id"] = gate_id if gate_id.startswith(prefix) else prefix + gate_id
            gates.append(gate)
    return gates


def _counts_from_gates(gates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        status: sum(1 for gate in gates if gate.get("status") == status)
        for status in _GATE_STATUSES
    }


def _infer_project_id(*paths: Path | None) -> str:
    # workflow 路径与 report 路径都可能包含 projects/<id>；前者语义更直接，
    # 调用方按 workflow, source_path 的顺序传入。
    for path in paths:
        if path is None:
            continue
        parts = _path_parts_for_inference(path)
        for index, part in enumerate(parts[:-1]):
            if part.lower() == "projects":
                candidate = parts[index + 1]
                if candidate:
                    return candidate
    # workflow 使用相对路径时，先尝试 source_path 的 reports/<id> 结构，避免
    # 把 <project>/reports/plans/workflow.yaml 误判成 project=reports。
    for path in reversed(paths):
        if path is None:
            continue
        parent = path.parent
        if parent.name.lower() in {"plans", "reports", "runs", "config", "data"}:
            candidate = parent.parent.name
            if candidate and candidate.lower() not in {"plans", "reports", "runs", "config", "data"}:
                return candidate
    return "unknown"


def _infer_run_id(workflow_path: Path, source_path: Path | None) -> str:
    if source_path is not None and source_path.stem:
        return source_path.stem
    if workflow_path.stem:
        return workflow_path.stem
    return "calibration-run"


@dataclass(slots=True)
class CalibrationStage:
    """一次 stage 的可追溯执行记录。

    ``metrics`` 和 ``quality_gates`` 保持字典结构，是因为不同 stage 的算法
    指标并不相同；其余常用元数据采用明确字段，未知旧字段放入 ``extra``。
    """

    stage: str
    status: str
    output_dir: Path | None = None
    result_file: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    quality_gates: list[dict[str, Any]] = field(default_factory=list)
    module: str | None = None
    started_utc: datetime | None = None
    completed_utc: datetime | None = None
    arguments: list[Any] = field(default_factory=list)
    schema_version: int = CALIBRATION_RUN_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.stage = str(self.stage).strip()
        if not self.stage:
            raise ConfigError("stage 名称不能为空")
        self.status = str(self.status or "unknown")
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir).expanduser()
        if self.result_file is not None:
            self.result_file = Path(self.result_file).expanduser()
        if not isinstance(self.metrics, Mapping):
            raise ConfigError(f"{self.stage}.metrics 必须是映射")
        self.metrics = deepcopy(dict(self.metrics))
        if not isinstance(self.quality_gates, Sequence) or isinstance(
            self.quality_gates, (str, bytes)
        ):
            raise ConfigError(f"{self.stage}.quality_gates 必须是列表")
        self.quality_gates = [
            deepcopy(dict(gate)) for gate in self.quality_gates if isinstance(gate, Mapping)
        ]
        if self.module is not None:
            self.module = str(self.module)
        self.started_utc = _coerce_datetime(self.started_utc, f"{self.stage}.started_utc")
        self.completed_utc = _coerce_datetime(self.completed_utc, f"{self.stage}.completed_utc")
        if not isinstance(self.arguments, Sequence) or isinstance(self.arguments, (str, bytes)):
            raise ConfigError(f"{self.stage}.arguments 必须是列表")
        self.arguments = deepcopy(list(self.arguments))
        if not isinstance(self.extra, Mapping):
            raise ConfigError(f"{self.stage}.extra 必须是映射")
        self.extra = deepcopy(dict(self.extra))

    @property
    def name(self) -> str:
        """与部分调用方约定兼容的 stage 名称别名。"""

        return self.stage

    @property
    def started_at(self) -> datetime | None:
        return self.started_utc

    @property
    def completed_at(self) -> datetime | None:
        return self.completed_utc

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        index: int = 1,
        base_dir: Path | None = None,
    ) -> "CalibrationStage":
        if not isinstance(record, Mapping):
            raise ConfigError(f"stages[{index}] 必须是映射")
        stage_value = _first_value(record, "stage", "name")
        if stage_value is None:
            raise ConfigError(f"stages[{index}].stage 不能为空")
        known = {
            "schema_version",
            "stage",
            "name",
            "module",
            "started_utc",
            "started_at",
            "completed_utc",
            "completed_at",
            "arguments",
            "output_dir",
            "result_file",
            "metrics",
            "quality_gates",
            "status",
        }
        extra = {key: deepcopy(value) for key, value in record.items() if key not in known}
        schema_version = record.get("schema_version", CALIBRATION_RUN_SCHEMA_VERSION)
        try:
            schema_version = int(schema_version)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"stages[{index}].schema_version 必须是整数") from exc
        metrics_value = record.get("metrics", {})
        if metrics_value is None:
            metrics_value = {}
        return cls(
            stage=str(stage_value),
            status=str(record.get("status") or "unknown"),
            output_dir=_coerce_optional_path(
                record.get("output_dir"),
                f"stages[{index}].output_dir",
                base_dir=base_dir,
            ),
            result_file=_coerce_optional_path(
                record.get("result_file"),
                f"stages[{index}].result_file",
                base_dir=base_dir,
            ),
            metrics=_copy_mapping(metrics_value, f"stages[{index}].metrics"),
            quality_gates=_coerce_gates(
                record.get("quality_gates", []),
                f"stages[{index}].quality_gates",
            ),
            module=(str(record["module"]) if record.get("module") is not None else None),
            started_utc=_coerce_datetime(
                _first_value(record, "started_utc", "started_at"),
                f"stages[{index}].started_utc",
            ),
            completed_utc=_coerce_datetime(
                _first_value(record, "completed_utc", "completed_at"),
                f"stages[{index}].completed_utc",
            ),
            arguments=deepcopy(record.get("arguments") or []),
            schema_version=schema_version,
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        document = deepcopy(self.extra)
        document.update(
            {
                "schema_version": self.schema_version,
                "stage": self.stage,
                "module": self.module,
                "started_utc": _as_serialized_datetime(self.started_utc),
                "completed_utc": _as_serialized_datetime(self.completed_utc),
                "arguments": deepcopy(self.arguments),
                "output_dir": _as_serialized_path(self.output_dir),
                "result_file": _as_serialized_path(self.result_file),
                "metrics": deepcopy(self.metrics),
                "quality_gates": deepcopy(self.quality_gates),
                "status": self.status,
            }
        )
        return document


# 让 A-2 或旧插件可以使用更具描述性的名称，不创建第二套模型。
CalibrationStageRecord = CalibrationStage


@dataclass(slots=True)
class CalibrationRun:
    """一次完整 calibration workflow 的统一数据模型。"""

    run_id: str
    project_id: str
    workflow_path: Path
    started_utc: datetime | None = None
    completed_utc: datetime | None = None
    status: str = "unknown"
    laser_orientation: str | None = None
    stages: list[CalibrationStage] = field(default_factory=list)
    gates: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)
    overall: str | None = None
    schema_version: int = CALIBRATION_RUN_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_id = str(self.run_id).strip()
        self.project_id = str(self.project_id).strip()
        if not self.run_id:
            raise ConfigError("run_id 不能为空")
        if not self.project_id:
            raise ConfigError("project_id 不能为空")
        if not isinstance(self.workflow_path, (str, Path)):
            raise ConfigError("workflow_path 必须是路径")
        self.workflow_path = Path(self.workflow_path).expanduser()
        self.started_utc = _coerce_datetime(self.started_utc, "started_utc")
        self.completed_utc = _coerce_datetime(self.completed_utc, "completed_utc")
        self.status = str(self.status or "unknown")
        if self.laser_orientation is not None:
            self.laser_orientation = str(self.laser_orientation)
        if not isinstance(self.stages, Sequence) or isinstance(self.stages, (str, bytes)):
            raise ConfigError("stages 必须是列表")
        if not all(isinstance(stage, CalibrationStage) for stage in self.stages):
            raise ConfigError("stages 必须由 CalibrationStage 组成")
        self.stages = list(self.stages)
        if not isinstance(self.gates, Sequence) or isinstance(self.gates, (str, bytes)):
            raise ConfigError("gates 必须是列表")
        self.gates = [deepcopy(dict(gate)) for gate in self.gates if isinstance(gate, Mapping)]
        if not isinstance(self.counts, Mapping):
            raise ConfigError("counts 必须是映射")
        self.counts = deepcopy(dict(self.counts))
        if self.overall is not None:
            self.overall = str(self.overall)
        if not isinstance(self.extra, Mapping):
            raise ConfigError("extra 必须是映射")
        self.extra = deepcopy(dict(self.extra))

    @property
    def workflow(self) -> Path:
        return self.workflow_path

    @property
    def started_at(self) -> datetime | None:
        return self.started_utc

    @property
    def completed_at(self) -> datetime | None:
        return self.completed_utc

    @property
    def overall_status(self) -> str:
        """质量整体状态；缺失时回退到 workflow 执行状态。"""

        return self.overall or self.status

    @classmethod
    def from_report(
        cls,
        report: Mapping[str, Any],
        *,
        run_id: str | None = None,
        project_id: str | None = None,
        workflow_path: str | Path | None = None,
        source_path: str | Path | None = None,
    ) -> "CalibrationRun":
        """从 ``run_workflow`` 返回值或旧的 workflow YAML 内容构造模型。

        当前 report 没有 ``run_id``/``project_id`` 字段，因此未显式传入时，
        分别使用源报告文件名（或 workflow 文件名）以及
        ``projects/<project-id>`` 路径推断。
        """

        if not isinstance(report, Mapping):
            raise ConfigError("workflow report 顶层必须是映射")
        source = (
            Path(source_path).expanduser().resolve() if source_path is not None else None
        )
        raw_workflow = (
            workflow_path
            if workflow_path is not None
            else _first_value(report, "workflow", "workflow_path")
        )
        if raw_workflow is None:
            raise ConfigError("workflow report 缺少 workflow 路径")
        workflow = _coerce_path(
            raw_workflow,
            "workflow",
            base_dir=source.parent if source is not None else None,
        )

        stage_values = report.get("stages", [])
        if stage_values is None:
            stage_values = []
        if isinstance(stage_values, (str, bytes)) or not isinstance(stage_values, Sequence):
            raise ConfigError("stages 必须是列表")
        stages = [
            CalibrationStage.from_record(
                record,
                index=index,
                base_dir=source.parent if source is not None else None,
            )
            for index, record in enumerate(stage_values, start=1)
        ]

        if "gates" in report and report.get("gates") is not None:
            gates = _coerce_gates(report.get("gates"), "gates")
        else:
            gates = _collect_stage_gates(stages)
        computed_counts = _counts_from_gates(gates)
        raw_counts = report.get("counts")
        if raw_counts is None:
            counts: dict[str, Any] = computed_counts
        else:
            counts = _copy_mapping(raw_counts, "counts")
            for status in _GATE_STATUSES:
                counts.setdefault(status, computed_counts[status])

        status = str(report.get("status", "unknown") or "unknown")
        raw_overall = _first_value(report, "overall", "overall_status")
        overall = str(raw_overall) if raw_overall is not None else (
            "fail"
            if status != "completed" or counts.get("fail", 0)
            else "warn" if counts.get("warn", 0) else "pass"
        )
        laser = report.get("laser")
        orientation = (
            _first_value(laser, "orientation")
            if isinstance(laser, Mapping)
            else report.get("laser_orientation")
        )

        inferred_source = source
        resolved_run_id = str(
            run_id
            or report.get("run_id")
            or _infer_run_id(workflow, inferred_source)
        )
        resolved_project_id = str(
            project_id
            or report.get("project_id")
            or _infer_project_id(workflow, inferred_source)
        )

        known = {
            "schema_version",
            "run_id",
            "project_id",
            "workflow",
            "workflow_path",
            "started_utc",
            "started_at",
            "completed_utc",
            "completed_at",
            "status",
            "laser",
            "laser_orientation",
            "stages",
            "gates",
            "counts",
            "overall",
            "overall_status",
        }
        extra = {key: deepcopy(value) for key, value in report.items() if key not in known}
        schema_version = report.get("schema_version", CALIBRATION_RUN_SCHEMA_VERSION)
        try:
            schema_version = int(schema_version)
        except (TypeError, ValueError) as exc:
            raise ConfigError("workflow report schema_version 必须是整数") from exc
        return cls(
            run_id=resolved_run_id,
            project_id=resolved_project_id,
            workflow_path=workflow,
            started_utc=_coerce_datetime(
                _first_value(report, "started_utc", "started_at"), "started_utc"
            ),
            completed_utc=_coerce_datetime(
                _first_value(report, "completed_utc", "completed_at"), "completed_utc"
            ),
            status=status,
            laser_orientation=(str(orientation) if orientation is not None else None),
            stages=stages,
            gates=gates,
            counts=counts,
            overall=overall,
            schema_version=schema_version,
            extra=extra,
        )

    @classmethod
    def from_workflow_report(cls, report: Mapping[str, Any], **kwargs: Any) -> "CalibrationRun":
        return cls.from_report(report, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        document = deepcopy(self.extra)
        document.update(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "project_id": self.project_id,
                "workflow": str(self.workflow_path),
                "started_utc": _as_serialized_datetime(self.started_utc),
                "completed_utc": _as_serialized_datetime(self.completed_utc),
                "status": self.status,
                "laser": {"orientation": self.laser_orientation},
                "stages": [stage.to_dict() for stage in self.stages],
                "gates": deepcopy(self.gates),
                "counts": deepcopy(self.counts),
                "overall": self.overall,
            }
        )
        return document

    def to_report(self) -> dict[str, Any]:
        """返回可与旧 workflow report 对接的普通字典。"""

        return self.to_dict()

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "CalibrationRun":
        """便捷加载入口；实际 YAML 实现位于 ``calibration_run_io``。"""

        from .calibration_run_io import load_calibration_run

        return load_calibration_run(path, **kwargs)

    def save(self, path: str | Path) -> Path:
        """便捷保存入口；实际 YAML 实现位于 ``calibration_run_io``。"""

        from .calibration_run_io import save_calibration_run

        return save_calibration_run(self, path)
