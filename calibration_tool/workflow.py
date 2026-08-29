from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .errors import ConfigError, StageExecutionError
from .io_utils import dump_yaml, load_document, resolve_relative
from .laser import normalize_laser_orientation
from .stages import ComputationService, options_to_argv


PATH_OPTIONS = {
    "fit_dir",
    "test_dir",
    "validation_dir",
    "input_dir",
    "output",
    "output_dir",
    "intrinsics",
    "laser_plane",
    "laser_model",
    "model_config",
    "extrinsics",
    "ground_bias_table",
    "config",
    "image",
    "adapter",
}


def run_workflow(
    plan_path: str | Path,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
    laser_orientation: str = "horizontal",
) -> dict[str, Any]:
    path = Path(plan_path).expanduser().resolve()
    plan = load_document(path)
    if plan.get("schema_version") != 1:
        raise ConfigError("workflow schema_version 必须为 1")
    calibration_src_value = plan.get("calibration_src")
    if not isinstance(calibration_src_value, str):
        raise ConfigError("workflow.calibration_src 必须是路径")
    service = ComputationService(resolve_relative(path, calibration_src_value))
    orientation = normalize_laser_orientation(laser_orientation)
    records: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    for index, raw in enumerate(plan.get("stages", []), start=1):
        if not isinstance(raw, Mapping):
            raise ConfigError(f"stages[{index}] 必须是映射")
        if raw.get("enabled", True) is False:
            continue
        name = str(raw.get("name", ""))
        options = raw.get("options", {})
        if not isinstance(options, Mapping):
            raise ConfigError(f"stages[{index}].options 必须是映射")
        try:
            if progress:
                progress({"event": "stage_started", "stage": name, "index": index})
            resolved_options = _resolve_stage_options(path, dict(options))
            if name == "laser_surface_models":
                configured = resolved_options.get("laser_orientation")
                if configured is not None and configured != orientation:
                    raise ConfigError(
                        "laser_surface_models.laser_orientation 与项目 laser.orientation 不一致："
                        f"{configured!r} != {orientation!r}"
                    )
                resolved_options["laser_orientation"] = orientation
            record = service.run(
                name,
                options_to_argv(resolved_options),
                allow_quality_failure=bool(raw.get("allow_quality_failure", False)),
            )
            records.append(record)
            if progress:
                progress({"event": "stage_finished", "stage": name, "status": record.get("status")})
        except StageExecutionError:
            if not bool(plan.get("continue_on_error", False)):
                raise
            record = {"stage": name, "status": "failed"}
            records.append(record)
            if progress:
                progress({"event": "stage_finished", "stage": name, "status": "failed"})
    report = {
        "schema_version": 1,
        "workflow": str(path),
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if all(item.get("status") == "completed" for item in records) else "failed",
        "laser": {"orientation": orientation},
        "stages": records,
    }
    gates: list[dict[str, Any]] = []
    for record in records:
        stage_name = str(record.get("stage", "unknown"))
        for raw_gate in record.get("quality_gates", []):
            if not isinstance(raw_gate, Mapping):
                continue
            gate = dict(raw_gate)
            gate_id = str(gate.get("id", "gate"))
            gate["id"] = gate_id if gate_id.startswith(f"{stage_name}.") else f"{stage_name}.{gate_id}"
            gates.append(gate)
    counts = {
        status: sum(1 for gate in gates if gate.get("status") == status)
        for status in ("pass", "warn", "fail")
    }
    report["gates"] = gates
    report["counts"] = counts
    report["overall"] = (
        "fail"
        if report["status"] != "completed" or counts["fail"]
        else "warn" if counts["warn"] else "pass"
    )
    output_value = plan.get("report")
    if isinstance(output_value, str) and output_value:
        dump_yaml(resolve_relative(path, output_value), report)
    return report


def _resolve_stage_options(plan_path: Path, options: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for name, value in options.items():
        normalized = name.removeprefix("--").replace("-", "_")
        if normalized in PATH_OPTIONS and isinstance(value, str):
            resolved[name] = str(resolve_relative(plan_path, value))
        else:
            resolved[name] = value
    return resolved
