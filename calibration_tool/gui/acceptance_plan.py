"""项目默认验收计划及 workflow 路径更新辅助逻辑。"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..io_utils import dump_yaml, load_document, resolve_relative


@dataclass(frozen=True, slots=True)
class AcceptancePlanUpdate:
    plan_path: Path
    changed: bool
    workflow_report: Path | None = None
    compensation_metrics: Path | None = None


def default_acceptance_plan_path(
    workspace: str | Path,
    existing_path: str | Path | None = None,
) -> Path:
    """返回项目验收计划路径；已有显式路径始终优先。"""

    if existing_path:
        return Path(existing_path).expanduser().resolve()
    return Path(workspace).expanduser().resolve() / "plans" / "acceptance_plan.yaml"


def ensure_default_acceptance_plan(
    workspace: str | Path,
    project_id: str,
    *,
    existing_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    """为未指定验收计划的项目创建一个安全的默认计划。

    已有显式计划不会被读取、改写或覆盖。默认计划的 policy 和 golden
    baseline 使用仓库内的正式文件，workflow/补偿等运行时输入留空，
    由 workflow 完成后再补齐。
    """

    path = default_acceptance_plan_path(workspace, existing_path)
    if existing_path or path.is_file():
        return path

    root = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    policy = root / "configs" / "acceptance_policy.yaml"
    golden = root / "golden" / "baseline.yaml"
    slug = _slug(project_id) or "calibration"
    document = {
        "schema_version": 1,
        "report_id": f"{slug}-acceptance",
        "title": f"{project_id.strip() or '当前项目'}标定验收报告",
        "output_dir": _relative_path(path.parent, path.parent.parent / "reports" / "acceptance"),
        "policy": _relative_path(path.parent, policy),
        "inputs": {
            "workflow_report": None,
            "quality_reports": [],
            "runtime_config": None,
            "expected_extractor": None,
            "golden_baseline": _relative_path(path.parent, golden) if golden.is_file() else None,
            "compensation_metrics": None,
            "artifacts": [],
        },
        "release": {"enabled": False},
    }
    _write_yaml_atomic(path, document)
    return path


def update_acceptance_plan_from_workflow(
    plan_path: str | Path,
    workflow_path: str | Path,
    workflow_result: Mapping[str, Any] | None = None,
) -> AcceptancePlanUpdate:
    """将本次 workflow 产生的报告和补偿产物填入验收计划。

    只填写空输入项，用户已明确配置的值保持不变。
    """

    plan_file = Path(plan_path).expanduser().resolve()
    workflow_file = Path(workflow_path).expanduser().resolve()
    document = load_document(plan_file)
    inputs = document.get("inputs")
    if inputs is None:
        inputs = {}
        document["inputs"] = inputs
    if not isinstance(inputs, dict):
        raise ValueError("acceptance plan.inputs 必须是映射")

    result = dict(workflow_result or {})
    report_path = _materialize_workflow_report(plan_file, workflow_file, result)
    records = _stage_records(result, report_path)
    compensation_path = _find_compensation_metrics(records)
    changed = False

    if report_path is not None:
        report_value = _relative_path(plan_file.parent, report_path)
        changed |= _set_if_empty(inputs, "workflow_report", report_value)
        changed |= _set_if_empty(inputs, "quality_reports", [report_value])

    if compensation_path is not None:
        compensation_value = _relative_path(plan_file.parent, compensation_path)
        changed |= _set_if_empty(inputs, "compensation_metrics", compensation_value)
    artifacts = _workflow_artifacts(records, compensation_path)
    if artifacts:
        changed |= _set_if_empty(
            inputs,
            "artifacts",
            [_relative_path(plan_file.parent, item) for item in artifacts],
        )

    if changed:
        _write_yaml_atomic(plan_file, document)
    return AcceptancePlanUpdate(plan_file, changed, report_path, compensation_path)


def _materialize_workflow_report(
    plan_file: Path,
    workflow_file: Path,
    result: Mapping[str, Any],
) -> Path | None:
    report_path: Path | None = None
    try:
        workflow = load_document(workflow_file)
    except Exception:
        workflow = {}
    report_value = workflow.get("report") if isinstance(workflow, Mapping) else None
    if isinstance(report_value, str) and report_value.strip():
        report_path = resolve_relative(workflow_file, report_value)
    elif result:
        report_path = plan_file.parent.parent / "reports" / f"{workflow_file.stem}_run.yaml"
    if report_path is not None and result and not report_path.is_file():
        _write_yaml_atomic(report_path, dict(result))
    return report_path if report_path is not None and report_path.is_file() else None


def _stage_records(result: Mapping[str, Any], report_path: Path | None) -> list[Mapping[str, Any]]:
    stages = result.get("stages")
    if (not isinstance(stages, list) or not stages) and report_path is not None:
        try:
            report = load_document(report_path)
        except Exception:
            report = {}
        stages = report.get("stages") if isinstance(report, Mapping) else None
    if not isinstance(stages, list):
        return []
    return [item for item in stages if isinstance(item, Mapping)]


def _find_compensation_metrics(records: list[Mapping[str, Any]]) -> Path | None:
    for record in records:
        if str(record.get("stage", "")) != "ground_bias":
            continue
        value = record.get("result_file")
        if not value:
            continue
        path = Path(str(value)).expanduser().resolve()
        if path.is_file():
            return path
    return None


def _compensation_artifacts(metrics_path: Path) -> list[Path]:
    names = (
        "ground_bias_table.csv",
        "ground_bias_table.npy",
        "z_profile_before_after.png",
        "pointcloud_before_after.png",
    )
    return [metrics_path.parent / name for name in names if (metrics_path.parent / name).is_file()]


def _workflow_artifacts(records: list[Mapping[str, Any]], compensation_path: Path | None) -> list[Path]:
    paths: list[Path] = []
    for record in records:
        value = record.get("result_file")
        if value:
            path = Path(str(value)).expanduser().resolve()
            if path == compensation_path:
                continue
            if path.is_file() and path not in paths:
                paths.append(path)
    if compensation_path is not None:
        for path in _compensation_artifacts(compensation_path):
            if path not in paths:
                paths.append(path)
    return paths


def _set_if_empty(mapping: dict[str, Any], key: str, value: Any) -> bool:
    current = mapping.get(key)
    if current not in (None, "", []):
        return False
    mapping[key] = value
    return True


def _relative_path(base: Path, target: Path) -> str:
    try:
        return target.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        try:
            return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()
        except ValueError:
            # Windows 临时目录可能位于与仓库不同的盘符；此时只能
            # 使用绝对路径，验收加载器同样支持绝对路径。
            return target.resolve().as_posix()


def _write_yaml_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        dump_yaml(temporary, document)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value.strip())


__all__ = [
    "AcceptancePlanUpdate",
    "default_acceptance_plan_path",
    "ensure_default_acceptance_plan",
    "update_acceptance_plan_from_workflow",
]
