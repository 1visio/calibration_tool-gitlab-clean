"""标定结果产物发现、命名解析和逐图指标关联。

本模块只读取 workflow/stage 已经生成的文件，不调用任何标定算法。解析失败或
文件损坏都保留为可展示的 artifact 记录，由 GUI 决定如何提示用户。
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..io_utils import load_document, resolve_relative


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})
DATA_SUFFIXES = frozenset({".csv", ".yaml", ".yml", ".json"})
_POSE_RE = re.compile(r"(?<!\d)(\d{1,6})(?!\d)")
_SPLIT_POSE_RE = re.compile(r"^(train|fit|validation|test|val)[_-](\d{1,6})(?:[_-].*)?$", re.IGNORECASE)
_CAPTURE_RE = re.compile(r"^(chess|nolaser|laser)[ _-]+(\d{1,6})(?:[_-]\d+)?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ResultArtifact:
    """一个可在结果页中选择的图像或诊断文件。"""

    stage: str
    split: str | None
    pose_id: str | None
    artifact_type: str
    source_path: Path
    image_path: Path | None = None
    status: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def display_path(self) -> Path:
        return self.image_path or self.source_path

    @property
    def label(self) -> str:
        pose = self.pose_id or "汇总"
        return f"{self.split or 'summary'} · {pose} · {self.artifact_type}"


def capture_artifacts_record(plan_path: str | Path | None, output_dir: str | Path) -> dict[str, str]:
    """构造采集完成后写入 WizardProject.extra 的路径记录。"""

    root = Path(output_dir).expanduser().resolve()
    record = {
        "dataset_root": str(root),
        "fit_dir": str(root / "fit"),
        "validation_dir": str(root / "validation"),
        "dataset_manifest": str(root / "dataset_manifest.yaml"),
        "frames_csv": str(root / "frames.csv"),
    }
    if plan_path:
        record["capture_plan"] = str(Path(plan_path).expanduser().resolve())
    return record


def discover_result_artifacts(
    result: Mapping[str, Any] | None,
    capture_artifacts: Mapping[str, Any] | None = None,
) -> list[ResultArtifact]:
    """从 workflow 报告、stage output_dir 和采集路径发现结果产物。"""

    result = result or {}
    roots: list[tuple[str, Path]] = []
    workflow_path = _result_workflow_path(result)
    workflow_document = result.get("workflow") if isinstance(result.get("workflow"), Mapping) else None
    stages = result.get("stages", [])
    if (not isinstance(stages, list) or not stages) and workflow_document is not None:
        stages = workflow_document.get("stages", [])
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, Mapping) or not stage.get("output_dir"):
                continue
            root = _resolve_result_path(workflow_path, stage["output_dir"])
            if root.is_dir():
                roots.append((str(stage.get("stage", root.name)), root))
    if capture_artifacts:
        for key, stage_name in (("fit_dir", "capture"), ("validation_dir", "capture"), ("dataset_root", "capture")):
            value = capture_artifacts.get(key)
            if value:
                root = _resolve_result_path(workflow_path, value)
                if root.is_dir():
                    roots.append((stage_name, root))

    explicit: list[tuple[str, Path]] = []
    raw_artifacts = result.get("artifacts", [])
    if isinstance(raw_artifacts, list):
        for raw in raw_artifacts:
            if isinstance(raw, Mapping) and raw.get("path"):
                path = _resolve_result_path(workflow_path, raw["path"])
                if path.is_file() or path.suffix.lower() in IMAGE_SUFFIXES:
                    explicit.append((str(raw.get("type", "artifact")), path))

    candidates: list[tuple[str, Path]] = explicit[:]
    for stage, root in roots:
        try:
            files = root.rglob("*")
        except OSError:
            continue
        for path in files:
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                candidates.append((stage, path))
            elif suffix in DATA_SUFFIXES and _is_diagnostic_data(path):
                candidates.append((stage, path))

    unique: dict[tuple[str, str], ResultArtifact] = {}
    metric_index = _build_metric_index([path for _, path in candidates if path.suffix.lower() in DATA_SUFFIXES])
    for stage, path in candidates:
        artifact = _make_artifact(stage, path, metric_index)
        key = (artifact.stage, str(artifact.display_path.resolve()))
        unique[key] = artifact
    return sorted(
        unique.values(),
        key=lambda item: (
            item.stage,
            _split_order(item.split),
            item.pose_id or "999999",
            item.artifact_type,
            str(item.display_path).casefold(),
        ),
    )


def _result_workflow_path(result: Mapping[str, Any]) -> Path | None:
    value = result.get("workflow")
    if isinstance(value, str) and value:
        return Path(value).expanduser().resolve()
    if isinstance(value, Mapping):
        nested = value.get("workflow")
        if isinstance(nested, str) and nested:
            return Path(nested).expanduser().resolve()
    return None


def _resolve_result_path(base: Path | None, value: Any) -> Path:
    raw = Path(str(value)).expanduser()
    if raw.is_absolute() or base is None:
        return raw.resolve()
    return resolve_relative(base, raw)


def _is_diagnostic_data(path: Path) -> bool:
    name = path.name.casefold()
    return any(
        token in name
        for token in (
            "metric",
            "error",
            "residual",
            "diagnostic",
            "per_image",
            "pointwise",
            "calibration_points",
            "images",
            "stage_run",
            "manifest",
            "report",
        )
    )


def _make_artifact(stage: str, path: Path, metric_index: Mapping[tuple[str, str], Mapping[str, Any]]) -> ResultArtifact:
    split, pose_id, artifact_type = _parse_name(path)
    metrics = dict(metric_index.get((split or "", pose_id or ""), {})) if pose_id else {}
    status = "missing" if not path.is_file() else _status_from_metrics(metrics)
    return ResultArtifact(
        stage=stage,
        split=split,
        pose_id=pose_id,
        artifact_type=artifact_type,
        source_path=path.resolve(),
        image_path=path.resolve() if path.suffix.lower() in IMAGE_SUFFIXES else None,
        status=status,
        metrics=metrics,
    )


def _parse_name(path: Path) -> tuple[str | None, str | None, str]:
    stem = path.stem
    lower = stem.casefold()
    parent_names = {part.casefold() for part in path.parts}
    if lower.startswith("validation_error_vs_"):
        metric = lower.removeprefix("validation_error_vs_")
        return "validation", None, f"validation_error_vs_{metric}"
    match = _SPLIT_POSE_RE.match(stem)
    if match:
        split = _normalize_split(match.group(1))
        pose = _normalize_pose(match.group(2))
        artifact_type = "extraction_preview" if "extraction" in lower or "preview" in parent_names else "image"
        return split, pose, artifact_type
    match = _CAPTURE_RE.match(stem)
    if match:
        if "reprojection" in parent_names:
            artifact_type = "reprojection_preview"
        elif "residual_vectors" in parent_names or "residual" in parent_names:
            artifact_type = "residual_preview"
        elif "detected" in parent_names:
            artifact_type = "corner_preview"
        elif "undistort_check" in parent_names:
            artifact_type = "undistort_preview"
        else:
            artifact_type = match.group(1).casefold()
        return "validation" if "validation" in parent_names or "test" in parent_names else "fit", _normalize_pose(match.group(2)), artifact_type
    if "preview" in parent_names or "extraction" in lower:
        pose_match = _POSE_RE.search(stem)
        return _infer_split_from_path(path), _normalize_pose(pose_match.group(1)) if pose_match else None, "extraction_preview"
    if "validation" in lower and "error" in lower:
        return "validation", None, lower
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return _infer_split_from_path(path), None, "image"
    if "per_image_metrics" in lower:
        return _infer_split_from_path(path), None, "per_image_metrics"
    if "pointwise_model_errors" in lower:
        return _infer_split_from_path(path), None, "pointwise_model_errors"
    if "calibration_points" in lower:
        return _infer_split_from_path(path), None, "calibration_points"
    if "stage_run" in lower:
        return _infer_split_from_path(path), None, "stage_run"
    return _infer_split_from_path(path), None, "diagnostic"


def _infer_split_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        lower = part.casefold()
        if lower in {"fit", "train"}:
            return "fit"
        if lower in {"validation", "val", "test"}:
            return "validation"
    return None


def _normalize_split(value: str | None) -> str | None:
    if value is None:
        return None
    return "fit" if value.casefold() in {"fit", "train"} else "validation" if value.casefold() in {"validation", "val", "test"} else value.casefold()


def _normalize_pose(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return f"{int(value):03d}"
    except (TypeError, ValueError):
        return str(value)


def _split_order(value: str | None) -> int:
    return {"fit": 0, "validation": 1, None: 2}.get(value, 3)


def _build_metric_index(paths: Iterable[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        try:
            if path.suffix.lower() == ".csv":
                rows = _read_csv(path)
            else:
                rows = list(_mapping_rows(load_document(path)))
        except Exception:
            continue
        for row in rows:
            split = _normalize_split(str(row.get("split")) if row.get("split") else _infer_split_from_path(path))
            pose = _row_pose(row)
            if pose is None:
                continue
            key = (split or "", pose)
            target = index.setdefault(key, {})
            _merge_metrics(target, row)
    return index


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _mapping_rows(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if any(key in value for key in ("pose_id", "image_id", "image", "frame_key", "id")):
            yield value
        for child in value.values():
            yield from _mapping_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from _mapping_rows(child)


def _row_pose(row: Mapping[str, Any]) -> str | None:
    for key in ("pose_id", "image_id", "frame_key", "image", "filename", "id"):
        value = row.get(key)
        if value in (None, ""):
            continue
        match = _POSE_RE.search(str(value))
        if match:
            return _normalize_pose(match.group(1))
    return None


def _merge_metrics(target: dict[str, Any], row: Mapping[str, Any]) -> None:
    aliases = {
        "pnp_rmse_px": ("pnp_rmse_px", "pnp_rmse", "reprojection_rmse_px"),
        "laser_center_point_count": (
            "laser_center_point_count",
            "center_point_count",
            "laser_point_count",
            "valid_intersections",
        ),
        "participates_in_fit": ("participates_in_fit", "included_in_fit", "used_for_fit"),
        "excluded": ("excluded", "excluded_from_fit"),
        "quality_warnings": ("quality_warnings", "warnings", "quality_warning"),
        "validation_error": ("validation_error", "rmse_mm", "board_error_mm", "surface_distance_mm", "ray_euclidean_error_mm"),
        "validation_p95": ("p95_abs_mm", "p95_mm", "board_p95_abs_mm", "surface_p95_abs_mm"),
        "valid_rate": ("valid_rate",),
    }
    for output, keys in aliases.items():
        if output in target:
            continue
        for key in keys:
            if key in row and row[key] not in (None, ""):
                target[output] = _coerce_value(row[key])
                break


def _coerce_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.casefold() in {"true", "false"}:
        return text.casefold() == "true"
    try:
        return float(text) if any(char in text for char in ".eE") else int(text)
    except ValueError:
        return text


def _status_from_metrics(metrics: Mapping[str, Any]) -> str | None:
    if not metrics:
        return None
    if metrics.get("excluded") is True:
        return "excluded"
    warnings = metrics.get("quality_warnings")
    if warnings not in (None, "", [], (), False):
        return "warning"
    return "ok"


__all__ = ["ResultArtifact", "capture_artifacts_record", "discover_result_artifacts", "IMAGE_SUFFIXES"]
