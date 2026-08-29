#!/usr/bin/env python3
"""Manage and audit the Phase-A baseline/laser-angle screening experiment."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml
from scipy.interpolate import BSpline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "geometry_baseline_angle"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration_tool.camera.models import (  # noqa: E402
    CameraConfig,
    CapturePlan,
    CaptureTask,
)
from calibration_tool.camera.plan_builder import save_generated_capture_plan  # noqa: E402
from calibration_tool.camera.quality import laser_column_metrics  # noqa: E402
from calibration_tool.io_utils import load_document, resolve_relative, sha256_file  # noqa: E402

BASELINE_LEVELS = (
    ("0", "B00"),
    ("5", "B05"),
    ("12.5", "B12p5"),
)
LASER_ANGLES = (5, 10, 15, 20)

# Future analysis code must preserve these columns from the existing CSV.
MANUAL_FIELDS = (
    "baseline_actual_mm",
    "manual_notes",
)

AUTOMATED_FIELDS = (
    "status",
    "capture_complete",
    "exclude_reason",
    "phaseA_selected",
    "data_dir",
    "captured_frame_count",
    "valid_frame_count",
    "laser_coverage",
    "laser_fwhm_p50_px",
    "laser_fwhm_p95_px",
    "laser_saturation_fraction",
    "board_reprojection_rmse_px",
    "geometry_score",
    "screening_status",
    "analyzed_at_utc",
)

CSV_FIELDS = (
    "config_id",
    "baseline_scale_reading",
    "laser_angle_deg",
    *MANUAL_FIELDS,
    *AUTOMATED_FIELDS,
)

EXPERIMENT_TYPE = "geometry_baseline_angle"
EXPERIMENT_PHASE = "phase_A_screening"
WORKING_DISTANCE_NOMINAL_MM = 1000
MEASUREMENT_PIECE_HEIGHTS_MM = (1, 10, 30)
REFERENCE_EXPOSURE_US = 1500.0
MEASUREMENT_EXPOSURE_US = 1500.0
EXPOSURE_OVERRIDES = {"B12p5_A20": 1900.0}
INVALID_FOV_CONFIG_ID = "B00_A05"
REFERENCE_DEVELOPMENT_CONFIG_ID = "B12p5_A20"
CAPTURED_CONFIG_IDS = tuple(
    f"{baseline_id}_A{angle:02d}"
    for _baseline, baseline_id in BASELINE_LEVELS
    for angle in LASER_ANGLES
    if f"{baseline_id}_A{angle:02d}" != INVALID_FOV_CONFIG_ID
)
MULTIHEIGHT_ROIS = (
    ("h001", "H1", 1.0, "#00bcd4"),
    ("h010", "H10", 10.0, "#ff9800"),
    ("h030", "H30", 30.0, "#e91e63"),
)
BAND_FIX_BACKGROUND_FALSE_RATE_MAX = 0.02
BAND_FIX_PRIMARY_CENTER_SHIFT_MAX_PX = 0.05
BAND_FIX_PRIMARY_SIGMA_P95_RATIO_MAX = 1.25
EXPECTED_TASK_FRAMES = {"reference": 50, "multiheight": 50}
CAMERA_FIELDS = (
    "exposure_us",
    "gain_db",
    "pixel_format",
    "width",
    "height",
    "offset_x",
    "offset_y",
)
ANALYSIS_NAN_FIELDS = (
    "valid_frame_count",
    "laser_coverage",
    "laser_fwhm_p50_px",
    "laser_fwhm_p95_px",
    "laser_saturation_fraction",
    "board_reprojection_rmse_px",
    "geometry_score",
    "screening_status",
    "analyzed_at_utc",
)
AUDIT_CSV_FIELDS = (
    "config_id",
    "status",
    "capture_complete",
    "exclude_reason",
    "phaseA_selected",
    "dataset_path",
    "dataset_exists",
    "manifest_exists",
    "manifest_status",
    "manifest_frame_count",
    "frames_csv_exists",
    "frames_csv_row_count",
    "reference_manifest_frames",
    "reference_csv_frames",
    "reference_image_count",
    "multiheight_manifest_frames",
    "multiheight_csv_frames",
    "multiheight_image_count",
    "missing_image_count",
    "extra_image_count",
    "exposure_us_values",
    "reference_exposure_us_values",
    "multiheight_exposure_us_values",
    "gain_db_values",
    "pixel_format_values",
    "width_values",
    "height_values",
    "offset_x_values",
    "offset_y_values",
    "camera_parameters_consistent_within_dataset",
    "camera_parameters_match_mode",
    "camera_mismatch_fields",
    "quality_passed_frame_count",
    "quality_warning_frame_count",
    "quality_warning_occurrence_count",
    "quality_warning_counts",
    "transport_warning_occurrence_count",
    "audit_warning_count",
    "audit_warnings",
    "audit_error_count",
    "audit_errors",
)
ROI_AUDIT_FIELDS = (
    "config_id", "reference_x_left", "reference_x_right",
    "h1_x_left", "h1_x_right", "h10_x_left", "h10_x_right",
    "h30_x_left", "h30_x_right", "all_rois_inside_reference", "roi_status",
)
GEOMETRY_SUMMARY_FIELDS = (
    "config_id", "baseline_scale_reading", "laser_angle_deg", "status",
    "reference_cv_interior_rmse_px", "reference_cv_interior_p95_px",
    "sensitivity_h1", "sensitivity_h10", "sensitivity_h30",
    "sensitivity_combined_px_per_mm",
    "sigma_pixel_h10_p95_px", "sigma_pixel_h30_p95_px",
    "sigma_z_pred_h10_p95_mm", "sigma_z_pred_h30_p95_mm",
    "sigma_z_pred_combined_mm", "roi_trim_change_h10", "roi_trim_change_h30",
    "warnings", "needs_manual_review",
)
REFERENCE_CV_WARN_RMSE_PX = 1.0
REFERENCE_CV_WARN_P95_PX = 2.0
STEGER_DIAGNOSTIC_THRESHOLDS = (30.0, 25.0, 20.0, 15.0)
STEGER_DIAGNOSTIC_DERIV_THRESHOLDS = (0.5, 0.4, 0.3)
STEGER_DIAGNOSTIC_VALID_FRACTIONS = (0.8, 0.6, 0.5)


def build_initial_rows() -> list[dict[str, str]]:
    """Return the fixed 3 x 4 experiment matrix with blank result fields."""
    rows: list[dict[str, str]] = []
    for baseline_value, baseline_id in BASELINE_LEVELS:
        for laser_angle in LASER_ANGLES:
            row = {field: "" for field in CSV_FIELDS}
            row.update(
                config_id=f"{baseline_id}_A{laser_angle:02d}",
                baseline_scale_reading=baseline_value,
                laser_angle_deg=str(laser_angle),
            )
            rows.append(row)
    return rows


def initialize_experiment(experiment_dir: Path = DEFAULT_EXPERIMENT_DIR) -> Path:
    """Create experiment directories and exclusively create geometry_master.csv."""
    experiment_dir = experiment_dir.resolve()
    master_path = experiment_dir / "geometry_master.csv"

    if master_path.exists():
        raise FileExistsError(f"拒绝覆盖已存在的文件：{master_path}")

    for relative_dir in ("configs", "configs/generated", "data", "results"):
        (experiment_dir / relative_dir).mkdir(parents=True, exist_ok=True)

    try:
        with master_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(build_initial_rows())
    except FileExistsError as exc:
        raise FileExistsError(f"拒绝覆盖已存在的文件：{master_path}") from exc

    return master_path


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})
    _atomic_write_text(path, stream.getvalue())


def _csv_value(value: Any) -> str | int | float:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _read_master_table(master_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not master_path.is_file():
        raise FileNotFoundError(f"geometry_master.csv 不存在，请先执行 init：{master_path}")
    with master_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or ())
        required = ("config_id", "baseline_scale_reading", "laser_angle_deg")
        missing = [field for field in required if field not in fieldnames]
        if missing:
            raise ValueError(f"geometry_master.csv 缺少字段：{', '.join(missing)}")
        rows = [dict(row) for row in reader]
    if any(None in row for row in rows):
        raise ValueError("geometry_master.csv 存在超出表头的列")

    expected_ids = [row["config_id"] for row in build_initial_rows()]
    actual_ids = [row["config_id"].strip() for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("geometry_master.csv 的12组 config_id 或顺序与固定实验矩阵不一致")
    for field in CSV_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
        for row in rows:
            row.setdefault(field, "")
    return fieldnames, rows


def _parse_number(value: str, field: str, config_id: str) -> int | float:
    text = value.strip()
    if not text:
        raise ValueError(f"{config_id} 的 {field} 不能为空")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"{config_id} 的 {field} 不是有效数字：{text}") from exc
    return int(number) if number.is_integer() else number


def _load_master_rows(master_path: Path) -> list[dict[str, str]]:
    _, rows = _read_master_table(master_path)
    return rows


def _capture_plan_for_row(row: dict[str, str], experiment_dir: Path) -> CapturePlan:
    config_id = row["config_id"].strip()
    baseline_scale = _parse_number(
        row["baseline_scale_reading"], "baseline_scale_reading", config_id
    )
    laser_angle = _parse_number(row["laser_angle_deg"], "laser_angle_deg", config_id)
    baseline_actual_text = row["baseline_actual_mm"].strip()
    baseline_actual = (
        _parse_number(baseline_actual_text, "baseline_actual_mm", config_id)
        if baseline_actual_text
        else None
    )
    measurement_camera = CameraConfig(
        exposure_us=EXPOSURE_OVERRIDES.get(config_id, MEASUREMENT_EXPOSURE_US)
    )
    reference_camera = CameraConfig(
        exposure_us=EXPOSURE_OVERRIDES.get(config_id, REFERENCE_EXPOSURE_US)
    )
    common_tags = {
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_phase": EXPERIMENT_PHASE,
        "config_id": config_id,
        "laser_state": "on",
        "measurement_pieces_fixed": True,
    }
    tasks = (
        CaptureTask(
            task_id="reference",
            pose_id="reference",
            role="reference",
            instruction=(
                "保持激光开启；将棋盘格沿固定导向移动到参考位置；"
                "激光只投射在裸露棋盘格参考面；"
                "1/10/30 mm 测量件保持固定在棋盘格上，不拆卸。"
            ),
            frames=50,
            settle_frames=5,
            image_format="tif",
            quality_mode="laser",
            filename_template="images/reference/{index04}{suffix}",
            config=reference_camera,
            tags={
                **common_tags,
                "measurement_piece_heights_mm": list(MEASUREMENT_PIECE_HEIGHTS_MM),
                "board_position": "reference",
                "laser_target": "bare_chessboard_reference_surface",
            },
        ),
        CaptureTask(
            task_id="multiheight",
            pose_id="measurement",
            role="multiheight",
            instruction=(
                "保持激光开启；将同一块棋盘格沿固定导向移动到 measurement 定位位置；"
                "激光线同时穿过 1 mm、10 mm、30 mm 测量件；"
                "测量件在整个12组实验中保持固定。"
            ),
            frames=50,
            settle_frames=5,
            image_format="tif",
            quality_mode="laser",
            filename_template="images/multiheight/{index04}{suffix}",
            config=measurement_camera,
            tags={
                **common_tags,
                "measurement_piece_heights_mm": list(MEASUREMENT_PIECE_HEIGHTS_MM),
                "board_position": "measurement",
                "laser_target": "multiheight_measurement_pieces",
            },
        ),
    )
    return CapturePlan(
        dataset_id=f"{EXPERIMENT_TYPE}_{config_id}",
        output_dir=(experiment_dir / "data" / config_id).resolve(),
        backend="mvs",
        serial_number="",
        base_config=measurement_camera,
        tasks=tasks,
        metadata={
            "experiment_type": EXPERIMENT_TYPE,
            "experiment_phase": EXPERIMENT_PHASE,
            "config_id": config_id,
            "baseline_scale_reading": baseline_scale,
            "baseline_actual_mm": baseline_actual,
            "laser_angle_deg": laser_angle,
            "working_distance_nominal_mm": WORKING_DISTANCE_NOMINAL_MM,
            "working_distance_calibrated": False,
            "baseline_scale_reading_note": "机械支架刻度，不是实际相机-激光光学基线",
        },
    )


def make_capture_plans(
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Generate one existing-schema capture plan for each matrix row."""
    experiment_dir = experiment_dir.resolve()
    rows = _load_master_rows(experiment_dir / "geometry_master.csv")
    generated_dir = experiment_dir / "configs" / "generated"
    plans = [_capture_plan_for_row(row, experiment_dir) for row in rows]
    targets = [generated_dir / f"{row['config_id'].strip()}.yaml" for row in rows]

    existing = [target for target in targets if target.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"拒绝覆盖已存在的采集计划：{names}")

    return [
        save_generated_capture_plan(plan, target, overwrite=overwrite)
        for plan, target in zip(plans, targets)
    ]


def _split_warnings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _camera_values(rows: Sequence[Mapping[str, str]], field: str) -> list[Any]:
    values: set[Any] = set()
    numeric = field != "pixel_format"
    integer = field in {"width", "height", "offset_x", "offset_y"}
    for row in rows:
        text = str(row.get(field, "")).strip()
        if not text:
            continue
        if not numeric:
            values.add(text)
            continue
        try:
            value = float(text)
        except ValueError:
            values.add(text)
            continue
        values.add(int(value) if integer and value.is_integer() else value)
    return sorted(values, key=lambda value: (str(type(value)), str(value)))


def _values_text(values: Sequence[Any]) -> str:
    return "|".join(str(value) for value in values)


def _new_audit_record(config_id: str, dataset_path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {field: "" for field in AUDIT_CSV_FIELDS}
    record.update(
        config_id=config_id,
        status="incomplete",
        capture_complete=False,
        exclude_reason="",
        phaseA_selected=False,
        dataset_path=str(dataset_path),
        dataset_exists=dataset_path.is_dir(),
        manifest_exists=False,
        frames_csv_exists=False,
        camera_parameters_consistent_within_dataset=False,
        camera_parameters_match_mode=False,
        audit_warnings=[],
        audit_errors=[],
        _camera_values={field: [] for field in CAMERA_FIELDS},
    )
    return record


def _finalize_audit_record(record: dict[str, Any]) -> dict[str, Any]:
    record["audit_warning_count"] = len(record["audit_warnings"])
    record["audit_error_count"] = len(record["audit_errors"])
    record["capture_complete"] = not record["audit_errors"]
    record["phaseA_selected"] = bool(record["capture_complete"])
    record["status"] = "captured" if record["capture_complete"] else "incomplete"
    return record


def _audit_dataset(config_id: str, dataset_path: Path) -> dict[str, Any]:
    record = _new_audit_record(config_id, dataset_path)
    errors: list[str] = record["audit_errors"]
    warnings: list[str] = record["audit_warnings"]
    if not dataset_path.is_dir():
        errors.append("dataset_missing")
        return _finalize_audit_record(record)

    manifest_path = dataset_path / "dataset_manifest.yaml"
    frames_csv_path = dataset_path / "frames.csv"
    record["manifest_exists"] = manifest_path.is_file()
    record["frames_csv_exists"] = frames_csv_path.is_file()

    manifest: dict[str, Any] = {}
    manifest_frames: list[Mapping[str, Any]] = []
    if not manifest_path.is_file():
        errors.append("dataset_manifest_missing")
    else:
        try:
            manifest = load_document(manifest_path)
        except Exception as exc:
            errors.append(f"dataset_manifest_invalid:{exc}")
        else:
            record["manifest_status"] = str(manifest.get("status", ""))
            if manifest.get("status") != "completed":
                errors.append(f"manifest_status_not_completed:{manifest.get('status')}")
            plan = manifest.get("plan")
            metadata = plan.get("metadata") if isinstance(plan, Mapping) else None
            manifest_config_id = metadata.get("config_id") if isinstance(metadata, Mapping) else None
            if manifest_config_id != config_id:
                errors.append(f"manifest_config_id_mismatch:{manifest_config_id}")
            raw_frames = manifest.get("frames")
            if isinstance(raw_frames, list) and all(isinstance(item, Mapping) for item in raw_frames):
                manifest_frames = list(raw_frames)
            else:
                errors.append("manifest_frames_invalid")
            record["manifest_frame_count"] = len(manifest_frames)
            task_states = manifest.get("tasks")
            if not isinstance(task_states, Mapping):
                errors.append("manifest_tasks_invalid")
                task_states = {}
            for task_id, expected in EXPECTED_TASK_FRAMES.items():
                state = task_states.get(task_id)
                if not isinstance(state, Mapping):
                    errors.append(f"manifest_task_missing:{task_id}")
                    continue
                captured = int(state.get("frames_captured") or 0)
                expected_in_manifest = int(state.get("frames_expected") or 0)
                if state.get("status") != "completed":
                    errors.append(f"manifest_task_not_completed:{task_id}")
                if captured != expected or expected_in_manifest != expected:
                    errors.append(
                        f"manifest_task_frame_count:{task_id}:{captured}/{expected_in_manifest}"
                    )

    csv_rows: list[dict[str, str]] = []
    if not frames_csv_path.is_file():
        errors.append("frames_csv_missing")
    else:
        try:
            with frames_csv_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                csv_fields = list(reader.fieldnames or ())
                required_fields = (
                    "task_id", "index", "filename", "quality_passed",
                    "quality_warnings", "transport_warnings", *CAMERA_FIELDS,
                )
                missing_fields = [field for field in required_fields if field not in csv_fields]
                if missing_fields:
                    errors.append(f"frames_csv_fields_missing:{'|'.join(missing_fields)}")
                csv_rows = [dict(row) for row in reader]
        except (OSError, UnicodeError, csv.Error) as exc:
            errors.append(f"frames_csv_invalid:{exc}")
    record["frames_csv_row_count"] = len(csv_rows)

    manifest_keys = {
        (str(item.get("task_id", "")), str(item.get("index", "")), str(item.get("filename", "")))
        for item in manifest_frames
    }
    csv_keys = {
        (row.get("task_id", ""), row.get("index", ""), row.get("filename", ""))
        for row in csv_rows
    }
    if manifest_frames and csv_rows and manifest_keys != csv_keys:
        errors.append("manifest_frames_csv_mismatch")

    csv_filenames = {row.get("filename", "") for row in csv_rows if row.get("filename")}
    missing_images = {
        filename for filename in csv_filenames if not (dataset_path / Path(filename)).is_file()
    }
    actual_images: set[str] = set()
    image_suffixes = {".tif", ".tiff", ".png"}
    for task_id, expected in EXPECTED_TASK_FRAMES.items():
        task_manifest_count = sum(
            str(item.get("task_id", "")) == task_id for item in manifest_frames
        )
        task_csv_rows = [row for row in csv_rows if row.get("task_id") == task_id]
        image_dir = dataset_path / "images" / task_id
        task_images = {
            path.relative_to(dataset_path).as_posix()
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in image_suffixes
        } if image_dir.is_dir() else set()
        actual_images.update(task_images)
        record[f"{task_id}_manifest_frames"] = task_manifest_count
        record[f"{task_id}_csv_frames"] = len(task_csv_rows)
        record[f"{task_id}_image_count"] = len(task_images)
        if not image_dir.is_dir():
            errors.append(f"image_directory_missing:{task_id}")
        if task_manifest_count != expected:
            errors.append(f"manifest_frame_count:{task_id}:{task_manifest_count}")
        if len(task_csv_rows) != expected:
            errors.append(f"csv_frame_count:{task_id}:{len(task_csv_rows)}")
        if len(task_images) != expected:
            errors.append(f"image_frame_count:{task_id}:{len(task_images)}")

    extra_images = actual_images - csv_filenames
    record["missing_image_count"] = len(missing_images)
    record["extra_image_count"] = len(extra_images)
    if missing_images:
        errors.append(f"images_missing:{len(missing_images)}")
    if extra_images:
        errors.append(f"images_not_indexed:{len(extra_images)}")

    camera_values = {field: _camera_values(csv_rows, field) for field in CAMERA_FIELDS}
    record["_camera_values"] = camera_values
    for field, values in camera_values.items():
        record[f"{field}_values"] = _values_text(values)
        if not values:
            errors.append(f"camera_field_empty:{field}")
    for task_id in EXPECTED_TASK_FRAMES:
        task_rows = [row for row in csv_rows if row.get("task_id") == task_id]
        record[f"{task_id}_exposure_us_values"] = _values_text(
            _camera_values(task_rows, "exposure_us")
        )
    record["camera_parameters_consistent_within_dataset"] = all(
        len(values) == 1 for values in camera_values.values()
    )
    if csv_rows and not record["camera_parameters_consistent_within_dataset"]:
        warnings.append("camera_parameters_vary_within_dataset")

    quality_warning_counts: Counter[str] = Counter()
    transport_warning_count = 0
    quality_warning_frames = 0
    quality_passed_frames = 0
    for row in csv_rows:
        quality_warnings = _split_warnings(row.get("quality_warnings"))
        quality_warning_counts.update(quality_warnings)
        quality_warning_frames += bool(quality_warnings)
        quality_passed_frames += str(row.get("quality_passed", "")).strip().lower() == "true"
        transport_warning_count += len(_split_warnings(row.get("transport_warnings")))
    record["quality_passed_frame_count"] = quality_passed_frames
    record["quality_warning_frame_count"] = quality_warning_frames
    record["quality_warning_occurrence_count"] = sum(quality_warning_counts.values())
    record["quality_warning_counts"] = dict(sorted(quality_warning_counts.items()))
    record["transport_warning_occurrence_count"] = transport_warning_count
    if quality_warning_frames:
        warnings.append(f"quality_warning_frames:{quality_warning_frames}")
    if transport_warning_count:
        warnings.append(f"transport_warning_occurrences:{transport_warning_count}")
    return _finalize_audit_record(record)


def _apply_cross_dataset_camera_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    captured = [
        record for record in records
        if record["config_id"] != INVALID_FOV_CONFIG_ID and record["dataset_exists"]
    ]
    field_summary: dict[str, Any] = {}
    for field in CAMERA_FIELDS:
        values = sorted(
            {value for record in captured for value in record["_camera_values"][field]},
            key=lambda value: (str(type(value)), str(value)),
        )
        field_summary[field] = {"consistent": len(values) <= 1, "values": values}

    signatures = [
        tuple((field, tuple(record["_camera_values"][field])) for field in CAMERA_FIELDS)
        for record in captured
        if all(len(record["_camera_values"][field]) == 1 for field in CAMERA_FIELDS)
    ]
    mode_signature = Counter(signatures).most_common(1)[0][0] if signatures else tuple()
    mode_values = {field: list(values) for field, values in mode_signature}
    for record in records:
        if record["config_id"] == INVALID_FOV_CONFIG_ID or not record["dataset_exists"]:
            record["camera_parameters_match_mode"] = False
            record["camera_mismatch_fields"] = ""
            continue
        mismatch = [
            field for field in CAMERA_FIELDS
            if record["_camera_values"][field] != mode_values.get(field, [])
        ]
        record["camera_parameters_match_mode"] = not mismatch
        record["camera_mismatch_fields"] = "|".join(mismatch)
        if mismatch:
            record["audit_warnings"].append(
                f"camera_parameters_differ_from_mode:{'|'.join(mismatch)}"
            )
            record["audit_warning_count"] = len(record["audit_warnings"])
    return {
        "consistent": all(item["consistent"] for item in field_summary.values()),
        "fields": field_summary,
        "mode": mode_values,
    }


def _public_audit_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _update_master_from_audit(
    master_path: Path,
    fieldnames: list[str],
    master_rows: list[dict[str, str]],
    records: Sequence[Mapping[str, Any]],
    root: Path,
) -> None:
    by_id = {str(record["config_id"]): record for record in records}
    try:
        data_root_text = root.relative_to(master_path.parent).as_posix()
    except ValueError:
        data_root_text = str(root)
    for row in master_rows:
        config_id = row["config_id"].strip()
        if config_id == INVALID_FOV_CONFIG_ID:
            row.update(
                status="invalid_fov",
                capture_complete="false",
                exclude_reason="laser_out_of_fov",
                phaseA_selected="false",
                data_dir="",
                captured_frame_count="0",
            )
            for field in ANALYSIS_NAN_FIELDS:
                row[field] = "NaN"
            continue
        record = by_id[config_id]
        complete = bool(record["capture_complete"])
        row.update(
            status="captured" if complete else "incomplete",
            capture_complete="true" if complete else "false",
            exclude_reason="" if complete else "capture_audit_incomplete",
            phaseA_selected="true" if complete else "false",
            data_dir=f"{data_root_text}/{config_id}",
            captured_frame_count=str(record["frames_csv_row_count"] or 0),
        )
    _write_csv(master_path, fieldnames, master_rows)


def audit_captures(root: Path, master_path: Path) -> dict[str, Any]:
    """Audit Phase-A capture provenance without reading or modifying image bytes."""
    root = root.expanduser().resolve()
    master_path = master_path.expanduser().resolve()
    fieldnames, master_rows = _read_master_table(master_path)
    records: list[dict[str, Any]] = []
    for row in master_rows:
        config_id = row["config_id"].strip()
        dataset_path = root / config_id
        if config_id == INVALID_FOV_CONFIG_ID:
            record = _new_audit_record(config_id, dataset_path)
            record.update(
                status="invalid_fov",
                capture_complete=False,
                exclude_reason="laser_out_of_fov",
                phaseA_selected=False,
                dataset_exists=False,
                audit_warning_count=0,
                audit_error_count=0,
            )
        else:
            record = _audit_dataset(config_id, dataset_path)
        records.append(record)

    camera_consistency = _apply_cross_dataset_camera_audit(records)
    _update_master_from_audit(master_path, fieldnames, master_rows, records, root)
    public_records = [_public_audit_record(record) for record in records]
    normal_records = [record for record in records if record["config_id"] != INVALID_FOV_CONFIG_ID]
    summary = {
        "expected_conditions": len(master_rows),
        "captured_conditions": sum(bool(record["dataset_exists"]) for record in normal_records),
        "invalid_fov": 1,
        "complete_datasets": sum(bool(record["capture_complete"]) for record in normal_records),
        "incomplete_datasets": sum(not bool(record["capture_complete"]) for record in normal_records),
    }
    unexpected = sorted(
        path.name for path in root.iterdir()
        if path.is_dir()
        and path.name not in {row["config_id"].strip() for row in master_rows}
    ) if root.is_dir() else []
    result = {
        "schema_version": 1,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_phase": EXPERIMENT_PHASE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "master": str(master_path),
        "summary": summary,
        "camera_consistency": camera_consistency,
        "unexpected_dataset_directories": unexpected,
        "datasets": public_records,
    }
    results_dir = master_path.parent / "results"
    _write_csv(results_dir / "capture_audit.csv", AUDIT_CSV_FIELDS, public_records)
    _atomic_write_text(
        results_dir / "capture_audit.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )
    return result


def _load_realtime_steger(calibration_src: Path):
    module_path = calibration_src.expanduser().resolve() / "realtime_steger.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"正式 realtime_steger.py 不存在：{module_path}")
    module_name = "_geometry_experiment_realtime_steger"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载正式 Steger 模块：{module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_reference_analysis_config(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    document = load_document(source)
    if int(document.get("schema_version") or 0) != 1:
        raise ValueError("analysis.yaml schema_version 必须为 1")
    reference = document.get("reference")
    if not isinstance(reference, Mapping):
        raise ValueError("analysis.yaml 缺少 reference 映射")
    valid_fraction = float(reference.get("valid_frame_fraction_min", 0.80))
    max_gap = int(reference.get("max_interp_gap_px", -1))
    if not 0.0 < valid_fraction <= 1.0:
        raise ValueError("valid_frame_fraction_min 必须位于 (0, 1]")
    if max_gap < 0:
        raise ValueError("max_interp_gap_px 必须是非负整数")
    reference_surface = document.get("reference_surface")
    if not isinstance(reference_surface, Mapping):
        raise ValueError("analysis.yaml 缺少 reference_surface 映射")
    raw_x_range = reference_surface.get("x_range")
    x_range: tuple[int, int] | None
    if raw_x_range is None:
        x_range = None
    elif (
        isinstance(raw_x_range, Sequence)
        and not isinstance(raw_x_range, (str, bytes))
        and len(raw_x_range) == 2
    ):
        try:
            x_left = int(raw_x_range[0])
            x_right = int(raw_x_range[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("reference_surface.x_range 必须是 [x_left, x_right]") from exc
        if x_left < 0 or x_right < x_left:
            raise ValueError("reference_surface.x_range 必须满足 0 <= x_left <= x_right")
        x_range = (x_left, x_right)
    else:
        raise ValueError("reference_surface.x_range 必须为 null 或 [x_left, x_right]")
    segment_edge_trim_px = int(reference_surface.get("segment_edge_trim_px", 2))
    smooth_spline_basis_count = int(reference_surface.get("smooth_spline_basis_count", 12))
    smooth_spline_penalty = float(reference_surface.get("smooth_spline_penalty", 1.0))
    robust_huber_delta = float(reference_surface.get("robust_huber_delta", 1.5))
    robust_max_iterations = int(reference_surface.get("robust_max_iterations", 15))
    cv_min_segment_width_px = int(reference_surface.get("cv_min_segment_width_px", 5))
    if segment_edge_trim_px < 0:
        raise ValueError("segment_edge_trim_px 必须是非负整数")
    if smooth_spline_basis_count < 4:
        raise ValueError("smooth_spline_basis_count 至少为 4")
    if smooth_spline_penalty < 0.0:
        raise ValueError("smooth_spline_penalty 必须是非负数")
    if robust_huber_delta <= 0.0 or robust_max_iterations < 1:
        raise ValueError("robust Huber 参数无效")
    if cv_min_segment_width_px < 4:
        raise ValueError("cv_min_segment_width_px 至少为 4")
    multiheight = document.get("multiheight") or {}
    if not isinstance(multiheight, Mapping):
        raise ValueError("analysis.yaml multiheight 必须为映射")
    plateau = multiheight.get("plateau_detection") or {}
    trim_check = multiheight.get("roi_trim_sensitivity") or {}
    if not isinstance(plateau, Mapping) or not isinstance(trim_check, Mapping):
        raise ValueError("multiheight plateau/trim 配置必须为映射")
    multiheight_valid_fraction = float(multiheight.get("valid_frame_fraction_min", 0.80))
    reference_band_margin = int(multiheight.get("reference_band_margin_px", 15))
    formal_roi_trim = int(multiheight.get("formal_roi_trim_px", 3))
    median_window = int(plateau.get("median_smoothing_window_px", 5))
    max_gradient = float(plateau.get("max_abs_gradient_px_per_column", 0.08))
    max_step = float(plateau.get("max_abs_step_px", 0.25))
    sigma_mad_scale = float(plateau.get("sigma_mad_scale", 4.0))
    sigma_floor_limit = float(plateau.get("sigma_floor_limit_px", 0.03))
    plateau_erosion = int(plateau.get("plateau_erosion_px", 2))
    min_stable_width = int(plateau.get("min_stable_width_px", 12))
    trim_values = tuple(int(value) for value in trim_check.get("trim_px", (0, 2, 3, 4)))
    trim_max_relative_change = float(trim_check.get("max_relative_change", 0.02))
    if not 0.0 < multiheight_valid_fraction <= 1.0:
        raise ValueError("multiheight.valid_frame_fraction_min 必须位于 (0, 1]")
    if formal_roi_trim < 0:
        raise ValueError("multiheight.formal_roi_trim_px 必须是非负整数")
    if reference_band_margin < 0:
        raise ValueError("multiheight.reference_band_margin_px 必须是非负整数")
    if median_window < 1 or median_window % 2 == 0:
        raise ValueError("median_smoothing_window_px 必须是正奇数")
    if max_gradient <= 0.0 or max_step <= 0.0:
        raise ValueError("plateau gradient/step 阈值必须为正数")
    if sigma_mad_scale <= 0.0 or sigma_floor_limit <= 0.0:
        raise ValueError("plateau sigma 阈值必须为正数")
    if plateau_erosion < 0 or min_stable_width < 1:
        raise ValueError("plateau erosion/min width 配置无效")
    if trim_values != (0, 2, 3, 4) or not 0.0 < trim_max_relative_change < 1.0:
        raise ValueError("roi_trim_sensitivity 必须使用 trim [0,2,3,4] 和有效相对变化阈值")
    if formal_roi_trim not in trim_values:
        raise ValueError("formal_roi_trim_px 必须包含在 roi_trim_sensitivity.trim_px 中")
    steger_value = document.get("steger_config")
    if not isinstance(steger_value, str) or not steger_value.strip():
        raise ValueError("analysis.yaml 必须指定正式 steger_config")
    return {
        "source": source,
        "valid_frame_fraction_min": valid_fraction,
        "max_interp_gap_px": max_gap,
        "reference_surface_x_range": x_range,
        "segment_edge_trim_px": segment_edge_trim_px,
        "smooth_spline_basis_count": smooth_spline_basis_count,
        "smooth_spline_penalty": smooth_spline_penalty,
        "robust_huber_delta": robust_huber_delta,
        "robust_max_iterations": robust_max_iterations,
        "cv_min_segment_width_px": cv_min_segment_width_px,
        "multiheight_valid_fraction_min": multiheight_valid_fraction,
        "reference_band_margin_px": reference_band_margin,
        "formal_roi_trim_px": formal_roi_trim,
        "plateau_median_window_px": median_window,
        "plateau_max_gradient_px_per_column": max_gradient,
        "plateau_max_step_px": max_step,
        "plateau_sigma_mad_scale": sigma_mad_scale,
        "plateau_sigma_floor_limit_px": sigma_floor_limit,
        "plateau_erosion_px": plateau_erosion,
        "min_stable_width_px": min_stable_width,
        "roi_trim_values_px": trim_values,
        "roi_trim_max_relative_change": trim_max_relative_change,
        "steger_config": resolve_relative(source, steger_value),
    }


def _task_frame_records(dataset: Path, task_id: str) -> list[dict[str, str]]:
    manifest = load_document(dataset / "dataset_manifest.yaml")
    if manifest.get("status") != "completed":
        raise ValueError(f"dataset_manifest status 不是 completed：{manifest.get('status')}")
    plan = manifest.get("plan")
    metadata = plan.get("metadata") if isinstance(plan, Mapping) else None
    config_id = metadata.get("config_id") if isinstance(metadata, Mapping) else None
    if dataset.name not in CAPTURED_CONFIG_IDS or config_id != dataset.name:
        raise ValueError(
            f"dataset/config_id 不属于已采集 Phase-A 组或不一致：{dataset.name} / {config_id}"
        )
    frames_path = dataset / "frames.csv"
    if not frames_path.is_file():
        raise FileNotFoundError(f"frames.csv 不存在：{frames_path}")
    with frames_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = ("task_id", "index", "filename", "pixel_format")
        missing = [field for field in required if field not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"frames.csv 缺少字段：{', '.join(missing)}")
        records = [dict(row) for row in reader if row.get("task_id") == task_id]
    records.sort(key=lambda row: int(row["index"]))
    if len(records) != 50:
        raise ValueError(f"{task_id} 应为50帧，实际 {len(records)}")
    expected_indices = list(range(1, 51))
    actual_indices = [int(row["index"]) for row in records]
    if actual_indices != expected_indices:
        raise ValueError(f"{task_id} 帧索引必须为1..50，实际 {actual_indices}")
    for row in records:
        relative = Path(row["filename"])
        if relative.parts[:2] != ("images", task_id) or ".." in relative.parts:
            raise ValueError(f"拒绝读取 images/{task_id} 之外的图像：{relative}")
        image_path = dataset / relative
        if not image_path.is_file():
            raise FileNotFoundError(f"{task_id} 图像不存在：{image_path}")
    return records


def _reference_frame_records(dataset: Path) -> list[dict[str, str]]:
    return _task_frame_records(dataset, "reference")


def _read_gray_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV 无法读取图像：{path}")
    if image.ndim != 2 or image.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"reference 图像必须是二维 uint8/uint16：{path}")
    return image


def _sensor_max_value(image: np.ndarray, pixel_format: str) -> float:
    if image.dtype == np.uint8 or pixel_format == "Mono8":
        return 255.0
    if pixel_format == "Mono12":
        return 4095.0
    return float(np.iinfo(image.dtype).max)


def _nan_text(value: Any) -> Any:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return numeric if np.isfinite(numeric) else "NaN"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _registry_x_range(value: Any, field: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{field} 必须为 null 或 [x_left, x_right]")
    try:
        left, right = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须为整数 [x_left, x_right]") from exc
    if left < 0 or right < left:
        raise ValueError(f"{field} 必须满足 0 <= x_left <= x_right")
    return left, right


def _roi_selected_x_range(roi: Mapping[str, Any], roi_id: str) -> tuple[int, int] | None:
    value = roi.get("selected_x_range") if "selected_x_range" in roi else roi.get("x_range")
    return _registry_x_range(value, f"{roi_id}.selected_x_range")


def _load_roi_registry(path: Path, config_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source = path.expanduser().resolve()
    document = load_document(source)
    if int(document.get("schema_version") or 0) != 1:
        raise ValueError("roi_registry.yaml schema_version 必须为 1")
    configs = document.get("configs")
    if not isinstance(configs, Mapping) or not isinstance(configs.get(config_id), Mapping):
        raise ValueError(f"roi_registry.yaml 缺少 configs.{config_id}")
    if not isinstance(configs, dict):
        configs = dict(configs)
        document["configs"] = configs
    raw_entry = configs[config_id]
    if isinstance(raw_entry, dict):
        config_entry = raw_entry
    else:
        config_entry = dict(raw_entry)
        configs[config_id] = config_entry
    reference_surface = config_entry.get("reference_surface")
    if not isinstance(reference_surface, Mapping):
        raise ValueError(f"configs.{config_id} 缺少 reference_surface")
    _registry_x_range(
        reference_surface.get("x_range"), f"configs.{config_id}.reference_surface.x_range"
    )
    multiheight = config_entry.get("multiheight")
    if not isinstance(multiheight, Mapping):
        raise ValueError(f"configs.{config_id} 缺少 multiheight")
    for roi_id, _label, expected_height, _color in MULTIHEIGHT_ROIS:
        roi = multiheight.get(roi_id)
        if not isinstance(roi, Mapping):
            raise ValueError(f"configs.{config_id}.multiheight 缺少 {roi_id}")
        if float(roi.get("height_mm")) != expected_height:
            raise ValueError(f"{roi_id}.height_mm 必须为 {expected_height}")
        _roi_selected_x_range(roi, roi_id)
    return source, document, config_entry


def _write_roi_registry(path: Path, document: Mapping[str, Any]) -> None:
    text = yaml.safe_dump(
        dict(document),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    _atomic_write_text(path, text)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_roi_config_entry(config_id: str) -> dict[str, Any]:
    return {
        "status": "invalid_fov" if config_id == INVALID_FOV_CONFIG_ID else "pending",
        "manual_confirmed": False,
        "updated_at": _utc_now(),
        "reference_surface": {"x_range": None, "method": None},
        "multiheight": {
            roi_id: {"height_mm": height_mm, "selected_x_range": None}
            for roi_id, _label, height_mm, _color in MULTIHEIGHT_ROIS
        },
    }


def ensure_roi_registry(path: Path) -> tuple[Path, dict[str, Any]]:
    """Add missing matrix entries without replacing any existing manual ROI."""
    source = path.expanduser().resolve()
    if source.is_file():
        document = load_document(source)
        if int(document.get("schema_version") or 0) != 1:
            raise ValueError("roi_registry.yaml schema_version 必须为 1")
    else:
        document = {
            "schema_version": 1,
            "experiment_type": EXPERIMENT_TYPE,
            "template_config": REFERENCE_DEVELOPMENT_CONFIG_ID,
            "configs": {},
        }
    configs = document.get("configs")
    if not isinstance(configs, dict):
        configs = dict(configs) if isinstance(configs, Mapping) else {}
        document["configs"] = configs
    changed = False
    for row in build_initial_rows():
        config_id = row["config_id"]
        if not isinstance(configs.get(config_id), Mapping):
            configs[config_id] = _new_roi_config_entry(config_id)
            changed = True
    template = configs[REFERENCE_DEVELOPMENT_CONFIG_ID]
    template_reference = _registry_x_range(
        template.get("reference_surface", {}).get("x_range"),
        f"configs.{REFERENCE_DEVELOPMENT_CONFIG_ID}.reference_surface.x_range",
    )
    template_rois_complete = all(
        _roi_selected_x_range(template["multiheight"][roi_id], roi_id) is not None
        for roi_id, _label, _height, _color in MULTIHEIGHT_ROIS
    )
    if template_reference == (935, 2236) and template_rois_complete:
        for key, value in (("status", "confirmed"), ("manual_confirmed", True)):
            if template.get(key) != value:
                template[key] = value
                changed = True
        if not template.get("updated_at"):
            template["updated_at"] = _utc_now()
            changed = True
    invalid = configs[INVALID_FOV_CONFIG_ID]
    if invalid.get("status") != "invalid_fov" or invalid.get("manual_confirmed") is not False:
        invalid["status"] = "invalid_fov"
        invalid["manual_confirmed"] = False
        invalid["updated_at"] = _utc_now()
        changed = True
    if changed or not source.is_file():
        _write_roi_registry(source, document)
    return source, document


def _roi_entry_issues(
    config_entry: Mapping[str, Any],
    image_width: int,
    formal_trim_px: int,
    minimum_formal_width_px: int,
) -> list[str]:
    issues: list[str] = []
    reference = _registry_x_range(
        config_entry.get("reference_surface", {}).get("x_range"),
        "reference_surface.x_range",
    )
    if reference is None:
        return ["reference_roi_missing"]
    if reference[1] >= image_width:
        issues.append("reference_roi_out_of_image")
    ranges: dict[str, tuple[int, int]] = {}
    multiheight = config_entry.get("multiheight")
    if not isinstance(multiheight, Mapping):
        return issues + ["multiheight_mapping_missing"]
    for roi_id, _label, _height, _color in MULTIHEIGHT_ROIS:
        roi = multiheight.get(roi_id)
        if not isinstance(roi, Mapping):
            issues.append(f"{roi_id}_missing")
            continue
        bounds = _roi_selected_x_range(roi, roi_id)
        if bounds is None:
            issues.append(f"{roi_id}_missing")
            continue
        ranges[roi_id] = bounds
        if bounds[1] >= image_width:
            issues.append(f"{roi_id}_out_of_image")
        if not (reference[0] <= bounds[0] <= bounds[1] <= reference[1]):
            issues.append(f"{roi_id}_outside_reference")
        formal_width = bounds[1] - bounds[0] + 1 - 2 * formal_trim_px
        if formal_width < minimum_formal_width_px:
            issues.append(f"{roi_id}_trimmed_width_below_min")
    ids = list(ranges)
    for index, roi_id in enumerate(ids):
        for other_id in ids[index + 1:]:
            left, right = ranges[roi_id], ranges[other_id]
            if min(left[1], right[1]) >= max(left[0], right[0]):
                issues.append(f"roi_overlap:{roi_id}:{other_id}")
    return issues


def audit_rois(
    root: Path,
    registry: Path,
    analysis_config: Path = DEFAULT_EXPERIMENT_DIR / "configs" / "analysis.yaml",
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    registry_path, document = ensure_roi_registry(registry)
    config = _load_reference_analysis_config(analysis_config)
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    configs = document["configs"]
    for matrix_row in build_initial_rows():
        config_id = matrix_row["config_id"]
        row = {field: "" for field in ROI_AUDIT_FIELDS}
        row["config_id"] = config_id
        if config_id == INVALID_FOV_CONFIG_ID:
            row.update(all_rois_inside_reference=False, roi_status="invalid_fov")
            counts["invalid_fov"] += 1
            rows.append(row)
            continue
        entry = configs[config_id]
        reference = _registry_x_range(
            entry["reference_surface"].get("x_range"),
            f"configs.{config_id}.reference_surface.x_range",
        )
        if reference is not None:
            row["reference_x_left"], row["reference_x_right"] = reference
        for roi_id, prefix in (("h001", "h1"), ("h010", "h10"), ("h030", "h30")):
            bounds = _roi_selected_x_range(entry["multiheight"][roi_id], roi_id)
            if bounds is not None:
                row[f"{prefix}_x_left"], row[f"{prefix}_x_right"] = bounds
        dataset = root / config_id
        try:
            records = _task_frame_records(dataset, "reference")
            image_width = _read_gray_image(dataset / records[0]["filename"]).shape[1]
            issues = _roi_entry_issues(
                entry,
                image_width,
                config["formal_roi_trim_px"],
                config["min_stable_width_px"],
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            issues = [f"dataset_invalid:{exc}"]
        row["all_rois_inside_reference"] = not any(
            "outside_reference" in issue or "missing" in issue for issue in issues
        ) and reference is not None
        if entry.get("status") == "failed":
            roi_status = "failed"
        elif issues:
            roi_status = "pending" if all("missing" in issue for issue in issues) else "failed"
        elif entry.get("status") == "confirmed" and entry.get("manual_confirmed") is True:
            roi_status = "confirmed"
        else:
            roi_status = "pending"
        row["roi_status"] = roi_status
        counts[roi_status] += 1
        rows.append(row)
    results_path = root.parent / "results" / "roi_audit.csv"
    _write_csv(results_path, ROI_AUDIT_FIELDS, rows)
    return {
        "registry": str(registry_path),
        "output": str(results_path),
        "rows": rows,
        "counts": dict(counts),
    }


def _validate_roi_candidate(
    config_entry: Mapping[str, Any],
    roi_id: str,
    candidate: tuple[int, int],
    image_width: int,
    *,
    reference_range: tuple[int, int] | None = None,
    formal_trim_px: int | None = None,
    minimum_formal_width_px: int | None = None,
) -> None:
    left, right = candidate
    if left < 0 or right >= image_width or left > right:
        raise ValueError(f"ROI 必须位于图像列范围 [0, {image_width - 1}] 内")
    if reference_range is not None and not (
        reference_range[0] <= left <= right <= reference_range[1]
    ):
        raise ValueError(f"ROI 必须完全位于 reference_surface {list(reference_range)} 内")
    if formal_trim_px is not None and minimum_formal_width_px is not None:
        formal_width = right - left + 1 - 2 * formal_trim_px
        if formal_width < minimum_formal_width_px:
            raise ValueError(
                f"ROI trim {formal_trim_px}px 后宽度 {formal_width}px，"
                f"小于最低 {minimum_formal_width_px}px"
            )
    multiheight = config_entry["multiheight"]
    for other_id, _label, _height, _color in MULTIHEIGHT_ROIS:
        if other_id == roi_id:
            continue
        other = _roi_selected_x_range(multiheight[other_id], other_id)
        if other is None:
            continue
        overlap = min(right, other[1]) - max(left, other[0]) + 1
        if overlap > 0:
            raise ValueError(
                f"{roi_id} 与 {other_id} 重叠 {overlap}px；三个测量件 ROI 不允许重叠"
            )


def _surface_bounds(config: Mapping[str, Any], width: int, *, required: bool) -> tuple[int, int] | None:
    x_range = config["reference_surface_x_range"]
    if x_range is None:
        if required:
            raise ValueError(
                "reference_surface.x_range 仍为 null；请先运行 preview-reference-roi，"
                "人工确认棋盘格基准面的 [x_left, x_right]"
            )
        return None
    left, right = x_range
    if right >= width:
        raise ValueError(f"reference_surface.x_range 超出图像宽度 {width}：[{left}, {right}]")
    return int(left), int(right)


def _extract_reference_stacks(
    dataset: Path,
    realtime: Any,
    steger_options: Mapping[str, Any],
    *,
    detail_output: Path | None = None,
    collect_median_image: bool = False,
    task_id: str = "reference",
    additional_band_bounds: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Run the formal extractor; an existing per-frame detail file is never overwritten."""
    records = _task_frame_records(dataset, task_id)
    per_frame_fields = (
        "frame_index", "filename", "u", "y_subpixel_px", "valid",
        "steger_response", "steger_offset_px", "steger_normal_y_abs",
        "fwhm_px", "background_dn", "peak_dn", "peak_contrast_dn",
        "quality_active", "peak_saturated", "peak_near_saturated",
        "saturated_width_px", "column_saturation_fraction",
    )
    temporary_csv: Path | None = None
    detail_stream: Any = None
    writer: csv.DictWriter[str] | None = None
    detail_was_preserved = bool(detail_output and detail_output.exists())
    if detail_output is not None and not detail_was_preserved:
        detail_output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".reference_frame_columns.", suffix=".csv.tmp", dir=str(detail_output.parent)
        )
        os.close(descriptor)
        temporary_csv = Path(temporary_name)
        detail_stream = temporary_csv.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(detail_stream, fieldnames=per_frame_fields, lineterminator="\n")
        writer.writeheader()

    y_stack: np.ndarray | None = None
    valid_stack: np.ndarray | None = None
    fwhm_stack: np.ndarray | None = None
    image_stack: np.ndarray | None = None
    image_shape: tuple[int, int] | None = None
    completed = False
    band_diagnostics: list[dict[str, float | None]] = []
    steger_time_ms = 0.0
    try:
        for frame_position, frame_record in enumerate(records):
            image_path = dataset / Path(frame_record["filename"])
            image = _read_gray_image(image_path)
            if image_shape is None:
                image_shape = image.shape
                width = image.shape[1]
                y_stack = np.full((len(records), width), np.nan, dtype=np.float64)
                valid_stack = np.zeros((len(records), width), dtype=bool)
                fwhm_stack = np.full((len(records), width), np.nan, dtype=np.float64)
                if collect_median_image:
                    image_stack = np.empty((len(records), *image.shape), dtype=image.dtype)
            elif image.shape != image_shape:
                raise ValueError(f"reference 图像尺寸不一致：{image_path} {image.shape} != {image_shape}")
            if image_stack is not None:
                image_stack[frame_position] = image
            sensor_max = _sensor_max_value(image, frame_record.get("pixel_format", ""))
            steger_started = time.perf_counter()
            if additional_band_bounds is None:
                extracted = realtime.extract_steger_columns(image, steger_options)
            else:
                extracted = realtime.extract_steger_columns(
                    image,
                    steger_options,
                    additional_band_bounds=additional_band_bounds,
                )
            steger_time_ms += (time.perf_counter() - steger_started) * 1000.0
            extracted_metadata = getattr(extracted, "metadata", {})
            band_diagnostics.append({
                field: extracted_metadata.get(field)
                for field in (
                    "original_band_top_px",
                    "original_band_bottom_exclusive_px",
                    "reference_envelope_top_px",
                    "reference_envelope_bottom_exclusive_px",
                    "final_band_top_px",
                    "final_band_bottom_exclusive_px",
                )
            })
            metrics = laser_column_metrics(image, sensor_max_value=sensor_max)
            valid = np.asarray(extracted.valid, dtype=bool) & np.isfinite(extracted.v_px)
            assert y_stack is not None and valid_stack is not None and fwhm_stack is not None
            y_stack[frame_position, valid] = extracted.v_px[valid]
            valid_stack[frame_position] = valid
            fwhm = np.asarray(metrics["fwhm_px"], dtype=np.float64)
            fwhm_stack[frame_position] = fwhm
            if writer is not None:
                background = np.asarray(metrics["background_dn"])
                peak = np.asarray(metrics["peak_dn"])
                peak_contrast = np.asarray(metrics["peak_contrast_dn"])
                active = np.asarray(metrics["active"])
                peak_saturated = np.asarray(metrics["peak_saturated"])
                peak_near_saturated = np.asarray(metrics["peak_near_saturated"])
                saturated_width = np.asarray(metrics["saturated_width_px"])
                saturation_fraction = np.asarray(metrics["column_saturation_fraction"])
                for column in range(image.shape[1]):
                    writer.writerow({
                        "frame_index": int(frame_record["index"]),
                        "filename": frame_record["filename"],
                        "u": column,
                        "y_subpixel_px": _nan_text(extracted.v_px[column]),
                        "valid": "true" if valid[column] else "false",
                        "steger_response": _nan_text(extracted.response[column]),
                        "steger_offset_px": _nan_text(extracted.offset_px[column]),
                        "steger_normal_y_abs": _nan_text(extracted.normal_y_abs[column]),
                        "fwhm_px": int(fwhm[column]),
                        "background_dn": float(background[column]),
                        "peak_dn": float(peak[column]),
                        "peak_contrast_dn": float(peak_contrast[column]),
                        "quality_active": "true" if active[column] else "false",
                        "peak_saturated": "true" if peak_saturated[column] else "false",
                        "peak_near_saturated": "true" if peak_near_saturated[column] else "false",
                        "saturated_width_px": int(saturated_width[column]),
                        "column_saturation_fraction": float(saturation_fraction[column]),
                    })
        completed = True
    finally:
        if detail_stream is not None:
            detail_stream.close()
        if completed and temporary_csv is not None and detail_output is not None:
            os.replace(temporary_csv, detail_output)
        if temporary_csv is not None:
            temporary_csv.unlink(missing_ok=True)

    assert image_shape is not None
    assert y_stack is not None and valid_stack is not None and fwhm_stack is not None
    median_image = None
    if image_stack is not None:
        median_image = np.median(image_stack, axis=0, overwrite_input=True)
    return {
        "records": records,
        "image_shape": image_shape,
        "y_stack": y_stack,
        "valid_stack": valid_stack,
        "fwhm_stack": fwhm_stack,
        "median_image": median_image,
        "detail_was_preserved": detail_was_preserved,
        "band_diagnostics": band_diagnostics,
        "steger_time_ms": steger_time_ms,
    }


def _aggregate_reference_stacks(stacks: Mapping[str, Any]) -> dict[str, np.ndarray]:
    y_stack = np.asarray(stacks["y_stack"], dtype=np.float64)
    valid_stack = np.asarray(stacks["valid_stack"], dtype=bool)
    fwhm_stack = np.asarray(stacks["fwhm_stack"], dtype=np.float64)
    width = y_stack.shape[1]
    valid_count = np.sum(valid_stack, axis=0)
    valid_fraction = valid_count.astype(np.float64) / float(y_stack.shape[0])
    y_median = np.full(width, np.nan, dtype=np.float64)
    sigma = np.full(width, np.nan, dtype=np.float64)
    fwhm_p50 = np.full(width, np.nan, dtype=np.float64)
    for column in range(width):
        mask = valid_stack[:, column]
        if np.any(mask):
            values = y_stack[mask, column]
            y_median[column] = float(np.median(values))
            sigma[column] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            fwhm_p50[column] = float(np.median(fwhm_stack[mask, column]))
    return {
        "u": np.arange(width, dtype=np.float64),
        "y_median": y_median,
        "sigma": sigma,
        "fwhm_p50": fwhm_p50,
        "valid_count": valid_count,
        "valid_fraction": valid_fraction,
    }


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _fit_robust_low_df_spline(
    x: np.ndarray,
    y: np.ndarray,
    domain: tuple[int, int],
    *,
    requested_basis_count: int,
    penalty: float,
    huber_delta: float,
    max_iterations: int,
) -> tuple[BSpline, dict[str, Any]]:
    if x.size < 4:
        raise ValueError(f"reference surface 内经 segment-edge trim 后可靠点不足：{x.size} < 4")
    degree = min(3, int(x.size) - 1)
    basis_count = min(max(degree + 1, requested_basis_count), int(x.size))
    internal_count = basis_count - degree - 1
    left, right = map(float, domain)
    internal = (
        np.linspace(left, right, internal_count + 2, dtype=np.float64)[1:-1]
        if internal_count
        else np.empty(0, dtype=np.float64)
    )
    knots = np.concatenate((np.repeat(left, degree + 1), internal, np.repeat(right, degree + 1)))
    design = BSpline.design_matrix(x, knots, degree, extrapolate=False).toarray()
    difference = np.diff(np.eye(basis_count, dtype=np.float64), n=2, axis=0)
    weights = np.ones(x.size, dtype=np.float64)
    coefficients = np.zeros(basis_count, dtype=np.float64)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        root_weights = np.sqrt(weights)
        matrix = design * root_weights[:, None]
        target = y * root_weights
        if penalty > 0.0 and difference.size:
            matrix = np.vstack((matrix, np.sqrt(penalty) * difference))
            target = np.concatenate((target, np.zeros(difference.shape[0], dtype=np.float64)))
        updated, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        residual = y - design @ updated
        centered = residual - np.median(residual)
        scale = 1.4826 * float(np.median(np.abs(centered)))
        if scale <= np.finfo(np.float64).eps:
            coefficients = updated
            break
        cutoff = huber_delta * scale
        absolute = np.abs(centered)
        new_weights = np.ones_like(weights)
        outliers = absolute > cutoff
        new_weights[outliers] = cutoff / absolute[outliers]
        converged = np.max(np.abs(updated - coefficients)) <= 1e-9 and np.max(
            np.abs(new_weights - weights)
        ) <= 1e-6
        coefficients = updated
        weights = new_weights
        if converged:
            break
    return BSpline(knots, coefficients, degree, extrapolate=False), {
        "model": "robust_penalized_cubic_bspline" if degree == 3 else "robust_penalized_bspline",
        "degree": degree,
        "basis_count": basis_count,
        "penalty": penalty,
        "huber_delta": huber_delta,
        "iterations": iterations,
    }


def _build_surface_reference(
    aggregates: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    u = aggregates["u"]
    raw_y = aggregates["y_median"]
    valid_fraction = aggregates["valid_fraction"]
    width = u.size
    bounds = _surface_bounds(config, width, required=True)
    assert bounds is not None
    left, right = bounds
    inside = (u >= left) & (u <= right)
    raw_reliable = (
        (valid_fraction >= config["valid_frame_fraction_min"])
        & np.isfinite(raw_y)
    )
    reliable_inside = raw_reliable & inside
    source = np.full(width, "outside_reference_surface", dtype=object)
    source[inside] = "invalid"
    source[reliable_inside] = "observed"

    runs = _true_runs(reliable_inside)
    trim = config["segment_edge_trim_px"]
    for start, end in runs:
        edge_end = min(end, start + trim - 1)
        if edge_end >= start:
            source[start:edge_end + 1] = "segment_edge_excluded"
        edge_start = max(start, end - trim + 1)
        if edge_start <= end:
            source[edge_start:end + 1] = "segment_edge_excluded"

    y_short_gap = np.full(width, np.nan, dtype=np.float64)
    for (left_start, left_end), (right_start, right_end) in zip(runs, runs[1:]):
        gap_start = left_end + 1
        gap_end = right_start - 1
        gap = gap_end - gap_start + 1
        left_anchor = left_end - trim
        right_anchor = right_start + trim
        if (
            gap <= 0
            or gap > config["max_interp_gap_px"]
            or left_anchor < left_start
            or right_anchor > right_end
            or source[left_anchor] != "observed"
            or source[right_anchor] != "observed"
        ):
            continue
        target = np.arange(gap_start, gap_end + 1)
        y_short_gap[target] = np.interp(
            target, (left_anchor, right_anchor), (raw_y[left_anchor], raw_y[right_anchor])
        )
        source[target] = "short_gap_interpolated"

    fit_mask = source == "observed"
    spline, model_info = _fit_robust_low_df_spline(
        u[fit_mask],
        raw_y[fit_mask],
        bounds,
        requested_basis_count=config["smooth_spline_basis_count"],
        penalty=config["smooth_spline_penalty"],
        huber_delta=config["robust_huber_delta"],
        max_iterations=config["robust_max_iterations"],
    )
    y_smooth = np.full(width, np.nan, dtype=np.float64)
    y_smooth[inside] = spline(u[inside])
    source[(source == "invalid") & inside & np.isfinite(y_smooth)] = "smooth_model_filled"
    residual = raw_y[fit_mask] - y_smooth[fit_mask]
    return {
        "bounds": bounds,
        "inside": inside,
        "raw_reliable": raw_reliable,
        "source": source,
        "y_ref_observed": np.where(reliable_inside, raw_y, np.nan),
        "y_ref_short_gap": y_short_gap,
        "y_ref_smooth": y_smooth,
        "fit_mask": fit_mask,
        "residual": residual,
        "model_info": model_info,
    }


def _cross_validate_reference_segments(
    aggregates: Mapping[str, np.ndarray],
    surface: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the smooth model without changing the final fit or any observed values."""
    u = np.asarray(aggregates["u"], dtype=np.float64)
    raw_y = np.asarray(aggregates["y_median"], dtype=np.float64)
    fit_mask = np.asarray(surface["fit_mask"], dtype=bool)
    segments = _true_runs(fit_mask)
    minimum_width = int(config["cv_min_segment_width_px"])
    eligible = [
        (segment, "boundary" if index in {0, len(segments) - 1} else "interior")
        for index, segment in enumerate(segments)
        if segment[1] - segment[0] + 1 >= minimum_width
    ]
    rows: list[dict[str, Any]] = []
    residual_u: list[np.ndarray] = []
    residual_values: list[np.ndarray] = []
    residual_segment_number: list[np.ndarray] = []
    residual_roles: list[np.ndarray] = []

    for number, ((start, end), segment_role) in enumerate(eligible, start=1):
        held_out = np.zeros_like(fit_mask)
        held_out[start:end + 1] = fit_mask[start:end + 1]
        training = fit_mask & ~held_out
        if np.count_nonzero(training) < 4:
            raise ValueError(
                f"CV 留出 segment {start}..{end} 后训练点不足，无法拟合相同 reference 模型"
            )
        spline, _ = _fit_robust_low_df_spline(
            u[training],
            raw_y[training],
            surface["bounds"],
            requested_basis_count=config["smooth_spline_basis_count"],
            penalty=config["smooth_spline_penalty"],
            huber_delta=config["robust_huber_delta"],
            max_iterations=config["robust_max_iterations"],
        )
        held_u = u[held_out]
        predicted = spline(held_u)
        residual = raw_y[held_out] - predicted
        absolute = np.abs(residual)
        if not np.all(np.isfinite(residual)):
            raise RuntimeError(f"CV segment {start}..{end} 产生非有限预测")
        segment_id = f"S{number:03d}"
        rows.append({
            "segment_id": segment_id,
            "segment_role": segment_role,
            "u_start": start,
            "u_end": end,
            "width_px": end - start + 1,
            "point_count": int(residual.size),
            "rmse_px": float(np.sqrt(np.mean(np.square(residual)))),
            "mae_px": float(np.mean(absolute)),
            "p95_abs_error_px": float(np.percentile(absolute, 95)),
            "max_abs_error_px": float(np.max(absolute)),
        })
        residual_u.append(held_u)
        residual_values.append(residual)
        residual_segment_number.append(np.full(residual.size, number, dtype=np.int32))
        residual_roles.append(np.full(residual.size, segment_role, dtype=object))

    if not rows:
        raise ValueError(
            f"没有宽度达到 {minimum_width}px 的 trim 后 observed segment，无法执行 segment CV"
        )
    all_residual = np.concatenate(residual_values)
    all_absolute = np.abs(all_residual)
    all_roles = np.concatenate(residual_roles)
    interior_residual = all_residual[all_roles == "interior"]
    boundary_residual = all_residual[all_roles == "boundary"]
    if interior_residual.size == 0:
        raise ValueError("没有同时被左右可靠 observed 数据包围的 interior segment，无法计算正式 CV")
    if boundary_residual.size == 0:
        raise ValueError("没有可诊断的 boundary segment，无法计算 boundary CV")
    interior_absolute = np.abs(interior_residual)
    statistics = {
        # Retain the all-segment values for provenance/backward compatibility.
        "reference_cv_rmse_px": float(np.sqrt(np.mean(np.square(all_residual)))),
        "reference_cv_mae_px": float(np.mean(all_absolute)),
        "reference_cv_p95_px": float(np.percentile(all_absolute, 95)),
        "reference_cv_max_px": float(np.max(all_absolute)),
        # Only these interior values are the formal interpolation CV metrics.
        "reference_cv_interior_rmse_px": float(
            np.sqrt(np.mean(np.square(interior_residual)))
        ),
        "reference_cv_interior_mae_px": float(np.mean(interior_absolute)),
        "reference_cv_interior_p95_px": float(np.percentile(interior_absolute, 95)),
        "reference_cv_interior_max_px": float(np.max(interior_absolute)),
        # Boundary hold-outs diagnose edge behavior but are not formal interpolation CV.
        "reference_cv_boundary_rmse_px": float(
            np.sqrt(np.mean(np.square(boundary_residual)))
        ),
        "reference_cv_boundary_max_px": float(np.max(np.abs(boundary_residual))),
    }
    return {
        "rows": rows,
        "u": np.concatenate(residual_u),
        "residual": all_residual,
        "segment_number": np.concatenate(residual_segment_number),
        "segment_role": all_roles,
        "statistics": statistics,
        "observed_segment_count": len(segments),
        "eligible_segment_count": len(eligible),
        "interior_segment_count": sum(row["segment_role"] == "interior" for row in rows),
        "boundary_segment_count": sum(row["segment_role"] == "boundary" for row in rows),
        "minimum_segment_width_px": minimum_width,
    }


def _save_figure(target: Path, figure: Any) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".png", dir=str(target.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=160)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _save_reference_cv_plot(
    output_dir: Path,
    cross_validation: Mapping[str, Any],
    bounds: tuple[int, int],
    config_id: str = REFERENCE_DEVELOPMENT_CONFIG_ID,
) -> Path:
    import matplotlib.pyplot as plt

    u = np.asarray(cross_validation["u"], dtype=np.float64)
    residual = np.asarray(cross_validation["residual"], dtype=np.float64)
    segment_number = np.asarray(cross_validation["segment_number"], dtype=np.int32)
    segment_role = np.asarray(cross_validation["segment_role"], dtype=object)
    interior = segment_role == "interior"
    boundary = segment_role == "boundary"
    p95 = float(cross_validation["statistics"]["reference_cv_interior_p95_px"])
    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    scatter = axis.scatter(
        u[interior],
        residual[interior],
        c=segment_number[interior],
        cmap="turbo",
        s=6,
        linewidths=0,
        alpha=0.85,
        label="interior segment CV",
    )
    axis.scatter(
        u[boundary],
        residual[boundary],
        marker="x",
        color="#d32f2f",
        s=15,
        linewidths=0.8,
        alpha=0.85,
        label="boundary segment CV (diagnostic only)",
    )
    axis.axhline(0.0, color="#424242", lw=0.8)
    axis.axhline(p95, color="#ef6c00", ls="--", lw=0.8, label="interior |residual| P95")
    axis.axhline(-p95, color="#ef6c00", ls="--", lw=0.8)
    axis.axvline(bounds[0], color="#1b5e20", ls="--", lw=1.0)
    axis.axvline(bounds[1], color="#1b5e20", ls="--", lw=1.0)
    axis.set_xlim(float(bounds[0]), float(bounds[1]))
    axis.set_xlabel("held-out observed u [px]")
    axis.set_ylabel("observed - CV prediction [px]")
    axis.set_title(
        f"{config_id} leave-one-observed-segment-out reference CV residual"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.01)
    colorbar.set_label("interior held-out segment number")
    target = _save_figure(output_dir / "reference_cv_residual.png", figure)
    plt.close(figure)
    return target


def _save_reference_roi_preview(
    output_dir: Path,
    median_image: np.ndarray,
    aggregates: Mapping[str, np.ndarray],
    valid_fraction_min: float,
    bounds: tuple[int, int] | None,
    config_id: str = REFERENCE_DEVELOPMENT_CONFIG_ID,
) -> Path:
    import matplotlib.pyplot as plt

    u = aggregates["u"]
    y = aggregates["y_median"]
    observed = (aggregates["valid_fraction"] >= valid_fraction_min) & np.isfinite(y)
    finite = median_image[np.isfinite(median_image)]
    display_min = float(np.percentile(finite, 1.0))
    display_max = float(np.percentile(finite, 99.8))
    figure, axis = plt.subplots(figsize=(18, 11), constrained_layout=True)
    axis.imshow(median_image, cmap="gray", vmin=display_min, vmax=display_max, origin="upper")
    axis.plot(u[observed], y[observed], ".", ms=1.4, color="#00e5ff", label="raw reliable Steger median")
    if bounds is not None:
        axis.axvspan(bounds[0], bounds[1], color="#76ff03", alpha=0.10, label="configured reference surface")
        axis.axvline(bounds[0], color="#76ff03", lw=1.2)
        axis.axvline(bounds[1], color="#76ff03", lw=1.2)
    axis.set_xlim(0, median_image.shape[1] - 1)
    axis.set_ylim(median_image.shape[0] - 1, 0)
    axis.set_xlabel("u [px]")
    axis.set_ylabel("v [px]")
    axis.set_title(f"{config_id} reference median image + raw Steger centerline (choose x_left / x_right)")
    axis.grid(True, alpha=0.20)
    axis.legend(loc="best")
    target = _save_figure(output_dir / "reference_roi_preview.png", figure)
    plt.close(figure)
    return target


def _save_multiheight_median(output_dir: Path, median_image: np.ndarray, pixel_format: str) -> Path:
    storage_dtype = np.uint8 if pixel_format == "Mono8" else np.uint16
    storage = np.rint(median_image).astype(storage_dtype)
    encoded_ok, encoded = cv2.imencode(".png", storage)
    if not encoded_ok:
        raise RuntimeError("OpenCV 无法编码 multiheight_median.png")
    target = output_dir / "multiheight_median.png"
    _atomic_write_bytes(target, encoded.tobytes())
    return target


def _draw_multiheight_roi_overlay(
    axis: Any,
    median_image: np.ndarray,
    aggregates: Mapping[str, np.ndarray],
    config_entry: Mapping[str, Any],
    valid_fraction_min: float,
    *,
    title: str,
) -> None:
    u = np.asarray(aggregates["u"], dtype=np.float64)
    y = np.asarray(aggregates["y_median"], dtype=np.float64)
    reliable = (
        np.asarray(aggregates["valid_fraction"], dtype=np.float64) >= valid_fraction_min
    ) & np.isfinite(y)
    finite = median_image[np.isfinite(median_image)]
    display_min = float(np.percentile(finite, 1.0))
    display_max = float(np.percentile(finite, 99.8))
    axis.imshow(median_image, cmap="gray", vmin=display_min, vmax=display_max, origin="upper")
    axis.plot(
        u[reliable], y[reliable], ".", ms=1.3, color="#00e5ff",
        label="raw reliable Steger median",
    )
    reference = _registry_x_range(
        config_entry["reference_surface"].get("x_range"), "reference_surface.x_range"
    )
    assert reference is not None
    axis.axvspan(
        reference[0], reference[1], color="#76ff03", alpha=0.06,
        label="reference surface",
    )
    axis.axvline(reference[0], color="#76ff03", lw=1.2, ls="--")
    axis.axvline(reference[1], color="#76ff03", lw=1.2, ls="--")
    unset: list[str] = []
    for roi_id, label, _height, color in MULTIHEIGHT_ROIS:
        roi = _roi_selected_x_range(config_entry["multiheight"][roi_id], roi_id)
        if roi is None:
            unset.append(label)
            continue
        axis.axvspan(roi[0], roi[1], color=color, alpha=0.18, label=f"{label} ROI")
        axis.axvline(roi[0], color=color, lw=1.1)
        axis.axvline(roi[1], color=color, lw=1.1)
        axis.text(
            (roi[0] + roi[1]) / 2.0,
            0.02,
            label,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            color=color,
            fontsize=10,
            fontweight="bold",
        )
    axis.set_xlim(0, median_image.shape[1] - 1)
    axis.set_ylim(median_image.shape[0] - 1, 0)
    axis.set_xlabel("u [px]")
    axis.set_ylabel("v [px]")
    axis.set_title(title + (f" | unset: {', '.join(unset)}" if unset else ""))
    axis.grid(True, alpha=0.20)
    axis.legend(loc="best")


def _multiheight_roi_summary(
    dataset: Path,
    output_dir: Path,
    config_entry: Mapping[str, Any],
    aggregates: Mapping[str, np.ndarray],
    frame_count: int,
    image_shape: tuple[int, int],
    steger_config: Path,
    valid_fraction_min: float,
) -> dict[str, Any]:
    width = image_shape[1]
    reference = _registry_x_range(
        config_entry["reference_surface"].get("x_range"), "reference_surface.x_range"
    )
    assert reference is not None
    valid_fraction = np.asarray(aggregates["valid_fraction"], dtype=np.float64)
    rois: dict[str, Any] = {}
    for roi_id, _label, height_mm, _color in MULTIHEIGHT_ROIS:
        x_range = _roi_selected_x_range(config_entry["multiheight"][roi_id], roi_id)
        if x_range is None:
            rois[roi_id] = {
                "height_mm": height_mm,
                "x_range": None,
                "width_px": None,
                "steger_valid_column_fraction": None,
                "fully_inside_reference_surface": None,
            }
            continue
        if x_range[1] >= width:
            raise ValueError(f"{roi_id}.x_range 超出图像宽度 {width}")
        roi_fraction = valid_fraction[x_range[0]:x_range[1] + 1]
        rois[roi_id] = {
            "height_mm": height_mm,
            "x_range": list(x_range),
            "width_px": x_range[1] - x_range[0] + 1,
            "steger_valid_column_fraction": float(
                np.mean(roi_fraction >= valid_fraction_min)
            ),
            "fully_inside_reference_surface": bool(
                reference[0] <= x_range[0] <= x_range[1] <= reference[1]
            ),
        }
    summary = {
        "schema_version": 1,
        "config_id": dataset.name,
        "task_id": "multiheight",
        "frame_count": frame_count,
        "image_shape": list(image_shape),
        "centre_extractor": "realtime_steger.extract_steger_columns",
        "steger_config": str(steger_config),
        "steger_config_sha256": sha256_file(steger_config),
        "reference_surface_x_range": list(reference),
        "valid_frame_fraction_min": valid_fraction_min,
        "rois": rois,
        "delta_y_computed": False,
        "sensitivity_px_per_mm_computed": False,
        "sigma_z_pred_computed": False,
        "laser_plane_used": False,
        "reconstruction_3d_performed": False,
        "pnp_used": False,
        "reference_model_modified": False,
        "outputs": {
            "median_image": str(output_dir / "multiheight_median.png"),
            "preview": str(output_dir / "multiheight_roi_preview.png"),
        },
    }
    _atomic_write_text(
        output_dir / "multiheight_roi_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary


def _write_multiheight_roi_outputs(
    dataset: Path,
    output_dir: Path,
    config_entry: Mapping[str, Any],
    median_image: np.ndarray,
    aggregates: Mapping[str, np.ndarray],
    frame_count: int,
    image_shape: tuple[int, int],
    steger_config: Path,
    valid_fraction_min: float,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(18, 11), constrained_layout=True)
    _draw_multiheight_roi_overlay(
        axis,
        median_image,
        aggregates,
        config_entry,
        valid_fraction_min,
        title=f"{dataset.name} multiheight median image + raw Steger centerline + ROI registry",
    )
    _save_figure(output_dir / "multiheight_roi_preview.png", figure)
    plt.close(figure)
    return _multiheight_roi_summary(
        dataset,
        output_dir,
        config_entry,
        aggregates,
        frame_count,
        image_shape,
        steger_config,
        valid_fraction_min,
    )


def _prepare_multiheight_roi_annotation(
    dataset: Path,
    registry: Path,
    analysis_config: Path,
    calibration_src: Path,
    *,
    require_reference_analysis: bool = True,
) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    if dataset.name not in CAPTURED_CONFIG_IDS:
        raise ValueError(f"不是已采集的 Phase-A 配置：{dataset.name}")
    config = _load_reference_analysis_config(analysis_config)
    registry_path, registry_document, config_entry = _load_roi_registry(
        registry, dataset.name
    )
    registry_reference = _registry_x_range(
        config_entry["reference_surface"].get("x_range"), "reference_surface.x_range"
    )
    if registry_reference is None:
        raise ValueError(f"{dataset.name} 尚未确认 reference_surface.x_range")
    configured_reference = config["reference_surface_x_range"]
    if dataset.name == REFERENCE_DEVELOPMENT_CONFIG_ID and configured_reference != registry_reference:
        raise ValueError(
            "模板组 roi_registry reference_surface 与 analysis.yaml 不一致："
            f"{registry_reference} != {configured_reference}"
        )
    config = {**config, "reference_surface_x_range": registry_reference}
    output_dir = dataset / "analysis"
    reference_analysis_path = output_dir / "reference_analysis.json"
    if require_reference_analysis:
        reference_analysis = load_document(reference_analysis_path)
        steger_hash = sha256_file(config["steger_config"])
        if reference_analysis.get("steger_config_sha256") != steger_hash:
            raise ValueError("multiheight Steger 配置 hash 与已验证 reference 不一致")
        if reference_analysis.get("centre_extractor") != "realtime_steger.extract_steger_columns":
            raise ValueError("已验证 reference 未使用要求的 realtime_steger.extract_steger_columns")
        if tuple(reference_analysis.get("reference_surface_x_range") or ()) != registry_reference:
            raise ValueError("reference_analysis.json 与 roi_registry reference_surface 不一致")

    realtime = _load_realtime_steger(calibration_src)
    steger_options = realtime.load_steger_options(config["steger_config"])
    first_record = _task_frame_records(dataset, "multiheight")[0]
    first_image = _read_gray_image(dataset / first_record["filename"])
    y_ref_for_band, _frozen_summary, _curve_path = _load_frozen_reference_curve(
        dataset,
        config,
        first_image.shape[1],
    )
    reference_envelope = _reference_vertical_envelope(
        y_ref_for_band,
        registry_reference,
        config["reference_band_margin_px"],
        first_image.shape[0],
    )
    original_stacks = _extract_reference_stacks(
        dataset,
        realtime,
        steger_options,
        collect_median_image=False,
        task_id="multiheight",
    )
    stacks = _extract_reference_stacks(
        dataset,
        realtime,
        steger_options,
        collect_median_image=True,
        task_id="multiheight",
        additional_band_bounds=reference_envelope,
    )
    original_aggregates = _aggregate_reference_stacks(original_stacks)
    aggregates = _aggregate_reference_stacks(stacks)
    median_image = np.asarray(stacks["median_image"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pixel_format = str(stacks["records"][0].get("pixel_format", ""))
    _save_multiheight_median(output_dir, median_image, pixel_format)
    summary = _write_multiheight_roi_outputs(
        dataset,
        output_dir,
        config_entry,
        median_image,
        aggregates,
        len(stacks["records"]),
        stacks["image_shape"],
        config["steger_config"],
        config["multiheight_valid_fraction_min"],
    )
    return {
        "dataset": dataset,
        "output_dir": output_dir,
        "registry_path": registry_path,
        "registry_document": registry_document,
        "config_entry": config_entry,
        "analysis_config": config,
        "median_image": median_image,
        "aggregates": aggregates,
        "original_aggregates": original_aggregates,
        "stacks": stacks,
        "original_stacks": original_stacks,
        "reference_envelope": reference_envelope,
        "frame_count": len(stacks["records"]),
        "image_shape": stacks["image_shape"],
        "summary": summary,
    }


def _prepare_reference_roi_annotation(
    dataset: Path,
    registry: Path,
    analysis_config: Path,
    calibration_src: Path,
) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    if dataset.name not in CAPTURED_CONFIG_IDS:
        raise ValueError(f"不是已采集的 Phase-A 配置：{dataset.name}")
    config = _load_reference_analysis_config(analysis_config)
    registry_path, registry_document, config_entry = _load_roi_registry(
        registry, dataset.name
    )
    realtime = _load_realtime_steger(calibration_src)
    steger_options = realtime.load_steger_options(config["steger_config"])
    stacks = _extract_reference_stacks(
        dataset,
        realtime,
        steger_options,
        collect_median_image=True,
        task_id="reference",
    )
    aggregates = _aggregate_reference_stacks(stacks)
    median_image = np.asarray(stacks["median_image"])
    bounds = _registry_x_range(
        config_entry["reference_surface"].get("x_range"),
        f"configs.{dataset.name}.reference_surface.x_range",
    )
    output_dir = dataset / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_reference_roi_preview(
        output_dir,
        median_image,
        aggregates,
        config["valid_frame_fraction_min"],
        bounds,
        dataset.name,
    )
    return {
        "dataset": dataset,
        "output_dir": output_dir,
        "registry_path": registry_path,
        "registry_document": registry_document,
        "config_entry": config_entry,
        "analysis_config": config,
        "median_image": median_image,
        "aggregates": aggregates,
        "image_shape": stacks["image_shape"],
    }


def _run_reference_roi_editor(context: dict[str, Any]) -> bool:
    import matplotlib.pyplot as plt

    dataset = context["dataset"]
    config_entry = context["config_entry"]
    registry_path = context["registry_path"]
    registry_document = context["registry_document"]
    median_image = context["median_image"]
    aggregates = context["aggregates"]
    valid_fraction_min = context["analysis_config"]["valid_frame_fraction_min"]
    width = context["image_shape"][1]
    current: dict[str, Any] = {"clicks": [], "skipped": False}
    figure, axis = plt.subplots(figsize=(16, 10), constrained_layout=True)
    try:
        figure.canvas.manager.set_window_title(f"{dataset.name} reference surface ROI")
    except AttributeError:
        pass
    status = figure.text(0.01, 0.005, "", ha="left", va="bottom", fontsize=9)

    def bounds() -> tuple[int, int] | None:
        return _registry_x_range(
            config_entry["reference_surface"].get("x_range"),
            "reference_surface.x_range",
        )

    def redraw(message: str) -> None:
        axis.clear()
        u = np.asarray(aggregates["u"], dtype=np.float64)
        y = np.asarray(aggregates["y_median"], dtype=np.float64)
        valid = (
            np.asarray(aggregates["valid_fraction"], dtype=np.float64) >= valid_fraction_min
        ) & np.isfinite(y)
        finite = median_image[np.isfinite(median_image)]
        axis.imshow(
            median_image,
            cmap="gray",
            vmin=float(np.percentile(finite, 1.0)),
            vmax=float(np.percentile(finite, 99.8)),
            origin="upper",
        )
        axis.plot(u[valid], y[valid], ".", ms=1.3, color="#00e5ff", label="raw reliable Steger")
        existing = bounds()
        if existing is not None:
            axis.axvspan(existing[0], existing[1], color="#76ff03", alpha=0.10, label="current reference_surface")
            axis.axvline(existing[0], color="#76ff03", lw=1.2)
            axis.axvline(existing[1], color="#76ff03", lw=1.2)
        for value in current["clicks"]:
            axis.axvline(value, color="#ffffff", lw=1.2, ls=":")
        axis.set_xlim(0, width - 1)
        axis.set_ylim(median_image.shape[0] - 1, 0)
        axis.set_title(f"{dataset.name}: click reference x_left / x_right | Enter accept | R reselect | S skip")
        axis.set_xlabel("u [px]")
        axis.set_ylabel("v [px]")
        axis.legend(loc="best")
        status.set_text(message)
        figure.canvas.draw_idle()

    def persist() -> None:
        _write_roi_registry(registry_path, registry_document)
        _save_reference_roi_preview(
            context["output_dir"],
            median_image,
            aggregates,
            valid_fraction_min,
            bounds(),
            dataset.name,
        )

    def on_click(event: Any) -> None:
        if event.inaxes is not axis or event.button != 1 or event.xdata is None:
            return
        value = int(round(float(event.xdata)))
        if not 0 <= value < width:
            redraw(f"x={value} 超出图像范围")
            return
        current["clicks"].append(value)
        if len(current["clicks"]) == 1:
            redraw(f"已选择第一边界 x={value}；请选择第二边界")
            return
        candidate = tuple(sorted(current["clicks"][:2]))
        config_entry["reference_surface"].update(
            x_range=list(candidate), method="manual_confirmed"
        )
        config_entry.update(
            status="reference_confirmed", manual_confirmed=False, updated_at=_utc_now()
        )
        current["clicks"] = []
        persist()
        redraw(f"已立即保存 reference_surface={list(candidate)}；Enter 接受，或重新点击覆盖")

    def on_key(event: Any) -> None:
        if event.key in {"r", "R", "escape"}:
            current["clicks"] = []
            redraw("已取消当前未完成点击；可重新选择，registry 已保存值不变")
        elif event.key == "enter":
            if bounds() is None:
                redraw("尚未选择 reference_surface，不能确认")
                return
            config_entry.update(
                status="reference_confirmed", manual_confirmed=False, updated_at=_utc_now()
            )
            persist()
            plt.close(figure)
        elif event.key in {"s", "S"}:
            config_entry.update(
                status="skipped", manual_confirmed=False, updated_at=_utc_now()
            )
            current["skipped"] = True
            persist()
            plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)
    redraw("选择棋盘格真实基准面；不要包含左右其他物理平面")
    plt.show()
    return not current["skipped"] and bounds() is not None


def annotate_all_rois(
    root: Path,
    registry: Path,
    analysis_config: Path,
    calibration_src: Path,
) -> dict[str, Any]:
    import matplotlib

    if matplotlib.get_backend().lower() == "agg":
        matplotlib.use("qtagg", force=True)
    root = root.expanduser().resolve()
    registry_path, _document = ensure_roi_registry(registry)
    for config_id in CAPTURED_CONFIG_IDS:
        if config_id == REFERENCE_DEVELOPMENT_CONFIG_ID:
            print(f"{config_id}: 保留已确认模板 ROI，跳过编辑")
            continue
        dataset = root / config_id
        print(f"{config_id}: 准备 reference ROI 预览...")
        try:
            reference_context = _prepare_reference_roi_annotation(
                dataset, registry_path, analysis_config, calibration_src
            )
            if not _run_reference_roi_editor(reference_context):
                print(f"{config_id}: 已跳过")
                continue
            print(f"{config_id}: 准备 multiheight ROI 预览...")
            multiheight_context = _prepare_multiheight_roi_annotation(
                dataset,
                registry_path,
                analysis_config,
                calibration_src,
                require_reference_analysis=False,
            )
            _run_multiheight_roi_editor(multiheight_context)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            _path, document, entry = _load_roi_registry(registry_path, config_id)
            entry.update(
                status="failed", manual_confirmed=False, updated_at=_utc_now(), error=str(exc)
            )
            _write_roi_registry(registry_path, document)
            print(f"{config_id}: 失败：{exc}")
    return audit_rois(root, registry_path, analysis_config)


def _run_multiheight_roi_editor(context: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    dataset = context["dataset"]
    output_dir = context["output_dir"]
    registry_path = context["registry_path"]
    registry_document = context["registry_document"]
    config_entry = context["config_entry"]
    config = context["analysis_config"]
    median_image = context["median_image"]
    aggregates = context["aggregates"]
    image_shape = context["image_shape"]
    roi_order = [item[0] for item in MULTIHEIGHT_ROIS]
    roi_labels = {item[0]: item[1] for item in MULTIHEIGHT_ROIS}
    current = {"roi_id": roi_order[0], "clicks": [], "skipped": False}

    figure, axis = plt.subplots(figsize=(16, 10), constrained_layout=True)
    try:
        figure.canvas.manager.set_window_title(f"{dataset.name} multiheight ROI annotation")
    except AttributeError:
        pass
    status = figure.text(0.01, 0.005, "", ha="left", va="bottom", fontsize=9)

    def set_status(message: str) -> None:
        status.set_text(message)

    def redraw() -> None:
        axis.clear()
        label = roi_labels[current["roi_id"]]
        _draw_multiheight_roi_overlay(
            axis,
            median_image,
            aggregates,
            config_entry,
            config["multiheight_valid_fraction_min"],
            title=f"Select {label}: click left then right boundary",
        )
        for value in current["clicks"]:
            axis.axvline(value, color="#ffffff", lw=1.2, ls=":")
        figure.canvas.draw_idle()

    def persist() -> dict[str, Any]:
        _write_roi_registry(registry_path, registry_document)
        return _write_multiheight_roi_outputs(
            dataset,
            output_dir,
            config_entry,
            median_image,
            aggregates,
            context["frame_count"],
            image_shape,
            config["steger_config"],
            config["multiheight_valid_fraction_min"],
        )

    def select_roi(roi_id: str) -> None:
        current["roi_id"] = roi_id
        current["clicks"] = []
        set_status(
            f"当前 {roi_labels[roi_id]}。左键依次点击左右边界；Esc 取消当前点击；"
            "数字键 1/2/3 重选 H1/H10/H30；Enter 确认；S 跳过。"
        )
        redraw()

    def on_click(event: Any) -> None:
        if event.inaxes is not axis or event.button != 1 or event.xdata is None:
            return
        value = int(round(float(event.xdata)))
        current["clicks"].append(value)
        if len(current["clicks"]) < 2:
            set_status(f"已选择第一边界 x={value}；请点击第二边界，Esc 可取消。")
            redraw()
            return
        candidate = tuple(sorted(current["clicks"][:2]))
        roi_id = current["roi_id"]
        try:
            reference = _registry_x_range(
                config_entry["reference_surface"].get("x_range"), "reference_surface.x_range"
            )
            _validate_roi_candidate(
                config_entry,
                roi_id,
                candidate,
                image_shape[1],
                reference_range=reference,
                formal_trim_px=config["formal_roi_trim_px"],
                minimum_formal_width_px=config["min_stable_width_px"],
            )
        except ValueError as exc:
            current["clicks"] = []
            set_status(f"未保存：{exc}。请重新点击 {roi_labels[roi_id]} 两个边界。")
            redraw()
            return
        config_entry["multiheight"][roi_id]["selected_x_range"] = list(candidate)
        config_entry["multiheight"][roi_id].pop("x_range", None)
        config_entry.update(
            status="in_progress", manual_confirmed=False, updated_at=_utc_now()
        )
        summary = persist()
        inside = summary["rois"][roi_id]["fully_inside_reference_surface"]
        current_index = roi_order.index(roi_id)
        next_roi = roi_order[min(current_index + 1, len(roi_order) - 1)]
        current["roi_id"] = next_roi
        current["clicks"] = []
        set_status(
            f"已立即保存 {roi_labels[roi_id]}={list(candidate)}；"
            f"完全位于 reference_surface 内={str(inside).lower()}。"
            f"当前选择 {roi_labels[next_roi]}。"
        )
        redraw()

    def on_key(event: Any) -> None:
        key_map = {"1": "h001", "2": "h010", "3": "h030"}
        if event.key in key_map:
            select_roi(key_map[event.key])
        elif event.key == "escape":
            current["clicks"] = []
            set_status("已取消当前未确认的边界修改，registry 中原值保持不变。")
            redraw()
        elif event.key == "enter":
            issues = _roi_entry_issues(
                config_entry,
                image_shape[1],
                config["formal_roi_trim_px"],
                config["min_stable_width_px"],
            )
            if issues:
                set_status(f"不能确认：{'; '.join(issues)}。请补选或重选 ROI。")
                return
            config_entry.update(
                status="confirmed", manual_confirmed=True, updated_at=_utc_now()
            )
            persist()
            plt.close(figure)
        elif event.key in {"s", "S"}:
            config_entry.update(
                status="skipped", manual_confirmed=False, updated_at=_utc_now()
            )
            current["skipped"] = True
            persist()
            plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)
    select_roi(roi_order[0])
    plt.show()
    return _multiheight_roi_summary(
        dataset,
        output_dir,
        config_entry,
        aggregates,
        context["frame_count"],
        image_shape,
        config["steger_config"],
        config["multiheight_valid_fraction_min"],
    )


def annotate_multiheight_rois(
    dataset: Path,
    registry: Path,
    analysis_config: Path,
    calibration_src: Path,
    *,
    preview_only: bool = False,
) -> dict[str, Any]:
    """Prepare multiheight ROI evidence and optionally launch the x-only annotation UI."""
    if not preview_only:
        import matplotlib

        if matplotlib.get_backend().lower() == "agg":
            matplotlib.use("qtagg", force=True)
    context = _prepare_multiheight_roi_annotation(
        dataset, registry, analysis_config, calibration_src
    )
    if preview_only:
        return context["summary"]
    return _run_multiheight_roi_editor(context)


def preview_reference_roi(
    dataset: Path,
    analysis_config: Path,
    calibration_src: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render the real 50-frame median image without selecting or building a surface model."""
    dataset = dataset.expanduser().resolve()
    if dataset.name != REFERENCE_DEVELOPMENT_CONFIG_ID:
        raise ValueError(f"本阶段只允许预览 {REFERENCE_DEVELOPMENT_CONFIG_ID}：{dataset}")
    config = _load_reference_analysis_config(analysis_config)
    output_dir = dataset / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / "reference_roi_preview.png", output_dir / "reference_roi_preview.json"]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("ROI 预览已存在，不会静默覆盖：" + ", ".join(path.name for path in existing))
    realtime = _load_realtime_steger(calibration_src)
    steger_options = realtime.load_steger_options(config["steger_config"])
    stacks = _extract_reference_stacks(
        dataset, realtime, steger_options, collect_median_image=True
    )
    aggregates = _aggregate_reference_stacks(stacks)
    width = int(stacks["image_shape"][1])
    bounds = _surface_bounds(config, width, required=False)
    preview_path = _save_reference_roi_preview(
        output_dir,
        np.asarray(stacks["median_image"]),
        aggregates,
        config["valid_frame_fraction_min"],
        bounds,
        dataset.name,
    )
    summary = {
        "schema_version": 1,
        "config_id": dataset.name,
        "frame_count": len(stacks["records"]),
        "image_shape": list(stacks["image_shape"]),
        "centre_extractor": "realtime_steger.extract_steger_columns",
        "steger_config": str(config["steger_config"]),
        "steger_config_sha256": sha256_file(config["steger_config"]),
        "reference_surface_x_range": list(bounds) if bounds is not None else None,
        "reference_model_built": False,
        "multiheight_analyzed": False,
        "output": str(preview_path),
    }
    _atomic_write_text(outputs[1], json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def _source_spans(axis: Any, source: np.ndarray, colors: Mapping[str, str]) -> None:
    start = 0
    for index in range(1, len(source) + 1):
        if index == len(source) or source[index] != source[start]:
            axis.axvspan(
                start - 0.5,
                index - 0.5,
                color=colors.get(str(source[start]), "#9e9e9e"),
                alpha=0.08,
            )
            start = index


def _save_reference_model_plots(
    output_dir: Path,
    aggregates: Mapping[str, np.ndarray],
    surface: Mapping[str, Any],
    valid_fraction_min: float,
    config_id: str = REFERENCE_DEVELOPMENT_CONFIG_ID,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = aggregates["u"]
    raw_y = aggregates["y_median"]
    sigma = aggregates["sigma"]
    valid_fraction = aggregates["valid_fraction"]
    raw_reliable = surface["raw_reliable"]
    source = surface["source"]
    fit_mask = surface["fit_mask"]
    y_smooth = surface["y_ref_smooth"]
    left, right = surface["bounds"]
    colors = {
        "observed": "#2e7d32",
        "short_gap_interpolated": "#fb8c00",
        "smooth_model_filled": "#8e24aa",
        "segment_edge_excluded": "#d32f2f",
        "outside_reference_surface": "#616161",
        "invalid": "#9e9e9e",
    }

    paths: list[Path] = []
    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    raw_source = np.where(raw_reliable, "observed", "invalid")
    _source_spans(axis, raw_source, colors)
    axis.plot(u[raw_reliable], raw_y[raw_reliable], ".", ms=2, color="#1565c0", label="raw observed")
    axis.set_xlim(float(u[0]), float(u[-1]))
    axis.invert_yaxis()
    axis.set_xlabel("u [px]")
    axis.set_ylabel("raw median Steger y [px]")
    axis.set_title(f"{config_id} raw reference centerline (all physical surfaces, no filtering hidden)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    paths.append(_save_figure(output_dir / "reference_centerline_raw.png", figure))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    _source_spans(axis, source, colors)
    axis.axvspan(left, right, color="#66bb6a", alpha=0.06)
    axis.axvline(left, color="#1b5e20", ls="--", lw=1.0)
    axis.axvline(right, color="#1b5e20", ls="--", lw=1.0)
    axis.plot(u[fit_mask], raw_y[fit_mask], ".", ms=2.5, color="#1565c0", label="trimmed reliable observed")
    axis.plot(u[surface["inside"]], y_smooth[surface["inside"]], color="#d81b60", lw=1.4, label="y_ref_smooth")
    excluded = source == "segment_edge_excluded"
    axis.plot(u[excluded], raw_y[excluded], "x", ms=3, color="#d32f2f", label="segment-edge excluded")
    axis.set_xlim(float(u[0]), float(u[-1]))
    axis.invert_yaxis()
    axis.set_xlabel("u [px]")
    axis.set_ylabel("reference y [px]")
    axis.set_title(f"{config_id} reference surface [{left}, {right}] and robust smooth curve")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    paths.append(_save_figure(output_dir / "reference_centerline_smooth.png", figure))
    plt.close(figure)

    residual_full = np.full(u.size, np.nan, dtype=np.float64)
    residual_full[fit_mask] = raw_y[fit_mask] - y_smooth[fit_mask]
    figure, axis = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    axis.axhline(0.0, color="#424242", lw=0.8)
    axis.plot(u[fit_mask], residual_full[fit_mask], ".", ms=2.5, color="#6a1b9a")
    axis.axvline(left, color="#1b5e20", ls="--", lw=1.0)
    axis.axvline(right, color="#1b5e20", ls="--", lw=1.0)
    axis.set_xlim(float(u[0]), float(u[-1]))
    axis.set_xlabel("u [px]")
    axis.set_ylabel("observed - y_ref_smooth [px]")
    axis.set_title(f"{config_id} robust reference model residual")
    axis.grid(True, alpha=0.25)
    paths.append(_save_figure(output_dir / "reference_model_residual.png", figure))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    axis.plot(u, valid_fraction, color="#1565c0", lw=0.8)
    axis.axhline(valid_fraction_min, color="#d32f2f", ls="--", lw=1.0)
    axis.axvspan(left, right, color="#66bb6a", alpha=0.08)
    axis.set_xlim(float(u[0]), float(u[-1]))
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("u [px]")
    axis.set_ylabel("valid frame fraction")
    axis.set_title(f"{config_id} reference Steger validity across 50 frames")
    axis.grid(True, alpha=0.25)
    paths.append(_save_figure(output_dir / "reference_valid_fraction.png", figure))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    axis.plot(u[fit_mask], sigma[fit_mask], ".", ms=2.5, color="#6a1b9a")
    axis.axvspan(left, right, color="#66bb6a", alpha=0.08)
    axis.set_xlim(float(u[0]), float(u[-1]))
    axis.set_xlabel("u [px]")
    axis.set_ylabel("sigma_ref [px]")
    axis.set_title(f"{config_id} repeatability of trimmed surface observations")
    axis.grid(True, alpha=0.25)
    paths.append(_save_figure(output_dir / "reference_repeatability.png", figure))
    plt.close(figure)
    return paths


def analyze_reference(
    dataset: Path,
    analysis_config: Path,
    calibration_src: Path,
    *,
    overwrite: bool = False,
    reference_x_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build one surface-bounded reference curve; never read multiheight."""
    dataset = dataset.expanduser().resolve()
    if dataset.name not in CAPTURED_CONFIG_IDS:
        raise ValueError(f"不是已采集的 Phase-A 配置：{dataset.name}")
    config = _load_reference_analysis_config(analysis_config)
    if reference_x_range is not None:
        config = {**config, "reference_surface_x_range": tuple(reference_x_range)}
    if config["reference_surface_x_range"] is None:
        _surface_bounds(config, 1, required=True)
    output_dir = dataset / "analysis"
    detail_output = output_dir / "reference_frame_columns.csv"
    derived_outputs = [
        output_dir / "reference_by_column.csv",
        output_dir / "reference_centerline_raw.png",
        output_dir / "reference_centerline_smooth.png",
        output_dir / "reference_model_residual.png",
        output_dir / "reference_cv_by_segment.csv",
        output_dir / "reference_cv_residual.png",
        output_dir / "reference_valid_fraction.png",
        output_dir / "reference_repeatability.png",
        output_dir / "reference_analysis.json",
    ]
    existing = [path for path in derived_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "reference 派生输出已存在，不会静默覆盖：" + ", ".join(path.name for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    realtime = _load_realtime_steger(calibration_src)
    steger_options = realtime.load_steger_options(config["steger_config"])
    stacks = _extract_reference_stacks(
        dataset, realtime, steger_options, detail_output=detail_output
    )
    aggregates = _aggregate_reference_stacks(stacks)
    surface = _build_surface_reference(aggregates, config)
    cross_validation = _cross_validate_reference_segments(aggregates, surface, config)
    source = surface["source"]
    fit_mask = surface["fit_mask"]
    sigma = aggregates["sigma"]
    fwhm = aggregates["fwhm_p50"]
    y_observed = surface["y_ref_observed"]
    y_smooth = surface["y_ref_smooth"]
    y_short_gap = surface["y_ref_short_gap"]

    rows = [
        {
            "u": column,
            "y_ref_observed_px": _nan_text(y_observed[column]),
            "y_ref_short_gap_px": _nan_text(y_short_gap[column]),
            "y_ref_smooth_px": _nan_text(y_smooth[column]),
            "sigma_ref_px": _nan_text(sigma[column] if np.isfinite(y_observed[column]) else np.nan),
            "valid_fraction": float(aggregates["valid_fraction"][column]),
            "valid_frame_count": int(aggregates["valid_count"][column]),
            "fwhm_p50_px": _nan_text(fwhm[column] if np.isfinite(y_observed[column]) else np.nan),
            "source": str(source[column]),
        }
        for column in range(len(source))
    ]
    _write_csv(
        output_dir / "reference_by_column.csv",
        (
            "u", "y_ref_observed_px", "y_ref_short_gap_px", "y_ref_smooth_px",
            "sigma_ref_px", "valid_fraction", "valid_frame_count", "fwhm_p50_px", "source",
        ),
        rows,
    )
    _write_csv(
        output_dir / "reference_cv_by_segment.csv",
        (
            "segment_id", "segment_role", "u_start", "u_end", "width_px", "point_count",
            "rmse_px", "mae_px", "p95_abs_error_px", "max_abs_error_px",
        ),
        cross_validation["rows"],
    )
    plot_paths = _save_reference_model_plots(
        output_dir, aggregates, surface, config["valid_frame_fraction_min"], dataset.name
    )
    cv_plot_path = _save_reference_cv_plot(
        output_dir, cross_validation, surface["bounds"], dataset.name
    )
    plot_paths.append(cv_plot_path)
    source_names = (
        "observed", "short_gap_interpolated", "smooth_model_filled",
        "segment_edge_excluded", "outside_reference_surface", "invalid",
    )
    source_counts = {name: int(np.count_nonzero(source == name)) for name in source_names}
    surface_width = int(surface["bounds"][1] - surface["bounds"][0] + 1)
    residual = np.asarray(surface["residual"], dtype=np.float64)
    observed_sigma = sigma[fit_mask]
    statistics = {
        "reference_surface_width_px": surface_width,
        "observed_fraction_inside_surface": float(source_counts["observed"] / surface_width),
        "model_filled_fraction": float(source_counts["smooth_model_filled"] / surface_width),
        "sigma_ref_p50_px": float(np.median(observed_sigma)),
        "sigma_ref_p95_px": float(np.percentile(observed_sigma, 95)),
        "reference_model_residual_rmse_px": float(np.sqrt(np.mean(np.square(residual)))),
        "reference_model_residual_p95_px": float(np.percentile(np.abs(residual), 95)),
        **cross_validation["statistics"],
    }
    summary = {
        "schema_version": 4,
        "config_id": dataset.name,
        "task_id": "reference",
        "frame_count": len(stacks["records"]),
        "image_shape": list(stacks["image_shape"]),
        "centre_extractor": "realtime_steger.extract_steger_columns",
        "steger_module": str((calibration_src / "realtime_steger.py").resolve()),
        "steger_config": str(config["steger_config"]),
        "steger_config_sha256": sha256_file(config["steger_config"]),
        "steger_options": steger_options,
        "analysis_config": str(config["source"]),
        "analysis_config_sha256": sha256_file(config["source"]),
        "reference_surface_x_range": list(surface["bounds"]),
        "segment_edge_trim_px": config["segment_edge_trim_px"],
        "valid_frame_fraction_min": config["valid_frame_fraction_min"],
        "max_interp_gap_px": config["max_interp_gap_px"],
        "source_counts": source_counts,
        "statistics": statistics,
        "smooth_model": surface["model_info"],
        "cross_validation": {
            "method": "leave_one_observed_segment_out",
            "observed_segment_count": cross_validation["observed_segment_count"],
            "eligible_segment_count": cross_validation["eligible_segment_count"],
            "interior_segment_count": cross_validation["interior_segment_count"],
            "boundary_segment_count": cross_validation["boundary_segment_count"],
            "minimum_segment_width_px": cross_validation["minimum_segment_width_px"],
            "formal_metrics_scope": "interior_segments_only",
            "boundary_cv_diagnostic_only": True,
            "used_for_final_model_fit": False,
        },
        "per_frame_steger_output_preserved": bool(stacks["detail_was_preserved"]),
        "global_line_fit_applied": False,
        "model_extrapolated_outside_reference_surface": False,
        "multiheight_analyzed": False,
        "outputs": [str(path) for path in (
            detail_output,
            output_dir / "reference_by_column.csv",
            output_dir / "reference_cv_by_segment.csv",
            *plot_paths,
        )],
    }
    _atomic_write_text(
        output_dir / "reference_analysis.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary


def _load_frozen_reference_curve(
    dataset: Path,
    config: Mapping[str, Any],
    image_width: int,
) -> tuple[np.ndarray, dict[str, Any], Path]:
    analysis_dir = dataset / "analysis"
    summary_path = analysis_dir / "reference_analysis.json"
    summary = load_document(summary_path)
    expected_range = config["reference_surface_x_range"]
    if tuple(summary.get("reference_surface_x_range") or ()) != expected_range:
        raise ValueError("冻结的 reference_analysis 与当前 reference_surface.x_range 不一致")
    if summary.get("steger_config_sha256") != sha256_file(config["steger_config"]):
        raise ValueError("冻结的 reference 与 multiheight Steger 配置 hash 不一致")
    if summary.get("model_extrapolated_outside_reference_surface") is not False:
        raise ValueError("冻结的 reference 未明确禁止 x_range 外外推")
    if summary.get("multiheight_analyzed") is not False:
        raise ValueError("冻结的 reference provenance 异常：multiheight_analyzed 不是 false")
    curve_path = analysis_dir / "reference_by_column.csv"
    with curve_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"u", "y_ref_smooth_px", "source"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"reference_by_column.csv 缺少字段：{sorted(missing)}")
        rows = list(reader)
    if len(rows) != image_width:
        raise ValueError(f"reference_by_column 应有 {image_width} 列，实际 {len(rows)}")
    y_ref = np.full(image_width, np.nan, dtype=np.float64)
    left, right = expected_range
    for expected_u, row in enumerate(rows):
        if int(row["u"]) != expected_u:
            raise ValueError("reference_by_column.csv 的 u 不是连续图像列")
        text = row["y_ref_smooth_px"].strip()
        if text and text.lower() != "nan":
            y_ref[expected_u] = float(text)
        if expected_u < left or expected_u > right:
            if row["source"] != "outside_reference_surface" or np.isfinite(y_ref[expected_u]):
                raise ValueError("reference x_range 外存在可用 reference 或错误 source")
    if not np.all(np.isfinite(y_ref[left:right + 1])):
        raise ValueError("reference_surface 内的 y_ref_smooth 存在 NaN")
    return y_ref, summary, curve_path


def _reference_vertical_envelope(
    y_ref: np.ndarray,
    reference_surface_x_range: tuple[int, int],
    margin_px: int,
    image_height: int,
) -> tuple[int, int]:
    """Return a clipped [top, bottom-exclusive) envelope from frozen in-surface y_ref."""
    left, right = reference_surface_x_range
    in_surface = np.asarray(y_ref[left:right + 1], dtype=np.float64)
    finite = in_surface[np.isfinite(in_surface)]
    if finite.size != in_surface.size or finite.size == 0:
        raise ValueError("reference vertical envelope 只能由 surface 内完整有限的 y_ref_smooth 构建")
    top = max(0, int(math.floor(float(np.min(finite)))) - margin_px)
    bottom = min(
        image_height,
        int(math.ceil(float(np.max(finite)))) + margin_px,
    )
    if bottom <= top:
        raise ValueError("reference vertical envelope 裁剪到图像后为空")
    return top, bottom


def _phase_a_band_fix_validation(
    context: Mapping[str, Any],
    background_roi: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    """Compare original auto-band and reference-envelope union on one multiheight dataset."""
    original_stacks = context["original_stacks"]
    final_stacks = context["stacks"]
    original_y = np.asarray(original_stacks["y_stack"], dtype=np.float64)
    original_valid = np.asarray(original_stacks["valid_stack"], dtype=bool)
    final_y = np.asarray(final_stacks["y_stack"], dtype=np.float64)
    final_valid = np.asarray(final_stacks["valid_stack"], dtype=bool)
    band_rows = final_stacks["band_diagnostics"]
    if not band_rows:
        raise ValueError("multiheight band diagnostic 为空")

    def band_value(field: str) -> int | None:
        values = [row.get(field) for row in band_rows]
        finite = [int(round(float(value))) for value in values if value is not None]
        return int(round(float(np.median(finite)))) if finite else None

    original_top = band_value("original_band_top_px")
    original_bottom = band_value("original_band_bottom_exclusive_px")
    envelope_top = band_value("reference_envelope_top_px")
    envelope_bottom = band_value("reference_envelope_bottom_exclusive_px")
    final_top = band_value("final_band_top_px")
    final_bottom = band_value("final_band_bottom_exclusive_px")
    steger_time_ms_before = float(original_stacks["steger_time_ms"])
    steger_time_ms_after = float(final_stacks["steger_time_ms"])
    runtime_change_percent = (
        (steger_time_ms_after - steger_time_ms_before) / steger_time_ms_before * 100.0
        if steger_time_ms_before > 0.0 else None
    )
    band_fields = {
        "original_band_top": original_top,
        "original_band_bottom": original_bottom,
        "reference_band_top": envelope_top,
        "reference_band_bottom": envelope_bottom,
        "reference_envelope_top": envelope_top,
        "reference_envelope_bottom": envelope_bottom,
        "final_band_top": final_top,
        "final_band_bottom": final_bottom,
        "original_band_height": (
            original_bottom - original_top
            if original_top is not None and original_bottom is not None else None
        ),
        "final_band_height": (
            final_bottom - final_top
            if final_top is not None and final_bottom is not None else None
        ),
        "steger_time_ms_before": steger_time_ms_before,
        "steger_time_ms_after": steger_time_ms_after,
        "runtime_change_percent": runtime_change_percent,
    }
    band_consistent = all(
        len({row.get(field) for row in band_rows}) == 1
        for field in (
            "original_band_top_px",
            "original_band_bottom_exclusive_px",
            "reference_envelope_top_px",
            "reference_envelope_bottom_exclusive_px",
            "final_band_top_px",
            "final_band_bottom_exclusive_px",
        )
    )

    config_entry = context["config_entry"]
    roi_results: dict[str, Any] = {}
    for roi_id, _label, _height_mm, _color in MULTIHEIGHT_ROIS:
        selected = _roi_selected_x_range(config_entry["multiheight"][roi_id], roi_id)
        if selected is None:
            raise ValueError(f"{roi_id}.selected_x_range 尚未人工标注")
        original_stats, _ = _steger_sweep_roi_statistics(
            original_y, original_valid, selected
        )
        final_stats, _ = _steger_sweep_roi_statistics(final_y, final_valid, selected)
        original_center = _finite_or_none(original_stats["median_center_position_px"])
        final_center = _finite_or_none(final_stats["median_center_position_px"])
        original_sigma = _finite_or_none(original_stats["sigma_pixel_p95_px"])
        final_sigma = _finite_or_none(final_stats["sigma_pixel_p95_px"])
        center_shift = (
            final_center - original_center
            if final_center is not None and original_center is not None else None
        )
        sigma_ratio = (
            final_sigma / original_sigma
            if final_sigma is not None and original_sigma is not None and original_sigma > 0 else None
        )
        roi_results[roi_id] = {
            "selected_x_range": list(selected),
            "original": original_stats,
            "final": final_stats,
            "center_shift_px": center_shift,
            "sigma_pixel_p95_ratio": sigma_ratio,
        }

    h1_center = _finite_or_none(
        roi_results["h001"]["final"]["median_center_position_px"]
    )
    h1_inside_original = bool(
        h1_center is not None
        and original_top is not None
        and original_bottom is not None
        and original_top <= h1_center < original_bottom
    )
    h1_inside_final = bool(
        h1_center is not None
        and final_top is not None
        and final_bottom is not None
        and final_top <= h1_center < final_bottom
    )

    background_rate: float | None = None
    if background_roi is not None:
        x_left, x_right, y_top, y_bottom = background_roi
        image_height, image_width = context["image_shape"]
        if not (
            0 <= x_left <= x_right < image_width
            and 0 <= y_top <= y_bottom < image_height
        ):
            raise ValueError(f"background ROI 超出图像范围：{background_roi}")
        columns = np.arange(x_left, x_right + 1)
        background_events = (
            final_valid[:, columns]
            & (final_y[:, columns] >= y_top)
            & (final_y[:, columns] <= y_bottom)
        )
        background_rate = float(np.mean(background_events))

    h1_valid = float(
        roi_results["h001"]["final"]["valid_column_fraction_at_0_8"]
    )
    primary_checks: dict[str, Any] = {}
    for roi_id in ("h010", "h030"):
        shift = roi_results[roi_id]["center_shift_px"]
        ratio = roi_results[roi_id]["sigma_pixel_p95_ratio"]
        primary_checks[roi_id] = {
            "center_shift_pass": bool(
                shift is not None
                and abs(shift) < BAND_FIX_PRIMARY_CENTER_SHIFT_MAX_PX
            ),
            "sigma_pixel_p95_pass": bool(
                ratio is not None
                and ratio <= BAND_FIX_PRIMARY_SIGMA_P95_RATIO_MAX
            ),
        }
    acceptance = {
        "h1_valid_column_fraction_pass": h1_valid >= 0.8,
        "background_false_detection_pass": bool(
            background_rate is not None
            and background_rate < BAND_FIX_BACKGROUND_FALSE_RATE_MAX
        ),
        "h10_center_shift_pass": primary_checks["h010"]["center_shift_pass"],
        "h30_center_shift_pass": primary_checks["h030"]["center_shift_pass"],
        "h10_sigma_pixel_p95_pass": primary_checks["h010"]["sigma_pixel_p95_pass"],
        "h30_sigma_pixel_p95_pass": primary_checks["h030"]["sigma_pixel_p95_pass"],
    }
    return {
        "method": "shared_steger_additional_band_union_with_frozen_reference_band",
        "bottom_semantics": "exclusive",
        **band_fields,
        "band_bounds_identical_across_frames": band_consistent,
        "reference_band_margin_px": context["analysis_config"]["reference_band_margin_px"],
        "h1_peak_y_median_px": h1_center,
        "h1_peak_inside_original_band": h1_inside_original,
        "h1_peak_inside_final_band": h1_inside_final,
        "rois": roi_results,
        "background_roi_xyxy": list(background_roi) if background_roi is not None else None,
        "background_false_detection_rate": background_rate,
        "acceptance_thresholds": {
            "h1_valid_column_fraction_min": 0.8,
            "background_false_detection_rate_max_exclusive": BAND_FIX_BACKGROUND_FALSE_RATE_MAX,
            "primary_abs_center_shift_max_px_exclusive": BAND_FIX_PRIMARY_CENTER_SHIFT_MAX_PX,
            "primary_sigma_pixel_p95_ratio_max": BAND_FIX_PRIMARY_SIGMA_P95_RATIO_MAX,
        },
        "acceptance": acceptance,
        "band_fix_validated": bool(all(acceptance.values())),
        "reference_union_band_validated": bool(all(acceptance.values())),
    }


def _median_smooth_for_detection(
    values: np.ndarray,
    eligible: np.ndarray,
    window: int,
) -> np.ndarray:
    smoothed = np.full(values.size, np.nan, dtype=np.float64)
    radius = window // 2
    for index in range(values.size):
        if not eligible[index]:
            continue
        start = max(0, index - radius)
        end = min(values.size, index + radius + 1)
        local = values[start:end][eligible[start:end] & np.isfinite(values[start:end])]
        if local.size:
            smoothed[index] = float(np.median(local))
    return smoothed


def _detect_stable_plateau(
    selected: tuple[int, int],
    delta_y: np.ndarray,
    sigma_obj: np.ndarray,
    valid_fraction: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    left, right = selected
    local_delta = np.asarray(delta_y[left:right + 1], dtype=np.float64)
    local_sigma = np.asarray(sigma_obj[left:right + 1], dtype=np.float64)
    local_valid_fraction = np.asarray(valid_fraction[left:right + 1], dtype=np.float64)
    eligible = (
        (local_valid_fraction >= config["multiheight_valid_fraction_min"])
        & np.isfinite(local_delta)
        & np.isfinite(local_sigma)
    )
    if not np.any(eligible):
        return {
            "status": "needs_manual_review",
            "auto_stable_x_range": None,
            "analysis_x_range": None,
            "stable_width_px": 0,
            "eligible": eligible,
            "stable_mask": np.zeros_like(eligible),
            "smoothed_delta": np.full_like(local_delta, np.nan),
            "gradient_abs": np.full_like(local_delta, np.nan),
            "sigma_threshold_px": np.nan,
        }
    sigma_values = local_sigma[eligible]
    sigma_median = float(np.median(sigma_values))
    sigma_mad = float(np.median(np.abs(sigma_values - sigma_median)))
    sigma_threshold = max(
        config["plateau_sigma_floor_limit_px"],
        sigma_median + config["plateau_sigma_mad_scale"] * 1.4826 * sigma_mad,
    )
    smoothed = _median_smooth_for_detection(
        local_delta, eligible, config["plateau_median_window_px"]
    )
    steps = np.abs(np.diff(smoothed))
    left_step = np.full(local_delta.size, np.nan, dtype=np.float64)
    right_step = np.full(local_delta.size, np.nan, dtype=np.float64)
    left_step[1:] = steps
    right_step[:-1] = steps
    gradient = np.fmax(left_step, right_step)
    if local_delta.size == 1:
        gradient[0] = 0.0
    elif local_delta.size > 1:
        gradient[0] = right_step[0]
        gradient[-1] = left_step[-1]
    step_ok = (
        (np.isnan(left_step) | (left_step <= config["plateau_max_step_px"]))
        & (np.isnan(right_step) | (right_step <= config["plateau_max_step_px"]))
    )
    stable = (
        eligible
        & np.isfinite(smoothed)
        & np.isfinite(gradient)
        & (gradient <= config["plateau_max_gradient_px_per_column"])
        & step_ok
        & (local_sigma <= sigma_threshold)
    )
    runs = _true_runs(stable)
    if not runs:
        best = None
    else:
        best = max(runs, key=lambda item: (item[1] - item[0] + 1, -item[0]))
    auto_range = None
    analysis_range = None
    stable_width = 0
    status = "needs_manual_review"
    if best is not None:
        auto_range = (left + best[0], left + best[1])
        stable_width = auto_range[1] - auto_range[0] + 1
        erosion = config["plateau_erosion_px"]
        eroded = (auto_range[0] + erosion, auto_range[1] - erosion)
        if eroded[0] <= eroded[1] and eroded[1] - eroded[0] + 1 >= config["min_stable_width_px"]:
            analysis_range = eroded
            status = "ok"
    return {
        "status": status,
        "auto_stable_x_range": auto_range,
        "analysis_x_range": analysis_range,
        "stable_width_px": stable_width,
        "eligible": eligible,
        "stable_mask": stable,
        "smoothed_delta": smoothed,
        "gradient_abs": gradient,
        "sigma_threshold_px": sigma_threshold,
    }


def _distribution_statistics(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("正式 analysis_x_range 内没有有限数据")
    return {
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


def _repeatability_statistics(
    sigma_pixel: np.ndarray,
    sensitivity: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    sigma_pixel = np.asarray(sigma_pixel, dtype=np.float64)
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    repeat_valid = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(sigma_pixel)
        & np.isfinite(sensitivity)
        & (sensitivity > np.finfo(np.float64).eps)
    )
    if not np.any(repeat_valid):
        raise ValueError("formal ROI 内没有可用重复性列")
    sigma_values = sigma_pixel[repeat_valid]
    sigma_z_values = sigma_values / sensitivity[repeat_valid]
    return repeat_valid, {
        "sigma_pixel_p50_px": float(np.median(sigma_values)),
        "sigma_pixel_p95_px": float(np.percentile(sigma_values, 95)),
        "sigma_z_pred_p50_mm": float(np.median(sigma_z_values)),
        "sigma_z_pred_p95_mm": float(np.percentile(sigma_z_values, 95)),
    }


def _phase_a_combined_metrics(
    roi_results: Mapping[str, Mapping[str, Any]],
) -> tuple[float | None, float | None]:
    primary = [roi_results[roi_id] for roi_id in ("h010", "h030")]
    if not all(result["formal_statistics_allowed"] for result in primary):
        return None, None
    return (
        float(np.median([result["sensitivity_median_px_per_mm"] for result in primary])),
        float(np.median([result["sigma_z_pred_p95_mm"] for result in primary])),
    )


def _roi_trim_sensitivity_rows(
    roi_id: str,
    height_mm: float,
    selected: tuple[int, int],
    delta_y: np.ndarray,
    valid_fraction: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool | None]:
    rows: list[dict[str, Any]] = []
    for trim in config["roi_trim_values_px"]:
        left, right = selected[0] + trim, selected[1] - trim
        if left > right:
            values = np.asarray([], dtype=np.float64)
            unavailable_reason = "empty_after_trim"
        else:
            mask = (
                (valid_fraction[left:right + 1] >= config["multiheight_valid_fraction_min"])
                & np.isfinite(delta_y[left:right + 1])
            )
            values = delta_y[left:right + 1][mask]
            unavailable_reason = None if values.size else "no_valid_delta_y"
        sensitivity = np.abs(values) / height_mm if values.size else values
        rows.append({
            "roi_id": roi_id,
            "height_mm": height_mm,
            "trim_px": trim,
            "x_start": left,
            "x_end": right,
            "width_px": max(0, right - left + 1),
            "valid_column_count": int(values.size),
            "available": bool(values.size),
            "unavailable_reason": unavailable_reason,
            "delta_y_median_px": float(np.median(values)) if values.size else None,
            "sensitivity_median_px_per_mm": (
                float(np.median(sensitivity)) if values.size else None
            ),
        })
    baseline = rows[0]["sensitivity_median_px_per_mm"] if rows else None
    changes: list[float] = []
    for row in rows:
        value = row["sensitivity_median_px_per_mm"]
        if baseline is None or value is None:
            relative = None
        elif abs(baseline) <= np.finfo(np.float64).eps:
            relative = 0.0 if abs(value) <= np.finfo(np.float64).eps else float("inf")
        else:
            relative = abs(value - baseline) / abs(baseline)
        row["relative_change_vs_trim0"] = relative
        if row["trim_px"] and relative is not None:
            changes.append(relative)
    all_trims_available = bool(rows) and all(row["available"] for row in rows)
    roi_stable = (
        bool(changes and max(changes) < config["roi_trim_max_relative_change"])
        if all_trims_available else None
    )
    for row in rows:
        row["roi_stable"] = roi_stable
        row["roi_sensitive"] = None if roi_stable is None else not roi_stable
    return rows, roi_stable


def _save_multiheight_analysis_plots(
    output_dir: Path,
    aggregates: Mapping[str, np.ndarray],
    y_ref: np.ndarray,
    delta_y: np.ndarray,
    roi_results: Mapping[str, Mapping[str, Any]],
    trim_rows: Sequence[Mapping[str, Any]],
    reference_range: tuple[int, int],
    config_id: str = REFERENCE_DEVELOPMENT_CONFIG_ID,
) -> list[Path]:
    import matplotlib.pyplot as plt

    u = np.asarray(aggregates["u"], dtype=np.float64)
    y_obj = np.asarray(aggregates["y_median"], dtype=np.float64)
    valid_fraction = np.asarray(aggregates["valid_fraction"], dtype=np.float64)
    colors = {item[0]: item[3] for item in MULTIHEIGHT_ROIS}
    labels = {item[0]: item[1] for item in MULTIHEIGHT_ROIS}
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    left, right = reference_range
    axis.plot(u[left:right + 1], y_ref[left:right + 1], color="#d81b60", lw=1.2, label="frozen y_ref_smooth")
    valid_obj = (valid_fraction >= 0.8) & np.isfinite(y_obj)
    axis.plot(u[valid_obj], y_obj[valid_obj], ".", ms=1.8, color="#1565c0", label="multiheight y_obj_median")
    for roi_id, result in roi_results.items():
        selected = result["selected_x_range"]
        axis.axvspan(selected[0], selected[1], color=colors[roi_id], alpha=0.12, label=f"{labels[roi_id]} selected")
    axis.set_xlim(left, right)
    axis.invert_yaxis()
    axis.set_xlabel("u [px]")
    axis.set_ylabel("subpixel center y [px]")
    axis.set_title(f"{config_id} frozen reference and multiheight centerline overlay")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    paths.append(_save_figure(output_dir / "centerline_overlay.png", figure))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    finite_delta = np.isfinite(delta_y) & (valid_fraction >= 0.8)
    axis.plot(u[finite_delta], delta_y[finite_delta], ".", ms=1.8, color="#424242")
    axis.axhline(0.0, color="#9e9e9e", lw=0.8)
    for roi_id, result in roi_results.items():
        analysis = result["analysis_x_range"]
        if analysis is not None:
            axis.axvspan(analysis[0], analysis[1], color=colors[roi_id], alpha=0.20, label=f"{labels[roi_id]} analysis")
    axis.set_xlim(left, right)
    axis.set_xlabel("u [px]")
    axis.set_ylabel("delta_y = y_obj - y_ref_smooth [px]")
    axis.set_title(f"{config_id} same-column laser displacement")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    paths.append(_save_figure(output_dir / "displacement_by_column.png", figure))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    for roi_id, result in roi_results.items():
        selected = result["selected_x_range"]
        columns = np.arange(selected[0], selected[1] + 1)
        sensitivity = np.abs(delta_y[columns]) / result["height_mm"]
        valid = (valid_fraction[columns] >= 0.8) & np.isfinite(sensitivity)
        axis.plot(columns[valid], sensitivity[valid], ".-", ms=3, lw=0.8, color=colors[roi_id], label=labels[roi_id])
        analysis = result["analysis_x_range"]
        if analysis is not None:
            axis.axvspan(analysis[0], analysis[1], color=colors[roi_id], alpha=0.12)
    axis.set_xlabel("u [px]")
    axis.set_ylabel("abs(delta_y) / height [px/mm]")
    axis.set_title(f"{config_id} per-column sensitivity (not used for plateau selection)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    paths.append(_save_figure(output_dir / "sensitivity_by_column.png", figure))
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
    for axis, (roi_id, result) in zip(axes, roi_results.items()):
        selected = result["selected_x_range"]
        columns = np.arange(selected[0], selected[1] + 1)
        detection = result["detection"]
        axis.axvspan(selected[0], selected[1], color="#bdbdbd", alpha=0.12, label="selected ROI")
        axis.plot(columns, delta_y[columns], ".", ms=4, color="#424242", label="raw delta_y")
        axis.plot(columns, detection["smoothed_delta"], color="#ff9800", lw=1.1, label="median smooth (detection only)")
        stable = detection["auto_stable_x_range"]
        diagnostic_analysis = detection["analysis_x_range"]
        analysis = result["analysis_x_range"]
        if stable is not None:
            axis.axvspan(stable[0], stable[1], color="#66bb6a", alpha=0.22, label="auto stable plateau")
            transition = (columns < stable[0]) | (columns > stable[1])
            axis.plot(columns[transition], delta_y[columns][transition], "x", ms=5, color="#d32f2f", label="excluded transition/anomaly")
        if diagnostic_analysis is not None:
            axis.axvspan(
                diagnostic_analysis[0], diagnostic_analysis[1], facecolor="none",
                edgecolor="#2e7d32", hatch="//", lw=0.8,
                label="diagnostic eroded stable ROI",
            )
        if analysis is not None:
            axis.axvspan(analysis[0], analysis[1], color="#1565c0", alpha=0.22, label="formal selected ROI trim3")
        axis.set_xlim(selected[0] - 1, selected[1] + 1)
        axis.set_ylabel("delta_y [px]")
        axis.set_title(
            f"{labels[roi_id]} plateau diagnostic | formal status={result['status']} | "
            f"warnings={len(result['warnings'])}"
        )
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", ncol=3)
    axes[-1].set_xlabel("u [px]")
    paths.append(_save_figure(output_dir / "multiheight_plateau_detection.png", figure))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for roi_id, _label, _height, color in MULTIHEIGHT_ROIS:
        rows = [row for row in trim_rows if row["roi_id"] == roi_id]
        axis.plot(
            [row["trim_px"] for row in rows],
            [row["sensitivity_median_px_per_mm"] for row in rows],
            "o-", color=color, label=labels[roi_id],
        )
    axis.set_xticks([0, 2, 3, 4])
    axis.set_xlabel("symmetric trim from selected ROI [px]")
    axis.set_ylabel("median sensitivity [px/mm]")
    axis.set_title(f"{config_id} manual ROI trim sensitivity check")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    paths.append(_save_figure(output_dir / "roi_trim_sensitivity_check.png", figure))
    plt.close(figure)
    return paths


def _save_repeatability_plots(
    output_dir: Path,
    repeatability_rows: Sequence[Mapping[str, Any]],
    roi_results: Mapping[str, Mapping[str, Any]],
    config_id: str = REFERENCE_DEVELOPMENT_CONFIG_ID,
) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = {item[0]: item[3] for item in MULTIHEIGHT_ROIS}
    labels = {item[0]: item[1] for item in MULTIHEIGHT_ROIS}
    paths: list[Path] = []
    for field, ylabel, filename, title in (
        ("sigma_pixel_px", "sigma_pixel [px]", "sigma_pixel_by_column.png", "50-frame per-column pixel repeatability in formal trim3 ROIs"),
        ("sigma_z_pred_mm", "sigma_z_pred [mm]", "sigma_z_pred_by_column.png", "Predicted height repeatability from per-column sensitivity"),
    ):
        figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
        for roi_id, _label, _height, _color in MULTIHEIGHT_ROIS:
            rows = [row for row in repeatability_rows if row["roi_id"] == roi_id]
            x = np.asarray([row["u"] for row in rows], dtype=np.float64)
            y = np.asarray([row[field] for row in rows], dtype=np.float64)
            valid = np.isfinite(y)
            axis.plot(x[valid], y[valid], ".-", ms=3, lw=0.8, color=colors[roi_id], label=labels[roi_id])
            formal = roi_results[roi_id]["analysis_x_range"]
            axis.axvspan(formal[0], formal[1], color=colors[roi_id], alpha=0.07)
        axis.set_xlabel("u [px]")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{config_id} {title}")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        paths.append(_save_figure(output_dir / filename, figure))
        plt.close(figure)
    return paths


def analyze_multiheight(
    dataset: Path,
    registry: Path,
    analysis_config: Path,
    calibration_src: Path,
    *,
    overwrite: bool = False,
    background_roi: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    """Analyze one confirmed multiheight dataset by same-column frozen-reference subtraction."""
    dataset = dataset.expanduser().resolve()
    output_dir = dataset / "analysis"
    geometry_summary_path = dataset.parent.parent / "results" / "geometry_master_summary.csv"
    geometry_summary_sha256_before = (
        sha256_file(geometry_summary_path) if geometry_summary_path.is_file() else None
    )
    outputs = [
        output_dir / "multiheight_by_column.csv",
        output_dir / "multiheight_analysis.json",
        output_dir / "roi_trim_sensitivity_check.csv",
        output_dir / "repeatability_by_column.csv",
        output_dir / "centerline_overlay.png",
        output_dir / "displacement_by_column.png",
        output_dir / "sensitivity_by_column.png",
        output_dir / "multiheight_plateau_detection.png",
        output_dir / "roi_trim_sensitivity_check.png",
        output_dir / "sigma_pixel_by_column.png",
        output_dir / "sigma_z_pred_by_column.png",
        output_dir / "band_fix_validation.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "multiheight 分析输出已存在，不会静默覆盖：" + ", ".join(path.name for path in existing)
        )
    context = _prepare_multiheight_roi_annotation(
        dataset, registry, analysis_config, calibration_src
    )
    config = context["analysis_config"]
    config_entry = context["config_entry"]
    aggregates = context["aggregates"]
    band_validation = _phase_a_band_fix_validation(context, background_roi)
    image_width = context["image_shape"][1]
    y_ref, _reference_summary, reference_curve_path = _load_frozen_reference_curve(
        dataset, config, image_width
    )
    y_obj = np.asarray(aggregates["y_median"], dtype=np.float64)
    sigma_obj = np.asarray(aggregates["sigma"], dtype=np.float64)
    valid_fraction = np.asarray(aggregates["valid_fraction"], dtype=np.float64)
    object_reliable = (
        (valid_fraction >= config["multiheight_valid_fraction_min"])
        & np.isfinite(y_obj)
    )
    delta_valid = object_reliable & np.isfinite(y_ref)
    delta_y = np.full(image_width, np.nan, dtype=np.float64)
    delta_y[delta_valid] = y_obj[delta_valid] - y_ref[delta_valid]

    roi_results: dict[str, dict[str, Any]] = {}
    trim_rows: list[dict[str, Any]] = []
    repeatability_rows: list[dict[str, Any]] = []
    for roi_id, label, height_mm, _color in MULTIHEIGHT_ROIS:
        roi_entry = config_entry["multiheight"][roi_id]
        selected = _roi_selected_x_range(roi_entry, roi_id)
        if selected is None:
            raise ValueError(f"{roi_id}.selected_x_range 尚未人工标注")
        if selected[1] >= image_width:
            raise ValueError(f"{roi_id}.selected_x_range 超出图像宽度")
        reference_range = config["reference_surface_x_range"]
        if not (reference_range[0] <= selected[0] <= selected[1] <= reference_range[1]):
            raise ValueError(f"{roi_id}.selected_x_range 不完全位于冻结 reference_surface 内")
        detection = _detect_stable_plateau(
            selected, delta_y, sigma_obj, valid_fraction, config
        )
        auto_range = detection["auto_stable_x_range"]
        selected_columns = np.arange(selected[0], selected[1] + 1)
        selected_valid_fraction = float(np.mean(delta_valid[selected_columns]))
        diagnostic_range = detection["analysis_x_range"]
        diagnostic_validity_range = diagnostic_range or auto_range or selected
        diagnostic_columns = np.arange(
            diagnostic_validity_range[0], diagnostic_validity_range[1] + 1
        )
        stable_valid_fraction = float(np.mean(delta_valid[diagnostic_columns]))
        roi_trim_rows, roi_stable = _roi_trim_sensitivity_rows(
            roi_id, height_mm, selected, delta_y, valid_fraction, config
        )
        trim_rows.extend(roi_trim_rows)
        trim_changes = [
            row["relative_change_vs_trim0"] for row in roi_trim_rows
            if row["trim_px"] and row["relative_change_vs_trim0"] is not None
        ]
        trim_max_relative_change = (
            float(max(trim_changes))
            if len(trim_changes) == len(config["roi_trim_values_px"]) - 1 else None
        )
        formal_trim = config["formal_roi_trim_px"]
        formal_row = next(row for row in roi_trim_rows if row["trim_px"] == formal_trim)
        analysis_range = (int(formal_row["x_start"]), int(formal_row["x_end"]))
        columns = np.arange(analysis_range[0], analysis_range[1] + 1)
        valid = delta_valid[columns]
        formal_valid_fraction = float(np.mean(valid))
        object_valid_column_count = int(np.count_nonzero(object_reliable[columns]))
        reference_valid_column_count = int(np.count_nonzero(np.isfinite(y_ref[columns])))
        common_valid_column_count = int(np.count_nonzero(valid))
        unavailable_reasons: list[str] = []
        if common_valid_column_count == 0:
            if object_valid_column_count == 0:
                unavailable_reasons.append("no_object_columns")
            if reference_valid_column_count == 0:
                unavailable_reasons.append("no_reference_columns")
            unavailable_reasons.append("no_common_columns")
        warnings: list[str] = []
        if detection["status"] != "ok":
            warnings.append("stable_plateau_width_below_min")
        if formal_valid_fraction < config["multiheight_valid_fraction_min"]:
            warnings.append("formal_roi_valid_column_fraction_low")
        if trim_max_relative_change is None:
            warnings.append("roi_trim_sensitivity_unavailable")
        elif trim_max_relative_change >= config["roi_trim_max_relative_change"]:
            warnings.append("formal_roi_trim_sensitive")
        warnings.extend(unavailable_reasons)
        formal_allowed = bool(
            common_valid_column_count > 0
            and formal_valid_fraction >= config["multiheight_valid_fraction_min"]
            and trim_max_relative_change is not None
            and trim_max_relative_change < config["roi_trim_max_relative_change"]
        )
        roi_status = (
            ("warning" if warnings else "ok")
            if formal_allowed else "unavailable"
        )
        result: dict[str, Any] = {
            "label": label,
            "height_mm": height_mm,
            "phase_a_role": (
                "small_height_resolution_diagnostic" if roi_id == "h001"
                else "primary_sensitivity"
            ),
            "included_in_combined": roi_id in {"h010", "h030"},
            "status": roi_status,
            "warnings": warnings,
            "unavailable_reasons": unavailable_reasons,
            "selected_x_range": selected,
            "auto_stable_x_range": detection["auto_stable_x_range"],
            "stable_plateau_analysis_x_range": diagnostic_range,
            "analysis_x_range": analysis_range,
            "stable_width_px": detection["stable_width_px"],
            "selected_valid_column_fraction": selected_valid_fraction,
            "stable_valid_column_fraction": stable_valid_fraction,
            "formal_roi_trim_px": formal_trim,
            "formal_statistics_allowed": formal_allowed,
            "roi_stable": roi_stable,
            "roi_sensitive": None if roi_stable is None else not roi_stable,
            "roi_trim_max_relative_change": trim_max_relative_change,
            "roi_trim_sensitivity": roi_trim_rows,
            "object_valid_column_count": object_valid_column_count,
            "reference_valid_column_count": reference_valid_column_count,
            "common_valid_column_count": common_valid_column_count,
            "detection": detection,
        }
        if formal_allowed:
            values = delta_y[columns][valid]
            sensitivity = np.abs(values) / height_mm
            delta_stats = _distribution_statistics(values)
            sensitivity_stats = _distribution_statistics(sensitivity)
            sensitivity_by_column = np.abs(delta_y[columns]) / height_mm
            try:
                repeat_valid, repeatability_statistics = _repeatability_statistics(
                    sigma_obj[columns], sensitivity_by_column, valid
                )
            except ValueError as exc:
                result["formal_statistics_allowed"] = False
                result["status"] = "unavailable"
                result["warnings"].append("repeatability_unavailable")
                result["unavailable_reasons"].append(str(exc))
                formal_allowed = False
            else:
                result.update({
                    "analysis_width_px": analysis_range[1] - analysis_range[0] + 1,
                    "valid_column_count": int(values.size),
                    "valid_column_fraction": formal_valid_fraction,
                    "delta_y_median_px": delta_stats["median"],
                    "delta_y_p05_px": delta_stats["p05"],
                    "delta_y_p95_px": delta_stats["p95"],
                    "sensitivity_median_px_per_mm": sensitivity_stats["median"],
                    "sensitivity_p05_px_per_mm": sensitivity_stats["p05"],
                    "sensitivity_p95_px_per_mm": sensitivity_stats["p95"],
                    "repeatability_valid_column_count": int(np.count_nonzero(repeat_valid)),
                    "repeatability_valid_column_fraction": float(np.mean(repeat_valid)),
                    **repeatability_statistics,
                })
        if not formal_allowed:
            result.update({
                "analysis_width_px": analysis_range[1] - analysis_range[0] + 1,
                "valid_column_count": common_valid_column_count,
                "valid_column_fraction": formal_valid_fraction,
                "delta_y_median_px": None,
                "delta_y_p05_px": None,
                "delta_y_p95_px": None,
                "sensitivity_median_px_per_mm": None,
                "sensitivity_p05_px_per_mm": None,
                "sensitivity_p95_px_per_mm": None,
                "repeatability_valid_column_count": 0,
                "repeatability_valid_column_fraction": 0.0,
                "sigma_pixel_p50_px": None,
                "sigma_pixel_p95_px": None,
                "sigma_z_pred_p50_mm": None,
                "sigma_z_pred_p95_mm": None,
            })
        for local_index, column in enumerate(columns):
            sensitivity_u = abs(delta_y[column]) / height_mm if delta_valid[column] else np.nan
            repeatability_valid = bool(
                formal_allowed
                and delta_valid[column]
                and np.isfinite(sigma_obj[column])
                and np.isfinite(sensitivity_u)
                and sensitivity_u > np.finfo(np.float64).eps
            )
            sigma_z_u = sigma_obj[column] / sensitivity_u if repeatability_valid else np.nan
            repeatability_rows.append({
                "roi_id": roi_id,
                "height_mm": height_mm,
                "u": int(column),
                "valid_fraction": float(valid_fraction[column]),
                "formal_column_valid": bool(valid[local_index]),
                "sigma_pixel_px": _nan_text(sigma_obj[column]),
                "sensitivity_px_per_mm": _nan_text(sensitivity_u),
                "sigma_z_pred_mm": _nan_text(sigma_z_u),
                "repeatability_valid": repeatability_valid,
            })
        roi_results[roi_id] = result

    sensitivity_combined, sigma_z_pred_combined = _phase_a_combined_metrics(roi_results)

    column_rows: list[dict[str, Any]] = []
    for column in range(image_width):
        containing = next(
            (
                roi_id for roi_id, result in roi_results.items()
                if result["selected_x_range"][0] <= column <= result["selected_x_range"][1]
            ),
            "",
        )
        result = roi_results.get(containing)
        auto_range = result["auto_stable_x_range"] if result else None
        analysis_range = result["analysis_x_range"] if result else None
        height_mm = result["height_mm"] if result else None
        sensitivity = (
            abs(delta_y[column]) / height_mm
            if height_mm is not None and np.isfinite(delta_y[column]) else np.nan
        )
        column_rows.append({
            "u": column,
            "y_obj_median_px": _nan_text(y_obj[column]),
            "sigma_obj_px": _nan_text(sigma_obj[column]),
            "valid_fraction": float(valid_fraction[column]),
            "obj_valid": bool(object_reliable[column]),
            "delta_valid": bool(delta_valid[column]),
            "y_ref_smooth_px": _nan_text(y_ref[column]),
            "delta_y_px": _nan_text(delta_y[column]),
            "roi_id": containing,
            "height_mm": "" if height_mm is None else height_mm,
            "sensitivity_px_per_mm": _nan_text(sensitivity),
            "in_selected_roi": bool(containing),
            "in_auto_stable_roi": bool(
                auto_range is not None and auto_range[0] <= column <= auto_range[1]
            ),
            "in_analysis_roi": bool(
                analysis_range is not None and analysis_range[0] <= column <= analysis_range[1]
            ),
        })
    _write_csv(
        output_dir / "multiheight_by_column.csv",
        (
            "u", "y_obj_median_px", "sigma_obj_px", "valid_fraction", "obj_valid", "delta_valid",
            "y_ref_smooth_px", "delta_y_px", "roi_id", "height_mm", "sensitivity_px_per_mm",
            "in_selected_roi",
            "in_auto_stable_roi", "in_analysis_roi",
        ),
        column_rows,
    )
    _write_csv(
        output_dir / "roi_trim_sensitivity_check.csv",
        (
            "roi_id", "height_mm", "trim_px", "x_start", "x_end", "width_px",
            "valid_column_count", "available", "unavailable_reason",
            "delta_y_median_px", "sensitivity_median_px_per_mm",
            "relative_change_vs_trim0", "roi_stable", "roi_sensitive",
        ),
        trim_rows,
    )
    _write_csv(
        output_dir / "repeatability_by_column.csv",
        (
            "roi_id", "height_mm", "u", "valid_fraction", "formal_column_valid",
            "sigma_pixel_px", "sensitivity_px_per_mm", "sigma_z_pred_mm",
            "repeatability_valid",
        ),
        repeatability_rows,
    )
    plot_paths = _save_multiheight_analysis_plots(
        output_dir,
        aggregates,
        y_ref,
        delta_y,
        roi_results,
        trim_rows,
        config["reference_surface_x_range"],
        dataset.name,
    )
    repeatability_plot_paths = _save_repeatability_plots(
        output_dir, repeatability_rows, roi_results, dataset.name
    )
    plot_paths.extend(repeatability_plot_paths)
    # Analysis is read-only with respect to the manually confirmed ROI registry.
    public_results = {
        roi_id: {key: (list(value) if isinstance(value, tuple) else value) for key, value in result.items() if key != "detection"}
        for roi_id, result in roi_results.items()
    }
    summary = {
        "schema_version": 1,
        "config_id": dataset.name,
        "task_id": "multiheight",
        "frame_count": context["frame_count"],
        "image_shape": list(context["image_shape"]),
        "centre_extractor": "realtime_steger.extract_steger_columns",
        "steger_config": str(config["steger_config"]),
        "steger_config_sha256": sha256_file(config["steger_config"]),
        "reference_surface_x_range": list(config["reference_surface_x_range"]),
        "reference_analysis_sha256": sha256_file(dataset / "analysis" / "reference_analysis.json"),
        "reference_by_column_sha256": sha256_file(reference_curve_path),
        "reference_model_modified": False,
        "same_column_reference_subtraction": True,
        "plateau_selection_uses_sensitivity": False,
        "plateau_detection_diagnostic_only": True,
        "formal_roi_rule": "selected_roi_symmetric_trim3_median",
        "analysis_config": {
            "valid_frame_fraction_min": config["multiheight_valid_fraction_min"],
            "reference_band_margin_px": config["reference_band_margin_px"],
            "formal_roi_trim_px": config["formal_roi_trim_px"],
            "median_smoothing_window_px": config["plateau_median_window_px"],
            "max_abs_gradient_px_per_column": config["plateau_max_gradient_px_per_column"],
            "max_abs_step_px": config["plateau_max_step_px"],
            "sigma_mad_scale": config["plateau_sigma_mad_scale"],
            "sigma_floor_limit_px": config["plateau_sigma_floor_limit_px"],
            "plateau_erosion_px": config["plateau_erosion_px"],
            "min_stable_width_px": config["min_stable_width_px"],
        },
        "band_detection": {
            key: band_validation[key]
            for key in (
                "method",
                "bottom_semantics",
                "original_band_top",
                "original_band_bottom",
                "reference_band_top",
                "reference_band_bottom",
                "reference_envelope_top",
                "reference_envelope_bottom",
                "final_band_top",
                "final_band_bottom",
                "original_band_height",
                "final_band_height",
                "steger_time_ms_before",
                "steger_time_ms_after",
                "runtime_change_percent",
                "band_bounds_identical_across_frames",
                "reference_band_margin_px",
                "h1_peak_y_median_px",
                "h1_peak_inside_original_band",
                "h1_peak_inside_final_band",
                "background_false_detection_rate",
            )
        },
        "band_fix_validation": band_validation,
        "band_fix_validated": band_validation["band_fix_validated"],
        "reference_union_band_validated": band_validation[
            "reference_union_band_validated"
        ],
        "rois": public_results,
        "sensitivity_combined_px_per_mm": sensitivity_combined,
        "sigma_pixel_h1_p50_px": roi_results["h001"]["sigma_pixel_p50_px"],
        "sigma_pixel_h1_p95_px": roi_results["h001"]["sigma_pixel_p95_px"],
        "sigma_pixel_h10_p50_px": roi_results["h010"]["sigma_pixel_p50_px"],
        "sigma_pixel_h10_p95_px": roi_results["h010"]["sigma_pixel_p95_px"],
        "sigma_pixel_h30_p50_px": roi_results["h030"]["sigma_pixel_p50_px"],
        "sigma_pixel_h30_p95_px": roi_results["h030"]["sigma_pixel_p95_px"],
        "sigma_z_pred_h1_p50_mm": roi_results["h001"]["sigma_z_pred_p50_mm"],
        "sigma_z_pred_h1_p95_mm": roi_results["h001"]["sigma_z_pred_p95_mm"],
        "sigma_z_pred_h10_p50_mm": roi_results["h010"]["sigma_z_pred_p50_mm"],
        "sigma_z_pred_h10_p95_mm": roi_results["h010"]["sigma_z_pred_p95_mm"],
        "sigma_z_pred_h30_p50_mm": roi_results["h030"]["sigma_z_pred_p50_mm"],
        "sigma_z_pred_h30_p95_mm": roi_results["h030"]["sigma_z_pred_p95_mm"],
        "sigma_z_pred_combined_mm": sigma_z_pred_combined,
        "combined_height_inputs": ["h010", "h030"],
        "h001_excluded_from_combined": True,
        "needs_manual_review": any(
            not roi_results[roi_id]["formal_statistics_allowed"]
            for roi_id in ("h010", "h030")
        ),
        "sigma_z_pred_computed": sensitivity_combined is not None,
        "other_configs_analyzed": False,
        "phase_a_ranking_performed": False,
        "laser_plane_used": False,
        "pnp_used": False,
        "reconstruction_3d_performed": False,
        "outputs": [str(path) for path in (
            output_dir / "multiheight_by_column.csv",
            output_dir / "multiheight_analysis.json",
            output_dir / "roi_trim_sensitivity_check.csv",
            output_dir / "repeatability_by_column.csv",
            output_dir / "band_fix_validation.json",
            *plot_paths,
        )],
    }
    _atomic_write_text(
        output_dir / "multiheight_analysis.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        output_dir / "band_fix_validation.json",
        json.dumps(band_validation, ensure_ascii=False, indent=2) + "\n",
    )
    geometry_summary_sha256_after = (
        sha256_file(geometry_summary_path) if geometry_summary_path.is_file() else None
    )
    if geometry_summary_sha256_after != geometry_summary_sha256_before:
        raise RuntimeError("单组 band fix 验证意外修改了 geometry_master_summary.csv")
    return summary


def _summary_nan_row(matrix_row: Mapping[str, Any], status: str) -> dict[str, Any]:
    row = {field: "NaN" for field in GEOMETRY_SUMMARY_FIELDS}
    row.update(
        config_id=matrix_row["config_id"],
        baseline_scale_reading=matrix_row["baseline_scale_reading"],
        laser_angle_deg=matrix_row["laser_angle_deg"],
        status=status,
        warnings="",
        needs_manual_review=status not in {"invalid_fov", "pending_roi"},
    )
    return row


def _load_template_analysis(dataset: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = load_document(dataset / "analysis" / "reference_analysis.json")
    multiheight = load_document(dataset / "analysis" / "multiheight_analysis.json")
    if reference.get("config_id") != REFERENCE_DEVELOPMENT_CONFIG_ID:
        raise ValueError("模板 reference_analysis.json config_id 异常")
    if multiheight.get("config_id") != REFERENCE_DEVELOPMENT_CONFIG_ID:
        raise ValueError("模板 multiheight_analysis.json config_id 异常")
    if multiheight.get("formal_roi_rule") != "selected_roi_symmetric_trim3_median":
        raise ValueError("模板 multiheight 正式 ROI 规则不是已验证 trim3 median")
    return reference, multiheight


def _validate_frozen_algorithm(
    reference: Mapping[str, Any],
    multiheight: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    expected_steger_hash = sha256_file(config["steger_config"])
    if reference.get("steger_config_sha256") != expected_steger_hash:
        raise ValueError("reference Steger 配置 hash 与冻结配置不一致")
    if multiheight.get("steger_config_sha256") != expected_steger_hash:
        raise ValueError("multiheight Steger 配置 hash 与冻结配置不一致")
    if float(reference.get("valid_frame_fraction_min")) != config["valid_frame_fraction_min"]:
        raise ValueError("reference valid_frame_fraction_min 与冻结配置不一致")
    if int(reference.get("segment_edge_trim_px")) != config["segment_edge_trim_px"]:
        raise ValueError("reference segment_edge_trim_px 与冻结配置不一致")
    model = reference.get("smooth_model")
    if not isinstance(model, Mapping):
        raise ValueError("reference smooth_model provenance 缺失")
    if (
        float(model.get("penalty")) != config["smooth_spline_penalty"]
        or float(model.get("huber_delta")) != config["robust_huber_delta"]
    ):
        raise ValueError("reference robust B-spline 参数与冻结配置不一致")
    cross_validation = reference.get("cross_validation")
    if not isinstance(cross_validation, Mapping) or cross_validation.get("method") != "leave_one_observed_segment_out":
        raise ValueError("reference interior CV 方法与冻结流程不一致")
    multi_config = multiheight.get("analysis_config")
    if not isinstance(multi_config, Mapping):
        raise ValueError("multiheight analysis_config provenance 缺失")
    if (
        multiheight.get("formal_roi_rule") != "selected_roi_symmetric_trim3_median"
        or int(multi_config.get("formal_roi_trim_px")) != config["formal_roi_trim_px"]
        or float(multi_config.get("valid_frame_fraction_min"))
        != config["multiheight_valid_fraction_min"]
    ):
        raise ValueError("multiheight 正式 ROI/valid_fraction 规则与冻结流程不一致")


def _geometry_summary_row(
    matrix_row: Mapping[str, Any],
    reference: Mapping[str, Any],
    multiheight: Mapping[str, Any],
) -> dict[str, Any]:
    statistics = reference["statistics"]
    rois = multiheight["rois"]
    warnings: list[str] = []
    reference_needs_review = False
    if float(statistics["reference_cv_interior_rmse_px"]) > REFERENCE_CV_WARN_RMSE_PX:
        warnings.append("reference_cv_interior_rmse_high")
        reference_needs_review = True
    if float(statistics["reference_cv_interior_p95_px"]) > REFERENCE_CV_WARN_P95_PX:
        warnings.append("reference_cv_interior_p95_high")
        reference_needs_review = True
    for roi_id in ("h001", "h010", "h030"):
        warnings.extend(str(item) for item in rois[roi_id].get("warnings", []))
    warnings = list(dict.fromkeys(warnings))
    primary_complete = all(
        bool(rois[roi_id].get("formal_statistics_allowed"))
        for roi_id in ("h010", "h030")
    )
    needs_manual_review = bool(reference_needs_review or not primary_complete)
    status = "failed" if not primary_complete else ("warning" if warnings else "ok")
    metric = lambda value: "NaN" if value is None else value
    return {
        "config_id": matrix_row["config_id"],
        "baseline_scale_reading": matrix_row["baseline_scale_reading"],
        "laser_angle_deg": matrix_row["laser_angle_deg"],
        "status": status,
        "reference_cv_interior_rmse_px": statistics["reference_cv_interior_rmse_px"],
        "reference_cv_interior_p95_px": statistics["reference_cv_interior_p95_px"],
        "sensitivity_h1": metric(rois["h001"]["sensitivity_median_px_per_mm"]),
        "sensitivity_h10": metric(rois["h010"]["sensitivity_median_px_per_mm"]),
        "sensitivity_h30": metric(rois["h030"]["sensitivity_median_px_per_mm"]),
        "sensitivity_combined_px_per_mm": metric(multiheight["sensitivity_combined_px_per_mm"]),
        "sigma_pixel_h10_p95_px": metric(rois["h010"]["sigma_pixel_p95_px"]),
        "sigma_pixel_h30_p95_px": metric(rois["h030"]["sigma_pixel_p95_px"]),
        "sigma_z_pred_h10_p95_mm": metric(rois["h010"]["sigma_z_pred_p95_mm"]),
        "sigma_z_pred_h30_p95_mm": metric(rois["h030"]["sigma_z_pred_p95_mm"]),
        "sigma_z_pred_combined_mm": metric(multiheight["sigma_z_pred_combined_mm"]),
        "roi_trim_change_h10": metric(rois["h010"]["roi_trim_max_relative_change"]),
        "roi_trim_change_h30": metric(rois["h030"]["roi_trim_max_relative_change"]),
        "warnings": warnings,
        "needs_manual_review": needs_manual_review,
    }


def analyze_all(
    root: Path,
    registry: Path,
    analysis_config: Path,
    calibration_src: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    registry_path, registry_document = ensure_roi_registry(registry)
    frozen_config = _load_reference_analysis_config(analysis_config)
    roi_audit = audit_rois(root, registry_path, analysis_config)
    roi_status = {row["config_id"]: row["roi_status"] for row in roi_audit["rows"]}
    matrix_rows = build_initial_rows()
    summary_rows: list[dict[str, Any]] = []
    configs = registry_document["configs"]
    for matrix_row in matrix_rows:
        config_id = matrix_row["config_id"]
        if config_id == INVALID_FOV_CONFIG_ID:
            summary_rows.append(_summary_nan_row(matrix_row, "invalid_fov"))
            continue
        if roi_status.get(config_id) != "confirmed":
            summary_rows.append(_summary_nan_row(matrix_row, "pending_roi"))
            continue
        dataset = root / config_id
        reference_range = _registry_x_range(
            configs[config_id]["reference_surface"].get("x_range"),
            f"configs.{config_id}.reference_surface.x_range",
        )
        assert reference_range is not None
        try:
            reference_path = dataset / "analysis" / "reference_analysis.json"
            if reference_path.is_file():
                reference_summary = load_document(reference_path)
                if tuple(reference_summary.get("reference_surface_x_range") or ()) != reference_range:
                    raise ValueError("已有 reference_analysis 与当前 registry ROI 不一致")
            else:
                reference_summary = analyze_reference(
                    dataset,
                    analysis_config,
                    calibration_src,
                    overwrite=False,
                    reference_x_range=reference_range,
                )
            multiheight_path = dataset / "analysis" / "multiheight_analysis.json"
            if multiheight_path.is_file() and not overwrite:
                multiheight_summary = load_document(multiheight_path)
                if multiheight_summary.get("formal_roi_rule") != "selected_roi_symmetric_trim3_median":
                    raise ValueError("已有 multiheight_analysis 不是冻结 trim3 规则；请使用 --overwrite")
            else:
                multiheight_summary = analyze_multiheight(
                    dataset,
                    registry_path,
                    analysis_config,
                    calibration_src,
                    overwrite=overwrite,
                )
            _validate_frozen_algorithm(reference_summary, multiheight_summary, frozen_config)
            summary_rows.append(
                _geometry_summary_row(matrix_row, reference_summary, multiheight_summary)
            )
        except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            failed = _summary_nan_row(matrix_row, "failed")
            failed["warnings"] = [str(exc)]
            failed["needs_manual_review"] = True
            summary_rows.append(failed)
    output = root.parent / "results" / "geometry_master_summary.csv"
    _write_csv(output, GEOMETRY_SUMMARY_FIELDS, summary_rows)
    raw_counts = Counter(str(row["status"]) for row in summary_rows)
    counts = {
        status: raw_counts.get(status, 0)
        for status in ("ok", "warning", "failed", "invalid_fov")
    }
    if raw_counts.get("pending_roi"):
        counts["pending_roi"] = raw_counts["pending_roi"]
    return {
        "output": str(output),
        "rows": summary_rows,
        "counts": counts,
        "roi_audit": roi_audit,
        "ranking_generated": False,
        "heatmap_generated": False,
    }


def _steger_diagnostic_stack(
    realtime: Any,
    images: Sequence[np.ndarray],
    options: Mapping[str, Any],
    *,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the formal extractor on full frames or one explicit x/y background crop."""
    if not images:
        raise ValueError("Steger diagnostic 没有输入图像")
    if crop is None:
        sample = images[0]
        x_left, x_right, y_top, y_bottom = 0, sample.shape[1] - 1, 0, sample.shape[0] - 1
    else:
        x_left, x_right, y_top, y_bottom = crop
    width = x_right - x_left + 1
    y_stack = np.full((len(images), width), np.nan, dtype=np.float64)
    valid_stack = np.zeros((len(images), width), dtype=bool)
    for frame_index, image in enumerate(images):
        target = image[y_top:y_bottom + 1, x_left:x_right + 1]
        extracted = realtime.extract_steger_columns(target, options)
        valid = np.asarray(extracted.valid, dtype=bool) & np.isfinite(extracted.v_px)
        valid_stack[frame_index] = valid
        y_stack[frame_index, valid] = np.asarray(extracted.v_px, dtype=np.float64)[valid] + y_top
    return y_stack, valid_stack


def _steger_sweep_roi_statistics(
    y_stack: np.ndarray,
    valid_stack: np.ndarray,
    x_range: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Summarize per-frame and per-column detection without imposing one formal gate."""
    left, right = x_range
    roi_valid = np.asarray(valid_stack[:, left:right + 1], dtype=bool)
    roi_y = np.asarray(y_stack[:, left:right + 1], dtype=np.float64)
    frame_count = roi_valid.shape[0]
    detected_count = np.sum(roi_valid, axis=0)
    detected_fraction = detected_count.astype(np.float64) / float(frame_count)
    per_frame_rate = np.mean(roi_valid, axis=1)
    sigma = np.full(roi_valid.shape[1], np.nan, dtype=np.float64)
    center = np.full(roi_valid.shape[1], np.nan, dtype=np.float64)
    for local_column in range(roi_valid.shape[1]):
        mask = roi_valid[:, local_column]
        if np.any(mask):
            values = roi_y[mask, local_column]
            center[local_column] = float(np.median(values))
            if values.size > 1:
                sigma[local_column] = float(np.std(values, ddof=1))
    sigma_finite = sigma[np.isfinite(sigma)]
    detected_y = roi_y[roi_valid]
    statistics: dict[str, Any] = {
        "frame_count": frame_count,
        "roi_width_px": roi_valid.shape[1],
        "detected_event_count": int(np.count_nonzero(roi_valid)),
        "per_frame_detection_rate_mean": float(np.mean(per_frame_rate)),
        "per_frame_detection_rate_p05": float(np.percentile(per_frame_rate, 5)),
        "per_frame_detection_rate_p50": float(np.median(per_frame_rate)),
        "per_frame_detection_rate_p95": float(np.percentile(per_frame_rate, 95)),
        "valid_column_fraction_at_0_8": float(np.mean(detected_fraction >= 0.8)),
        "valid_column_fraction_at_0_6": float(np.mean(detected_fraction >= 0.6)),
        "valid_column_fraction_at_0_5": float(np.mean(detected_fraction >= 0.5)),
        "sigma_pixel_column_count": int(sigma_finite.size),
        "sigma_pixel_p50_px": (
            float(np.median(sigma_finite)) if sigma_finite.size else None
        ),
        "sigma_pixel_p95_px": (
            float(np.percentile(sigma_finite, 95)) if sigma_finite.size else None
        ),
        "median_center_position_px": (
            float(np.median(detected_y)) if detected_y.size else None
        ),
    }
    return statistics, {
        "detected_count": detected_count,
        "detected_fraction": detected_fraction,
        "sigma": sigma,
        "center": center,
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def diagnose_auto_band_generation(
    dataset: Path,
    analysis_config: Path,
    calibration_src: Path,
    *,
    overwrite_diagnostic: bool = False,
) -> dict[str, Any]:
    """Trace the unchanged realtime_steger auto-band path on reference and multiheight."""
    dataset = dataset.expanduser().resolve()
    if dataset.name not in CAPTURED_CONFIG_IDS:
        raise ValueError(f"不是已采集的 Phase-A 配置：{dataset.name}")
    output_dir = dataset / "analysis" / "auto_band_generation_diagnostic"
    csv_path = output_dir / "auto_band_generation_by_frame.csv"
    json_path = output_dir / "auto_band_generation_summary.json"
    existing = [path for path in (csv_path, json_path) if path.exists()]
    if existing and not overwrite_diagnostic:
        raise FileExistsError(
            "auto-band diagnostic 已存在，不会静默覆盖："
            + ", ".join(str(path) for path in existing)
        )
    config = _load_reference_analysis_config(analysis_config)
    realtime = _load_realtime_steger(calibration_src)
    options = realtime.load_steger_options(config["steger_config"])
    protected_paths = (
        Path(config["source"]),
        Path(config["steger_config"]),
        dataset / "analysis" / "reference_analysis.json",
        dataset / "analysis" / "reference_by_column.csv",
        dataset / "analysis" / "multiheight_analysis.json",
        dataset.parent.parent / "results" / "geometry_master_summary.csv",
    )
    protected_before = {
        str(path): sha256_file(path) for path in protected_paths if path.is_file()
    }
    rows: list[dict[str, Any]] = []
    trace_fields = (
        "raw_candidate_top",
        "raw_candidate_bottom",
        "margin_before_clip",
        "margin_after_clip",
        "roi_max_height_applied",
        "final_band_top",
        "final_band_bottom",
    )
    for task_id in ("reference", "multiheight"):
        for record in _task_frame_records(dataset, task_id):
            image = _read_gray_image(dataset / record["filename"])
            extracted = realtime.extract_steger_columns(image, options)
            metadata = extracted.metadata
            if any(field not in metadata for field in trace_fields):
                raise RuntimeError("realtime_steger 未返回完整 auto-band diagnostic trace")
            if (
                metadata["final_band_top"] != metadata["original_band_top_px"]
                or metadata["final_band_bottom"]
                != metadata["original_band_bottom_exclusive_px"]
            ):
                raise RuntimeError("纯 auto-band 诊断中 trace 与 extractor original band 不一致")
            rows.append({
                "task_id": task_id,
                "frame_index": int(record["index"]),
                "filename": record["filename"],
                "image_height": image.shape[0],
                "image_width": image.shape[1],
                "seed_row": metadata.get("seed_row"),
                "adaptive_row_threshold": metadata.get("adaptive_row_threshold"),
                "raw_candidate_top": metadata["raw_candidate_top"],
                "raw_candidate_bottom": metadata["raw_candidate_bottom"],
                "margin_before_clip": json.dumps(metadata["margin_before_clip"]),
                "margin_after_clip": json.dumps(metadata["margin_after_clip"]),
                "roi_max_height_applied": metadata["roi_max_height_applied"],
                "final_band_top": metadata["final_band_top"],
                "final_band_bottom": metadata["final_band_bottom"],
            })

    task_summaries: dict[str, Any] = {}
    for task_id in ("reference", "multiheight"):
        task_rows = [row for row in rows if row["task_id"] == task_id]
        combinations = Counter(
            (
                row["raw_candidate_top"],
                row["raw_candidate_bottom"],
                row["margin_before_clip"],
                row["margin_after_clip"],
                bool(row["roi_max_height_applied"]),
                row["final_band_top"],
                row["final_band_bottom"],
            )
            for row in task_rows
        )
        dominant, dominant_count = combinations.most_common(1)[0]
        task_summaries[task_id] = {
            "frame_count": len(task_rows),
            "all_frames_same_trace": len(combinations) == 1,
            "unique_trace_count": len(combinations),
            "dominant_trace_frame_count": dominant_count,
            "dominant_trace": {
                "raw_candidate_top": dominant[0],
                "raw_candidate_bottom": dominant[1],
                "margin_before_clip": json.loads(dominant[2]),
                "margin_after_clip": json.loads(dominant[3]),
                "roi_max_height_applied": dominant[4],
                "final_band_top": dominant[5],
                "final_band_bottom": dominant[6],
            },
        }
    output_dir.mkdir(parents=True, exist_ok=overwrite_diagnostic)
    _write_csv(
        csv_path,
        (
            "task_id", "frame_index", "filename", "image_height", "image_width",
            "seed_row", "adaptive_row_threshold", "raw_candidate_top",
            "raw_candidate_bottom", "margin_before_clip", "margin_after_clip",
            "roi_max_height_applied", "final_band_top", "final_band_bottom",
        ),
        rows,
    )
    protected_after = {
        str(path): sha256_file(path) for path in protected_paths if path.is_file()
    }
    if protected_after != protected_before:
        raise RuntimeError("auto-band diagnostic 意外修改了正式配置或分析结果")
    summary = {
        "schema_version": 1,
        "config_id": dataset.name,
        "diagnostic_only": True,
        "auto_band_implementation": {
            "file": str((calibration_src / "realtime_steger.py").resolve()),
            "function": "_detect_steger_band",
            "caller": "_extract_columnwise",
        },
        "projection": {
            "row_peak": "max(gray, axis=1)",
            "seed": "argmax(sum(gray, axis=1))",
            "adaptive_threshold": "max(threshold, 0.3 * row_peak[seed])",
        },
        "bottom_semantics": "exclusive",
        "effective_steger_options": options,
        "task_summaries": task_summaries,
        "reference_and_multiheight_share_same_band_function": True,
        "steger_derivatives_run_on_cropped_band": True,
        "protected_artifacts_sha256": protected_after,
        "formal_results_modified": False,
        "outputs": [str(csv_path), str(json_path)],
    }
    _atomic_write_text(
        json_path,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary


def _classify_steger_h1_failure(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify H1 failure using explicit recovery and primary-stability criteria."""
    by_parameter: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_parameter.setdefault(str(row["parameter_id"]), {})[str(row["roi_id"])] = row
    formal_id = next(
        parameter_id for parameter_id, parameter_rows in by_parameter.items()
        if bool(parameter_rows["h001"].get("formal_parameters"))
    )
    formal = by_parameter[formal_id]
    formal_h1 = formal["h001"]
    formal_background_rate = float(formal["background"]["per_frame_detection_rate_mean"])
    background_limit = max(0.02, formal_background_rate + 0.02)
    acceptable: list[tuple[tuple[float, ...], str, dict[str, float]]] = []
    best_h1_at_0_5 = 0.0
    for parameter_id, parameter_rows in by_parameter.items():
        if parameter_id == formal_id:
            continue
        h1 = parameter_rows["h001"]
        h1_recovery = float(h1["valid_column_fraction_at_0_8"])
        best_h1_at_0_5 = max(best_h1_at_0_5, float(h1["valid_column_fraction_at_0_5"]))
        background_rate = float(parameter_rows["background"]["per_frame_detection_rate_mean"])
        primary_shifts: list[float] = []
        primary_sigma_ratios: list[float] = []
        for roi_id in ("h010", "h030"):
            shift = _finite_or_none(parameter_rows[roi_id].get("center_shift_vs_formal_px"))
            ratio = _finite_or_none(parameter_rows[roi_id].get("sigma_p95_ratio_vs_formal"))
            primary_shifts.append(abs(shift) if shift is not None else float("inf"))
            primary_sigma_ratios.append(ratio if ratio is not None else float("inf"))
        max_shift = max(primary_shifts)
        max_sigma_ratio = max(primary_sigma_ratios)
        if (
            h1_recovery >= 0.5
            and background_rate <= background_limit
            and max_shift <= 0.10
            and max_sigma_ratio <= 1.25
        ):
            impact = (
                max_shift,
                max(0.0, max_sigma_ratio - 1.0),
                background_rate,
                -float(h1_recovery),
                -float(parameter_rows["h001"]["threshold"]),
                -float(parameter_rows["h001"]["deriv_thresh"]),
            )
            acceptable.append((impact, parameter_id, {
                "h1_valid_column_fraction_at_0_8": h1_recovery,
                "background_false_detection_rate": background_rate,
                "max_primary_center_shift_px": max_shift,
                "max_primary_sigma_p95_ratio": max_sigma_ratio,
            }))
    candidate_id: str | None = None
    candidate_metrics: dict[str, float] | None = None
    if acceptable:
        _impact, candidate_id, candidate_metrics = min(acceptable, key=lambda item: item[0])

    formal_h1_at_0_8 = float(formal_h1["valid_column_fraction_at_0_8"])
    formal_h1_at_0_5 = float(formal_h1["valid_column_fraction_at_0_5"])
    formal_h1_event_rate = float(formal_h1["per_frame_detection_rate_mean"])
    if formal_h1_at_0_8 < 0.1 and formal_h1_at_0_5 >= 0.5 and formal_h1_event_rate >= 0.5:
        verdict = "B. mainly_valid_fraction_threshold"
        rationale = "正式单帧提取已有足够检测，但0.8跨帧门槛淘汰了大部分列"
    elif candidate_id is not None:
        verdict = "A. mainly_per_frame_steger_threshold"
        rationale = "降低单帧Steger阈值后H1在0.8门槛恢复，且背景与H10/H30保持稳定"
    elif best_h1_at_0_5 < 0.2:
        verdict = "C. optical_or_signal_quality_issue"
        rationale = "扫描范围内即使使用0.5跨帧门槛也无法稳定恢复H1"
    else:
        verdict = "D. inconclusive"
        rationale = "H1有部分恢复，但伴随背景误检或primary中心/重复性变化，证据不足"
    return {
        "verdict": verdict,
        "rationale": rationale,
        "formal_parameter_id": formal_id,
        "formal_h1_per_frame_detection_rate": formal_h1_event_rate,
        "formal_h1_valid_column_fraction_at_0_8": formal_h1_at_0_8,
        "formal_h1_valid_column_fraction_at_0_5": formal_h1_at_0_5,
        "background_false_detection_rate_limit": background_limit,
        "candidate_parameter_id": candidate_id,
        "candidate_metrics": candidate_metrics,
        "candidate_acceptance_criteria": {
            "h1_valid_column_fraction_at_0_8_min": 0.5,
            "background_false_detection_rate_max": background_limit,
            "h10_h30_max_abs_center_shift_px": 0.10,
            "h10_h30_sigma_p95_ratio_max": 1.25,
        },
    }


def _save_steger_h1_diagnostic_plots(
    output_dir: Path,
    detection_rows: Sequence[Mapping[str, Any]],
    sweep_rows: Sequence[Mapping[str, Any]],
    verdict: Mapping[str, Any],
) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = {item[0]: item[3] for item in MULTIHEIGHT_ROIS}
    labels = {item[0]: item[1] for item in MULTIHEIGHT_ROIS}
    paths: list[Path] = []
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    for axis, (roi_id, label, _height, color) in zip(axes, MULTIHEIGHT_ROIS):
        rows = [row for row in detection_rows if row["roi_id"] == roi_id]
        axis.plot(
            [row["u"] for row in rows],
            [row["detected_frame_fraction"] for row in rows],
            ".-", color=color, ms=3, lw=0.8,
        )
        for gate, line_style in ((0.8, "-"), (0.6, "--"), (0.5, ":")):
            axis.axhline(gate, color="#616161", lw=0.8, ls=line_style, label=f"gate {gate}")
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("detected fraction")
        axis.set_title(f"{label} formal Steger detection over 50 frames")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", ncol=3)
    axes[-1].set_xlabel("u [px]")
    paths.append(_save_figure(output_dir / "detected_frame_fraction_by_column.png", figure))
    plt.close(figure)

    parameter_ids = list(dict.fromkeys(str(row["parameter_id"]) for row in sweep_rows))
    x = np.arange(len(parameter_ids))
    by_key = {(str(row["parameter_id"]), str(row["roi_id"])): row for row in sweep_rows}
    figure, axes = plt.subplots(4, 1, figsize=(15, 14), constrained_layout=True)
    h1_rows = [by_key[(parameter_id, "h001")] for parameter_id in parameter_ids]
    for field, label, marker in (
        ("valid_column_fraction_at_0_8", "H1 gate 0.8", "o"),
        ("valid_column_fraction_at_0_6", "H1 gate 0.6", "s"),
        ("valid_column_fraction_at_0_5", "H1 gate 0.5", "^"),
    ):
        axes[0].plot(x, [row[field] for row in h1_rows], marker=marker, label=label)
    axes[0].set_ylabel("valid column fraction")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(loc="best", ncol=3)

    for roi_id in ("h010", "h030"):
        axes[1].plot(
            x,
            [_finite_or_none(by_key[(parameter_id, roi_id)]["center_shift_vs_formal_px"]) for parameter_id in parameter_ids],
            "o-", color=colors[roi_id], label=f"{labels[roi_id]} center shift",
        )
    axes[1].axhline(0.0, color="#616161", lw=0.8)
    axes[1].set_ylabel("center shift [px]")
    axes[1].legend(loc="best")

    for roi_id in ("h001", "h010", "h030"):
        axes[2].plot(
            x,
            [_finite_or_none(by_key[(parameter_id, roi_id)]["sigma_pixel_p95_px"]) for parameter_id in parameter_ids],
            "o-", color=colors[roi_id], label=f"{labels[roi_id]} sigma P95",
        )
    axes[2].set_ylabel("sigma_pixel P95 [px]")
    axes[2].legend(loc="best", ncol=3)

    axes[3].plot(
        x,
        [by_key[(parameter_id, "background")]["per_frame_detection_rate_mean"] for parameter_id in parameter_ids],
        "o-", color="#d32f2f", label="background false detection rate",
    )
    axes[3].axhline(
        float(verdict["background_false_detection_rate_limit"]),
        color="#616161", ls="--", lw=0.8, label="acceptance limit",
    )
    axes[3].set_ylabel("false detection rate")
    axes[3].legend(loc="best")
    axes[3].set_xticks(x, parameter_ids, rotation=45, ha="right")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[0].set_title(f"B05_A15 offline Steger sweep | {verdict['verdict']}")
    paths.append(_save_figure(output_dir / "steger_threshold_sweep.png", figure))
    plt.close(figure)
    return paths


def diagnose_steger_h1_failure(
    dataset: Path,
    registry: Path,
    analysis_config: Path,
    calibration_src: Path,
    background_roi: tuple[int, int, int, int],
) -> dict[str, Any]:
    """Run a read-only offline Steger threshold sweep for one multiheight dataset."""
    dataset = dataset.expanduser().resolve()
    if dataset.name != "B05_A15":
        raise ValueError("当前 H1 diagnostic 第一版仅允许 B05_A15")
    output_dir = dataset / "analysis" / "steger_h1_diagnostic"
    outputs = (
        output_dir / "steger_detection_by_column.csv",
        output_dir / "detected_frame_fraction_by_column.png",
        output_dir / "steger_threshold_sweep.csv",
        output_dir / "steger_threshold_sweep.png",
        output_dir / "steger_threshold_sweep.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "诊断输出已存在，不会覆盖：" + ", ".join(str(path) for path in existing)
        )
    config = _load_reference_analysis_config(analysis_config)
    registry_path, _registry_document, config_entry = _load_roi_registry(registry, dataset.name)
    base_options = _load_realtime_steger(calibration_src).load_steger_options(config["steger_config"])
    for key, expected in (("sigma", 1.5), ("threshold", 30.0), ("deriv_thresh", 0.5)):
        if float(base_options[key]) != expected:
            raise ValueError(f"正式 Steger {key}={base_options[key]}，不符合本诊断冻结值 {expected}")
    records = _task_frame_records(dataset, "multiheight")
    if len(records) != 50:
        raise ValueError(f"B05_A15 multiheight 应为50帧，实际 {len(records)}")
    images = [_read_gray_image(dataset / record["filename"]) for record in records]
    image_height, image_width = images[0].shape
    if any(image.shape != (image_height, image_width) for image in images):
        raise ValueError("B05_A15 multiheight 图像尺寸不一致")
    x_left, x_right, y_top, y_bottom = background_roi
    if not (0 <= x_left <= x_right < image_width and 0 <= y_top <= y_bottom < image_height):
        raise ValueError(f"background ROI 超出图像范围：{background_roi}")
    roi_ranges: dict[str, tuple[int, int]] = {}
    for roi_id, _label, _height, _color in MULTIHEIGHT_ROIS:
        selected = _roi_selected_x_range(config_entry["multiheight"][roi_id], roi_id)
        if selected is None:
            raise ValueError(f"{roi_id}.selected_x_range 尚未标注")
        roi_ranges[roi_id] = selected

    protected_paths = (
        registry_path,
        Path(config["source"]),
        Path(config["steger_config"]),
        dataset / "analysis" / "reference_analysis.json",
        dataset / "analysis" / "reference_by_column.csv",
        dataset / "analysis" / "multiheight_analysis.json",
        dataset.parent.parent / "results" / "geometry_master_summary.csv",
    )
    protected_before = {
        str(path): sha256_file(path) for path in protected_paths if path.is_file()
    }
    realtime = _load_realtime_steger(calibration_src)
    sweep_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    formal_centers: dict[str, float | None] = {}
    formal_sigma_p95: dict[str, float | None] = {}
    for threshold in STEGER_DIAGNOSTIC_THRESHOLDS:
        for deriv_thresh in STEGER_DIAGNOSTIC_DERIV_THRESHOLDS:
            options = dict(base_options)
            options.update(sigma=1.5, threshold=threshold, deriv_thresh=deriv_thresh)
            parameter_id = f"T{threshold:g}_D{deriv_thresh:g}"
            formal_parameters = threshold == 30.0 and deriv_thresh == 0.5
            y_stack, valid_stack = _steger_diagnostic_stack(realtime, images, options)
            background_y, background_valid = _steger_diagnostic_stack(
                realtime, images, options, crop=background_roi
            )
            parameter_rows: list[dict[str, Any]] = []
            for roi_id, _label, _height, _color in MULTIHEIGHT_ROIS:
                statistics, columns = _steger_sweep_roi_statistics(
                    y_stack, valid_stack, roi_ranges[roi_id]
                )
                row = {
                    "parameter_id": parameter_id,
                    "sigma": 1.5,
                    "threshold": threshold,
                    "deriv_thresh": deriv_thresh,
                    "formal_parameters": formal_parameters,
                    "roi_id": roi_id,
                    "roi_kind": "measurement_piece",
                    "x_left": roi_ranges[roi_id][0],
                    "x_right": roi_ranges[roi_id][1],
                    "y_top": "",
                    "y_bottom": "",
                    **statistics,
                }
                parameter_rows.append(row)
                if formal_parameters:
                    formal_centers[roi_id] = statistics["median_center_position_px"]
                    formal_sigma_p95[roi_id] = statistics["sigma_pixel_p95_px"]
                    for local_column, column in enumerate(
                        range(roi_ranges[roi_id][0], roi_ranges[roi_id][1] + 1)
                    ):
                        detection_rows.append({
                            "roi_id": roi_id,
                            "u": column,
                            "detected_frame_count": int(columns["detected_count"][local_column]),
                            "detected_frame_fraction": float(columns["detected_fraction"][local_column]),
                            "sigma_pixel_px": _nan_text(columns["sigma"][local_column]),
                            "median_center_position_px": _nan_text(columns["center"][local_column]),
                        })
            background_statistics, _background_columns = _steger_sweep_roi_statistics(
                background_y,
                background_valid,
                (0, x_right - x_left),
            )
            parameter_rows.append({
                "parameter_id": parameter_id,
                "sigma": 1.5,
                "threshold": threshold,
                "deriv_thresh": deriv_thresh,
                "formal_parameters": formal_parameters,
                "roi_id": "background",
                "roi_kind": "background_false_positive_control",
                "x_left": x_left,
                "x_right": x_right,
                "y_top": y_top,
                "y_bottom": y_bottom,
                **background_statistics,
            })
            sweep_rows.extend(parameter_rows)

    for row in sweep_rows:
        roi_id = str(row["roi_id"])
        if roi_id == "background":
            row["center_shift_vs_formal_px"] = None
            row["sigma_p95_ratio_vs_formal"] = None
            continue
        center = _finite_or_none(row["median_center_position_px"])
        formal_center = formal_centers.get(roi_id)
        sigma_p95 = _finite_or_none(row["sigma_pixel_p95_px"])
        formal_sigma = formal_sigma_p95.get(roi_id)
        row["center_shift_vs_formal_px"] = (
            center - formal_center
            if center is not None and formal_center is not None else None
        )
        row["sigma_p95_ratio_vs_formal"] = (
            sigma_p95 / formal_sigma
            if sigma_p95 is not None and formal_sigma is not None and formal_sigma > 0 else None
        )
    verdict = _classify_steger_h1_failure(sweep_rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(
        output_dir / "steger_detection_by_column.csv",
        (
            "roi_id", "u", "detected_frame_count", "detected_frame_fraction",
            "sigma_pixel_px", "median_center_position_px",
        ),
        detection_rows,
    )
    sweep_fields = (
        "parameter_id", "sigma", "threshold", "deriv_thresh", "formal_parameters",
        "roi_id", "roi_kind", "x_left", "x_right", "y_top", "y_bottom",
        "frame_count", "roi_width_px", "detected_event_count",
        "per_frame_detection_rate_mean", "per_frame_detection_rate_p05",
        "per_frame_detection_rate_p50", "per_frame_detection_rate_p95",
        "valid_column_fraction_at_0_8", "valid_column_fraction_at_0_6",
        "valid_column_fraction_at_0_5", "sigma_pixel_column_count",
        "sigma_pixel_p50_px", "sigma_pixel_p95_px", "median_center_position_px",
        "center_shift_vs_formal_px", "sigma_p95_ratio_vs_formal",
    )
    _write_csv(output_dir / "steger_threshold_sweep.csv", sweep_fields, sweep_rows)
    plot_paths = _save_steger_h1_diagnostic_plots(
        output_dir, detection_rows, sweep_rows, verdict
    )
    protected_after = {
        str(path): sha256_file(path) for path in protected_paths if path.is_file()
    }
    if protected_after != protected_before:
        raise RuntimeError("H1 diagnostic 意外修改了受保护的正式输入/结果")
    summary = {
        "schema_version": 1,
        "config_id": dataset.name,
        "task_id": "multiheight",
        "diagnostic_only": True,
        "frame_count": len(images),
        "image_shape": [image_height, image_width],
        "centre_extractor": "realtime_steger.extract_steger_columns",
        "formal_steger_config": str(config["steger_config"]),
        "formal_steger_config_sha256": sha256_file(config["steger_config"]),
        "formal_options": base_options,
        "scan": {
            "sigma": [1.5],
            "threshold": list(STEGER_DIAGNOSTIC_THRESHOLDS),
            "deriv_thresh": list(STEGER_DIAGNOSTIC_DERIV_THRESHOLDS),
            "valid_column_fraction_gates": list(STEGER_DIAGNOSTIC_VALID_FRACTIONS),
        },
        "measurement_rois": {roi_id: list(bounds) for roi_id, bounds in roi_ranges.items()},
        "background_roi_xyxy": list(background_roi),
        "background_roi_selection_method": "manual_from_multiheight_median_preview",
        "sigma_pixel_scope": "per-column std over every parameter-valid detection; columns require >=2 detections",
        "median_center_scope": "median over all parameter-valid frame-column detections inside ROI",
        "verdict": verdict,
        "protected_artifacts_sha256": protected_after,
        "ranking_generated": False,
        "formal_reference_modified": False,
        "formal_multiheight_analysis_modified": False,
        "geometry_master_summary_modified": False,
        "outputs": [str(path) for path in outputs],
    }
    _atomic_write_text(
        output_dir / "steger_threshold_sweep.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    return summary


def _local_raw_profile_metrics(profile: np.ndarray) -> dict[str, float]:
    values = np.asarray(profile, dtype=np.float64)
    edge_count = max(3, min(10, values.size // 4))
    background_samples = np.concatenate([values[:edge_count], values[-edge_count:]])
    background = float(np.median(background_samples))
    peak_index = int(np.argmax(values))
    peak = float(values[peak_index])
    half_max = background + 0.5 * (peak - background)
    left = peak_index
    while left > 0 and values[left - 1] >= half_max:
        left -= 1
    right = peak_index
    while right < values.size - 1 and values[right + 1] >= half_max:
        right += 1
    return {
        "peak_intensity": peak,
        "local_background": background,
        "peak_minus_background": peak - background,
        "fwhm_px": float(right - left + 1),
    }


def _profile_summary(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None, None, None
    return (
        float(np.median(finite)),
        float(np.percentile(finite, 5)),
        float(np.percentile(finite, 95)),
    )


def _classify_steger_root_cause(
    profile_rows: Sequence[Mapping[str, Any]],
    rejection_summary: Mapping[str, Any],
    sigma_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    profiles = {str(row["roi_id"]): row for row in profile_rows}
    h1_contrast = float(profiles["h001"]["peak_minus_background"])
    primary_contrast = float(np.median([
        float(profiles["h010"]["peak_minus_background"]),
        float(profiles["h030"]["peak_minus_background"]),
    ]))
    contrast_ratio = h1_contrast / primary_contrast if primary_contrast > 0 else 0.0
    by_sigma: dict[float, dict[str, Mapping[str, Any]]] = {}
    for row in sigma_rows:
        by_sigma.setdefault(float(row["sigma"]), {})[str(row["roi_id"])] = row
    candidates: list[tuple[tuple[float, ...], float, dict[str, float]]] = []
    for sigma, rows in by_sigma.items():
        if sigma == 1.5:
            continue
        h1_recovery = float(rows["h001"]["valid_column_fraction_at_0_8"])
        background_rate = float(rows["background"]["per_frame_detection_rate_mean"])
        shifts = [
            abs(float(rows[roi_id]["center_shift_vs_formal_px"]))
            if rows[roi_id].get("center_shift_vs_formal_px") is not None else float("inf")
            for roi_id in ("h010", "h030")
        ]
        ratios = [
            float(rows[roi_id]["sigma_p95_ratio_vs_formal"])
            if rows[roi_id].get("sigma_p95_ratio_vs_formal") is not None else float("inf")
            for roi_id in ("h010", "h030")
        ]
        max_shift = max(shifts)
        max_sigma_ratio = max(ratios)
        if (
            h1_recovery >= 0.5
            and background_rate <= 0.02
            and max_shift <= 0.10
            and max_sigma_ratio <= 1.25
        ):
            candidates.append((
                (max_shift, max(0.0, max_sigma_ratio - 1.0), background_rate, -h1_recovery),
                sigma,
                {
                    "h1_valid_column_fraction_at_0_8": h1_recovery,
                    "background_false_detection_rate": background_rate,
                    "max_primary_center_shift_px": max_shift,
                    "max_primary_sigma_p95_ratio": max_sigma_ratio,
                },
            ))
    candidate_sigma: float | None = None
    candidate_metrics: dict[str, float] | None = None
    if candidates:
        _impact, candidate_sigma, candidate_metrics = min(candidates, key=lambda item: item[0])

    h1_reasons = rejection_summary["rois"]["h001"]["rejection_reason_counts"]
    rejected_reasons = {
        key: int(value) for key, value in h1_reasons.items() if key != "accepted"
    }
    dominant_reason = max(rejected_reasons, key=rejected_reasons.get)
    formal_threshold = 30.0
    weak_peak = bool(
        float(profiles["h001"]["peak_intensity"]) < formal_threshold
        or contrast_ratio < 0.5
    )
    if candidate_sigma is not None:
        verdict = "B. steger_scale_mismatch"
        rationale = "改变 sigma 后H1在正式0.8门槛恢复，且primary与背景对照保持稳定"
    elif weak_peak:
        verdict = "A. weak_or_missing_optical_peak"
        rationale = "H1原始局部峰值/对比度显著弱于H10/H30，sigma扫描也未形成稳定中心"
    elif dominant_reason in {
        "intensity_peak_outside_detected_band",
        "ridge_response_below_threshold",
        "derivative_condition_failed",
        "invalid_subpixel_offset",
        "no_valid_ridge_candidate",
    }:
        verdict = "C. other_steger_rejection_rule"
        rationale = "H1存在可见光强峰，但主要被正式Steger的检测带或其他非尺度门控拒绝"
    else:
        verdict = "D. unresolved"
        rationale = "原始峰值、尺度扫描与拒绝原因未形成一致证据"
    return {
        "verdict": verdict,
        "rationale": rationale,
        "h1_peak_minus_background": h1_contrast,
        "primary_peak_minus_background_median": primary_contrast,
        "h1_to_primary_contrast_ratio": contrast_ratio,
        "h1_dominant_rejection_reason": dominant_reason,
        "candidate_sigma": candidate_sigma,
        "candidate_metrics": candidate_metrics,
        "sigma_candidate_acceptance_criteria": {
            "h1_valid_column_fraction_at_0_8_min": 0.5,
            "background_false_detection_rate_max": 0.02,
            "h10_h30_max_abs_center_shift_px": 0.10,
            "h10_h30_sigma_p95_ratio_max": 1.25,
        },
    }


def _save_steger_root_cause_plots(
    output_dir: Path,
    profile_plot_data: Mapping[str, Mapping[str, np.ndarray]],
    sigma_rows: Sequence[Mapping[str, Any]],
    verdict: Mapping[str, Any],
) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = {item[0]: item[3] for item in MULTIHEIGHT_ROIS}
    labels = {item[0]: item[1] for item in MULTIHEIGHT_ROIS}
    paths: list[Path] = []
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    for axis, (roi_id, label, _height, color) in zip(axes, MULTIHEIGHT_ROIS):
        data = profile_plot_data[roi_id]
        axis.fill_between(
            data["y"], data["p05"], data["p95"], color=color, alpha=0.18,
            label="frame/column P05-P95",
        )
        axis.plot(data["y"], data["median"], color=color, lw=1.3, label="median raw profile")
        axis.axvline(float(data["expected_y"][0]), color="#616161", ls="--", lw=0.8, label="expected center")
        axis.set_ylabel("raw DN")
        axis.set_title(f"{label} representative-column raw intensity profile")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xlabel("image y [px]")
    paths.append(_save_figure(output_dir / "raw_intensity_profile_h1_h10_h30.png", figure))
    plt.close(figure)

    sigmas = list(dict.fromkeys(float(row["sigma"]) for row in sigma_rows))
    by_key = {(float(row["sigma"]), str(row["roi_id"])): row for row in sigma_rows}
    figure, axes = plt.subplots(4, 1, figsize=(12, 13), constrained_layout=True)
    for roi_id, label, _height, color in MULTIHEIGHT_ROIS:
        axes[0].plot(
            sigmas,
            [by_key[(sigma, roi_id)]["valid_column_fraction_at_0_8"] for sigma in sigmas],
            "o-", color=color, label=label,
        )
        axes[1].plot(
            sigmas,
            [_finite_or_none(by_key[(sigma, roi_id)]["center_shift_vs_formal_px"]) for sigma in sigmas],
            "o-", color=color, label=label,
        )
        axes[2].plot(
            sigmas,
            [_finite_or_none(by_key[(sigma, roi_id)]["sigma_pixel_p95_px"]) for sigma in sigmas],
            "o-", color=color, label=label,
        )
    axes[3].plot(
        sigmas,
        [by_key[(sigma, "background")]["per_frame_detection_rate_mean"] for sigma in sigmas],
        "o-", color="#d32f2f", label="background false detection",
    )
    axes[0].set_ylabel("valid columns @0.8")
    axes[1].set_ylabel("center shift [px]")
    axes[1].axhline(0.0, color="#616161", lw=0.8)
    axes[2].set_ylabel("sigma_pixel P95 [px]")
    axes[3].set_ylabel("false detection rate")
    axes[3].set_xlabel("Steger sigma [px]")
    for axis in axes:
        axis.axvline(1.5, color="#616161", ls="--", lw=0.8, label="formal sigma=1.5")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", ncol=4)
    axes[0].set_title(f"B05_A15 diagnostic-only sigma sweep | {verdict['verdict']}")
    paths.append(_save_figure(output_dir / "steger_sigma_sweep.png", figure))
    plt.close(figure)
    return paths


def diagnose_steger_h1_root_cause(
    dataset: Path,
    registry: Path,
    analysis_config: Path,
    calibration_src: Path,
    background_roi: tuple[int, int, int, int],
    *,
    representative_column_count: int = 5,
    overwrite_diagnostic: bool = False,
) -> dict[str, Any]:
    """Diagnose B05_A15 H1 optical profile, formal rejection gate and sigma scale."""
    dataset = dataset.expanduser().resolve()
    if dataset.name != "B05_A15":
        raise ValueError("当前 root-cause diagnostic 第一版仅允许 B05_A15")
    if representative_column_count < 1 or representative_column_count % 2 == 0:
        raise ValueError("representative_column_count 必须为正奇数")
    output_dir = dataset / "analysis" / "steger_h1_root_cause_diagnostic"
    outputs = (
        output_dir / "raw_intensity_profile_summary.csv",
        output_dir / "raw_intensity_profile_h1_h10_h30.png",
        output_dir / "steger_rejection_reason_by_column.csv",
        output_dir / "steger_rejection_reason_summary.json",
        output_dir / "steger_sigma_sweep.csv",
        output_dir / "steger_sigma_sweep.png",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite_diagnostic:
        raise FileExistsError("root-cause 诊断输出已存在，不会覆盖：" + ", ".join(map(str, existing)))
    config = _load_reference_analysis_config(analysis_config)
    registry_path, _registry_document, config_entry = _load_roi_registry(registry, dataset.name)
    realtime = _load_realtime_steger(calibration_src)
    formal_options = realtime.load_steger_options(config["steger_config"])
    if any(float(formal_options[key]) != expected for key, expected in (
        ("sigma", 1.5), ("threshold", 30.0), ("deriv_thresh", 0.5)
    )):
        raise ValueError("正式 Steger 参数不再是 sigma=1.5/threshold=30/deriv_thresh=0.5")
    records = _task_frame_records(dataset, "multiheight")
    if len(records) != 50:
        raise ValueError(f"B05_A15 multiheight 应为50帧，实际 {len(records)}")
    images = [_read_gray_image(dataset / record["filename"]) for record in records]
    image_height, image_width = images[0].shape
    x_left, x_right, y_top, y_bottom = background_roi
    if not (0 <= x_left <= x_right < image_width and 0 <= y_top <= y_bottom < image_height):
        raise ValueError(f"background ROI 超出图像范围：{background_roi}")
    roi_ranges: dict[str, tuple[int, int]] = {}
    for roi_id, _label, _height, _color in MULTIHEIGHT_ROIS:
        selected = _roi_selected_x_range(config_entry["multiheight"][roi_id], roi_id)
        if selected is None:
            raise ValueError(f"{roi_id}.selected_x_range 尚未标注")
        roi_ranges[roi_id] = selected

    protected_paths = (
        registry_path,
        Path(config["source"]),
        Path(config["steger_config"]),
        dataset / "analysis" / "reference_analysis.json",
        dataset / "analysis" / "reference_by_column.csv",
        dataset / "analysis" / "multiheight_analysis.json",
        dataset.parent.parent / "results" / "geometry_master_summary.csv",
    )
    protected_before = {str(path): sha256_file(path) for path in protected_paths if path.is_file()}

    formal_y, formal_valid = _steger_diagnostic_stack(realtime, images, formal_options)
    reference_y = np.full(image_width, np.nan, dtype=np.float64)
    with (dataset / "analysis" / "reference_by_column.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            column = int(row["u"])
            try:
                reference_y[column] = float(row["y_ref_smooth_px"])
            except (TypeError, ValueError):
                pass

    profile_rows: list[dict[str, Any]] = []
    profile_plot_data: dict[str, dict[str, np.ndarray]] = {}
    for roi_id, _label, _height, _color in MULTIHEIGHT_ROIS:
        left, right = roi_ranges[roi_id]
        center_u = (left + right) // 2
        radius = representative_column_count // 2
        representative_columns = list(range(center_u - radius, center_u + radius + 1))
        roi_columns = np.arange(left, right + 1)
        detected_values = formal_y[:, roi_columns][formal_valid[:, roi_columns]]
        if detected_values.size:
            expected_y = float(np.median(detected_values))
        else:
            expected_y = float(np.nanmedian(reference_y[representative_columns]))
        window_top = max(0, int(round(expected_y)) - 25)
        window_bottom = min(image_height - 1, int(round(expected_y)) + 25)
        profiles: list[np.ndarray] = []
        metrics: dict[str, list[float]] = {
            "peak_intensity": [], "local_background": [],
            "peak_minus_background": [], "fwhm_px": [],
        }
        for image in images:
            for column in representative_columns:
                profile = image[window_top:window_bottom + 1, column].astype(np.float64)
                profiles.append(profile)
                values = _local_raw_profile_metrics(profile)
                for field in metrics:
                    metrics[field].append(values[field])
        stacked_profiles = np.stack(profiles)
        row: dict[str, Any] = {
            "roi_id": roi_id,
            "representative_columns": representative_columns,
            "profile_sample_count": len(profiles),
            "y_top": window_top,
            "y_bottom": window_bottom,
            "expected_center_y_px": expected_y,
        }
        for field, field_values in metrics.items():
            p50, p05, p95 = _profile_summary(field_values)
            row[field] = p50
            row[f"{field}_p05"] = p05
            row[f"{field}_p95"] = p95
        profile_rows.append(row)
        profile_plot_data[roi_id] = {
            "y": np.arange(window_top, window_bottom + 1, dtype=np.float64),
            "median": np.median(stacked_profiles, axis=0),
            "p05": np.percentile(stacked_profiles, 5, axis=0),
            "p95": np.percentile(stacked_profiles, 95, axis=0),
            "expected_y": np.asarray([expected_y]),
        }

    reason_names = (
        "accepted", "no_intensity_peak", "intensity_peak_outside_detected_band",
        "ridge_response_below_threshold",
        "derivative_condition_failed", "invalid_subpixel_offset",
        "no_valid_ridge_candidate", "other",
    )
    accumulators: dict[tuple[str, int], dict[str, Any]] = {}
    for roi_id, bounds in roi_ranges.items():
        for column in range(bounds[0], bounds[1] + 1):
            accumulators[(roi_id, column)] = {
                "reasons": Counter(), "full_max_intensity": [], "max_intensity": [],
                "outside_band": 0, "max_response": [],
                "min_offset": [], "intensity_pass": 0, "derivative_pass": 0,
                "response_pass": 0, "offset_pass": 0,
            }
    band_tops: list[float] = []
    band_bottoms: list[float] = []
    for image in images:
        extracted = realtime.extract_steger_columns(image, formal_options, diagnostic=True)
        diagnostics = extracted.diagnostics
        if diagnostics is None:
            raise RuntimeError("正式 realtime_steger 未返回 diagnostic-only 中间量")
        band_tops.append(float(diagnostics.band_top_px) if diagnostics.band_top_px is not None else float("nan"))
        band_bottoms.append(float(diagnostics.band_bottom_px) if diagnostics.band_bottom_px is not None else float("nan"))
        for (roi_id, column), accumulator in accumulators.items():
            reason = str(diagnostics.rejection_reason[column])
            accumulator["reasons"][reason] += 1
            accumulator["full_max_intensity"].append(
                float(diagnostics.full_image_max_intensity_dn[column])
            )
            accumulator["max_intensity"].append(float(diagnostics.max_intensity_dn[column]))
            accumulator["outside_band"] += int(
                diagnostics.intensity_peak_outside_detected_band[column]
            )
            accumulator["max_response"].append(float(diagnostics.max_ridge_response[column]))
            accumulator["min_offset"].append(float(diagnostics.min_subpixel_offset_px[column]))
            accumulator["intensity_pass"] += int(diagnostics.intensity_peak_present[column])
            accumulator["derivative_pass"] += int(diagnostics.derivative_condition_passed[column])
            accumulator["response_pass"] += int(diagnostics.ridge_response_passed[column])
            accumulator["offset_pass"] += int(diagnostics.subpixel_offset_passed[column])

    rejection_rows: list[dict[str, Any]] = []
    rejection_rois: dict[str, Any] = {}
    for roi_id, bounds in roi_ranges.items():
        roi_reason_counts = Counter()
        for column in range(bounds[0], bounds[1] + 1):
            accumulator = accumulators[(roi_id, column)]
            roi_reason_counts.update(accumulator["reasons"])
            rejected = Counter({
                reason: count for reason, count in accumulator["reasons"].items()
                if reason != "accepted"
            })
            dominant = rejected.most_common(1)[0][0] if rejected else "accepted"
            rejection_rows.append({
                "roi_id": roi_id,
                "u": column,
                "frame_count": len(images),
                **{f"{reason}_count": int(accumulator["reasons"].get(reason, 0)) for reason in reason_names},
                "dominant_rejection_reason": dominant,
                "full_image_intensity_peak_p50_dn": _profile_summary(accumulator["full_max_intensity"])[0],
                "intensity_peak_outside_band_gate_count": accumulator["outside_band"],
                "intensity_peak_present_count": accumulator["intensity_pass"],
                "derivative_condition_passed_count": accumulator["derivative_pass"],
                "ridge_response_passed_count": accumulator["response_pass"],
                "subpixel_offset_passed_count": accumulator["offset_pass"],
                "max_intensity_p50_dn": _profile_summary(accumulator["max_intensity"])[0],
                "max_ridge_response_p50": _profile_summary(accumulator["max_response"])[0],
                "min_subpixel_offset_p50_px": _profile_summary(accumulator["min_offset"])[0],
            })
        total = sum(roi_reason_counts.values())
        rejection_rois[roi_id] = {
            "column_count": bounds[1] - bounds[0] + 1,
            "frame_column_opportunity_count": total,
            "rejection_reason_counts": {
                reason: int(roi_reason_counts.get(reason, 0)) for reason in reason_names
            },
            "rejection_reason_fractions": {
                reason: float(roi_reason_counts.get(reason, 0) / total) for reason in reason_names
            },
        }

    sigma_values = (0.8, 1.0, 1.2, 1.5, 2.0, 2.5)
    sigma_rows: list[dict[str, Any]] = []
    formal_centers: dict[str, float | None] = {}
    formal_sigma_p95: dict[str, float | None] = {}
    for sigma in sigma_values:
        options = dict(formal_options)
        options["sigma"] = sigma
        y_stack, valid_stack = _steger_diagnostic_stack(realtime, images, options)
        background_y, background_valid = _steger_diagnostic_stack(
            realtime, images, options, crop=background_roi
        )
        for roi_id, bounds in roi_ranges.items():
            statistics, _columns = _steger_sweep_roi_statistics(y_stack, valid_stack, bounds)
            row = {"sigma": sigma, "roi_id": roi_id, **statistics}
            sigma_rows.append(row)
            if sigma == 1.5:
                formal_centers[roi_id] = statistics["median_center_position_px"]
                formal_sigma_p95[roi_id] = statistics["sigma_pixel_p95_px"]
        background_statistics, _columns = _steger_sweep_roi_statistics(
            background_y, background_valid, (0, x_right - x_left)
        )
        sigma_rows.append({"sigma": sigma, "roi_id": "background", **background_statistics})
    for row in sigma_rows:
        roi_id = str(row["roi_id"])
        if roi_id == "background":
            row["center_shift_vs_formal_px"] = None
            row["sigma_p95_ratio_vs_formal"] = None
            continue
        center = _finite_or_none(row["median_center_position_px"])
        sigma_p95 = _finite_or_none(row["sigma_pixel_p95_px"])
        formal_center = formal_centers.get(roi_id)
        formal_sigma = formal_sigma_p95.get(roi_id)
        row["center_shift_vs_formal_px"] = (
            center - formal_center if center is not None and formal_center is not None else None
        )
        row["sigma_p95_ratio_vs_formal"] = (
            sigma_p95 / formal_sigma
            if sigma_p95 is not None and formal_sigma is not None and formal_sigma > 0 else None
        )

    rejection_summary = {
        "schema_version": 1,
        "config_id": dataset.name,
        "formal_parameters": formal_options,
        "rejection_reason_definitions": {
            "no_intensity_peak": "column has no pixel at or above formal intensity threshold in detected band",
            "intensity_peak_outside_detected_band": "column reaches formal intensity threshold somewhere in the image, but not inside the extractor's automatically selected band",
            "ridge_response_below_threshold": "negative ridge exists but response does not exceed deriv_thresh",
            "derivative_condition_failed": "intensity peak exists but no safe negative-curvature ridge direction",
            "invalid_subpixel_offset": "ridge response passes but subpixel offset exceeds formal +/-0.6 px gate",
            "no_valid_ridge_candidate": "all recorded component gates pass but no formal candidate remains",
            "other": "unclassified fallback",
        },
        "rois": rejection_rois,
        "formal_detected_band_px": {
            "top_p05": _profile_summary(band_tops)[1],
            "top_p50": _profile_summary(band_tops)[0],
            "top_p95": _profile_summary(band_tops)[2],
            "bottom_exclusive_p05": _profile_summary(band_bottoms)[1],
            "bottom_exclusive_p50": _profile_summary(band_bottoms)[0],
            "bottom_exclusive_p95": _profile_summary(band_bottoms)[2],
        },
    }
    verdict = _classify_steger_root_cause(profile_rows, rejection_summary, sigma_rows)
    rejection_summary["raw_profile_summary"] = profile_rows
    rejection_summary["sigma_sweep"] = sigma_rows
    rejection_summary["verdict"] = verdict
    rejection_summary["background_roi_xyxy"] = list(background_roi)
    rejection_summary["diagnostic_only"] = True

    output_dir.mkdir(parents=True, exist_ok=overwrite_diagnostic)
    profile_fields = (
        "roi_id", "representative_columns", "profile_sample_count", "y_top", "y_bottom",
        "expected_center_y_px", "peak_intensity", "peak_intensity_p05", "peak_intensity_p95",
        "local_background", "local_background_p05", "local_background_p95",
        "peak_minus_background", "peak_minus_background_p05", "peak_minus_background_p95",
        "fwhm_px", "fwhm_px_p05", "fwhm_px_p95",
    )
    _write_csv(output_dir / "raw_intensity_profile_summary.csv", profile_fields, profile_rows)
    rejection_fields = (
        "roi_id", "u", "frame_count", *(f"{reason}_count" for reason in reason_names),
        "dominant_rejection_reason", "full_image_intensity_peak_p50_dn",
        "intensity_peak_outside_band_gate_count", "intensity_peak_present_count",
        "derivative_condition_passed_count", "ridge_response_passed_count",
        "subpixel_offset_passed_count", "max_intensity_p50_dn",
        "max_ridge_response_p50", "min_subpixel_offset_p50_px",
    )
    _write_csv(
        output_dir / "steger_rejection_reason_by_column.csv",
        rejection_fields,
        rejection_rows,
    )
    sigma_fields = (
        "sigma", "roi_id", "frame_count", "roi_width_px", "detected_event_count",
        "per_frame_detection_rate_mean", "valid_column_fraction_at_0_8",
        "sigma_pixel_p50_px", "sigma_pixel_p95_px", "median_center_position_px",
        "center_shift_vs_formal_px", "sigma_p95_ratio_vs_formal",
    )
    _write_csv(output_dir / "steger_sigma_sweep.csv", sigma_fields, sigma_rows)
    _save_steger_root_cause_plots(output_dir, profile_plot_data, sigma_rows, verdict)
    protected_after = {str(path): sha256_file(path) for path in protected_paths if path.is_file()}
    if protected_after != protected_before:
        raise RuntimeError("root-cause diagnostic 意外修改了正式配置或结果")
    rejection_summary.update({
        "protected_artifacts_sha256": protected_after,
        "formal_results_modified": False,
        "formal_config_modified": False,
        "ranking_generated": False,
        "outputs": [str(path) for path in outputs],
    })
    _atomic_write_text(
        output_dir / "steger_rejection_reason_summary.json",
        json.dumps(rejection_summary, ensure_ascii=False, indent=2) + "\n",
    )
    return rejection_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基线刻度—激光倾角 Phase-A 快速筛选实验工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="初始化实验目录和 geometry_master.csv")
    make_plan_parser = subparsers.add_parser(
        "make-plan", help="根据 geometry_master.csv 生成12份采集计划"
    )
    make_plan_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式覆盖 configs/generated 中已有的采集计划",
    )
    audit_parser = subparsers.add_parser(
        "audit-captures", help="审计 Phase-A 数据集、帧索引和实际相机参数"
    )
    audit_parser.add_argument("--root", type=Path, required=True, help="11组已采集 dataset 的根目录")
    audit_parser.add_argument("--master", type=Path, required=True, help="geometry_master.csv 路径")
    preview_parser = subparsers.add_parser(
        "preview-reference-roi",
        help="输出 B12p5_A20 的50帧中位数图和原始 Steger 中心线，供人工选择 x_range",
    )
    preview_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "data" / REFERENCE_DEVELOPMENT_CONFIG_ID,
    )
    preview_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "analysis.yaml",
    )
    preview_parser.add_argument(
        "--calibration-src",
        type=Path,
        default=PROJECT_ROOT.parent / "calibration" / "src",
    )
    preview_parser.add_argument("--overwrite", action="store_true")
    annotate_parser = subparsers.add_parser(
        "annotate-multiheight-rois",
        help="仅标注 B12p5_A20 multiheight 的 H1/H10/H30 横向 ROI",
    )
    annotate_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "data" / REFERENCE_DEVELOPMENT_CONFIG_ID,
    )
    annotate_parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "roi_registry.yaml",
    )
    annotate_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "analysis.yaml",
    )
    annotate_parser.add_argument(
        "--calibration-src",
        type=Path,
        default=PROJECT_ROOT.parent / "calibration" / "src",
    )
    annotate_parser.add_argument(
        "--preview-only",
        action="store_true",
        help="只生成中位图、raw Steger 预览和当前 registry summary，不打开交互窗口",
    )
    annotate_all_parser = subparsers.add_parser(
        "annotate-all-rois",
        help="逐组交互确认 reference 与 H1/H10/H30 ROI；模板组保持冻结",
    )
    annotate_all_parser.add_argument("--root", type=Path, required=True)
    annotate_all_parser.add_argument("--registry", type=Path, required=True)
    annotate_all_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "analysis.yaml",
    )
    annotate_all_parser.add_argument(
        "--calibration-src",
        type=Path,
        default=PROJECT_ROOT.parent / "calibration" / "src",
    )
    audit_roi_parser = subparsers.add_parser(
        "audit-rois", help="只审计当前 ROI registry，不打开交互窗口"
    )
    audit_roi_parser.add_argument("--root", type=Path, required=True)
    audit_roi_parser.add_argument("--registry", type=Path, required=True)
    audit_roi_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "analysis.yaml",
    )
    multiheight_parser = subparsers.add_parser(
        "analyze-multiheight",
        help="仅分析一个 confirmed multiheight，并与冻结 reference 做同列差分",
    )
    multiheight_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "data" / REFERENCE_DEVELOPMENT_CONFIG_ID,
    )
    multiheight_parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "roi_registry.yaml",
    )
    multiheight_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "analysis.yaml",
    )
    multiheight_parser.add_argument(
        "--calibration-src",
        type=Path,
        default=PROJECT_ROOT.parent / "calibration" / "src",
    )
    multiheight_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式覆盖已有 multiheight 派生结果；不覆盖原始图像或 reference 分析",
    )
    multiheight_parser.add_argument(
        "--background-roi",
        type=int,
        nargs=4,
        metavar=("X_LEFT", "X_RIGHT", "Y_TOP", "Y_BOTTOM"),
        default=None,
        help="可选背景二维 ROI，用于 band fix false-detection 验证（包含端点）",
    )
    reference_parser = subparsers.add_parser(
        "analyze-reference",
        help="仅分析 B12p5_A20/reference，不读取 multiheight",
    )
    reference_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "data" / REFERENCE_DEVELOPMENT_CONFIG_ID,
    )
    reference_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "analysis.yaml",
    )
    reference_parser.add_argument(
        "--calibration-src",
        type=Path,
        default=PROJECT_ROOT.parent / "calibration" / "src",
    )
    reference_parser.add_argument("--overwrite", action="store_true")
    analyze_all_parser = subparsers.add_parser(
        "analyze-all", help="仅分析 ROI registry 中 confirmed 的 Phase-A 配置"
    )
    analyze_all_parser.add_argument("--root", type=Path, required=True)
    analyze_all_parser.add_argument("--registry", type=Path, required=True)
    analyze_all_parser.add_argument("--config", type=Path, required=True)
    analyze_all_parser.add_argument(
        "--calibration-src",
        type=Path,
        default=PROJECT_ROOT.parent / "calibration" / "src",
    )
    analyze_all_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="重新生成 confirmed 配置的 multiheight 派生分析；已有 reference 保持冻结",
    )
    auto_band_parser = subparsers.add_parser(
        "diagnose-auto-band",
        help="只读追踪 reference/multiheight 的 realtime_steger auto band 生成过程",
    )
    auto_band_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "data" / "B05_A15",
    )
    auto_band_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "analysis.yaml",
    )
    auto_band_parser.add_argument(
        "--calibration-src",
        type=Path,
        default=PROJECT_ROOT.parent / "calibration" / "src",
    )
    auto_band_parser.add_argument(
        "--overwrite-diagnostic",
        action="store_true",
        help="仅覆盖本命令自己的 diagnostic 日志",
    )
    diagnostic_parser = subparsers.add_parser(
        "diagnose-steger-h1",
        help="对 B05_A15 做只读离线 Steger H1 threshold/valid-fraction 诊断",
    )
    diagnostic_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "data" / "B05_A15",
    )
    diagnostic_parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "roi_registry.yaml",
    )
    diagnostic_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "analysis.yaml",
    )
    diagnostic_parser.add_argument(
        "--calibration-src",
        type=Path,
        default=PROJECT_ROOT.parent / "calibration" / "src",
    )
    diagnostic_parser.add_argument(
        "--background-roi",
        type=int,
        nargs=4,
        metavar=("X_LEFT", "X_RIGHT", "Y_TOP", "Y_BOTTOM"),
        required=True,
        help="量块邻近无激光区域的二维 ROI，包含端点",
    )
    root_cause_parser = subparsers.add_parser(
        "diagnose-steger-h1-root-cause",
        help="对 B05_A15 做只读 H1 原始剖面、拒绝门控与 sigma 尺度诊断",
    )
    root_cause_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "data" / "B05_A15",
    )
    root_cause_parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "roi_registry.yaml",
    )
    root_cause_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "configs" / "analysis.yaml",
    )
    root_cause_parser.add_argument(
        "--calibration-src",
        type=Path,
        default=PROJECT_ROOT.parent / "calibration" / "src",
    )
    root_cause_parser.add_argument(
        "--background-roi",
        type=int,
        nargs=4,
        metavar=("X_LEFT", "X_RIGHT", "Y_TOP", "Y_BOTTOM"),
        required=True,
        help="量块邻近无激光区域的二维 ROI，包含端点",
    )
    root_cause_parser.add_argument(
        "--representative-columns",
        type=int,
        default=5,
        help="每个量块 ROI 中央用于原始灰度剖面的列数（正奇数，默认5）",
    )
    root_cause_parser.add_argument(
        "--overwrite-diagnostic",
        action="store_true",
        help="仅覆盖此命令自己的 diagnostic 产物，不触碰正式配置和分析结果",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        try:
            master_path = initialize_experiment()
        except FileExistsError as exc:
            print(f"错误：{exc}")
            return 2
        print(f"已创建：{master_path}")
        return 0
    if args.command == "make-plan":
        try:
            paths = make_capture_plans(overwrite=args.overwrite)
        except (FileExistsError, FileNotFoundError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        print(f"已生成 {len(paths)} 份采集计划：{paths[0].parent}")
        return 0
    if args.command == "audit-captures":
        try:
            result = audit_captures(args.root, args.master)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        summary = result["summary"]
        print(f"expected conditions = {summary['expected_conditions']}")
        print(f"captured conditions = {summary['captured_conditions']}")
        print(f"invalid_fov = {summary['invalid_fov']}")
        print(f"complete datasets = {summary['complete_datasets']}")
        print(f"incomplete datasets = {summary['incomplete_datasets']}")
        return 0 if summary["incomplete_datasets"] == 0 else 1
    if args.command == "preview-reference-roi":
        try:
            summary = preview_reference_roi(
                args.dataset,
                args.config,
                args.calibration_src,
                overwrite=args.overwrite,
            )
        except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        print(f"config_id = {summary['config_id']}")
        print(f"reference frames = {summary['frame_count']}")
        print(f"reference_surface.x_range = {summary['reference_surface_x_range']}")
        print(f"preview = {summary['output']}")
        print("multiheight analyzed = false")
        return 0
    if args.command == "annotate-multiheight-rois":
        try:
            summary = annotate_multiheight_rois(
                args.dataset,
                args.registry,
                args.config,
                args.calibration_src,
                preview_only=args.preview_only,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        print(f"config_id = {summary['config_id']}")
        print(f"multiheight frames = {summary['frame_count']}")
        for roi_id, label, _height, _color in MULTIHEIGHT_ROIS:
            roi = summary["rois"][roi_id]
            print(
                f"{label} x_range = {roi['x_range']}, width_px = {roi['width_px']}, "
                f"valid_column_fraction = {roi['steger_valid_column_fraction']}, "
                f"inside_reference_surface = {roi['fully_inside_reference_surface']}"
            )
        print("delta_y computed = false")
        print("multiheight height analysis = false")
        return 0
    if args.command == "annotate-all-rois":
        try:
            result = annotate_all_rois(
                args.root, args.registry, args.config, args.calibration_src
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        print(f"roi audit = {result['output']}")
        print(f"roi counts = {result['counts']}")
        return 0
    if args.command == "audit-rois":
        try:
            result = audit_rois(args.root, args.registry, args.config)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        print(f"roi audit = {result['output']}")
        print(f"roi counts = {result['counts']}")
        return 0
    if args.command == "analyze-multiheight":
        try:
            summary = analyze_multiheight(
                args.dataset,
                args.registry,
                args.config,
                args.calibration_src,
                overwrite=args.overwrite,
                background_roi=(
                    tuple(args.background_roi) if args.background_roi is not None else None
                ),
            )
        except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        print(f"config_id = {summary['config_id']}")
        print(f"multiheight frames = {summary['frame_count']}")
        for roi_id, label, _height, _color in MULTIHEIGHT_ROIS:
            roi = summary["rois"][roi_id]
            print(
                f"{label}: selected_x_range = {roi['selected_x_range']}, "
                f"auto_stable_x_range = {roi['auto_stable_x_range']}, "
                f"analysis_x_range = {roi['analysis_x_range']}, "
                f"delta_y_median_px = {roi['delta_y_median_px']}, "
                "sensitivity_median_px_per_mm = "
                f"{roi['sensitivity_median_px_per_mm']}, "
                f"sigma_pixel_p50/p95_px = {roi['sigma_pixel_p50_px']}/{roi['sigma_pixel_p95_px']}, "
                f"sigma_z_pred_p50/p95_mm = {roi['sigma_z_pred_p50_mm']}/{roi['sigma_z_pred_p95_mm']}, "
                f"roi_stable = {str(roi['roi_stable']).lower()}, "
                f"warnings = {roi['warnings']}, status = {roi['status']}"
            )
        print(
            "sensitivity_combined_px_per_mm = "
            f"{summary['sensitivity_combined_px_per_mm']}"
        )
        print(f"sigma_z_pred_combined_mm = {summary['sigma_z_pred_combined_mm']}")
        band = summary["band_detection"]
        print(
            "band original/reference/final = "
            f"[{band['original_band_top']},{band['original_band_bottom']}) / "
            f"[{band['reference_envelope_top']},{band['reference_envelope_bottom']}) / "
            f"[{band['final_band_top']},{band['final_band_bottom']})"
        )
        print(
            "steger time before/after/change = "
            f"{band['steger_time_ms_before']:.3f} ms / "
            f"{band['steger_time_ms_after']:.3f} ms / "
            f"{band['runtime_change_percent']:.2f}%"
        )
        print(
            "reference_union_band_validated = "
            f"{str(summary['reference_union_band_validated']).lower()}"
        )
        print(f"needs_manual_review = {str(summary['needs_manual_review']).lower()}")
        return 1 if summary["needs_manual_review"] else 0
    if args.command == "analyze-reference":
        try:
            summary = analyze_reference(
                args.dataset,
                args.config,
                args.calibration_src,
                overwrite=args.overwrite,
            )
        except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        counts = summary["source_counts"]
        print(f"config_id = {summary['config_id']}")
        print(f"reference frames = {summary['frame_count']}")
        print(f"observed columns = {counts['observed']}")
        print(f"short-gap interpolated columns = {counts['short_gap_interpolated']}")
        print(f"smooth-model filled columns = {counts['smooth_model_filled']}")
        print(f"segment-edge excluded columns = {counts['segment_edge_excluded']}")
        print(f"outside reference surface columns = {counts['outside_reference_surface']}")
        print(f"invalid columns = {counts['invalid']}")
        print(f"CV segments = {summary['cross_validation']['eligible_segment_count']}")
        print(f"CV interior segments = {summary['cross_validation']['interior_segment_count']}")
        print(f"CV boundary segments = {summary['cross_validation']['boundary_segment_count']}")
        print(
            "reference_cv_interior_rmse_px = "
            f"{summary['statistics']['reference_cv_interior_rmse_px']:.6f}"
        )
        print(
            "reference_cv_interior_p95_px = "
            f"{summary['statistics']['reference_cv_interior_p95_px']:.6f}"
        )
        print(
            "reference_cv_boundary_rmse_px = "
            f"{summary['statistics']['reference_cv_boundary_rmse_px']:.6f}"
        )
        print(f"multiheight analyzed = {str(summary['multiheight_analyzed']).lower()}")
        return 0
    if args.command == "diagnose-auto-band":
        try:
            summary = diagnose_auto_band_generation(
                args.dataset,
                args.config,
                args.calibration_src,
                overwrite_diagnostic=args.overwrite_diagnostic,
            )
        except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        print(f"diagnostic output = {Path(summary['outputs'][0]).parent}")
        print(f"effective roi_margin = {summary['effective_steger_options']['roi_margin']}")
        for task_id in ("reference", "multiheight"):
            trace = summary["task_summaries"][task_id]["dominant_trace"]
            print(
                f"{task_id}: raw=[{trace['raw_candidate_top']},{trace['raw_candidate_bottom']}), "
                f"margin_before_clip={trace['margin_before_clip']}, "
                f"margin_after_clip={trace['margin_after_clip']}, "
                f"roi_max_height_applied={str(trace['roi_max_height_applied']).lower()}, "
                f"final=[{trace['final_band_top']},{trace['final_band_bottom']})"
            )
        print("formal results modified = false")
        return 0
    if args.command == "diagnose-steger-h1":
        try:
            summary = diagnose_steger_h1_failure(
                args.dataset,
                args.registry,
                args.config,
                args.calibration_src,
                tuple(args.background_roi),
            )
        except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        verdict = summary["verdict"]
        print(f"diagnostic output = {Path(summary['outputs'][0]).parent}")
        print(f"verdict = {verdict['verdict']}")
        print(f"candidate = {verdict['candidate_parameter_id']}")
        print("ranking generated = false")
        print("formal results modified = false")
        return 0
    if args.command == "diagnose-steger-h1-root-cause":
        try:
            summary = diagnose_steger_h1_root_cause(
                args.dataset,
                args.registry,
                args.config,
                args.calibration_src,
                tuple(args.background_roi),
                representative_column_count=args.representative_columns,
                overwrite_diagnostic=args.overwrite_diagnostic,
            )
        except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        verdict = summary["verdict"]
        print(f"diagnostic output = {Path(summary['outputs'][0]).parent}")
        print(f"verdict = {verdict['verdict']}")
        print(f"dominant H1 rejection = {verdict['h1_dominant_rejection_reason']}")
        print(f"candidate sigma = {verdict['candidate_sigma']}")
        print("formal config modified = false")
        print("formal results modified = false")
        return 0
    if args.command == "analyze-all":
        try:
            result = analyze_all(
                args.root,
                args.registry,
                args.config,
                args.calibration_src,
                overwrite=args.overwrite,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            print(f"错误：{exc}")
            return 2
        print(f"geometry summary = {result['output']}")
        print(f"analysis counts = {result['counts']}")
        print("ranking generated = false")
        print("heatmap generated = false")
        return 1 if result["counts"].get("failed", 0) else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
