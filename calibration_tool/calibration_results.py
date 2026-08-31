"""CalibrationRun 的只读结构化结果适配层。

本模块只读取 ``CalibrationRun`` 已有的 stage 状态、路径和 metrics，不读取或
复制 stage 的详细 YAML，也不修改 CalibrationRun 本身。
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .calibration_run import CalibrationRun, CalibrationStage
from .io_utils import load_document
from .laser_models import SUPPORTED_LASER_MODEL_TYPES


NOT_EXECUTED = "not_executed"
DETAILS_LOADED = "loaded"
DETAILS_UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class StageResultSummary:
    """Stage 的最小公共摘要；未找到 stage 时 status 为 ``not_executed``。"""

    stage: str
    status: str = NOT_EXECUTED
    output_dir: Path | None = None
    result_file: Path | None = None


@dataclass(frozen=True, slots=True)
class IntrinsicsSummary(StageResultSummary):
    fit_rmse_px: float | int | None = None
    test_rmse_px: float | int | None = None
    fit_image_count: int | None = None
    test_image_count: int | None = None


@dataclass(frozen=True, slots=True)
class IntrinsicsReprojectionRow:
    """已有逐图重投影误差的只读行，不重新计算误差。"""

    split: str
    image: str
    status: str
    rmse_px: float | None = None
    mean_error_px: float | None = None
    error_stage: str | None = None


@dataclass(frozen=True, slots=True)
class IntrinsicsDetails:
    """从 intrinsics result_file 读取的轻量详细结果。"""

    status: str
    result_file: Path | None = None
    error: str | None = None
    notes: tuple[str, ...] = ()
    camera_matrix: tuple[tuple[float, ...], ...] = ()
    dist_coeffs: tuple[float, ...] = ()
    fit_rmse_px: float | None = None
    test_rmse_px: float | None = None
    fit_image_count: int | None = None
    test_image_count: int | None = None
    fit_reprojection: tuple[IntrinsicsReprojectionRow, ...] = ()
    test_reprojection: tuple[IntrinsicsReprojectionRow, ...] = ()

    @property
    def reprojection_rows(self) -> tuple[IntrinsicsReprojectionRow, ...]:
        return self.fit_reprojection + self.test_reprojection


@dataclass(frozen=True, slots=True)
class LaserSurfaceSummary(StageResultSummary):
    model_type: str | None = None
    validation_rmse_mm: float | int | None = None
    validation_p95_mm: float | int | None = None
    validation_valid_rate: float | int | None = None


@dataclass(frozen=True, slots=True)
class LaserModelParameters:
    """三种激光面模型的少量可读参数。

    这里仅保留结果页需要的参数，不把模型 YAML 原样复制到结果对象。
    """

    model: str
    source_file: Path | None = None
    plane_abcd: tuple[float, ...] = ()
    apex_camera_mm: tuple[float, ...] = ()
    axis_unit_camera: tuple[float, ...] = ()
    half_apex_angle_deg: float | None = None
    coefficients: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class LaserModelComparison:
    """一个模型在训练集和独立验证集上的已有指标。"""

    model: str
    train_rmse_mm: float | None = None
    validation_rmse_mm: float | None = None
    train_p95_mm: float | None = None
    validation_p95_mm: float | None = None
    train_max_mm: float | None = None
    validation_max_mm: float | None = None
    train_valid_rate: float | None = None
    validation_valid_rate: float | None = None
    available: bool = False
    parameters: LaserModelParameters | None = None


@dataclass(frozen=True, slots=True)
class LaserSurfaceDetails:
    """激光表面 stage 的只读详细结果。"""

    status: str
    output_dir: Path | None = None
    result_file: Path | None = None
    error: str | None = None
    notes: tuple[str, ...] = ()
    selected_model: str | None = None
    validation_rmse_mm: float | None = None
    validation_p95_mm: float | None = None
    validation_max_mm: float | None = None
    validation_valid_rate: float | None = None
    model_comparisons: tuple[LaserModelComparison, ...] = ()
    error_vs_u: Path | None = None
    error_vs_v: Path | None = None
    error_vs_depth: Path | None = None

    @property
    def models(self) -> tuple[LaserModelComparison, ...]:
        """兼容结果页使用的简短别名。"""

        return self.model_comparisons

    @property
    def selected_parameters(self) -> LaserModelParameters | None:
        for comparison in self.model_comparisons:
            if comparison.model == self.selected_model:
                return comparison.parameters
        return None


@dataclass(frozen=True, slots=True)
class GroundExtrinsicsSummary(StageResultSummary):
    validation_rmse_mm: float | int | None = None
    validation_p95_mm: float | int | None = None


@dataclass(frozen=True, slots=True)
class GroundValidationRow:
    """地面外参已有逐帧 validation 结果的轻量只读表示。"""

    frame: str
    status: str = ""
    detection_method: str | None = None
    pnp_rmse_px: float | None = None
    rmse_mm: float | None = None
    p95_mm: float | None = None
    max_mm: float | None = None
    mean_mm: float | None = None
    std_mm: float | None = None
    point_count: int | None = None

    @property
    def image(self) -> str:
        """兼容现有产物使用的 image 字段命名。"""

        return self.frame


@dataclass(frozen=True, slots=True)
class GroundExtrinsicsDetails:
    """从地面外参 stage 产物读取的结构化结果，不重新求解外参。"""

    status: str
    stage: str = "ground_extrinsics"
    output_dir: Path | None = None
    result_file: Path | None = None
    error: str | None = None
    notes: tuple[str, ...] = ()
    transform_name: str | None = None
    rotation_matrix: tuple[tuple[float, ...], ...] = ()
    translation_mm: tuple[float, ...] = ()
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    fit_rmse_mm: float | None = None
    validation_rmse_mm: float | None = None
    validation_p95_mm: float | None = None
    validation_max_mm: float | None = None
    fit_frame_count: int | None = None
    validation_frame_count: int | None = None
    validation_rows: tuple[GroundValidationRow, ...] = ()
    validation_error_plot: Path | None = None

    @property
    def rotation(self) -> tuple[tuple[float, ...], ...]:
        return self.rotation_matrix

    @property
    def r(self) -> tuple[tuple[float, ...], ...]:
        return self.rotation_matrix

    @property
    def translation(self) -> tuple[float, ...]:
        return self.translation_mm

    @property
    def t(self) -> tuple[float, ...]:
        return self.translation_mm

    @property
    def validation_frames(self) -> tuple[GroundValidationRow, ...]:
        return self.validation_rows

    @property
    def validation_plot(self) -> Path | None:
        return self.validation_error_plot


@dataclass(frozen=True, slots=True)
class GroundBiasSummary(StageResultSummary):
    loaded_frame_count: int | None = None
    independent_validation_frame_count: int | None = None


@dataclass(frozen=True, slots=True)
class CalibrationResultsSummary:
    """供总览和后续详细页面使用的轻量结果摘要。"""

    run_id: str
    project_id: str
    status: str
    overall: str
    laser_orientation: str | None
    started_utc: datetime | None
    completed_utc: datetime | None
    intrinsics: IntrinsicsSummary
    laser_surface: LaserSurfaceSummary
    ground_extrinsics: GroundExtrinsicsSummary
    ground_bias: GroundBiasSummary


def _find_stage(run: CalibrationRun, names: tuple[str, ...]) -> CalibrationStage | None:
    for stage in run.stages:
        if stage.stage in names:
            return stage
    return None


def _stage_kwargs(stage: CalibrationStage | None, default_name: str) -> dict[str, Any]:
    if stage is None:
        return {"stage": default_name}
    return {
        "stage": stage.stage,
        "status": stage.status,
        "output_dir": stage.output_dir,
        "result_file": stage.result_file,
    }


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_matrix(value: Any) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return ()
    rows: list[tuple[float, ...]] = []
    for raw_row in value:
        if isinstance(raw_row, (str, bytes)) or not isinstance(raw_row, Sequence):
            return ()
        row = tuple(_finite_float(item) for item in raw_row)
        if any(item is None for item in row):
            return ()
        rows.append(tuple(item for item in row if item is not None))
    return tuple(rows)


def _as_vector(value: Any) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return ()
    if (
        len(value) == 1
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        value = value[0]
    values = tuple(_finite_float(item) for item in value)
    if any(item is None for item in values):
        return ()
    return tuple(item for item in values if item is not None)


def _as_count(value: Any) -> int | None:
    numeric = _finite_float(value)
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


def _first_mapping_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _load_reprojection_rows(
    path: Path,
    split: str,
    *,
    limit: int,
) -> tuple[IntrinsicsReprojectionRow, ...]:
    if limit <= 0 or not path.is_file():
        return ()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows: list[IntrinsicsReprojectionRow] = []
            for raw_row in reader:
                if len(rows) >= limit:
                    break
                image = str(_first_mapping_value(raw_row, "image", "filename") or "")
                rows.append(
                    IntrinsicsReprojectionRow(
                        split=split,
                        image=image,
                        status=str(raw_row.get("status") or ""),
                        rmse_px=_finite_float(
                            _first_mapping_value(
                                raw_row,
                                "per_image_rmse",
                                "rmse_px",
                                "reprojection_rmse",
                            )
                        ),
                        mean_error_px=_finite_float(
                            _first_mapping_value(
                                raw_row,
                                "mean_euclidean_reprojection_error",
                                "mean_error_px",
                            )
                        ),
                        error_stage=(
                            str(raw_row["error_stage"])
                            if raw_row.get("error_stage") not in (None, "")
                            else None
                        ),
                    )
                )
    except (OSError, UnicodeError, csv.Error):
        return ()
    return tuple(rows)


def load_intrinsics_details(
    intrinsics: IntrinsicsSummary,
    *,
    max_rows: int = 200,
) -> IntrinsicsDetails:
    """读取内参 stage 的详细结果和已有逐图误差，失败时返回可展示状态。"""

    if not isinstance(intrinsics, IntrinsicsSummary):
        raise TypeError("intrinsics 必须是 IntrinsicsSummary")
    if intrinsics.status == NOT_EXECUTED:
        return IntrinsicsDetails(
            status=NOT_EXECUTED,
            result_file=intrinsics.result_file,
        )

    source = intrinsics.result_file
    if source is None:
        return IntrinsicsDetails(
            status=DETAILS_UNAVAILABLE,
            error="内参 stage 未记录 result_file",
        )
    source = Path(source).expanduser()
    if not source.is_file():
        return IntrinsicsDetails(
            status=DETAILS_UNAVAILABLE,
            result_file=source,
            error=f"内参结果文件不存在：{source}",
        )
    try:
        document = load_document(source)
    except Exception as exc:
        return IntrinsicsDetails(
            status=DETAILS_UNAVAILABLE,
            result_file=source,
            error=f"内参结果读取失败：{exc}",
        )

    notes: list[str] = []
    camera_matrix = _as_matrix(document.get("camera_matrix"))
    if not camera_matrix:
        notes.append("结果文件未包含有效 camera_matrix")
    dist_coeffs = _as_vector(document.get("dist_coeffs"))
    if not dist_coeffs:
        notes.append("结果文件未包含有效 dist_coeffs")
    fit_metrics = document.get("fit_metrics")
    fit_metrics = fit_metrics if isinstance(fit_metrics, Mapping) else {}
    test_metrics = document.get("test_metrics")
    test_metrics = test_metrics if isinstance(test_metrics, Mapping) else {}
    fit_reprojection = _load_reprojection_rows(
        source.parent / "fit_images.csv",
        "fit",
        limit=max_rows,
    )
    test_reprojection = _load_reprojection_rows(
        source.parent / "test_images.csv",
        "test",
        limit=max_rows,
    )
    if not fit_reprojection and not test_reprojection:
        notes.append("未找到已有逐图重投影误差")
    return IntrinsicsDetails(
        status=DETAILS_LOADED,
        result_file=source,
        notes=tuple(notes),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        fit_rmse_px=_finite_float(fit_metrics.get("overall_reprojection_rmse")),
        test_rmse_px=_finite_float(test_metrics.get("overall_reprojection_rmse")),
        fit_image_count=_as_count(fit_metrics.get("image_count")),
        test_image_count=_as_count(test_metrics.get("image_count")),
        fit_reprojection=fit_reprojection,
        test_reprojection=test_reprojection,
    )


def _canonical_laser_model(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    aliases = {
        "global_plane": "global_plane",
        "plane": "global_plane",
        "plane_abcd": "global_plane",
        "laser_plane": "global_plane",
        "quadratic_graph": "quadratic_graph",
        "quadratic": "quadratic_graph",
        "quadratic_surface": "quadratic_graph",
        "circular_cone": "circular_cone",
        "cone": "circular_cone",
    }
    return aliases.get(value.strip().lower().replace("-", "_").replace(" ", "_"))


def _laser_metric(row: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = _finite_float(row.get(name))
        if value is not None:
            return value
    return None


def _laser_plane_abcd(document: Mapping[str, Any]) -> tuple[float, ...]:
    value = document.get("plane_abcd")
    if isinstance(value, Mapping):
        value = [value.get(name) for name in ("a", "b", "c", "d")]
    plane = _as_vector(value)
    if len(plane) == 4:
        return plane
    normal = _as_vector(document.get("normal"))
    d_value = _finite_float(document.get("d_mm"))
    if len(normal) == 3 and d_value is not None:
        return normal + (d_value,)
    return ()


def _laser_model_parameters(
    model: str,
    document: Mapping[str, Any],
    source_file: Path | None,
) -> LaserModelParameters:
    return LaserModelParameters(
        model=model,
        source_file=source_file,
        plane_abcd=_laser_plane_abcd(document) if model == "global_plane" else (),
        apex_camera_mm=(
            (
                _as_vector(document.get("apex_camera_mm"))
                or _as_vector(document.get("apex"))
            )
            if model == "circular_cone"
            else ()
        ),
        axis_unit_camera=(
            (
                _as_vector(document.get("axis_unit_camera"))
                or _as_vector(document.get("axis"))
            )
            if model == "circular_cone"
            else ()
        ),
        half_apex_angle_deg=(
            _laser_metric(
                document,
                "half_apex_angle_deg",
                "half_angle_deg",
                "half_angle",
            )
            if model == "circular_cone"
            else None
        ),
        coefficients=(
            (
                _as_vector(document.get("coefficients"))
                or _as_vector(document.get("main_coefficients"))
            )
            if model == "quadratic_graph"
            else ()
        ),
    )


def _parameter_file_candidates(base: Path, model: str) -> tuple[Path, ...]:
    names = [f"{model}.yaml"]
    if model == "global_plane":
        names.extend(("plane.yaml", "laser_plane.yaml"))
    candidates: list[Path] = []
    for directory in (base / "models", base):
        for name in names:
            candidate = directory / name
            if candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def _load_laser_parameters(
    base: Path | None,
    selected_model: str | None,
    result_file: Path | None,
    result_document: Mapping[str, Any],
    notes: list[str],
) -> dict[str, LaserModelParameters]:
    parameters: dict[str, LaserModelParameters] = {}
    for model in SUPPORTED_LASER_MODEL_TYPES:
        source: Path | None = None
        document: Mapping[str, Any] | None = None
        if base is not None:
            for candidate in _parameter_file_candidates(base, model):
                if not candidate.is_file():
                    continue
                source = candidate
                try:
                    document = load_document(candidate)
                except Exception as exc:
                    notes.append(f"{model} 参数读取失败：{exc}")
                    continue
                break
        if document is None and model == selected_model and result_document:
            document = result_document
            source = result_file
        if document is not None:
            parameters[model] = _laser_model_parameters(model, document, source)
    return parameters


def _load_laser_comparison_rows(
    path: Path,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = {str(value).strip().lower() for value in (reader.fieldnames or [])}
        if "model" not in fieldnames and "model_type" not in fieldnames:
            raise ValueError("model_comparison.csv 缺少 model 列")
        rows: dict[str, dict[str, Mapping[str, Any]]] = {}
        for raw_row in reader:
            row = {
                str(key).strip().lower(): value
                for key, value in raw_row.items()
                if key is not None
            }
            model = _canonical_laser_model(
                _first_mapping_value(row, "model", "model_type")
            )
            split_value = str(_first_mapping_value(row, "split", "dataset") or "")
            split = split_value.strip().lower()
            split = {"val": "validation", "valid": "validation"}.get(split, split)
            if model is None or split not in ("train", "validation"):
                continue
            rows.setdefault(model, {}).setdefault(split, row)
    if not rows:
        raise ValueError("model_comparison.csv 没有有效的 train/validation 模型行")
    return rows


def _find_laser_plot(base: Path | None, filename: str) -> Path | None:
    if base is None:
        return None
    for candidate in (
        base / filename,
        base / "plots" / filename,
        base / "diagnostics" / filename,
    ):
        if candidate.is_file():
            return candidate
    return None


def _laser_comparison(
    model: str,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    parameters: LaserModelParameters | None,
) -> LaserModelComparison:
    has_metric = any(
        value is not None
        for value in (
            _laser_metric(
                train,
                "board_rmse_mm",
                "surface_rmse_mm",
                "ray_rmse_mm",
                "train_rmse_mm",
            ),
            _laser_metric(
                validation,
                "board_rmse_mm",
                "surface_rmse_mm",
                "ray_rmse_mm",
                "validation_rmse_mm",
            ),
            _laser_metric(train, "valid_rate", "train_valid_rate"),
            _laser_metric(validation, "valid_rate", "validation_valid_rate"),
        )
    )
    return LaserModelComparison(
        model=model,
        train_rmse_mm=_laser_metric(
            train, "board_rmse_mm", "surface_rmse_mm", "ray_rmse_mm", "train_rmse_mm"
        ),
        validation_rmse_mm=_laser_metric(
            validation,
            "board_rmse_mm",
            "surface_rmse_mm",
            "ray_rmse_mm",
            "validation_rmse_mm",
        ),
        train_p95_mm=_laser_metric(
            train,
            "board_p95_abs_mm",
            "surface_p95_abs_mm",
            "ray_p95_abs_mm",
            "train_p95_mm",
        ),
        validation_p95_mm=_laser_metric(
            validation,
            "board_p95_abs_mm",
            "surface_p95_abs_mm",
            "ray_p95_abs_mm",
            "validation_p95_mm",
        ),
        train_max_mm=_laser_metric(
            train,
            "board_max_abs_mm",
            "surface_max_abs_mm",
            "ray_max_abs_mm",
            "train_max_mm",
        ),
        validation_max_mm=_laser_metric(
            validation,
            "board_max_abs_mm",
            "surface_max_abs_mm",
            "ray_max_abs_mm",
            "validation_max_mm",
        ),
        train_valid_rate=_laser_metric(train, "valid_rate", "train_valid_rate"),
        validation_valid_rate=_laser_metric(
            validation, "valid_rate", "validation_valid_rate"
        ),
        available=has_metric or parameters is not None,
        parameters=parameters,
    )


def load_laser_surface_details(
    laser_surface: LaserSurfaceSummary,
) -> LaserSurfaceDetails:
    """读取激光模型比较、参数和已有误差图；失败时返回可展示状态。"""

    if not isinstance(laser_surface, LaserSurfaceSummary):
        raise TypeError("laser_surface 必须是 LaserSurfaceSummary")
    if laser_surface.status == NOT_EXECUTED:
        return LaserSurfaceDetails(
            status=NOT_EXECUTED,
            output_dir=laser_surface.output_dir,
            result_file=laser_surface.result_file,
        )

    source = (
        Path(laser_surface.result_file).expanduser().resolve()
        if laser_surface.result_file is not None
        else None
    )
    recorded_output_dir = (
        Path(laser_surface.output_dir).expanduser().resolve()
        if laser_surface.output_dir is not None
        else None
    )
    # result_file 是本次 stage 的明确产物锚点；当旧 report 的 output_dir
    # 与它不一致时，优先使用 result_file 所在目录读取同批次 CSV/YAML/图片。
    base = source.parent if source is not None and source.is_file() else recorded_output_dir
    if base is None and source is not None:
        base = source.parent
    notes: list[str] = []
    result_document: Mapping[str, Any] = {}
    error: str | None = None
    if source is None:
        error = "激光表面 stage 未记录 result_file"
    elif not source.is_file():
        error = f"激光模型结果文件不存在：{source}"
    else:
        try:
            result_document = load_document(source)
        except Exception as exc:
            error = f"激光模型结果读取失败：{exc}"

    selected_model = _canonical_laser_model(laser_surface.model_type)
    if selected_model is None:
        selected_model = _canonical_laser_model(result_document.get("model_type"))
    if selected_model is None:
        model_selection = result_document.get("model_selection")
        if isinstance(model_selection, Mapping):
            selected_model = _canonical_laser_model(model_selection.get("default_model"))

    comparison_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
    comparison_path = base / "model_comparison.csv" if base is not None else None
    if comparison_path is None or not comparison_path.is_file():
        notes.append("未找到 model_comparison.csv")
    else:
        try:
            comparison_rows = _load_laser_comparison_rows(comparison_path)
        except Exception as exc:
            notes.append(f"模型比较结果读取失败：{exc}")

    result_metrics = result_document.get("metrics")
    result_metrics = result_metrics if isinstance(result_metrics, Mapping) else {}
    result_train = result_metrics.get("train")
    result_train = result_train if isinstance(result_train, Mapping) else {}
    result_validation = result_metrics.get("validation")
    result_validation = (
        result_validation if isinstance(result_validation, Mapping) else {}
    )
    parameters = _load_laser_parameters(
        base,
        selected_model,
        source,
        result_document,
        notes,
    )

    comparisons: list[LaserModelComparison] = []
    for model in SUPPORTED_LASER_MODEL_TYPES:
        model_rows = comparison_rows.get(model, {})
        train = model_rows.get("train", {})
        validation = model_rows.get("validation", {})
        if model == selected_model:
            train = train or result_train
            validation = validation or result_validation
        comparisons.append(
            _laser_comparison(model, train, validation, parameters.get(model))
        )

    selected_comparison = next(
        (item for item in comparisons if item.model == selected_model), None
    )
    validation_rmse = laser_surface.validation_rmse_mm
    validation_p95 = laser_surface.validation_p95_mm
    validation_rate = laser_surface.validation_valid_rate
    validation_max = None
    if selected_comparison is not None:
        validation_rmse = (
            validation_rmse
            if validation_rmse is not None
            else selected_comparison.validation_rmse_mm
        )
        validation_p95 = (
            validation_p95
            if validation_p95 is not None
            else selected_comparison.validation_p95_mm
        )
        validation_rate = (
            validation_rate
            if validation_rate is not None
            else selected_comparison.validation_valid_rate
        )
        validation_max = selected_comparison.validation_max_mm

    unique_notes = tuple(dict.fromkeys(notes))
    return LaserSurfaceDetails(
        status=DETAILS_UNAVAILABLE if error else DETAILS_LOADED,
        output_dir=base,
        result_file=source,
        error=error,
        notes=unique_notes,
        selected_model=selected_model,
        validation_rmse_mm=_finite_float(validation_rmse),
        validation_p95_mm=_finite_float(validation_p95),
        validation_max_mm=_finite_float(validation_max),
        validation_valid_rate=_finite_float(validation_rate),
        model_comparisons=tuple(comparisons),
        error_vs_u=_find_laser_plot(base, "validation_error_vs_u.png"),
        error_vs_v=_find_laser_plot(base, "validation_error_vs_v.png"),
        error_vs_depth=_find_laser_plot(base, "validation_error_vs_depth.png"),
    )


def _ground_metric(
    documents: Sequence[Mapping[str, Any]],
    *names: str,
) -> float | None:
    for document in documents:
        value = _first_mapping_value(document, *names)
        numeric = _finite_float(value)
        if numeric is not None:
            return numeric
    return None


def _ground_count(
    documents: Sequence[Mapping[str, Any]],
    *names: str,
) -> int | None:
    for document in documents:
        value = _first_mapping_value(document, *names)
        count = _as_count(value)
        if count is not None:
            return count
    return None


def _ground_row_from_mapping(
    raw_row: Mapping[str, Any],
    *,
    index: int,
    split: str,
) -> GroundValidationRow:
    frame = _first_mapping_value(raw_row, "image", "frame", "filename", "frame_id")
    if frame in (None, ""):
        frame = f"{split} #{index}"
    status_value = _first_mapping_value(raw_row, "status", "accepted")
    status = "" if status_value in (None, "") else str(status_value)
    return GroundValidationRow(
        frame=str(frame),
        status=status,
        detection_method=(
            str(raw_row["detection_method"])
            if raw_row.get("detection_method") not in (None, "")
            else None
        ),
        pnp_rmse_px=_finite_float(
            _first_mapping_value(raw_row, "pnp_rmse_px", "reprojection_rmse_px")
        ),
        rmse_mm=_finite_float(
            _first_mapping_value(
                raw_row,
                "zg_rmse_mm",
                "rmse_mm",
                "validation_rmse_mm",
                "plane_rmse_mm",
            )
        ),
        p95_mm=_finite_float(
            _first_mapping_value(
                raw_row,
                "zg_p95_abs_mm",
                "p95_abs_mm",
                "p95_mm",
                "validation_p95_mm",
            )
        ),
        max_mm=_finite_float(
            _first_mapping_value(
                raw_row,
                "zg_max_abs_mm",
                "max_abs_mm",
                "max_mm",
                "validation_max_mm",
            )
        ),
        mean_mm=_finite_float(
            _first_mapping_value(raw_row, "zg_mean_mm", "mean_mm")
        ),
        std_mm=_finite_float(_first_mapping_value(raw_row, "zg_std_mm", "std_mm")),
        point_count=_as_count(
            _first_mapping_value(raw_row, "zg_point_count", "point_count", "points")
        ),
    )


def _load_ground_csv_rows(
    path: Path,
    *,
    split: str,
    limit: int,
    notes: list[str],
) -> tuple[GroundValidationRow, ...]:
    if not path.is_file() or limit <= 0:
        return ()
    rows: list[GroundValidationRow] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for index, raw_row in enumerate(reader, start=1):
                if len(rows) >= limit:
                    break
                rows.append(_ground_row_from_mapping(raw_row, index=index, split=split))
    except (OSError, UnicodeError, csv.Error) as exc:
        notes.append(f"{split} 逐帧结果读取失败：{exc}")
        return ()
    return tuple(rows)


def _ground_document_rows(
    document: Mapping[str, Any],
    *,
    split: str,
    limit: int,
) -> tuple[GroundValidationRow, ...]:
    candidates: list[Any] = []
    if split == "validation":
        validation = document.get("validation")
        if isinstance(validation, Mapping):
            candidates.append(validation.get("frames"))
        candidates.append(document.get("validation_frames"))
    else:
        fit_selection = document.get("fit_selection")
        if isinstance(fit_selection, Mapping):
            candidates.append(fit_selection.get("frames"))
        normal_estimation = document.get("normal_estimation")
        if isinstance(normal_estimation, Mapping):
            candidates.append(normal_estimation.get("frames"))
        candidates.append(document.get("fit_frames"))
    for candidate in candidates:
        if isinstance(candidate, (str, bytes)) or not isinstance(candidate, Sequence):
            continue
        rows: list[GroundValidationRow] = []
        for index, raw_row in enumerate(candidate, start=1):
            if len(rows) >= limit:
                break
            if isinstance(raw_row, Mapping):
                rows.append(_ground_row_from_mapping(raw_row, index=index, split=split))
        if rows:
            return tuple(rows)
    return ()


def _ground_transform(
    document: Mapping[str, Any],
) -> tuple[
    str | None,
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
]:
    roots: list[Mapping[str, Any]] = [document]
    for key in ("extrinsics", "pose", "transform"):
        nested = document.get(key)
        if isinstance(nested, Mapping):
            roots.append(nested)

    transform_keys = (
        "T_ground_from_camera",
        "transform_ground_from_camera",
        "ground_from_camera",
    )
    for root in roots:
        for key in transform_keys:
            matrix = _as_matrix(root.get(key))
            is_homogeneous = (
                len(matrix) >= 4
                and all(len(row) >= 4 for row in matrix[:4])
                and all(abs(value) <= 1.0e-9 for value in matrix[3][:3])
                and abs(matrix[3][3] - 1.0) <= 1.0e-9
            )
            if is_homogeneous:
                return (
                    key,
                    tuple(tuple(row[:3]) for row in matrix[:3]),
                    tuple(row[3] for row in matrix[:3]),
                )

    # 某些新/旧 stage 只保存显式 R/t。只有在字段已经明确存在时读取，
    # 不从反向 T_camera_from_ground 求逆，也不猜测坐标轴定义。
    rotation_keys = (
        "R_ground_from_camera",
        "rotation_ground_from_camera",
        "R",
        "rotation_matrix",
    )
    translation_keys = (
        "t_ground_from_camera",
        "translation_ground_from_camera",
        "t",
        "translation",
        "translation_mm",
        "translation_vector",
        "t_mm",
    )
    for root in roots:
        for rotation_key in rotation_keys:
            rotation = _as_matrix(root.get(rotation_key))
            if len(rotation) < 3 or not all(len(row) >= 3 for row in rotation[:3]):
                continue
            for translation_key in translation_keys:
                translation = _as_vector(root.get(translation_key))
                if len(translation) >= 3:
                    return (
                        f"{rotation_key} + {translation_key}",
                        tuple(tuple(row[:3]) for row in rotation[:3]),
                        tuple(translation[:3]),
                    )
    return None, (), ()


def _explicit_ground_euler(
    document: Mapping[str, Any],
) -> tuple[float | None, float | None, float | None]:
    # 只接受结果中明确以 degree 命名的值。项目没有现成的 Euler 轴顺序，
    # 因此绝不从 rotation matrix 自行转换。
    roots: list[Mapping[str, Any]] = [document]
    for key in (
        "pose",
        "orientation",
        "euler_angles",
        "euler_angles_deg",
        "rpy",
        "rpy_deg",
    ):
        nested = document.get(key)
        if isinstance(nested, Mapping):
            roots.append(nested)
    for root in roots:
        values = tuple(
            _finite_float(root.get(name))
            for name in ("roll_deg", "pitch_deg", "yaw_deg")
        )
        if all(value is not None for value in values):
            return values  # type: ignore[return-value]
    return None, None, None


def _find_ground_validation_plot(base: Path | None) -> Path | None:
    if base is None:
        return None
    for filename in (
        "validation_zg_residual.png",
        "validation_error.png",
        "validation_residual.png",
    ):
        for candidate in (
            base / filename,
            base / "plots" / filename,
            base / "diagnostics" / filename,
        ):
            if candidate.is_file():
                return candidate
    return None


def load_ground_extrinsics_details(
    ground_extrinsics: GroundExtrinsicsSummary,
    *,
    max_rows: int = 200,
) -> GroundExtrinsicsDetails:
    """读取地面外参已有 YAML/CSV/图表，失败时返回可降级展示的结果。"""

    if not isinstance(ground_extrinsics, GroundExtrinsicsSummary):
        raise TypeError("ground_extrinsics 必须是 GroundExtrinsicsSummary")
    if ground_extrinsics.status == NOT_EXECUTED:
        return GroundExtrinsicsDetails(
            status=NOT_EXECUTED,
            stage=ground_extrinsics.stage,
            output_dir=ground_extrinsics.output_dir,
            result_file=ground_extrinsics.result_file,
        )

    recorded_output_dir = (
        Path(ground_extrinsics.output_dir).expanduser().resolve()
        if ground_extrinsics.output_dir is not None
        else None
    )
    source = None
    if ground_extrinsics.result_file is not None:
        source = Path(ground_extrinsics.result_file).expanduser()
        if not source.is_absolute() and recorded_output_dir is not None:
            output_candidate = recorded_output_dir / source
            # 直接构造 stage 摘要时 result_file 也可能已经是相对于当前
            # 工作目录的完整相对路径；优先使用存在的路径，否则按
            # output_dir 下的文件名解析。
            source = (
                source
                if source.is_file() and not output_candidate.is_file()
                else output_candidate
            )
        source = source.resolve()
    base = source.parent if source is not None and source.is_file() else recorded_output_dir
    if base is None and source is not None:
        base = source.parent

    error: str | None = None
    document: Mapping[str, Any] = {}
    if source is None:
        error = "地面外参 stage 未记录 result_file"
    elif not source.is_file():
        error = f"地面外参结果文件不存在：{source}"
    else:
        try:
            document = load_document(source)
        except Exception as exc:
            error = f"地面外参结果读取失败：{exc}"

    notes: list[str] = []
    transform_name, rotation, translation = _ground_transform(document)
    if not rotation:
        notes.append("结果文件未提供有效 T_ground_from_camera，未推导 R/t")
    roll_deg, pitch_deg, yaw_deg = _explicit_ground_euler(document)
    if roll_deg is None or pitch_deg is None or yaw_deg is None:
        notes.append("结果文件未提供带明确坐标系定义的 Roll/Pitch/Yaw，未自行换算")

    fit_rows = _load_ground_csv_rows(
        base / "fit_frames.csv" if base is not None else Path(),
        split="fit",
        limit=max_rows,
        notes=notes,
    )
    validation_rows = _load_ground_csv_rows(
        base / "validation_frames.csv" if base is not None else Path(),
        split="validation",
        limit=max_rows,
        notes=notes,
    )
    if not fit_rows:
        fit_rows = _ground_document_rows(document, split="fit", limit=max_rows)
    if not validation_rows:
        validation_rows = _ground_document_rows(
            document,
            split="validation",
            limit=max_rows,
        )
    if not validation_rows:
        notes.append("未找到已有独立 validation 逐帧结果")

    quality_checks = document.get("quality_checks")
    quality_checks = quality_checks if isinstance(quality_checks, Mapping) else {}
    fit_quality = quality_checks.get("fit_zg")
    fit_quality = fit_quality if isinstance(fit_quality, Mapping) else {}
    validation_quality = quality_checks.get("validation_zg")
    validation_quality = (
        validation_quality if isinstance(validation_quality, Mapping) else {}
    )
    fit_selection = document.get("fit_selection")
    fit_selection = fit_selection if isinstance(fit_selection, Mapping) else {}
    normal_estimation = document.get("normal_estimation")
    normal_estimation = (
        normal_estimation if isinstance(normal_estimation, Mapping) else {}
    )
    validation = document.get("validation")
    validation = validation if isinstance(validation, Mapping) else {}
    validation_metrics = validation.get("metrics")
    validation_metrics = (
        validation_metrics if isinstance(validation_metrics, Mapping) else {}
    )

    fit_frame_count = _ground_count(
        (fit_selection, normal_estimation, document),
        "accepted_frame_count",
        "fit_frame_count",
        "frame_count",
        "detected_frame_count",
    )
    if fit_frame_count is None and fit_rows:
        fit_frame_count = len(fit_rows)
    validation_frame_count = _ground_count(
        (validation, document),
        "validation_frame_count",
        "frame_count",
        "detected_frame_count",
    )
    if validation_frame_count is None and validation_rows:
        validation_frame_count = len(validation_rows)

    fit_rmse = _ground_metric(
        (fit_quality, fit_selection, document),
        "rmse_mm",
        "fit_rmse_mm",
    )
    validation_rmse = ground_extrinsics.validation_rmse_mm
    if validation_rmse is None:
        validation_rmse = _ground_metric(
            (validation_quality, validation_metrics, validation, document),
            "rmse_mm",
            "validation_rmse_mm",
        )
    validation_p95 = ground_extrinsics.validation_p95_mm
    if validation_p95 is None:
        validation_p95 = _ground_metric(
            (validation_quality, validation_metrics, validation, document),
            "p95_abs_mm",
            "validation_p95_mm",
            "p95_mm",
        )
    validation_max = _ground_metric(
        (validation_quality, validation_metrics, validation, document),
        "max_abs_mm",
        "validation_max_mm",
        "max_mm",
    )

    unique_notes = tuple(dict.fromkeys(notes))
    return GroundExtrinsicsDetails(
        status=DETAILS_UNAVAILABLE if error else DETAILS_LOADED,
        stage=ground_extrinsics.stage,
        output_dir=recorded_output_dir or base,
        result_file=source,
        error=error,
        notes=unique_notes,
        transform_name=transform_name,
        rotation_matrix=rotation,
        translation_mm=translation,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        fit_rmse_mm=fit_rmse,
        validation_rmse_mm=_finite_float(validation_rmse),
        validation_p95_mm=_finite_float(validation_p95),
        validation_max_mm=_finite_float(validation_max),
        fit_frame_count=fit_frame_count,
        validation_frame_count=validation_frame_count,
        validation_rows=validation_rows,
        validation_error_plot=_find_ground_validation_plot(base),
    )


def summarize_calibration_run(run: CalibrationRun) -> CalibrationResultsSummary:
    """将 CalibrationRun 映射为只读的结构化结果摘要。"""

    if not isinstance(run, CalibrationRun):
        raise TypeError("run 必须是 CalibrationRun")

    intrinsics_stage = _find_stage(run, ("intrinsics",))
    intrinsics_metrics = intrinsics_stage.metrics if intrinsics_stage is not None else {}
    intrinsics = IntrinsicsSummary(
        **_stage_kwargs(intrinsics_stage, "intrinsics"),
        fit_rmse_px=intrinsics_metrics.get("fit_rmse_px"),
        test_rmse_px=intrinsics_metrics.get("test_rmse_px"),
        fit_image_count=intrinsics_metrics.get("fit_image_count"),
        test_image_count=intrinsics_metrics.get("test_image_count"),
    )

    laser_stage = _find_stage(
        run,
        ("laser_surface_models", "laser_plane_shared_steger"),
    )
    laser_metrics = laser_stage.metrics if laser_stage is not None else {}
    laser_surface = LaserSurfaceSummary(
        **_stage_kwargs(laser_stage, "laser_surface_models"),
        model_type=laser_metrics.get("model_type"),
        validation_rmse_mm=laser_metrics.get("validation_rmse_mm"),
        validation_p95_mm=laser_metrics.get("validation_p95_mm"),
        validation_valid_rate=laser_metrics.get("validation_valid_rate"),
    )

    ground_stage = _find_stage(
        run,
        ("ground_extrinsics_board_only", "ground_extrinsics_shared_steger"),
    )
    ground_metrics = ground_stage.metrics if ground_stage is not None else {}
    ground_extrinsics = GroundExtrinsicsSummary(
        **_stage_kwargs(ground_stage, "ground_extrinsics"),
        validation_rmse_mm=ground_metrics.get("validation_rmse_mm"),
        validation_p95_mm=ground_metrics.get("validation_p95_mm"),
    )

    bias_stage = _find_stage(run, ("ground_bias",))
    bias_metrics = bias_stage.metrics if bias_stage is not None else {}
    ground_bias = GroundBiasSummary(
        **_stage_kwargs(bias_stage, "ground_bias"),
        loaded_frame_count=bias_metrics.get("loaded_frame_count"),
        independent_validation_frame_count=bias_metrics.get(
            "independent_validation_frame_count"
        ),
    )

    return CalibrationResultsSummary(
        run_id=run.run_id,
        project_id=run.project_id,
        status=run.status,
        overall=run.overall_status,
        laser_orientation=run.laser_orientation,
        started_utc=run.started_utc,
        completed_utc=run.completed_utc,
        intrinsics=intrinsics,
        laser_surface=laser_surface,
        ground_extrinsics=ground_extrinsics,
        ground_bias=ground_bias,
    )


# 为后续页面保留更直观的调用名称；实现保持单一。
build_calibration_results_summary = summarize_calibration_run


__all__ = [
    "DETAILS_LOADED",
    "DETAILS_UNAVAILABLE",
    "NOT_EXECUTED",
    "CalibrationResultsSummary",
    "GroundBiasSummary",
    "GroundExtrinsicsDetails",
    "GroundExtrinsicsSummary",
    "GroundValidationRow",
    "IntrinsicsDetails",
    "IntrinsicsReprojectionRow",
    "IntrinsicsSummary",
    "LaserModelComparison",
    "LaserModelParameters",
    "LaserSurfaceDetails",
    "LaserSurfaceSummary",
    "StageResultSummary",
    "build_calibration_results_summary",
    "load_intrinsics_details",
    "load_laser_surface_details",
    "load_ground_extrinsics_details",
    "summarize_calibration_run",
]
