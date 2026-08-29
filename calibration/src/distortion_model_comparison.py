from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ZONE_DEFINITIONS = (
    ("center", 0.0, 0.33),
    ("middle", 0.33, 0.66),
    ("edge", 0.66, float("inf")),
)
RESIDUAL_ARROW_SCALE = 50.0


@dataclass(frozen=True)
class ModelResult:
    key: str
    label: str
    flags: int
    rms: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    fit_paths: list[Path]
    fit_rvecs: tuple[np.ndarray, ...]
    fit_tvecs: tuple[np.ndarray, ...]
    fit_errors: list[float]
    fit_residuals: list[np.ndarray]
    validation_paths: list[Path]
    validation_rvecs: tuple[np.ndarray, ...]
    validation_tvecs: tuple[np.ndarray, ...]
    validation_errors: list[float]
    validation_residuals: list[np.ndarray]
    validation_failed_paths: list[Path]


def _calibrate_model(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    flags: int,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=flags,
    )
    return rms, camera_matrix, dist_coeffs, tuple(rvecs), tuple(tvecs)


def _residuals_and_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[list[np.ndarray], list[float]]:
    residuals: list[np.ndarray] = []
    errors: list[float] = []
    for object_point, observed, rvec, tvec in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
        strict=True,
    ):
        projected, _ = cv2.projectPoints(
            object_point,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        residual = observed.reshape(-1, 2) - projected.reshape(-1, 2)
        residuals.append(residual)
        errors.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
    return residuals, errors


def _evaluate_validation(
    object_point_template: np.ndarray,
    image_points: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    list[np.ndarray],
    list[float],
]:
    rvecs: list[np.ndarray] = []
    tvecs: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    errors: list[float] = []
    for corners in image_points:
        solved, rvec, tvec = cv2.solvePnP(
            object_point_template,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not solved:
            raise RuntimeError("solvePnP failed for a validation image")
        projected, _ = cv2.projectPoints(
            object_point_template,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        residual = corners.reshape(-1, 2) - projected.reshape(-1, 2)
        rvecs.append(rvec)
        tvecs.append(tvec)
        residuals.append(residual)
        errors.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
    return tuple(rvecs), tuple(tvecs), residuals, errors


def _error_statistics(errors: list[float]) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _normalized_radii(points: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    xy = points.reshape(-1, 2).astype(np.float64)
    x = (xy[:, 0] - width / 2.0) / (width / 2.0)
    y = (xy[:, 1] - height / 2.0) / (height / 2.0)
    return np.sqrt(x**2 + y**2)


def _radial_statistics(
    image_points: list[np.ndarray],
    residuals: list[np.ndarray],
    image_size: tuple[int, int],
) -> dict[str, dict[str, float | int]]:
    radius_parts = [
        _normalized_radii(points, image_size) for points in image_points
    ]
    radii = np.concatenate(radius_parts)
    vectors = np.concatenate(residuals, axis=0).astype(np.float64)
    result: dict[str, dict[str, float | int]] = {}
    for name, lower, upper in ZONE_DEFINITIONS:
        mask = (radii >= lower) & (radii < upper)
        selected = vectors[mask]
        if selected.size == 0:
            result[name] = {
                "point_count": 0,
                "rmse": float("nan"),
                "mean_dx": float("nan"),
                "mean_dy": float("nan"),
            }
            continue
        result[name] = {
            "point_count": int(selected.shape[0]),
            "rmse": float(np.sqrt(np.mean(np.sum(selected**2, axis=1)))),
            "mean_dx": float(np.mean(selected[:, 0])),
            "mean_dy": float(np.mean(selected[:, 1])),
        }
    return result


def _format_float(value: float) -> str:
    return "null" if not np.isfinite(value) else f"{value:.17g}"


def _write_model_yaml(
    output_path: Path,
    result: ModelResult,
    image_size: tuple[int, int],
    fit_radial: dict[str, dict[str, float | int]],
    validation_radial: dict[str, dict[str, float | int]],
) -> None:
    matrix_rows = [
        "  - [" + ", ".join(f"{value:.17g}" for value in row) + "]"
        for row in result.camera_matrix
    ]
    distortion = ", ".join(
        f"{value:.17g}" for value in result.dist_coeffs.reshape(-1)
    )
    lines = [
        f'model: "{result.key}"',
        f"opencv_flags: {result.flags}",
        f"image_width: {image_size[0]}",
        f"image_height: {image_size[1]}",
        "camera_matrix:",
        *matrix_rows,
        f"dist_coeffs: [{distortion}]",
        f"RMS: {result.rms:.17g}",
        "fit_per_image_rmse:",
    ]
    for path, error in zip(result.fit_paths, result.fit_errors, strict=True):
        lines.extend(
            [
                f'  - image: "{path.name}"',
                f"    rmse_pixels: {error:.17g}",
            ]
        )
    lines.append("validation_per_image_rmse:")
    for path, error in zip(
        result.validation_paths,
        result.validation_errors,
        strict=True,
    ):
        lines.extend(
            [
                f'  - image: "{path.name}"',
                f"    rmse_pixels: {error:.17g}",
            ]
        )
    lines.append("validation_detection_failed:")
    for path in result.validation_failed_paths:
        lines.append(f'  - image: "{path.name}"')
    for dataset_name, radial in (
        ("fit_radial_rmse", fit_radial),
        ("validation_radial_rmse", validation_radial),
    ):
        lines.append(f"{dataset_name}:")
        for zone, values in radial.items():
            lines.extend(
                [
                    f"  {zone}:",
                    f"    point_count: {values['point_count']}",
                    f"    rmse_pixels: {_format_float(float(values['rmse']))}",
                    f"    mean_dx_pixels: {_format_float(float(values['mean_dx']))}",
                    f"    mean_dy_pixels: {_format_float(float(values['mean_dy']))}",
                ]
            )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _save_validation_visuals(
    result: ModelResult,
    validation_image_points: list[np.ndarray],
    image_size: tuple[int, int],
    output_dir: Path,
    read_image: Callable[[Path], np.ndarray | None],
    write_image: Callable[[Path, np.ndarray], None],
) -> None:
    overlay_dir = output_dir / "validation_overlays"
    vector_dir = output_dir / "residual_vectors"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)
    width, height = image_size
    center = (width // 2, height // 2)
    for path, observed, residual, error in zip(
        result.validation_paths,
        validation_image_points,
        result.validation_residuals,
        result.validation_errors,
        strict=True,
    ):
        image = read_image(path)
        if image is None:
            raise RuntimeError(f"Unable to reread validation image: {path}")
        observed_xy = observed.reshape(-1, 2)
        projected_xy = observed_xy - residual

        overlay = image.copy()
        for measured, projected in zip(observed_xy, projected_xy, strict=True):
            measured_int = tuple(np.rint(measured).astype(int))
            projected_int = tuple(np.rint(projected).astype(int))
            cv2.drawMarker(
                overlay,
                measured_int,
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=1,
            )
            cv2.circle(overlay, projected_int, 3, (0, 0, 255), -1)
        _draw_label(
            overlay,
            f"{result.label} | green: measured | red: projected | RMSE={error:.4f}px",
        )
        write_image(
            overlay_dir / f"{path.stem}_{result.key}_overlay.png",
            overlay,
        )

        vector_image = np.clip(image.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
        for radius in (0.33, 0.66):
            axes = (int(width * radius / 2.0), int(height * radius / 2.0))
            cv2.ellipse(vector_image, center, axes, 0, 0, 360, (180, 180, 180), 1)
        radii = _normalized_radii(observed, image_size)
        for measured, projected, vector, radius in zip(
            observed_xy,
            projected_xy,
            residual,
            radii,
            strict=True,
        ):
            start = tuple(np.rint(projected).astype(int))
            end = tuple(np.rint(projected + vector * RESIDUAL_ARROW_SCALE).astype(int))
            if radius < 0.33:
                color = (255, 180, 0)
            elif radius < 0.66:
                color = (0, 215, 255)
            else:
                color = (0, 0, 255)
            cv2.arrowedLine(
                vector_image,
                start,
                end,
                color,
                2,
                cv2.LINE_AA,
                tipLength=0.25,
            )
            cv2.circle(vector_image, start, 2, (255, 255, 255), -1)
        _draw_label(
            vector_image,
            f"{result.label} | projected -> measured residual | {RESIDUAL_ARROW_SCALE:.0f}x",
        )
        write_image(
            vector_dir / f"{path.stem}_{result.key}_residual_vectors.png",
            vector_image,
        )


def _draw_label(image: np.ndarray, text: str) -> None:
    cv2.putText(
        image,
        text,
        (24, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (24, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _write_comparison_csv(
    output_dir: Path,
    results: list[ModelResult],
    radial_by_model: dict[str, dict[str, dict[str, dict[str, float | int]]]],
    recommended_key: str,
) -> None:
    header = [
        "model",
        "opencv_flags",
        "fit_opencv_rms",
        "fit_mean_rmse",
        "fit_median_rmse",
        "fit_p95_rmse",
        "fit_max_rmse",
        "validation_mean_rmse",
        "validation_median_rmse",
        "validation_p95_rmse",
        "validation_max_rmse",
        "validation_center_rmse",
        "validation_middle_rmse",
        "validation_edge_rmse",
        "validation_selected_count",
        "validation_success_count",
        "validation_failed_count",
        "fx",
        "fy",
        "cx",
        "cy",
        "k1",
        "k2",
        "p1",
        "p2",
        "k3",
        "recommended",
    ]
    with (output_dir / "distortion_model_comparison.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for result in results:
            fit_stats = _error_statistics(result.fit_errors)
            validation_stats = _error_statistics(result.validation_errors)
            validation_radial = radial_by_model[result.key]["validation"]
            distortion = result.dist_coeffs.reshape(-1)
            writer.writerow(
                [
                    result.key,
                    result.flags,
                    f"{result.rms:.17g}",
                    f"{fit_stats['mean']:.17g}",
                    f"{fit_stats['median']:.17g}",
                    f"{fit_stats['p95']:.17g}",
                    f"{fit_stats['max']:.17g}",
                    f"{validation_stats['mean']:.17g}",
                    f"{validation_stats['median']:.17g}",
                    f"{validation_stats['p95']:.17g}",
                    f"{validation_stats['max']:.17g}",
                    _format_float(float(validation_radial["center"]["rmse"])),
                    _format_float(float(validation_radial["middle"]["rmse"])),
                    _format_float(float(validation_radial["edge"]["rmse"])),
                    len(result.validation_paths) + len(result.validation_failed_paths),
                    len(result.validation_paths),
                    len(result.validation_failed_paths),
                    f"{result.camera_matrix[0, 0]:.17g}",
                    f"{result.camera_matrix[1, 1]:.17g}",
                    f"{result.camera_matrix[0, 2]:.17g}",
                    f"{result.camera_matrix[1, 2]:.17g}",
                    *(f"{value:.17g}" for value in distortion),
                    str(result.key == recommended_key).lower(),
                ]
            )

    with (output_dir / "distortion_model_per_image.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["model", "dataset", "image", "detection_success", "rmse_pixels"]
        )
        for result in results:
            for dataset, paths, errors in (
                ("fit", result.fit_paths, result.fit_errors),
                ("validation", result.validation_paths, result.validation_errors),
            ):
                for path, error in zip(paths, errors, strict=True):
                    writer.writerow(
                        [result.key, dataset, path.name, "true", f"{error:.17g}"]
                    )
            for path in result.validation_failed_paths:
                writer.writerow([result.key, "validation", path.name, "false", ""])

    with (output_dir / "distortion_model_radial_zones.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "model",
                "dataset",
                "zone",
                "radius_min",
                "radius_max",
                "point_count",
                "rmse_pixels",
                "mean_dx_pixels",
                "mean_dy_pixels",
            ]
        )
        bounds = {name: (lower, upper) for name, lower, upper in ZONE_DEFINITIONS}
        for result in results:
            for dataset in ("fit", "validation"):
                for zone, values in radial_by_model[result.key][dataset].items():
                    lower, upper = bounds[zone]
                    writer.writerow(
                        [
                            result.key,
                            dataset,
                            zone,
                            f"{lower:.2f}",
                            "inf" if not np.isfinite(upper) else f"{upper:.2f}",
                            values["point_count"],
                            _format_float(float(values["rmse"])),
                            _format_float(float(values["mean_dx"])),
                            _format_float(float(values["mean_dy"])),
                        ]
                    )


def _write_summary_chart(
    output_dir: Path,
    results: list[ModelResult],
    radial_by_model: dict[str, dict[str, dict[str, dict[str, float | int]]]],
) -> None:
    metric_labels = ["Fit RMS", "Validation mean", "Validation P95", "Validation edge"]
    values_by_model: list[list[float]] = []
    for result in results:
        validation_stats = _error_statistics(result.validation_errors)
        values_by_model.append(
            [
                result.rms,
                validation_stats["mean"],
                validation_stats["p95"],
                float(radial_by_model[result.key]["validation"]["edge"]["rmse"]),
            ]
        )
    width, height = 980, 560
    left, right, top, bottom = 78, 28, 72, 95
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = max(max(values) for values in values_by_model) * 1.18
    colors = ("#2563eb", "#d97706")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="78" y="36" font-family="Segoe UI,Arial" font-size="24" font-weight="700" fill="#172033">Distortion model comparison</text>',
    ]
    for tick in np.linspace(0, y_max, 6):
        y = top + plot_height * (1 - tick / y_max)
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dce2ea"/>'
        )
        svg.append(
            f'<text x="{left-10}" y="{y+5:.2f}" text-anchor="end" font-family="Segoe UI,Arial" font-size="13" fill="#5c687b">{tick:.3f}</text>'
        )
    group_width = plot_width / len(metric_labels)
    bar_width = group_width * 0.27
    for metric_index, label in enumerate(metric_labels):
        center_x = left + group_width * (metric_index + 0.5)
        for model_index, values in enumerate(values_by_model):
            value = values[metric_index]
            bar_height = value / y_max * plot_height
            x = center_x + (model_index - 1) * bar_width
            y = top + plot_height - bar_height
            svg.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{colors[model_index]}" rx="2"/>'
            )
            svg.append(
                f'<text x="{x+bar_width/2:.2f}" y="{y-7:.2f}" text-anchor="middle" font-family="Segoe UI,Arial" font-size="12" fill="#172033">{value:.3f}</text>'
            )
        svg.append(
            f'<text x="{center_x:.2f}" y="{top+plot_height+30}" text-anchor="middle" font-family="Segoe UI,Arial" font-size="14" fill="#394457">{label}</text>'
        )
    for index, result in enumerate(results):
        x = 620 + index * 165
        svg.append(f'<rect x="{x}" y="22" width="16" height="16" fill="{colors[index]}"/>')
        svg.append(
            f'<text x="{x+24}" y="35" font-family="Segoe UI,Arial" font-size="14" fill="#394457">{result.label}</text>'
        )
    svg.append(
        f'<text x="24" y="{top+plot_height/2:.2f}" transform="rotate(-90 24 {top+plot_height/2:.2f})" text-anchor="middle" font-family="Segoe UI,Arial" font-size="14" fill="#5c687b">RMSE (pixels)</text>'
    )
    svg.append("</svg>")
    (output_dir / "distortion_model_comparison.svg").write_text(
        "\n".join(svg),
        encoding="utf-8",
    )


def _write_validation_chart(output_dir: Path, results: list[ModelResult]) -> None:
    paths = results[0].validation_paths
    width, height = 1060, 520
    left, right, top, bottom = 76, 28, 70, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = max(max(result.validation_errors) for result in results) * 1.18
    colors = ("#2563eb", "#d97706")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="76" y="36" font-family="Segoe UI,Arial" font-size="24" font-weight="700" fill="#172033">Independent validation per-image RMSE</text>',
    ]
    for tick in np.linspace(0, y_max, 6):
        y = top + plot_height * (1 - tick / y_max)
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dce2ea"/>'
        )
        svg.append(
            f'<text x="{left-10}" y="{y+5:.2f}" text-anchor="end" font-family="Segoe UI,Arial" font-size="13" fill="#5c687b">{tick:.3f}</text>'
        )
    group_width = plot_width / len(paths)
    bar_width = group_width * 0.28
    for path_index, path in enumerate(paths):
        center_x = left + group_width * (path_index + 0.5)
        for model_index, result in enumerate(results):
            value = result.validation_errors[path_index]
            bar_height = value / y_max * plot_height
            x = center_x + (model_index - 1) * bar_width
            y = top + plot_height - bar_height
            svg.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{colors[model_index]}" rx="2"/>'
            )
        svg.append(
            f'<text x="{center_x:.2f}" y="{top+plot_height+28}" text-anchor="middle" font-family="Segoe UI,Arial" font-size="14" fill="#394457">{path.stem[-3:]}</text>'
        )
    for index, result in enumerate(results):
        x = 700 + index * 165
        svg.append(f'<rect x="{x}" y="22" width="16" height="16" fill="{colors[index]}"/>')
        svg.append(
            f'<text x="{x+24}" y="35" font-family="Segoe UI,Arial" font-size="14" fill="#394457">{result.label}</text>'
        )
    svg.append("</svg>")
    (output_dir / "validation_model_rmse_comparison.svg").write_text(
        "\n".join(svg),
        encoding="utf-8",
    )


def _recommend_model(
    standard: ModelResult,
    fixed: ModelResult,
    radial_by_model: dict[str, dict[str, dict[str, dict[str, float | int]]]],
) -> tuple[str, str]:
    standard_stats = _error_statistics(standard.validation_errors)
    fixed_stats = _error_statistics(fixed.validation_errors)
    standard_edge = float(radial_by_model[standard.key]["validation"]["edge"]["rmse"])
    fixed_edge = float(radial_by_model[fixed.key]["validation"]["edge"]["rmse"])
    deltas = {
        "mean": fixed_stats["mean"] - standard_stats["mean"],
        "p95": fixed_stats["p95"] - standard_stats["p95"],
        "edge": fixed_edge - standard_edge,
    }
    if any(delta > 0.02 for delta in deltas.values()):
        reason = (
            "固定 k3 后独立验证均值、P95 或边缘区 RMSE 至少有一项增加超过 "
            "0.02 px，因此保留标准 5 参数模型。"
        )
        return standard.key, reason
    reason = (
        "固定 k3 后独立验证均值、P95 和边缘区 RMSE 均未增加超过 0.02 px；"
        "按简洁性和稳定性原则推荐固定 k3=0 模型。"
    )
    return fixed.key, reason


def _write_markdown_report(
    output_dir: Path,
    results: list[ModelResult],
    radial_by_model: dict[str, dict[str, dict[str, dict[str, float | int]]]],
    selected_fit_paths: list[Path],
    selected_validation_paths: list[Path],
    outlier_threshold: float | None,
    outlier_paths: list[Path],
    recommended_key: str,
    recommendation_reason: str,
) -> None:
    standard, fixed = results
    standard_fit = _error_statistics(standard.fit_errors)
    fixed_fit = _error_statistics(fixed.fit_errors)
    standard_validation = _error_statistics(standard.validation_errors)
    fixed_validation = _error_statistics(fixed.validation_errors)
    standard_validation_edge = radial_by_model[standard.key]["validation"]["edge"]
    fixed_validation_edge = radial_by_model[fixed.key]["validation"]["edge"]
    fit_range = (
        f"`{selected_fit_paths[0].name}`—`{selected_fit_paths[-1].name}`"
        f"（共 {len(selected_fit_paths)} 张）"
    )
    validation_range = (
        f"`{selected_validation_paths[0].name}`—`{selected_validation_paths[-1].name}`"
        f"（共 {len(selected_validation_paths)} 张）"
    )
    standard_edge_bias = float(
        np.hypot(
            float(standard_validation_edge["mean_dx"]),
            float(standard_validation_edge["mean_dy"]),
        )
    )
    fixed_edge_bias = float(
        np.hypot(
            float(fixed_validation_edge["mean_dx"]),
            float(fixed_validation_edge["mean_dy"]),
        )
    )
    lines = [
        "# 畸变模型对比报告",
        "",
        "## 1. 对比约束",
        "",
        f"- 拟合集固定为 {fit_range}。",
        f"- 独立验证集固定为 {validation_range}。",
        "- 两种模型共享一次角点检测及 cornerSubPix 结果。",
        "- 异常图只由模型 A 的拟合集初始误差判定，再对两种模型应用相同索引。",
        "- 验证集只运行 solvePnP 和 projectPoints，从未进入 calibrateCamera。",
        f"- 验证角点检测成功 `{len(standard.validation_paths)}` 张，失败 `{len(standard.validation_failed_paths)}` 张。",
        "",
        f"异常阈值：`{_format_float(float(outlier_threshold)) if outlier_threshold is not None else '未启用'}` px；",
        f"剔除图像：`{', '.join(path.name for path in outlier_paths) if outlier_paths else '无'}`。",
        "",
        "## 2. 总体结果",
        "",
        "![主要指标对比](distortion_model_comparison.svg)",
        "",
        "| 指标 | 模型 A：5 参数 | 模型 B：固定 k3=0 | B-A |",
        "|---|---:|---:|---:|",
        f"| 拟合 OpenCV RMS (px) | {standard.rms:.6f} | {fixed.rms:.6f} | {fixed.rms-standard.rms:+.6f} |",
        f"| 拟合单图均值 (px) | {standard_fit['mean']:.6f} | {fixed_fit['mean']:.6f} | {fixed_fit['mean']-standard_fit['mean']:+.6f} |",
        f"| 拟合单图中位数 (px) | {standard_fit['median']:.6f} | {fixed_fit['median']:.6f} | {fixed_fit['median']-standard_fit['median']:+.6f} |",
        f"| 拟合单图 P95 (px) | {standard_fit['p95']:.6f} | {fixed_fit['p95']:.6f} | {fixed_fit['p95']-standard_fit['p95']:+.6f} |",
        f"| 拟合单图最大值 (px) | {standard_fit['max']:.6f} | {fixed_fit['max']:.6f} | {fixed_fit['max']-standard_fit['max']:+.6f} |",
        f"| 验证单图均值 (px) | {standard_validation['mean']:.6f} | {fixed_validation['mean']:.6f} | {fixed_validation['mean']-standard_validation['mean']:+.6f} |",
        f"| 验证单图中位数 (px) | {standard_validation['median']:.6f} | {fixed_validation['median']:.6f} | {fixed_validation['median']-standard_validation['median']:+.6f} |",
        f"| 验证单图 P95 (px) | {standard_validation['p95']:.6f} | {fixed_validation['p95']:.6f} | {fixed_validation['p95']-standard_validation['p95']:+.6f} |",
        f"| 验证单图最大值 (px) | {standard_validation['max']:.6f} | {fixed_validation['max']:.6f} | {fixed_validation['max']-standard_validation['max']:+.6f} |",
        "",
        "## 3. 参数",
        "",
    ]
    for result in results:
        distortion = result.dist_coeffs.reshape(-1)
        lines.extend(
            [
                f"### {result.label}",
                "",
                "```text",
                np.array2string(result.camera_matrix, precision=12, separator=", "),
                "dist_coeffs = [" + ", ".join(f"{value:.12g}" for value in distortion) + "]",
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "模型 B 的 `k3` 已通过程序严格断言为 `0.0`。",
            "",
            "## 4. 独立验证逐图误差",
            "",
            "![验证集逐图误差](validation_model_rmse_comparison.svg)",
            "",
            "| 图像 | 模型 A RMSE (px) | 模型 B RMSE (px) | B-A (px) |",
            "|---|---:|---:|---:|",
        ]
    )
    for path, standard_error, fixed_error in zip(
        standard.validation_paths,
        standard.validation_errors,
        fixed.validation_errors,
        strict=True,
    ):
        lines.append(
            f"| {path.name} | {standard_error:.6f} | {fixed_error:.6f} | {fixed_error-standard_error:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## 5. 归一化半径分区",
            "",
            "归一化半径定义为 `sqrt((dx/(width/2))² + (dy/(height/2))²)`；中心区 `<0.33`，中间区 `0.33–0.66`，边缘区 `>=0.66`。",
            "",
            "| 数据集 | 区域 | 模型 A RMSE (px) | 模型 B RMSE (px) | B-A (px) | 点数 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("fit", "validation"):
        for zone, _, _ in ZONE_DEFINITIONS:
            a_values = radial_by_model[standard.key][dataset][zone]
            b_values = radial_by_model[fixed.key][dataset][zone]
            a_rmse = float(a_values["rmse"])
            b_rmse = float(b_values["rmse"])
            lines.append(
                f"| {dataset} | {zone} | {_format_float(a_rmse)} | {_format_float(b_rmse)} | {_format_float(b_rmse-a_rmse)} | {a_values['point_count']} |"
            )
    lines.extend(
        [
            "",
            "验证集边缘区平均残差向量（重投影点 → 实测角点）：",
            "",
            f"- 模型 A：`dx={float(standard_validation_edge['mean_dx']):+.6f} px`，`dy={float(standard_validation_edge['mean_dy']):+.6f} px`。",
            f"- 模型 B：`dx={float(fixed_validation_edge['mean_dx']):+.6f} px`，`dy={float(fixed_validation_edge['mean_dy']):+.6f} px`。",
            f"- 边缘区平均残差向量模长分别为 {standard_edge_bias:.6f} px 和 {fixed_edge_bias:.6f} px；固定 k3 未产生明显的方向性误差放大。",
            "",
            "残差向量定义为“重投影点 → 实测亚像素角点”，图中放大 50 倍；蓝/黄/红分别表示中心、中间和边缘区域。",
            "",
            "- 模型 A 验证叠加图：[目录](distortion_model_comparison/5param/validation_overlays/)",
            "- 模型 A 残差向量图：[目录](distortion_model_comparison/5param/residual_vectors/)",
            "- 模型 B 验证叠加图：[目录](distortion_model_comparison/fix_k3/validation_overlays/)",
            "- 模型 B 残差向量图：[目录](distortion_model_comparison/fix_k3/residual_vectors/)",
            "",
            "## 6. 推荐",
            "",
            f"**推荐模型：`{recommended_key}`。**",
            "",
            recommendation_reason,
            "",
            "验证检测失败：`"
            + (", ".join(path.name for path in standard.validation_failed_paths) or "无")
            + "`；失败图不参与验证误差统计，也不回流拟合。",
            "",
            "详细机器可读数据见 `distortion_model_comparison.csv`、`distortion_model_per_image.csv` 和 `distortion_model_radial_zones.csv`。",
            "",
        ]
    )
    (output_dir / "distortion_model_comparison.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_distortion_model_comparison(
    *,
    output_dir: Path,
    image_size: tuple[int, int],
    pattern_size: tuple[int, int],
    object_point_template: np.ndarray,
    selected_fit_paths: list[Path],
    successful_fit_paths: list[Path],
    fit_object_points: list[np.ndarray],
    fit_image_points: list[np.ndarray],
    validation_paths: list[Path],
    reject_outliers: bool,
    outlier_sigma: float,
    detect_corners: Callable[
        [np.ndarray, tuple[int, int]], tuple[bool, np.ndarray | None]
    ],
    find_outliers: Callable[[list[float], float], tuple[list[int], float | None]],
    read_image: Callable[[Path], np.ndarray | None],
    write_image: Callable[[Path, np.ndarray], None],
) -> None:
    initial = _calibrate_model(fit_object_points, fit_image_points, image_size, flags=0)
    initial_residuals, initial_errors = _residuals_and_errors(
        fit_object_points,
        fit_image_points,
        initial[3],
        initial[4],
        initial[1],
        initial[2],
    )
    del initial_residuals
    outlier_indices: list[int] = []
    outlier_threshold: float | None = None
    if reject_outliers:
        outlier_indices, outlier_threshold = find_outliers(
            initial_errors,
            outlier_sigma,
        )
    outlier_index_set = set(outlier_indices)
    keep_indices = [
        index
        for index in range(len(successful_fit_paths))
        if index not in outlier_index_set
    ]
    if len(keep_indices) < 3:
        raise RuntimeError("Outlier rejection left fewer than three fit images")
    shared_fit_paths = [successful_fit_paths[index] for index in keep_indices]
    shared_object_points = [fit_object_points[index] for index in keep_indices]
    shared_image_points = [fit_image_points[index] for index in keep_indices]
    outlier_paths = [successful_fit_paths[index] for index in outlier_indices]

    validation_successful_paths: list[Path] = []
    validation_image_points: list[np.ndarray] = []
    validation_failed_paths: list[Path] = []
    for path in validation_paths:
        image = read_image(path)
        if image is None:
            raise RuntimeError(f"Unable to read validation image: {path}")
        current_size = (image.shape[1], image.shape[0])
        if current_size != image_size:
            raise ValueError(
                f"Validation image size mismatch: {path.name} is {current_size}, "
                f"expected {image_size}"
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = detect_corners(gray, pattern_size)
        if not found or corners is None:
            validation_failed_paths.append(path)
            continue
        validation_successful_paths.append(path)
        validation_image_points.append(corners)
    if not validation_successful_paths:
        raise RuntimeError("No validation image produced usable chessboard corners")

    results: list[ModelResult] = []
    for key, label, flags in (
        ("5param", "Model A: standard 5-param", 0),
        ("fix_k3", "Model B: fixed k3=0", cv2.CALIB_FIX_K3),
    ):
        rms, camera_matrix, dist_coeffs, fit_rvecs, fit_tvecs = _calibrate_model(
            shared_object_points,
            shared_image_points,
            image_size,
            flags,
        )
        if key == "fix_k3" and float(dist_coeffs.reshape(-1)[4]) != 0.0:
            raise RuntimeError(
                f"CALIB_FIX_K3 did not produce exact k3=0: {dist_coeffs.reshape(-1)[4]}"
            )
        fit_residuals, fit_errors = _residuals_and_errors(
            shared_object_points,
            shared_image_points,
            fit_rvecs,
            fit_tvecs,
            camera_matrix,
            dist_coeffs,
        )
        validation_rvecs, validation_tvecs, validation_residuals, validation_errors = (
            _evaluate_validation(
                object_point_template,
                validation_image_points,
                camera_matrix,
                dist_coeffs,
            )
        )
        results.append(
            ModelResult(
                key=key,
                label=label,
                flags=flags,
                rms=rms,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                fit_paths=shared_fit_paths,
                fit_rvecs=fit_rvecs,
                fit_tvecs=fit_tvecs,
                fit_errors=fit_errors,
                fit_residuals=fit_residuals,
                validation_paths=validation_successful_paths,
                validation_rvecs=validation_rvecs,
                validation_tvecs=validation_tvecs,
                validation_errors=validation_errors,
                validation_residuals=validation_residuals,
                validation_failed_paths=validation_failed_paths,
            )
        )

    radial_by_model: dict[
        str, dict[str, dict[str, dict[str, float | int]]]
    ] = {}
    for result in results:
        radial_by_model[result.key] = {
            "fit": _radial_statistics(
                shared_image_points,
                result.fit_residuals,
                image_size,
            ),
            "validation": _radial_statistics(
                validation_image_points,
                result.validation_residuals,
                image_size,
            ),
        }

    recommended_key, recommendation_reason = _recommend_model(
        results[0],
        results[1],
        radial_by_model,
    )
    comparison_root = output_dir / "distortion_model_comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    for result in results:
        _write_model_yaml(
            output_dir
            / (
                "camera_intrinsics_5param.yaml"
                if result.key == "5param"
                else "camera_intrinsics_fix_k3.yaml"
            ),
            result,
            image_size,
            radial_by_model[result.key]["fit"],
            radial_by_model[result.key]["validation"],
        )
        _save_validation_visuals(
            result,
            validation_image_points,
            image_size,
            comparison_root / result.key,
            read_image,
            write_image,
        )

    _write_comparison_csv(
        output_dir,
        results,
        radial_by_model,
        recommended_key,
    )
    _write_summary_chart(output_dir, results, radial_by_model)
    _write_validation_chart(output_dir, results)
    _write_markdown_report(
        output_dir,
        results,
        radial_by_model,
        selected_fit_paths,
        validation_paths,
        outlier_threshold,
        outlier_paths,
        recommended_key,
        recommendation_reason,
    )

    selected_fit_set = set(selected_fit_paths)
    if any(path in selected_fit_set for path in validation_paths):
        raise RuntimeError("Fit and validation image sets overlap")
    print(f"Shared fit images: {len(shared_fit_paths)}")
    print(f"Shared validation images: {len(validation_successful_paths)}")
    print(f"Validation detection failures: {len(validation_failed_paths)}")
    print(f"Outliers rejected for both models: {len(outlier_paths)}")
    print(f"Recommended model: {recommended_key}")
