from __future__ import annotations

import argparse
import csv
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Observation:
    path: Path
    corners: np.ndarray
    detection_method: str


@dataclass(frozen=True)
class ViewResult:
    observation: Observation
    rvec: np.ndarray
    tvec: np.ndarray
    projected: np.ndarray
    residuals: np.ndarray
    mean_euclidean_error: float
    rmse: float


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须大于0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用拟合集完成OpenCV单目棋盘格相机标定；如提供测试集，则额外评估测试集"
    )
    parser.add_argument("--fit-dir", type=Path, required=True, help="拟合图像目录")
    parser.add_argument("--test-dir", type=Path, help="独立测试图像目录；不提供时仅执行拟合集标定")
    parser.add_argument("--output", type=Path, required=True, help="结果输出目录")
    parser.add_argument("--fit-pattern", default="chess *.tif", help="拟合集文件glob")
    parser.add_argument("--test-pattern", default="chess *.tif", help="测试集文件glob")
    parser.add_argument("--pattern-cols", type=positive_int, default=6, help="横向内角点数")
    parser.add_argument("--pattern-rows", type=positive_int, default=5, help="纵向内角点数")
    parser.add_argument(
        "--square-size-mm", type=positive_float, default=30.0, help="棋盘格单格尺寸/mm"
    )
    parser.add_argument("--exclude-fit", nargs="*", default=[], metavar="FILENAME")
    parser.add_argument("--exclude-test", nargs="*", default=[], metavar="FILENAME")
    parser.add_argument("--max-fit-images", type=positive_int, default=None)
    parser.add_argument("--max-test-images", type=positive_int, default=None)
    parser.add_argument(
        "--free-k3",
        action="store_true",
        help="允许估计k3；默认使用CALIB_FIX_K3将k3固定为0",
    )
    parser.add_argument(
        "--no-reject-outliers",
        action="store_true",
        help="关闭拟合集稳健重投影异常剔除",
    )
    parser.add_argument(
        "--outlier-sigma",
        type=positive_float,
        default=3.0,
        help="MAD稳健异常阈值倍数（默认3.0）",
    )
    parser.add_argument(
        "--undistort-samples",
        type=positive_int,
        default=3,
        help="保存的去畸变示例数量（默认3）",
    )
    return parser


def calibration_flags(free_k3: bool) -> int:
    return 0 if free_k3 else cv2.CALIB_FIX_K3


def natural_sort_key(path: Path) -> list[tuple[int, str | int]]:
    return [
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    ]


def discover_images(directory: Path, pattern: str, maximum: int | None) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"图像目录不存在或不是目录：{directory}")
    paths = sorted(
        (
            path
            for path in directory.glob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )
    if maximum is not None:
        paths = paths[:maximum]
    if not paths:
        raise ValueError(f"{directory} 中没有匹配 {pattern!r} 的支持图像")
    return paths


def apply_exclusions(paths: list[Path], names: list[str], label: str) -> tuple[list[Path], set[Path]]:
    by_name = {path.name.casefold(): path for path in paths}
    requested = {name.casefold() for name in names}
    unknown = requested - by_name.keys()
    if unknown:
        raise ValueError(f"{label}排除列表中不存在：{', '.join(sorted(unknown))}")
    excluded = {by_name[name] for name in requested}
    selected = [path for path in paths if path not in excluded]
    if not selected:
        raise ValueError(f"{label}应用排除列表后没有剩余图像")
    return selected, excluded


def read_image(path: Path) -> np.ndarray | None:
    """兼容Windows中文路径读取。"""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        return None if encoded.size == 0 else cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def write_image(path: Path, image: np.ndarray) -> None:
    """兼容Windows中文路径写入。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise RuntimeError(f"OpenCV无法编码图像：{path}")
    encoded.tofile(path)


def detect_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None, str]:
    """SB成功时直接使用其亚像素结果；仅传统检测结果再做cornerSubPix。"""
    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            found, corners = cv2.findChessboardCornersSB(
                gray, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if found and corners is not None:
                return True, corners.astype(np.float32), "SB"
        except cv2.error:
            pass

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found or corners is None:
        return False, None, "failed"
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(
        gray, corners.astype(np.float32), (11, 11), (-1, -1), criteria
    )
    return True, refined, "legacy+cornerSubPix"


def create_object_points(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    return points


def scan_images(
    paths: list[Path],
    pattern_size: tuple[int, int],
    detected_dir: Path,
    expected_size: tuple[int, int] | None,
) -> tuple[list[Observation], dict[Path, str], tuple[int, int]]:
    observations: list[Observation] = []
    failures: dict[Path, str] = {}
    image_size = expected_size
    for path in paths:
        image = read_image(path)
        if image is None:
            failures[path] = "read_failed"
            print(f"[失败] 无法读取：{path.name}")
            continue
        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError(
                f"图像尺寸不一致：{path.name} 为 {current_size[0]}x{current_size[1]}，"
                f"期望 {image_size[0]}x{image_size[1]}"
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners, method = detect_corners(gray, pattern_size)
        if not found or corners is None:
            failures[path] = "corner_detection_failed"
            print(f"[失败] 未检测到完整棋盘格：{path.name}")
            continue
        observation = Observation(path, corners, method)
        observations.append(observation)
        overlay = image.copy()
        cv2.drawChessboardCorners(overlay, pattern_size, corners, True)
        cv2.putText(
            overlay,
            f"detector: {method}",
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        write_image(detected_dir / f"{path.stem}.png", overlay)
        print(f"[成功/{method}] {path.name}")
    if image_size is None:
        raise RuntimeError("所有图像均无法读取")
    return observations, failures, image_size


def build_view_results(
    observations: list[Observation],
    object_template: np.ndarray,
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[ViewResult]:
    results: list[ViewResult] = []
    for observation, rvec, tvec in zip(observations, rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(
            object_template, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected_xy = projected.reshape(-1, 2)
        residuals = observation.corners.reshape(-1, 2) - projected_xy
        distances = np.linalg.norm(residuals, axis=1)
        results.append(
            ViewResult(
                observation=observation,
                rvec=rvec,
                tvec=tvec,
                projected=projected_xy,
                residuals=residuals,
                mean_euclidean_error=float(np.mean(distances)),
                rmse=float(np.sqrt(np.mean(distances**2))),
            )
        )
    return results


def metric_summary(results: list[ViewResult]) -> dict[str, float | int]:
    if not results:
        raise ValueError("没有可计算误差的图像")
    distances = np.concatenate(
        [np.linalg.norm(result.residuals, axis=1) for result in results]
    )
    per_image_rmse = np.asarray([result.rmse for result in results])
    return {
        "image_count": len(results),
        "point_count": len(distances),
        "mean_euclidean_reprojection_error": float(np.mean(distances)),
        "overall_reprojection_rmse": float(np.sqrt(np.mean(distances**2))),
        "mean_per_image_rmse": float(np.mean(per_image_rmse)),
        "median_per_image_rmse": float(np.median(per_image_rmse)),
        "p95_per_image_rmse": float(np.percentile(per_image_rmse, 95)),
        "max_per_image_rmse": float(np.max(per_image_rmse)),
    }


def empty_metric_summary() -> dict[str, float | int]:
    return {
        "image_count": 0,
        "point_count": 0,
        "mean_euclidean_reprojection_error": float("nan"),
        "overall_reprojection_rmse": float("nan"),
        "mean_per_image_rmse": float("nan"),
        "median_per_image_rmse": float("nan"),
        "p95_per_image_rmse": float("nan"),
        "max_per_image_rmse": float("nan"),
    }


def find_outliers(results: list[ViewResult], sigma_multiplier: float) -> tuple[list[int], float | None]:
    if len(results) < 4:
        return [], None
    values = np.asarray([result.rmse for result in results], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= np.finfo(np.float64).eps:
        return [], None
    threshold = median + sigma_multiplier * robust_sigma
    return [index for index, value in enumerate(values) if value > threshold], threshold


def calibrate_observations(
    observations: list[Observation],
    object_template: np.ndarray,
    image_size: tuple[int, int],
    flags: int,
) -> tuple[float, np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    object_points = [object_template.copy() for _ in observations]
    image_points = [observation.corners for observation in observations]
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=flags
    )
    return rms, camera_matrix, dist_coeffs, tuple(rvecs), tuple(tvecs)


def evaluate_test_set(
    observations: list[Observation],
    object_template: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[ViewResult]:
    rvecs: list[np.ndarray] = []
    tvecs: list[np.ndarray] = []
    for observation in observations:
        solved, rvec, tvec = cv2.solvePnP(
            object_template,
            observation.corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not solved:
            raise RuntimeError(f"测试图位姿求解失败：{observation.path.name}")
        rvecs.append(rvec)
        tvecs.append(tvec)
    return build_view_results(
        observations,
        object_template,
        tuple(rvecs),
        tuple(tvecs),
        camera_matrix,
        dist_coeffs,
    )


def save_reprojection_visuals(results: list[ViewResult], output_dir: Path) -> None:
    reprojection_dir = output_dir / "reprojection"
    residual_dir = output_dir / "residual_vectors"
    for result in results:
        image = read_image(result.observation.path)
        if image is None:
            raise RuntimeError(f"无法重新读取：{result.observation.path}")
        observed = result.observation.corners.reshape(-1, 2)
        overlay = image.copy()
        vectors = image.copy()
        for measured, projected in zip(observed, result.projected, strict=True):
            measured_pt = tuple(np.rint(measured).astype(int))
            projected_pt = tuple(np.rint(projected).astype(int))
            cv2.drawMarker(overlay, measured_pt, (0, 255, 0), cv2.MARKER_CROSS, 9, 1)
            cv2.circle(overlay, projected_pt, 3, (0, 0, 255), -1)
            end = tuple(np.rint(projected + (measured - projected) * 50).astype(int))
            cv2.arrowedLine(vectors, projected_pt, end, (0, 0, 255), 2, tipLength=0.2)
        for target, text in (
            (overlay, f"green=observed red=projected RMSE={result.rmse:.4f}px"),
            (vectors, f"residual vectors x50 RMSE={result.rmse:.4f}px"),
        ):
            cv2.putText(
                target, text, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
        write_image(reprojection_dir / f"{result.observation.path.stem}.png", overlay)
        write_image(residual_dir / f"{result.observation.path.stem}.png", vectors)


def save_undistort_samples(
    results: list[ViewResult],
    output_dir: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    count: int,
) -> None:
    for result in results[:count]:
        image = read_image(result.observation.path)
        if image is None:
            continue
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        write_image(output_dir / f"{result.observation.path.stem}_undistorted.png", undistorted)


def write_extrinsics(path: Path, results: list[ViewResult]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["image", "rvec_x", "rvec_y", "rvec_z", "tvec_x_mm", "tvec_y_mm", "tvec_z_mm"]
        )
        for result in results:
            values = (*result.rvec.reshape(-1), *result.tvec.reshape(-1))
            writer.writerow([result.observation.path.name, *(f"{value:.17g}" for value in values)])


def write_image_report(
    path: Path,
    all_paths: list[Path],
    excluded: set[Path],
    observations: list[Observation],
    failures: dict[Path, str],
    final_results: list[ViewResult],
    initial_results: list[ViewResult] | None = None,
    outlier_paths: set[Path] | None = None,
    dataset_role: str = "fit",
) -> None:
    if dataset_role not in {"fit", "test"}:
        raise ValueError(f"未知数据集角色：{dataset_role}")
    is_fit = dataset_role == "fit"
    observation_by_path = {item.path: item for item in observations}
    final_by_path = {item.observation.path: item for item in final_results}
    initial_by_path = (
        {item.observation.path: item for item in initial_results} if initial_results else {}
    )
    outliers = outlier_paths or set()
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image", "status", "detection_method", "used_for_intrinsics",
                "mean_euclidean_reprojection_error", "per_image_rmse", "error_stage",
            ]
        )
        for image_path in all_paths:
            if image_path in excluded:
                writer.writerow([image_path.name, "excluded", "", "false", "", "", ""])
                continue
            if image_path in failures:
                writer.writerow([image_path.name, failures[image_path], "", "false", "", "", ""])
                continue
            observation = observation_by_path[image_path]
            if image_path in outliers:
                result = initial_by_path[image_path]
                writer.writerow(
                    [image_path.name, "reprojection_outlier", observation.detection_method, "false",
                     f"{result.mean_euclidean_error:.17g}", f"{result.rmse:.17g}", "initial_fit"]
                )
                continue
            result = final_by_path[image_path]
            writer.writerow(
                [
                    image_path.name,
                    "used" if is_fit else "evaluated",
                    observation.detection_method,
                    str(is_fit).lower(),
                    f"{result.mean_euclidean_error:.17g}",
                    f"{result.rmse:.17g}",
                    "final_fit" if is_fit else "frozen_intrinsics",
                ]
            )


def write_parameters_csv(
    path: Path,
    image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    opencv_rms: float,
    fit_metrics: dict[str, float | int],
    test_metrics: dict[str, float | int],
    fixed_k3: bool,
) -> None:
    distortion = dist_coeffs.reshape(-1)
    rows = [
        ("image_width", image_size[0], "px", "图像宽度"),
        ("image_height", image_size[1], "px", "图像高度"),
        ("fx", camera_matrix[0, 0], "px", "x方向焦距"),
        ("fy", camera_matrix[1, 1], "px", "y方向焦距"),
        ("cx", camera_matrix[0, 2], "px", "主点x"),
        ("cy", camera_matrix[1, 2], "px", "主点y"),
        ("k1", distortion[0], "", "径向畸变"),
        ("k2", distortion[1], "", "径向畸变"),
        ("p1", distortion[2], "", "切向畸变"),
        ("p2", distortion[3], "", "切向畸变"),
        ("k3", distortion[4], "", "径向畸变；默认固定为0"),
        ("k3_fixed", str(fixed_k3).lower(), "", "是否固定k3"),
        ("opencv_calibrate_rms", opencv_rms, "px", "OpenCV calibrateCamera返回值"),
    ]
    for dataset, metrics in (("fit", fit_metrics), ("test", test_metrics)):
        for name in (
            "mean_euclidean_reprojection_error",
            "overall_reprojection_rmse",
            "mean_per_image_rmse",
        ):
            rows.append((f"{dataset}_{name}", metrics[name], "px", name))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["parameter", "value", "unit", "description"])
        for name, value, unit, description in rows:
            formatted = f"{value:.17g}" if isinstance(value, (float, np.floating)) else value
            writer.writerow([name, formatted, unit, description])


def write_metrics_csv(
    path: Path,
    fit_metrics: dict[str, float | int],
    test_metrics: dict[str, float | int],
) -> None:
    fieldnames = ["dataset", *fit_metrics.keys()]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, metrics in (("fit", fit_metrics), ("test", test_metrics)):
            writer.writerow({"dataset": dataset, **metrics})


def write_yaml_result(
    path: Path,
    args: argparse.Namespace,
    image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    opencv_rms: float,
    fit_metrics: dict[str, float | int],
    test_metrics: dict[str, float | int],
) -> None:
    data = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "pattern_cols": args.pattern_cols,
        "pattern_rows": args.pattern_rows,
        "square_size_mm": args.square_size_mm,
        "k3_fixed": not args.free_k3,
        "corner_detection_policy": "SB direct; legacy fallback plus cornerSubPix",
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "opencv_calibrate_rms": opencv_rms,
        "fit_metrics": fit_metrics,
        "test_metrics": test_metrics,
    }
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def format_matrix(matrix: np.ndarray) -> str:
    return np.array2string(matrix, precision=12, separator=", ")


def write_reports(
    output_dir: Path,
    args: argparse.Namespace,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    opencv_rms: float,
    fit_metrics: dict[str, float | int],
    test_metrics: dict[str, float | int],
    fit_count: int,
    test_count: int,
    outlier_names: list[str],
    detector_counts: dict[str, int],
) -> None:
    metric_rows = []
    for label, metrics in (("拟合集", fit_metrics), ("独立测试集", test_metrics)):
        metric_rows.append(
            f"| {label} | {float(metrics['mean_euclidean_reprojection_error']):.6f} | "
            f"{float(metrics['overall_reprojection_rmse']):.6f} | "
            f"{float(metrics['mean_per_image_rmse']):.6f} |"
        )
    markdown = [
        "# 相机标定报告",
        "",
        "## 流程",
        "",
        "```mermaid",
        "flowchart LR",
        '  A["拟合集"] --> B["SB直接/传统+cornerSubPix"] --> C["异常剔除"] --> D["calibrateCamera"]',
        '  D --> E["冻结内参"] --> F["独立测试集solvePnP"] --> G["三种误差统计"]',
        "```",
        "",
        f"- 最终拟合图像：{fit_count}张；测试图像：{test_count}张。",
        f"- 角点方法统计：{', '.join(f'{key}={value}' for key, value in detector_counts.items())}。",
        f"- 拟合异常图：{', '.join(outlier_names) if outlier_names else '无'}。",
        f"- k3：{'自由估计' if args.free_k3 else '固定为0'}。",
        "",
        "## 三种误差",
        "",
        "| 数据集 | mean_euclidean_reprojection_error | overall_reprojection_rmse | mean_per_image_rmse |",
        "|---|---:|---:|---:|",
        *metric_rows,
        "",
        f"OpenCV `calibrateCamera()` 返回RMS：`{opencv_rms:.9f} px`。",
        "",
        "## 内参",
        "",
        "```text",
        format_matrix(camera_matrix),
        "dist_coeffs = [" + ", ".join(f"{value:.12g}" for value in dist_coeffs.reshape(-1)) + "]",
        "```",
        "",
        "## 图例",
        "",
        "- `detected`：棋盘格检测角点。",
        "- `reprojection`：绿色十字为实测角点，红色圆点为重投影点。",
        "- `residual_vectors`：从重投影点指向实测角点，向量放大50倍。",
        "- 测试集仅求外参并评价，从未参与相机内参拟合。",
        "",
    ]
    (output_dir / "calibration_report.md").write_text("\n".join(markdown), encoding="utf-8")

    html_rows = "".join(
        f"<tr><td>{label}</td><td>{float(metrics['mean_euclidean_reprojection_error']):.6f}</td>"
        f"<td>{float(metrics['overall_reprojection_rmse']):.6f}</td>"
        f"<td>{float(metrics['mean_per_image_rmse']):.6f}</td></tr>"
        for label, metrics in (("拟合集", fit_metrics), ("独立测试集", test_metrics))
    )
    html_text = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>相机标定报告</title><style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1050px;margin:32px auto;color:#1f2937;line-height:1.6}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d1d5db;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.flow{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}.box{{padding:10px 14px;background:#eff6ff;border:1px solid #93c5fd;border-radius:8px}}pre{{background:#f3f4f6;padding:14px;overflow:auto}}</style></head><body>
<h1>相机标定报告</h1><div class="flow"><div class="box">拟合集角点</div>→<div class="box">异常剔除与标定</div>→<div class="box">冻结内参</div>→<div class="box">独立测试</div>→<div class="box">三种误差</div></div>
<p>最终拟合{fit_count}张，测试{test_count}张；k3 {'自由估计' if args.free_k3 else '固定为0'}。</p>
<table><tr><th>数据集</th><th>平均欧氏距离</th><th>总体RMSE</th><th>逐图RMSE平均</th></tr>{html_rows}</table>
<p>OpenCV calibrateCamera RMS：<strong>{opencv_rms:.9f} px</strong></p>
<h2>内参</h2><pre>{html.escape(format_matrix(camera_matrix))}\ndist_coeffs = {html.escape(str(dist_coeffs.reshape(-1).tolist()))}</pre>
<h2>图例</h2><p>重投影图：绿色十字=实测角点，红色圆点=重投影点；残差向量从重投影点指向实测点并放大50倍。</p></body></html>"""
    (output_dir / "calibration_report.html").write_text(html_text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern_size = (args.pattern_cols, args.pattern_rows)
    object_template = create_object_points(*pattern_size, args.square_size_mm)

    fit_all = discover_images(args.fit_dir, args.fit_pattern, args.max_fit_images)
    fit_paths, fit_excluded = apply_exclusions(fit_all, args.exclude_fit, "拟合集")

    test_all: list[Path] = []
    test_paths: list[Path] = []
    test_excluded: set[Path] = set()
    test_observations: list[Observation] = []
    test_failures: dict[Path, str] = {}
    test_results: list[ViewResult] = []
    test_metrics = empty_metric_summary()
    test_count = 0

    if args.test_dir is not None:
        test_all = discover_images(args.test_dir, args.test_pattern, args.max_test_images)
        test_paths, test_excluded = apply_exclusions(test_all, args.exclude_test, "测试集")
        overlap = {path.resolve() for path in fit_paths} & {path.resolve() for path in test_paths}
        if overlap:
            names = ", ".join(sorted(path.name for path in overlap))
            raise ValueError(f"拟合集与测试集存在重叠，已拒绝标定：{names}")
    else:
        print("未提供 --test-dir，仅进行拟合集标定。")

    fit_observations, fit_failures, image_size = scan_images(
        fit_paths, pattern_size, output_dir / "fit" / "detected", None
    )
    if len(fit_observations) < 3:
        raise RuntimeError("成功检测的拟合图少于3张，无法标定")
    flags = calibration_flags(args.free_k3)
    initial_rms, initial_k, initial_d, initial_rvecs, initial_tvecs = calibrate_observations(
        fit_observations, object_template, image_size, flags
    )
    initial_results = build_view_results(
        fit_observations, object_template, initial_rvecs, initial_tvecs, initial_k, initial_d
    )
    outlier_indices: list[int] = []
    outlier_threshold: float | None = None
    if not args.no_reject_outliers:
        outlier_indices, outlier_threshold = find_outliers(initial_results, args.outlier_sigma)
    outlier_set = set(outlier_indices)
    final_observations = [
        item for index, item in enumerate(fit_observations) if index not in outlier_set
    ]
    if len(final_observations) < 3:
        raise RuntimeError("异常剔除后拟合图少于3张")
    if outlier_indices:
        for index in outlier_indices:
            print(f"[异常剔除] {fit_observations[index].path.name}: {initial_results[index].rmse:.9f} px")
        opencv_rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_observations(
            final_observations, object_template, image_size, flags
        )
    else:
        opencv_rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
            initial_rms, initial_k, initial_d, initial_rvecs, initial_tvecs
        )
    if not args.free_k3 and float(dist_coeffs.reshape(-1)[4]) != 0.0:
        raise RuntimeError("CALIB_FIX_K3已启用，但最终k3不为0")
    fit_results = build_view_results(
        final_observations, object_template, rvecs, tvecs, camera_matrix, dist_coeffs
    )

    if args.test_dir is not None:
        test_observations, test_failures, _ = scan_images(
            test_paths, pattern_size, output_dir / "test" / "detected", image_size
        )
        if not test_observations:
            raise RuntimeError("测试集没有成功检测到棋盘格的图像")
        test_results = evaluate_test_set(
            test_observations, object_template, camera_matrix, dist_coeffs
        )
        test_metrics = metric_summary(test_results)
        test_count = len(test_results)
    else:
        test_count = 0

    fit_metrics = metric_summary(fit_results)

    save_reprojection_visuals(fit_results, output_dir / "fit")
    if test_results:
        save_reprojection_visuals(test_results, output_dir / "test")
        save_undistort_samples(
            test_results, output_dir / "undistort_check", camera_matrix, dist_coeffs,
            args.undistort_samples,
        )
        write_extrinsics(output_dir / "test_extrinsics.csv", test_results)
    else:
        (output_dir / "test").mkdir(parents=True, exist_ok=True)
        (output_dir / "undistort_check").mkdir(parents=True, exist_ok=True)
    write_extrinsics(output_dir / "fit_extrinsics.csv", fit_results)
    outlier_paths = {fit_observations[index].path for index in outlier_indices}
    write_image_report(
        output_dir / "fit_images.csv", fit_all, fit_excluded, fit_observations,
        fit_failures, fit_results, initial_results, outlier_paths, "fit",
    )
    write_image_report(
        output_dir / "test_images.csv", test_all, test_excluded, test_observations,
        test_failures, test_results, dataset_role="test",
    )
    write_parameters_csv(
        output_dir / "camera_parameters.csv", image_size, camera_matrix, dist_coeffs,
        opencv_rms, fit_metrics, test_metrics, not args.free_k3,
    )
    write_metrics_csv(output_dir / "calibration_metrics.csv", fit_metrics, test_metrics)
    write_yaml_result(
        output_dir / "calibration_result.yaml", args, image_size, camera_matrix,
        dist_coeffs, opencv_rms, fit_metrics, test_metrics,
    )
    np.save(output_dir / "camera_matrix.npy", camera_matrix)
    np.save(output_dir / "dist_coeffs.npy", dist_coeffs)
    detector_counts: dict[str, int] = {}
    for observation in (*fit_observations, *test_observations):
        detector_counts[observation.detection_method] = detector_counts.get(observation.detection_method, 0) + 1
    write_reports(
        output_dir, args, camera_matrix, dist_coeffs, opencv_rms, fit_metrics,
        test_metrics, len(fit_results), len(test_results),
        [path.name for path in sorted(outlier_paths, key=natural_sort_key)], detector_counts,
    )
    metadata = {
        "outlier_threshold_pixels": outlier_threshold,
        "fit_detection_failures": [path.name for path in fit_failures],
        "test_detection_failures": [path.name for path in test_failures],
    }
    (output_dir / "run_metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"标定完成：{output_dir.resolve()}")
    print(f"OpenCV RMS：{opencv_rms:.9f} px")
    print(
        "拟合集三指标："
        f"mean_euclidean={float(fit_metrics['mean_euclidean_reprojection_error']):.9f}, "
        f"overall_rmse={float(fit_metrics['overall_reprojection_rmse']):.9f}, "
        f"mean_per_image_rmse={float(fit_metrics['mean_per_image_rmse']):.9f} px"
    )
    if args.test_dir is not None:
        print(
            "测试集三指标："
            f"mean_euclidean={float(test_metrics['mean_euclidean_reprojection_error']):.9f}, "
            f"overall_rmse={float(test_metrics['overall_reprojection_rmse']):.9f}, "
            f"mean_per_image_rmse={float(test_metrics['mean_per_image_rmse']):.9f} px"
        )
    else:
        print("测试集未提供，已跳过测试集评估。")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, cv2.error, yaml.YAMLError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
