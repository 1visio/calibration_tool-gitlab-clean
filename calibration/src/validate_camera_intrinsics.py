from __future__ import annotations

import argparse
import csv
import html
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from calibrate_chessboard_opencv import (
    create_object_points,
    detect_corners,
    read_image,
    write_image,
)


@dataclass(frozen=True)
class Intrinsics:
    key: str
    label: str
    source: Path
    image_size: tuple[int, int]
    pattern_size: tuple[int, int]
    square_size_mm: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass(frozen=True)
class ImageResult:
    path: Path
    rmse: float
    residuals: np.ndarray
    projected: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在同一棋盘格验证集上比较两份冻结相机内参"
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--image-pattern", default="chess *.tif")
    parser.add_argument("--opencv-yaml", type=Path, required=True)
    parser.add_argument("--matlab-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_intrinsics(path: Path, key: str, label: str) -> Intrinsics:
    if not path.is_file():
        raise FileNotFoundError(f"内参文件不存在：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "image_width",
        "image_height",
        "pattern_cols",
        "pattern_rows",
        "square_size_mm",
        "camera_matrix",
        "dist_coeffs",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{path.name} 缺少字段：{', '.join(sorted(missing))}")
    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"{path.name} 的 camera_matrix 不是 3x3")
    if dist_coeffs.size < 5:
        raise ValueError(f"{path.name} 的 dist_coeffs 少于5项")
    return Intrinsics(
        key=key,
        label=label,
        source=path.resolve(),
        image_size=(int(data["image_width"]), int(data["image_height"])),
        pattern_size=(int(data["pattern_cols"]), int(data["pattern_rows"])),
        square_size_mm=float(data["square_size_mm"]),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )


def statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def evaluate_image(
    path: Path,
    object_points: np.ndarray,
    corners: np.ndarray,
    intrinsics: Intrinsics,
) -> ImageResult:
    solved, rvec, tvec = cv2.solvePnP(
        object_points,
        corners,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        raise RuntimeError(f"{intrinsics.label} 对 {path.name} 求解位姿失败")
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    )
    observed_xy = corners.reshape(-1, 2)
    projected_xy = projected.reshape(-1, 2)
    residuals = observed_xy - projected_xy
    rmse = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    return ImageResult(path, rmse, residuals, projected_xy)


def radial_statistics(
    results: list[ImageResult],
    corners_by_path: dict[Path, np.ndarray],
    image_size: tuple[int, int],
) -> dict[str, dict[str, float | int]]:
    width, height = image_size
    values: dict[str, list[np.ndarray]] = {"center": [], "middle": [], "edge": []}
    for result in results:
        points = corners_by_path[result.path].reshape(-1, 2)
        radius = np.sqrt(
            ((points[:, 0] - width / 2) / (width / 2)) ** 2
            + ((points[:, 1] - height / 2) / (height / 2)) ** 2
        )
        masks = {
            "center": radius < 0.33,
            "middle": (radius >= 0.33) & (radius < 0.66),
            "edge": radius >= 0.66,
        }
        for zone, mask in masks.items():
            if np.any(mask):
                values[zone].append(result.residuals[mask])
    output: dict[str, dict[str, float | int]] = {}
    for zone, chunks in values.items():
        residuals = np.concatenate(chunks, axis=0)
        output[zone] = {
            "point_count": int(len(residuals)),
            "rmse": float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))),
            "mean_dx": float(np.mean(residuals[:, 0])),
            "mean_dy": float(np.mean(residuals[:, 1])),
        }
    return output


def save_visuals(
    output_dir: Path,
    model: Intrinsics,
    results: list[ImageResult],
    corners_by_path: dict[Path, np.ndarray],
) -> None:
    overlay_dir = output_dir / model.key / "reprojection_overlays"
    residual_dir = output_dir / model.key / "residual_vectors"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    residual_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        image = read_image(result.path)
        if image is None:
            raise RuntimeError(f"无法重新读取验证图：{result.path}")
        observed = corners_by_path[result.path].reshape(-1, 2)
        overlay = image.copy()
        for measured, projected in zip(observed, result.projected, strict=True):
            measured_pt = tuple(np.rint(measured).astype(int))
            projected_pt = tuple(np.rint(projected).astype(int))
            cv2.circle(overlay, measured_pt, 5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.drawMarker(
                overlay,
                projected_pt,
                (0, 0, 255),
                cv2.MARKER_CROSS,
                9,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            overlay,
            f"{model.label} RMSE={result.rmse:.4f}px  green=observed red=projected",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        write_image(overlay_dir / f"{result.path.stem}.png", overlay)

        vectors = image.copy()
        for measured, projected in zip(observed, result.projected, strict=True):
            start = tuple(np.rint(projected).astype(int))
            end = tuple(np.rint(projected + (measured - projected) * 50).astype(int))
            cv2.arrowedLine(vectors, start, end, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.2)
            cv2.circle(vectors, start, 3, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            vectors,
            f"{model.label} residual vectors x50",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        write_image(residual_dir / f"{result.path.stem}.png", vectors)


def write_chart(output_dir: Path, paths: list[Path], by_model: dict[str, list[ImageResult]]) -> None:
    width, height = 1100, 540
    left, right, top, bottom = 80, 30, 70, 100
    plot_w = width - left - right
    plot_h = height - top - bottom
    maximum = max(result.rmse for results in by_model.values() for result in results) * 1.18
    colors = {"opencv": "#2563eb", "matlab": "#d97706"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="35" text-anchor="middle" font-family="sans-serif" font-size="22">Validation per-image reprojection RMSE</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_h - plot_h * tick / 5
        svg.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2f}</text>',
            ]
        )
    group_w = plot_w / len(paths)
    bar_w = group_w * 0.32
    for index, path in enumerate(paths):
        center = left + group_w * (index + 0.5)
        for offset, key in ((-0.5, "opencv"), (0.5, "matlab")):
            value = by_model[key][index].rmse
            bar_h = plot_h * value / maximum
            x = center + offset * bar_w - bar_w / 2
            y = top + plot_h - bar_h
            svg.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="{colors[key]}"/>'
            )
        svg.append(
            f'<text x="{center:.2f}" y="{top+plot_h+22}" text-anchor="middle" font-family="sans-serif" font-size="11" transform="rotate(35 {center:.2f} {top+plot_h+22})">{html.escape(path.stem)}</text>'
        )
    svg.extend(
        [
            f'<rect x="{width-250}" y="48" width="14" height="14" fill="{colors["opencv"]}"/><text x="{width-230}" y="60" font-family="sans-serif" font-size="13">OpenCV</text>',
            f'<rect x="{width-150}" y="48" width="14" height="14" fill="{colors["matlab"]}"/><text x="{width-130}" y="60" font-family="sans-serif" font-size="13">MATLAB</text>',
            f'<text x="18" y="{top+plot_h/2}" font-family="sans-serif" font-size="14" transform="rotate(-90 18 {top+plot_h/2})">RMSE (pixels)</text>',
            "</svg>",
        ]
    )
    (output_dir / "intrinsics_validation_comparison.svg").write_text(
        "\n".join(svg), encoding="utf-8"
    )


def write_reports(
    output_dir: Path,
    models: list[Intrinsics],
    successful_paths: list[Path],
    failed_paths: list[Path],
    by_model: dict[str, list[ImageResult]],
    radial: dict[str, dict[str, dict[str, float | int]]],
) -> None:
    summary_rows: list[dict[str, str]] = []
    for model in models:
        results = by_model[model.key]
        stats = statistics([result.rmse for result in results])
        all_residuals = np.concatenate([result.residuals for result in results], axis=0)
        overall = float(np.sqrt(np.mean(np.sum(all_residuals**2, axis=1))))
        summary_rows.append(
            {
                "model": model.key,
                "successful_images": str(len(results)),
                "point_count": str(len(all_residuals)),
                "overall_point_rmse": f"{overall:.17g}",
                **{name: f"{value:.17g}" for name, value in stats.items()},
                **{
                    f"{zone}_rmse": f"{float(radial[model.key][zone]['rmse']):.17g}"
                    for zone in ("center", "middle", "edge")
                },
            }
        )
    with (output_dir / "validation_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_dir / "validation_per_image.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "detection_success", "opencv_rmse", "matlab_rmse", "matlab_minus_opencv"])
        for index, path in enumerate(successful_paths):
            opencv_rmse = by_model["opencv"][index].rmse
            matlab_rmse = by_model["matlab"][index].rmse
            writer.writerow([path.name, "true", f"{opencv_rmse:.17g}", f"{matlab_rmse:.17g}", f"{matlab_rmse-opencv_rmse:.17g}"])
        for path in failed_paths:
            writer.writerow([path.name, "false", "", "", ""])

    with (output_dir / "validation_radial_zones.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "zone", "point_count", "rmse", "mean_dx", "mean_dy"])
        for model in models:
            for zone in ("center", "middle", "edge"):
                values = radial[model.key][zone]
                writer.writerow([model.key, zone, values["point_count"], f"{float(values['rmse']):.17g}", f"{float(values['mean_dx']):.17g}", f"{float(values['mean_dy']):.17g}"])

    opencv_summary, matlab_summary = summary_rows
    delta_mean = float(matlab_summary["mean"]) - float(opencv_summary["mean"])
    better = "MATLAB" if delta_mean < 0 else "OpenCV"
    markdown = [
        "# 双内参独立验证报告",
        "",
        "## 验证约束",
        "",
        f"- 验证图：`{successful_paths[0].name}`—`{successful_paths[-1].name}`；检测成功 {len(successful_paths)} 张，失败 {len(failed_paths)} 张。",
        "- 每张图只检测一次棋盘格角点，两套内参共享完全相同的 cornerSubPix 结果。",
        "- 内参全程冻结；每套模型仅使用 solvePnP 求棋盘格位姿，再用 projectPoints 重投影。",
        "- 验证集不做高误差剔除，也不进入 calibrateCamera。",
        "",
        "```mermaid",
        "flowchart LR",
        '  A["验证图像"] --> B["共享角点检测 + cornerSubPix"]',
        '  B --> C["OpenCV 内参 solvePnP"]',
        '  B --> D["MATLAB 内参 solvePnP"]',
        '  C --> E["重投影与残差统计"]',
        '  D --> E',
        "```",
        "",
        "## 汇总结果",
        "",
        "![逐图RMSE对比](intrinsics_validation_comparison.svg)",
        "",
        "| 模型 | 逐图均值 | 中位数 | P95 | 最大值 | 全角点RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, row in zip(models, summary_rows, strict=True):
        markdown.append(f"| {model.label} | {float(row['mean']):.6f} | {float(row['median']):.6f} | {float(row['p95']):.6f} | {float(row['max']):.6f} | {float(row['overall_point_rmse']):.6f} |")
    markdown.extend(["", "## 逐图结果", "", "| 图像 | OpenCV | MATLAB | MATLAB-OpenCV |", "|---|---:|---:|---:|"])
    for index, path in enumerate(successful_paths):
        a = by_model["opencv"][index].rmse
        b = by_model["matlab"][index].rmse
        markdown.append(f"| {path.name} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    markdown.extend(["", "## 径向分区", "", "| 区域 | OpenCV | MATLAB | MATLAB-OpenCV |", "|---|---:|---:|---:|"])
    for zone in ("center", "middle", "edge"):
        a = float(radial["opencv"][zone]["rmse"])
        b = float(radial["matlab"][zone]["rmse"])
        markdown.append(f"| {zone} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    markdown.extend(["", "## 结论", "", f"本验证集上，{better} 内参的逐图平均RMSE较低；MATLAB-OpenCV差值为 `{delta_mean:+.6f} px`。该结论仅针对本验证集的棋盘格重投影表现。", ""])
    (output_dir / "intrinsics_validation_report.md").write_text("\n".join(markdown), encoding="utf-8")

    table_rows = "".join(
        f"<tr><td>{html.escape(path.name)}</td><td>{by_model['opencv'][index].rmse:.6f}</td><td>{by_model['matlab'][index].rmse:.6f}</td><td>{by_model['matlab'][index].rmse-by_model['opencv'][index].rmse:+.6f}</td></tr>"
        for index, path in enumerate(successful_paths)
    )
    summary_html = "".join(
        f"<tr><td>{model.label}</td><td>{float(row['mean']):.6f}</td><td>{float(row['median']):.6f}</td><td>{float(row['p95']):.6f}</td><td>{float(row['max']):.6f}</td><td>{float(row['overall_point_rmse']):.6f}</td></tr>"
        for model, row in zip(models, summary_rows, strict=True)
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>双内参独立验证报告</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1100px;margin:32px auto;color:#1f2937}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #d1d5db;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.flow{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}.box{{padding:12px 18px;background:#eff6ff;border:1px solid #93c5fd;border-radius:8px}}code{{background:#f3f4f6;padding:2px 4px}}img{{max-width:100%}}</style></head>
<body><h1>双内参独立验证报告</h1><p>同一组亚像素角点，冻结内参，仅通过 solvePnP 求每张棋盘格位姿。</p>
<div class="flow"><div class="box">验证图像</div>→<div class="box">共享角点检测 + cornerSubPix</div>→<div class="box">两套冻结内参 solvePnP</div>→<div class="box">重投影与残差统计</div></div>
<h2>汇总结果</h2><img src="intrinsics_validation_comparison.svg" alt="逐图误差对比"><table><tr><th>模型</th><th>均值</th><th>中位数</th><th>P95</th><th>最大值</th><th>全角点RMSE</th></tr>{summary_html}</table>
<h2>逐图结果</h2><table><tr><th>图像</th><th>OpenCV</th><th>MATLAB</th><th>MATLAB-OpenCV</th></tr>{table_rows}</table>
<h2>结论</h2><p>本验证集上，{better} 内参逐图平均RMSE较低；MATLAB-OpenCV为 <code>{delta_mean:+.6f} px</code>。验证图不参与内参拟合或异常剔除。</p></body></html>"""
    (output_dir / "intrinsics_validation_report.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        models = [
            load_intrinsics(args.opencv_yaml, "opencv", "OpenCV fixed k3"),
            load_intrinsics(args.matlab_yaml, "matlab", "MATLAB fixed k3"),
        ]
        reference = models[0]
        for model in models[1:]:
            if model.image_size != reference.image_size:
                raise ValueError("两份内参的图像尺寸不一致")
            if model.pattern_size != reference.pattern_size:
                raise ValueError("两份内参的棋盘格规格不一致")
            if model.square_size_mm != reference.square_size_mm:
                raise ValueError("两份内参的格子尺寸不一致")
        image_paths = sorted(args.image_dir.glob(args.image_pattern), key=lambda path: path.name.lower())
        if not image_paths:
            raise ValueError("验证目录中没有匹配的棋盘格图像")
        output_dir = args.output
        detected_dir = output_dir / "detected"
        detected_dir.mkdir(parents=True, exist_ok=True)
        object_points = create_object_points(*reference.pattern_size, reference.square_size_mm)
        corners_by_path: dict[Path, np.ndarray] = {}
        failed_paths: list[Path] = []
        for path in image_paths:
            image = read_image(path)
            if image is None:
                failed_paths.append(path)
                continue
            image_size = (image.shape[1], image.shape[0])
            if image_size != reference.image_size:
                raise ValueError(f"图像尺寸不一致：{path.name} 为 {image_size}，内参为 {reference.image_size}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found, corners = detect_corners(gray, reference.pattern_size)
            if not found or corners is None:
                failed_paths.append(path)
                continue
            corners_by_path[path] = corners
            overlay = image.copy()
            cv2.drawChessboardCorners(overlay, reference.pattern_size, corners, True)
            write_image(detected_dir / f"{path.stem}.png", overlay)
            print(f"[角点成功] {path.name}")
        successful_paths = list(corners_by_path)
        if not successful_paths:
            raise RuntimeError("没有验证图成功检测到棋盘格角点")
        by_model: dict[str, list[ImageResult]] = {}
        radial: dict[str, dict[str, dict[str, float | int]]] = {}
        for model in models:
            results = [evaluate_image(path, object_points, corners_by_path[path], model) for path in successful_paths]
            by_model[model.key] = results
            radial[model.key] = radial_statistics(results, corners_by_path, model.image_size)
            save_visuals(output_dir, model, results, corners_by_path)
        write_chart(output_dir, successful_paths, by_model)
        write_reports(output_dir, models, successful_paths, failed_paths, by_model, radial)
        (output_dir / "failed_images.txt").write_text("\n".join(path.name for path in failed_paths), encoding="utf-8")
        print(f"验证完成：成功 {len(successful_paths)} 张，失败 {len(failed_paths)} 张")
        print(f"结果目录：{output_dir.resolve()}")
    except (OSError, ValueError, RuntimeError, cv2.error, yaml.YAMLError) as exc:
        print(f"错误：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
