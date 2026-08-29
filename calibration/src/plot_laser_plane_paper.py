#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制论文风格的激光面标定结果图。

输出：
1. train_feature_points_plane.png / .pdf
   拟合集代表姿态的三维特征线与拟合光平面
2. train_point_plane_distance.png / .pdf
   按图像列坐标排序并轻度平滑后的点面距离曲线
3. train_laser_plane_figure.png
   左右组合图，包含 (a)、(b) 与中文图注

默认代表姿态：1、4、8、11、15、17
默认使用训练集中 used_for_fit=True 且 status=accepted_final 的点。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


REQUIRED_COLUMNS = {
    "image_id",
    "split",
    "column_px",
    "x_mm",
    "y_mm",
    "z_mm",
    "signed_distance_mm",
    "abs_distance_mm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="绘制简洁的论文风格激光面标定结果图。"
    )
    parser.add_argument(
        "--plane-yaml",
        required=True,
        type=Path,
        help="激光平面 YAML 文件路径。",
    )
    parser.add_argument(
        "--points-csv",
        required=True,
        type=Path,
        help="feature_point_distances.csv 文件路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_plane_figure"),
        help="输出目录，默认：outputs/paper_plane_figure",
    )
    parser.add_argument(
        "--poses",
        nargs="+",
        type=int,
        default=[1, 4, 8, 11, 15, 17],
        help="要绘制的拟合集姿态编号，默认：1 4 8 11 15 17",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=51,
        help="距离曲线 Savitzky-Golay 平滑窗口，默认：51；0 或 1 表示不平滑。",
    )
    parser.add_argument(
        "--smooth-3d-window",
        type=int,
        default=21,
        help="三维特征线轻度平滑窗口，默认：21；0 或 1 表示不平滑。",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay 多项式阶数，默认：2。",
    )
    parser.add_argument(
        "--distance",
        choices=["absolute", "signed"],
        default="absolute",
        help="右图距离类型，默认 absolute；可选 signed。",
    )
    parser.add_argument(
        "--max-3d-points",
        type=int,
        default=260,
        help="每组姿态在三维图中的最大绘制点数，默认：260。",
    )
    parser.add_argument(
        "--plane-grid-size",
        type=int,
        default=30,
        help="拟合平面的网格密度，默认：30。",
    )
    parser.add_argument(
        "--max-column-gap",
        type=int,
        default=120,
        help="相邻点列坐标差超过该值时断开曲线，默认：120 pixel。",
    )
    parser.add_argument(
        "--min-segment-points",
        type=int,
        default=12,
        help="连续片段的最小点数，过短片段不绘制，默认：12。",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="右图纵轴上限，单位 mm；默认自动确定。",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="在右图中同时绘制较淡的原始距离曲线。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG 输出分辨率，默认：300 dpi。",
    )
    parser.add_argument(
        "--no-combine",
        action="store_true",
        help="不生成左右组合 PNG。",
    )
    return parser.parse_args()


def configure_matplotlib() -> str:
    """设置适合中文论文图的字体与矢量字体嵌入。"""
    candidate_families = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]

    chosen_family = "DejaVu Sans"
    for family in candidate_families:
        try:
            path = font_manager.findfont(
                FontProperties(family=family),
                fallback_to_default=False,
            )
            if path and Path(path).exists():
                chosen_family = family
                break
        except Exception:
            continue

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chosen_family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )
    return chosen_family


def read_plane(path: Path) -> tuple[float, float, float, float]:
    if not path.exists():
        raise FileNotFoundError(f"找不到激光平面文件：{path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict) or "plane" not in data:
        raise ValueError("YAML 中缺少 plane 字段。")

    plane = data["plane"]
    try:
        a = float(plane["a"])
        b = float(plane["b"])
        c = float(plane["c"])
        d = float(plane["d"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("plane 字段中必须包含可转换为浮点数的 a、b、c、d。") from exc

    norm = math.sqrt(a * a + b * b + c * c)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("激光平面法向量无效。")

    # 统一归一化，确保点面距离与平面显示一致
    return a / norm, b / norm, c / norm, d / norm


def read_and_filter_points(
    path: Path,
    poses: Sequence[int],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到特征点文件：{path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "CSV 缺少必要列：" + ", ".join(sorted(missing))
        )

    mask = df["split"].astype(str).str.lower().eq("train")

    if "used_for_fit" in df.columns:
        mask &= df["used_for_fit"].fillna(False).astype(bool)

    if "status" in df.columns:
        mask &= df["status"].astype(str).eq("accepted_final")

    mask &= df["image_id"].isin(poses)

    selected = df.loc[mask].copy()
    if selected.empty:
        raise ValueError("筛选后没有可绘制的拟合集数据。请检查姿态编号和筛选字段。")

    selected["image_id"] = selected["image_id"].astype(int)
    selected = selected.sort_values(["image_id", "column_px"]).reset_index(drop=True)

    existing = sorted(selected["image_id"].unique().tolist())
    missing_poses = [pose for pose in poses if pose not in existing]
    if missing_poses:
        print(
            "警告：以下姿态在筛选后的拟合数据中不存在，将跳过："
            + ", ".join(map(str, missing_poses)),
            file=sys.stderr,
        )

    return selected


def valid_odd_window(length: int, requested: int, polyorder: int) -> int:
    """将窗口调整为不超过数据长度的合法奇数。"""
    if requested <= 1 or length <= polyorder + 2:
        return 1

    window = min(int(requested), int(length))
    if window % 2 == 0:
        window -= 1

    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1

    if window < minimum:
        return 1
    return window


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """SciPy 不可用时的居中移动平均回退。"""
    if window <= 1:
        return values.copy()

    series = pd.Series(values)
    return (
        series.rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def smooth_values(
    values: Iterable[float],
    requested_window: int,
    polyorder: int,
) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return array

    window = valid_odd_window(array.size, requested_window, polyorder)
    if window <= 1:
        return array.copy()

    if savgol_filter is not None:
        return savgol_filter(
            array,
            window_length=window,
            polyorder=min(polyorder, window - 2),
            mode="interp",
        )

    return moving_average(array, window)


def downsample_group(group: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(group) <= max_points:
        return group

    indices = np.linspace(0, len(group) - 1, max_points)
    indices = np.unique(np.round(indices).astype(int))
    return group.iloc[indices]


def split_by_column_gap(
    group: pd.DataFrame,
    max_gap: int,
    min_points: int,
) -> list[pd.DataFrame]:
    """
    按图像列坐标的不连续位置切分曲线。

    这样可避免将两个相隔很远的激光条纹片段强行连成一条斜线，
    也可避免 Savitzky-Golay 滤波跨越大缺口产生不真实的平滑结果。
    """
    if group.empty:
        return []

    ordered = group.sort_values("column_px").reset_index(drop=True)
    columns = ordered["column_px"].to_numpy(dtype=float)

    if max_gap <= 0 or len(ordered) <= 1:
        return [ordered] if len(ordered) >= min_points else []

    breaks = np.where(np.diff(columns) > max_gap)[0] + 1
    index_groups = np.split(np.arange(len(ordered)), breaks)

    segments: list[pd.DataFrame] = []
    for indices in index_groups:
        if len(indices) >= max(2, min_points):
            segments.append(ordered.iloc[indices].copy())
    return segments


def add_plane_surface(
    ax,
    plane: tuple[float, float, float, float],
    points: pd.DataFrame,
    grid_size: int,
) -> None:
    """根据法向量最大分量选择数值稳定的平面参数化方式。"""
    a, b, c, d = plane
    grid_size = max(8, int(grid_size))

    x = points["x_mm"].to_numpy(dtype=float)
    y = points["y_mm"].to_numpy(dtype=float)
    z = points["z_mm"].to_numpy(dtype=float)

    def expanded_limits(values: np.ndarray, ratio: float = 0.08) -> tuple[float, float]:
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
        span = max(high - low, 1.0)
        return low - ratio * span, high + ratio * span

    xlim = expanded_limits(x)
    ylim = expanded_limits(y)
    zlim = expanded_limits(z)

    dominant = int(np.argmax(np.abs([a, b, c])))

    if dominant == 2:  # 解 z
        xx, yy = np.meshgrid(
            np.linspace(*xlim, grid_size),
            np.linspace(*ylim, grid_size),
        )
        zz = -(a * xx + b * yy + d) / c
        ax.plot_surface(
            xx,
            yy,
            zz,
            alpha=0.20,
            linewidth=0,
            antialiased=True,
            shade=True,
        )
    elif dominant == 1:  # 解 y
        xx, zz = np.meshgrid(
            np.linspace(*xlim, grid_size),
            np.linspace(*zlim, grid_size),
        )
        yy = -(a * xx + c * zz + d) / b
        ax.plot_surface(
            xx,
            yy,
            zz,
            alpha=0.20,
            linewidth=0,
            antialiased=True,
            shade=True,
        )
    else:  # 解 x
        yy, zz = np.meshgrid(
            np.linspace(*ylim, grid_size),
            np.linspace(*zlim, grid_size),
        )
        xx = -(b * yy + c * zz + d) / a
        ax.plot_surface(
            xx,
            yy,
            zz,
            alpha=0.20,
            linewidth=0,
            antialiased=True,
            shade=True,
        )


def draw_3d_figure(
    points: pd.DataFrame,
    plane: tuple[float, float, float, float],
    poses: Sequence[int],
    output_dir: Path,
    smooth_window: int,
    polyorder: int,
    max_points: int,
    grid_size: int,
    max_column_gap: int,
    min_segment_points: int,
    dpi: int,
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_subplot(111, projection="3d")

    add_plane_surface(ax, plane, points, grid_size)

    for pose in poses:
        group = points.loc[points["image_id"].eq(pose)].sort_values("column_px")
        if group.empty:
            continue

        segments = split_by_column_gap(
            group,
            max_gap=max_column_gap,
            min_points=min_segment_points,
        )
        pose_color = None

        for segment in segments:
            segment = downsample_group(segment, max_points)
            x = smooth_values(segment["x_mm"], smooth_window, polyorder)
            y = smooth_values(segment["y_mm"], smooth_window, polyorder)
            z = smooth_values(segment["z_mm"], smooth_window, polyorder)

            line = ax.plot(
                x,
                y,
                z,
                linewidth=1.55,
                color=pose_color,
            )[0]
            if pose_color is None:
                pose_color = line.get_color()

    ax.set_title("特征点与拟合光平面", pad=12)
    ax.set_xlabel("X (mm)", labelpad=7)
    ax.set_ylabel("Y (mm)", labelpad=7)
    ax.set_zlabel("Z (mm)", labelpad=7)

    # 固定为更接近论文示例的观察方向
    ax.view_init(elev=24, azim=-128)
    ax.set_box_aspect((1.55, 0.75, 1.15))
    ax.grid(True, alpha=0.28)

    fig.tight_layout()

    png_path = output_dir / "train_feature_points_plane.png"
    pdf_path = output_dir / "train_feature_points_plane.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def draw_distance_figure(
    points: pd.DataFrame,
    poses: Sequence[int],
    output_dir: Path,
    smooth_window: int,
    polyorder: int,
    distance_mode: str,
    show_raw: bool,
    y_max: float | None,
    max_column_gap: int,
    min_segment_points: int,
    dpi: int,
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(7.6, 5.6))
    ax = fig.add_axes([0.12, 0.13, 0.82, 0.78])

    distance_column = (
        "abs_distance_mm"
        if distance_mode == "absolute"
        else "signed_distance_mm"
    )

    all_smoothed: list[np.ndarray] = []

    for pose in poses:
        group = points.loc[points["image_id"].eq(pose)].sort_values("column_px")
        if group.empty:
            continue

        # 同一列若存在多个点，取中位数，避免重复列导致竖直折线
        curve = (
            group.groupby("column_px", as_index=False)[distance_column]
            .median()
            .sort_values("column_px")
        )

        segments = split_by_column_gap(
            curve,
            max_gap=max_column_gap,
            min_points=min_segment_points,
        )

        pose_color = None
        label_pending = True

        for segment in segments:
            x = segment["column_px"].to_numpy(dtype=float)
            raw = segment[distance_column].to_numpy(dtype=float)
            smooth = smooth_values(raw, smooth_window, polyorder)
            all_smoothed.append(smooth)

            label = f"{pose:03d}" if label_pending else None

            if show_raw:
                raw_line = ax.plot(
                    x,
                    raw,
                    linewidth=0.65,
                    alpha=0.18,
                    color=pose_color,
                )[0]
                if pose_color is None:
                    pose_color = raw_line.get_color()
                ax.plot(
                    x,
                    smooth,
                    linewidth=1.55,
                    label=label,
                    color=pose_color,
                )
            else:
                line = ax.plot(
                    x,
                    smooth,
                    linewidth=1.55,
                    label=label,
                    color=pose_color,
                )[0]
                if pose_color is None:
                    pose_color = line.get_color()

            label_pending = False

    ax.set_title("特征点到光平面的距离", pad=10)
    ax.set_xlabel("图像列坐标 (pixel)")
    ylabel = "绝对距离 (mm)" if distance_mode == "absolute" else "有符号距离 (mm)"
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    if distance_mode == "absolute":
        ax.set_ylim(bottom=0.0)

    if y_max is not None:
        if distance_mode == "absolute":
            ax.set_ylim(0.0, y_max)
        else:
            current_bottom, _ = ax.get_ylim()
            ax.set_ylim(current_bottom, y_max)
    elif all_smoothed and distance_mode == "absolute":
        combined = np.concatenate(all_smoothed)
        finite = combined[np.isfinite(combined)]
        if finite.size:
            robust_top = float(np.percentile(finite, 99.7))
            if robust_top > 0:
                ax.set_ylim(0.0, robust_top * 1.08)

    ax.legend(
        title="姿态",
        ncol=2,
        frameon=True,
        loc="upper right",
        borderaxespad=0.5,
        handlelength=2.2,
    )

    png_path = output_dir / "train_point_plane_distance.png"
    pdf_path = output_dir / "train_point_plane_distance.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def find_font_path(family: str) -> Path | None:
    candidates = [
        family,
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ]
    for item in candidates:
        try:
            path = Path(
                font_manager.findfont(
                    FontProperties(family=item),
                    fallback_to_default=False,
                )
            )
            if path.exists():
                return path
        except Exception:
            continue
    return None


def combine_figures(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    font_family: str,
) -> None:
    if Image is None or ImageDraw is None or ImageFont is None:
        print(
            "提示：未安装 Pillow，已跳过组合图。可执行：python -m pip install pillow",
            file=sys.stderr,
        )
        return

    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    target_height = max(left.height, right.height)

    def resize_to_height(image: Image.Image, height: int) -> Image.Image:
        if image.height == height:
            return image
        width = int(round(image.width * height / image.height))
        return image.resize((width, height), Image.Resampling.LANCZOS)

    left = resize_to_height(left, target_height)
    right = resize_to_height(right, target_height)

    outer_margin = max(30, target_height // 30)
    gap = max(18, target_height // 45)
    label_band = max(165, target_height // 5)

    canvas_width = left.width + right.width + gap + 2 * outer_margin
    canvas_height = target_height + label_band + outer_margin

    # 使用原图左上角像素作为背景，不强制指定图形颜色
    background = left.getpixel((0, 0))
    canvas = Image.new("RGB", (canvas_width, canvas_height), background)

    left_x = outer_margin
    right_x = outer_margin + left.width + gap
    top_y = outer_margin // 2

    canvas.paste(left, (left_x, top_y))
    canvas.paste(right, (right_x, top_y))

    draw = ImageDraw.Draw(canvas)
    font_path = find_font_path(font_family)

    if font_path is not None:
        sublabel_font = ImageFont.truetype(
            str(font_path),
            size=max(28, target_height // 25),
        )
        caption_font = ImageFont.truetype(
            str(font_path),
            size=max(24, target_height // 31),
        )
    else:
        sublabel_font = ImageFont.load_default()
        caption_font = ImageFont.load_default()

    label_y = top_y + target_height + max(15, label_band // 10)
    left_center = left_x + left.width // 2
    right_center = right_x + right.width // 2

    draw.text(
        (left_center, label_y),
        "(a)",
        font=sublabel_font,
        anchor="mm",
        fill=0,
    )
    draw.text(
        (right_center, label_y),
        "(b)",
        font=sublabel_font,
        anchor="mm",
        fill=0,
    )

    caption = "拟合集代表姿态的特征点与拟合光平面及点面距离分布"
    caption_y = label_y + max(55, label_band // 3)
    draw.text(
        (canvas_width // 2, caption_y),
        caption,
        font=caption_font,
        anchor="mm",
        fill=0,
    )

    canvas.save(output_path, dpi=(300, 300))


def save_selection_summary(
    points: pd.DataFrame,
    poses: Sequence[int],
    output_dir: Path,
) -> Path:
    rows = []
    for pose in poses:
        group = points.loc[points["image_id"].eq(pose)]
        if group.empty:
            continue
        abs_distance = group["abs_distance_mm"].to_numpy(dtype=float)
        rows.append(
            {
                "image_id": pose,
                "point_count": len(group),
                "column_min_px": int(group["column_px"].min()),
                "column_max_px": int(group["column_px"].max()),
                "mean_abs_distance_mm": float(np.mean(abs_distance)),
                "rmse_mm": float(np.sqrt(np.mean(abs_distance ** 2))),
                "p95_mm": float(np.percentile(abs_distance, 95)),
                "max_mm": float(np.max(abs_distance)),
            }
        )

    summary = pd.DataFrame(rows)
    path = output_dir / "selected_pose_summary.csv"
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> int:
    args = parse_args()

    if args.smooth_window < 0 or args.smooth_3d_window < 0:
        raise ValueError("平滑窗口不能为负数。")
    if args.polyorder < 0:
        raise ValueError("polyorder 不能为负数。")
    if args.dpi < 72:
        raise ValueError("dpi 建议不低于 72。")

    font_family = configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plane = read_plane(args.plane_yaml)
    points = read_and_filter_points(args.points_csv, args.poses)

    available_poses = [
        pose for pose in args.poses
        if pose in set(points["image_id"].unique())
    ]

    left_png, left_pdf = draw_3d_figure(
        points=points,
        plane=plane,
        poses=available_poses,
        output_dir=args.output_dir,
        smooth_window=args.smooth_3d_window,
        polyorder=args.polyorder,
        max_points=args.max_3d_points,
        grid_size=args.plane_grid_size,
        max_column_gap=args.max_column_gap,
        min_segment_points=args.min_segment_points,
        dpi=args.dpi,
    )

    right_png, right_pdf = draw_distance_figure(
        points=points,
        poses=available_poses,
        output_dir=args.output_dir,
        smooth_window=args.smooth_window,
        polyorder=args.polyorder,
        distance_mode=args.distance,
        show_raw=args.show_raw,
        y_max=args.y_max,
        max_column_gap=args.max_column_gap,
        min_segment_points=args.min_segment_points,
        dpi=args.dpi,
    )

    combined_path = args.output_dir / "train_laser_plane_figure.png"
    if not args.no_combine:
        combine_figures(
            left_path=left_png,
            right_path=right_png,
            output_path=combined_path,
            font_family=font_family,
        )

    summary_path = save_selection_summary(
        points=points,
        poses=available_poses,
        output_dir=args.output_dir,
    )

    a, b, c, d = plane
    print("绘图完成。")
    print(f"归一化光平面：{a:+.9f} x {b:+.9f} y {c:+.9f} z {d:+.9f} = 0")
    print("绘制姿态：" + ", ".join(f"{pose:03d}" for pose in available_poses))
    print(f"三维图 PNG：{left_png}")
    print(f"三维图 PDF：{left_pdf}")
    print(f"距离图 PNG：{right_png}")
    print(f"距离图 PDF：{right_pdf}")
    if not args.no_combine and combined_path.exists():
        print(f"组合图 PNG：{combined_path}")
    print(f"姿态指标：{summary_path}")

    if savgol_filter is None:
        print(
            "提示：当前环境未安装 SciPy，本次使用居中移动平均作为平滑回退。"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
