#!/usr/bin/env python3
"""验证 vertical Steger 路径与旋转 90° 后 horizontal 路径的等价性。

对原图执行 ``scan_axis=row``，再把同一图像旋转 90° 后执行
``scan_axis=column``。第二组中心点映射回原图坐标后，按原图行号 ``v``
配对，比较亚像素法向坐标和二维点距。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_IMAGE = (
    ROOT
    / "projects"
    / "daheng"
    / "data"
    / "obs"
    / "test_350"
    / "fit"
    / "laser 003.tif"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "projects" / "daheng" / "analysis" / "rotation_equivalence_laser_003"
)
ROTATIONS = ("clockwise", "counterclockwise")


@dataclass(frozen=True, slots=True)
class Comparison:
    rows: list[dict[str, float | int | str]]
    only_native_v: list[int]
    only_rotated_v: list[int]


def load_grayscale(path: Path) -> np.ndarray:
    """读取单通道图像，同时支持 Windows 非 ASCII 路径。"""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"图像不存在：{resolved}")
    image = cv2.imdecode(
        np.fromfile(resolved, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if image is None:
        raise ValueError(f"无法解码图像：{resolved}")
    if image.ndim != 2:
        raise ValueError(f"只支持单通道灰度图，实际 shape={image.shape}")
    return np.ascontiguousarray(image)


def load_steger(calibration_src: Path) -> tuple[Any, dict[str, Any]]:
    """加载正式共享 realtime_steger 及其冻结配置。"""

    source = calibration_src.expanduser().resolve()
    module_path = source / "realtime_steger.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"共享 realtime_steger.py 不存在：{module_path}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module("realtime_steger")
    loaded_path = Path(module.__file__).resolve()
    if loaded_path != module_path:
        raise RuntimeError(
            f"realtime_steger 加载路径错误：期望 {module_path}，实际 {loaded_path}"
        )
    return module, dict(module.merge_options(module.load_steger_options()))


def rotate_image(image: np.ndarray, rotation: str) -> np.ndarray:
    if rotation == "clockwise":
        return np.ascontiguousarray(np.rot90(image, k=-1))
    if rotation == "counterclockwise":
        return np.ascontiguousarray(np.rot90(image, k=1))
    raise ValueError(f"未知旋转方向：{rotation}")


def restore_rotated_points(
    points: np.ndarray,
    original_shape: tuple[int, int],
    rotation: str,
) -> np.ndarray:
    """把旋转图上的 ``(u_r, v_r)`` 映射回原图 ``(u, v)``。"""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"points 必须是 (N,2)，实际 {values.shape}")
    height, width = original_shape
    if rotation == "clockwise":
        # u_r = H - 1 - v, v_r = u
        restored = np.column_stack((values[:, 1], height - 1.0 - values[:, 0]))
    elif rotation == "counterclockwise":
        # u_r = v, v_r = W - 1 - u
        restored = np.column_stack((width - 1.0 - values[:, 1], values[:, 0]))
    else:
        raise ValueError(f"未知旋转方向：{rotation}")
    return np.ascontiguousarray(restored)


def _index_by_scanline(points: np.ndarray) -> dict[int, tuple[float, float]]:
    indexed: dict[int, tuple[float, float]] = {}
    for u, v in np.asarray(points, dtype=np.float64):
        key = int(round(float(v)))
        if abs(float(v) - key) > 1.0e-6:
            raise ValueError(f"扫描轴 v 应为整数，收到 {v}")
        if key in indexed:
            raise ValueError(f"同一扫描行 v={key} 出现多个中心点")
        indexed[key] = (float(u), float(v))
    return indexed


def compare_points(
    native: np.ndarray,
    restored: np.ndarray,
    rotation: str,
) -> Comparison:
    """按原图整数行 ``v`` 配对，不按数组下标配对。"""

    native_by_v = _index_by_scanline(native)
    rotated_by_v = _index_by_scanline(restored)
    native_keys = set(native_by_v)
    rotated_keys = set(rotated_by_v)
    rows: list[dict[str, float | int | str]] = []
    for v in sorted(native_keys & rotated_keys):
        native_u, native_v = native_by_v[v]
        rotated_u, rotated_v = rotated_by_v[v]
        delta_u = rotated_u - native_u
        delta_v = rotated_v - native_v
        rows.append(
            {
                "rotation": rotation,
                "v": v,
                "native_u_px": native_u,
                "rotated_back_u_px": rotated_u,
                "delta_u_px": delta_u,
                "delta_v_px": delta_v,
                "delta_p_px": math.hypot(delta_u, delta_v),
            }
        )
    return Comparison(
        rows=rows,
        only_native_v=sorted(native_keys - rotated_keys),
        only_rotated_v=sorted(rotated_keys - native_keys),
    )


def bin_low_frequency(
    rows: Iterable[Mapping[str, float | int | str]],
    bin_width_px: int,
) -> list[dict[str, float | int | str]]:
    """按原图 v 固定宽度分箱，提取低频 Δu 趋势。"""

    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        v = int(row["v"])
        start = (v // bin_width_px) * bin_width_px
        grouped.setdefault((str(row["rotation"]), start), []).append(
            float(row["delta_u_px"])
        )
    result: list[dict[str, float | int | str]] = []
    for (rotation, start), values in sorted(grouped.items()):
        data = np.asarray(values, dtype=np.float64)
        result.append(
            {
                "rotation": rotation,
                "v_start": start,
                "v_end": start + bin_width_px,
                "count": len(data),
                "mean_delta_u_px": float(np.mean(data)),
                "median_delta_u_px": float(np.median(data)),
                "rms_delta_u_px": float(np.sqrt(np.mean(np.square(data)))),
                "max_abs_delta_u_px": float(np.max(np.abs(data))),
            }
        )
    return result


def summarize(
    comparison: Comparison,
    bins: list[dict[str, float | int | str]],
    *,
    native_count: int,
    rotated_count: int,
    tolerance_px: float,
    low_frequency_warning_px: float,
) -> dict[str, Any]:
    delta_u = np.asarray(
        [float(row["delta_u_px"]) for row in comparison.rows], dtype=np.float64
    )
    delta_p = np.asarray(
        [float(row["delta_p_px"]) for row in comparison.rows], dtype=np.float64
    )
    bin_means = np.asarray(
        [float(row["mean_delta_u_px"]) for row in bins], dtype=np.float64
    )
    max_abs = float(np.max(delta_p)) if delta_p.size else float("nan")
    low_frequency_max_abs = (
        float(np.max(np.abs(bin_means))) if bin_means.size else float("nan")
    )
    no_missing = not comparison.only_native_v and not comparison.only_rotated_v
    coordinate_pass = bool(delta_p.size and max_abs <= tolerance_px and no_missing)
    low_frequency_warning = bool(
        bin_means.size and low_frequency_max_abs >= low_frequency_warning_px
    )
    return {
        "native_point_count": native_count,
        "rotated_point_count": rotated_count,
        "common_scanline_count": len(comparison.rows),
        "only_native_count": len(comparison.only_native_v),
        "only_rotated_count": len(comparison.only_rotated_v),
        "only_native_v": comparison.only_native_v,
        "only_rotated_v": comparison.only_rotated_v,
        "mean_delta_u_px": float(np.mean(delta_u)) if delta_u.size else None,
        "median_delta_u_px": float(np.median(delta_u)) if delta_u.size else None,
        "rms_delta_p_px": (
            float(np.sqrt(np.mean(np.square(delta_p)))) if delta_p.size else None
        ),
        "p95_abs_delta_p_px": (
            float(np.percentile(delta_p, 95)) if delta_p.size else None
        ),
        "p99_abs_delta_p_px": (
            float(np.percentile(delta_p, 99)) if delta_p.size else None
        ),
        "max_abs_delta_p_px": max_abs if delta_p.size else None,
        "low_frequency_max_abs_bin_mean_px": (
            low_frequency_max_abs if bin_means.size else None
        ),
        "low_frequency_bin_mean_peak_to_peak_px": (
            float(np.ptp(bin_means)) if bin_means.size else None
        ),
        "coordinate_equivalence_pass": coordinate_pass,
        "low_frequency_warning": low_frequency_warning,
        "verdict": "PASS" if coordinate_pass and not low_frequency_warning else "FAIL",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_px(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.9f}"


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# 90° 旋转等价测试",
        "",
        f"- 图像：`{payload['image']['path']}`",
        f"- 尺寸：`{payload['image']['width']}×{payload['image']['height']}`，"
        f"`{payload['image']['dtype']}`",
        "- 对照：原图 `scan_axis=row`（vertical pipeline）。",
        "- 候选：旋转 90° 后 `scan_axis=column`（horizontal pipeline），再映射回原图坐标。",
        f"- 逐点阈值：`{payload['thresholds']['point_tolerance_px']:.6f} px`；"
        f"低频告警阈值：`{payload['thresholds']['low_frequency_warning_px']:.6f} px`。",
        f"- 总结论：**{payload['verdict']}**。",
        "",
        "| rotation | native | rotated | common | only native | only rotated | max abs Δp px | P95 abs Δp px | low-freq max abs bin mean px | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rotation, result in payload["rotations"].items():
        lines.append(
            f"| {rotation} | {result['native_point_count']} | "
            f"{result['rotated_point_count']} | {result['common_scanline_count']} | "
            f"{result['only_native_count']} | {result['only_rotated_count']} | "
            f"{_format_px(result['max_abs_delta_p_px'])} | "
            f"{_format_px(result['p95_abs_delta_p_px'])} | "
            f"{_format_px(result['low_frequency_max_abs_bin_mean_px'])} | "
            f"{result['verdict']} |"
        )
    lines.extend(
        [
            "",
            "低频量使用固定 v 宽度分箱后的 `mean(Δu)`；逐点明细见 "
            "`point_deltas.csv`，分箱明细见 `low_frequency_bins.csv`。",
            "当前 `row` 路径会先转置原图，再复用 columnwise 核心；因此本测试主要验证 "
            "row adapter、旋转/坐标恢复以及正式入口的端到端方向等价性。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_test(args: argparse.Namespace) -> dict[str, Any]:
    image_path = args.image.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_grayscale(image_path)
    steger, base_options = load_steger(args.calibration_src)

    native_options = {**base_options, "scan_axis": "row"}
    native_extraction = steger.extract_steger(image, native_options)
    native_points = native_extraction.pixels
    rotations = ROTATIONS if args.rotation == "both" else (args.rotation,)
    point_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    rotation_summaries: dict[str, Any] = {}

    for rotation in rotations:
        rotated_image = rotate_image(image, rotation)
        rotated_extraction = steger.extract_steger(
            rotated_image,
            {**base_options, "scan_axis": "column"},
        )
        restored = restore_rotated_points(
            rotated_extraction.pixels,
            image.shape,
            rotation,
        )
        comparison = compare_points(native_points, restored, rotation)
        current_bins = bin_low_frequency(comparison.rows, args.low_frequency_bin_px)
        point_rows.extend(comparison.rows)
        bin_rows.extend(current_bins)
        result = summarize(
            comparison,
            current_bins,
            native_count=len(native_points),
            rotated_count=len(rotated_extraction.pixels),
            tolerance_px=args.tolerance_px,
            low_frequency_warning_px=args.low_frequency_warning_px,
        )
        result["rotated_search_region_px"] = [
            rotated_extraction.metadata.get("final_search_region_start_px"),
            rotated_extraction.metadata.get("final_search_region_end_px"),
        ]
        rotation_summaries[rotation] = result

    payload = {
        "test": "90_degree_rotation_equivalence",
        "image": {
            "path": str(image_path),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "dtype": str(image.dtype),
        },
        "pipeline": {
            "module": str(Path(steger.__file__).resolve()),
            "native_scan_axis": "row",
            "rotated_scan_axis": "column",
            "options": base_options,
            "native_search_region_px": [
                native_extraction.metadata.get("final_search_region_start_px"),
                native_extraction.metadata.get("final_search_region_end_px"),
            ],
        },
        "thresholds": {
            "point_tolerance_px": args.tolerance_px,
            "low_frequency_warning_px": args.low_frequency_warning_px,
            "low_frequency_bin_width_px": args.low_frequency_bin_px,
        },
        "rotations": rotation_summaries,
        "verdict": (
            "PASS"
            if rotation_summaries
            and all(result["verdict"] == "PASS" for result in rotation_summaries.values())
            else "FAIL"
        ),
    }

    _write_csv(
        output_dir / "point_deltas.csv",
        point_rows,
        [
            "rotation",
            "v",
            "native_u_px",
            "rotated_back_u_px",
            "delta_u_px",
            "delta_v_px",
            "delta_p_px",
        ],
    )
    _write_csv(
        output_dir / "low_frequency_bins.csv",
        bin_rows,
        [
            "rotation",
            "v_start",
            "v_end",
            "count",
            "mean_delta_u_px",
            "median_delta_u_px",
            "rms_delta_u_px",
            "max_abs_delta_u_px",
        ],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "report.md", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--calibration-src",
        type=Path,
        default=WORKSPACE / "calibration" / "src",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--rotation",
        choices=(*ROTATIONS, "both"),
        default="both",
        help="旋转方向；both 可排除镜像方向偶然性",
    )
    parser.add_argument("--tolerance-px", type=float, default=0.02)
    parser.add_argument("--low-frequency-warning-px", type=float, default=0.05)
    parser.add_argument("--low-frequency-bin-px", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tolerance_px < 0.0:
        raise SystemExit("--tolerance-px 不能为负数")
    if args.low_frequency_warning_px < 0.0:
        raise SystemExit("--low-frequency-warning-px 不能为负数")
    if args.low_frequency_bin_px < 1:
        raise SystemExit("--low-frequency-bin-px 必须为正整数")
    result = run_test(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n输出目录：{args.output_dir.expanduser().resolve()}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
