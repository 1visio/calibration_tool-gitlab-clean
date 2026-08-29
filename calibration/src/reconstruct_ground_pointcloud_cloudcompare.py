#!/usr/bin/env python3
"""在线激光三维截面重建结果中额外导出 CloudCompare 可读的 PLY 点云。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

import reconstruct_ground_pointcloud_interactive as core


PLY_COLUMNS = ["Xg_mm", "Yg_mm", "Zg_mm", "u", "v"]


def save_ground_pointcloud_ply(frame: pd.DataFrame, path: Path) -> None:
    """将地面坐标点保存为 ASCII PLY；坐标单位为毫米。"""
    missing = [column for column in PLY_COLUMNS if column not in frame.columns]
    if missing:
        raise core.ReconstructionError(
            f"PLY 导出缺少数据列：{', '.join(missing)}"
        )

    values = frame[PLY_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise core.ReconstructionError("PLY 导出数据包含 NaN 或无穷值")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format ascii 1.0",
        "comment coordinate_system ground",
        "comment units millimeter",
        "comment single_frame_laser_profile_no_surface_interpolation",
        f"element vertex {len(values)}",
        "property double x",
        "property double y",
        "property double z",
        "property double u",
        "property double v",
        "property double height_mm",
        "end_header",
    ]
    try:
        with path.open("w", encoding="ascii", newline="\n") as stream:
            stream.write("\n".join(header) + "\n")
            for xg, yg, zg, u, v in values:
                stream.write(
                    f"{xg:.9f} {yg:.9f} {zg:.9f} "
                    f"{u:.9f} {v:.9f} {zg:.9f}\n"
                )
    except OSError as exc:
        raise core.ReconstructionError(f"无法写入 PLY 点云：{path}：{exc}") from exc


def _append_export_metadata(statistics_path: Path, point_count: int) -> None:
    """在每帧统计文件中记录点云文件、单位和点数。"""
    try:
        with statistics_path.open("r", encoding="utf-8") as stream:
            statistics = yaml.safe_load(stream)
        if not isinstance(statistics, dict):
            raise core.ReconstructionError(f"统计文件格式无效：{statistics_path}")
        statistics["ground_pointcloud_export"] = {
            "file": "points_ground.ply",
            "format": "PLY ASCII 1.0",
            "coordinate_system": "ground",
            "coordinate_unit": "mm",
            "point_count": point_count,
            "surface_interpolation": False,
        }
        with statistics_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(statistics, stream, allow_unicode=True, sort_keys=False)
    except OSError as exc:
        raise core.ReconstructionError(
            f"无法更新点云导出统计：{statistics_path}：{exc}"
        ) from exc


def _make_cloudcompare_process_frame(
    original_process_frame: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    def process_frame_with_ply(
        image_path: Path,
        output_dir: Path,
        calibration: core.Calibration,
        extraction: core.ExtractionParams,
        reconstruction: core.ReconstructionParams,
        obstacle_fit_params: core.ObstacleFitParams,
        show_3d: bool,
    ) -> dict[str, Any]:
        statistics = original_process_frame(
            image_path,
            output_dir,
            calibration,
            extraction,
            reconstruction,
            obstacle_fit_params,
            show_3d,
        )
        frame_dir = output_dir / image_path.stem
        csv_path = frame_dir / "points_ground.csv"
        try:
            frame = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise core.ReconstructionError(
                f"无法读取待导出的地面点 CSV：{csv_path}：{exc}"
            ) from exc

        if len(frame) != statistics["valid_point_count"]:
            raise core.ReconstructionError(
                "PLY 导出前点数校验失败："
                f"CSV={len(frame)}，有效点统计={statistics['valid_point_count']}"
            )
        save_ground_pointcloud_ply(frame, frame_dir / "points_ground.ply")
        _append_export_metadata(frame_dir / "statistics.yaml", len(frame))
        statistics["ground_pointcloud_file"] = "points_ground.ply"
        statistics["ground_pointcloud_point_count"] = len(frame)
        return statistics

    return process_frame_with_ply


def main(argv: list[str] | None = None) -> int:
    """复用已验证的重建入口，仅为每帧追加 PLY 导出。"""
    original_process_frame = core.process_frame
    core.process_frame = _make_cloudcompare_process_frame(original_process_frame)
    try:
        return core.main(argv)
    finally:
        core.process_frame = original_process_frame


if __name__ == "__main__":
    raise SystemExit(main())
