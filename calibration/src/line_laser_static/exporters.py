from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .models import PointCloud


RESULT_FIELDS = ("x_mm", "y_mm", "z_mm", "intensity", "confidence", "valid")


def write_csv(path: str | Path, cloud: PointCloud) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(RESULT_FIELDS)
        for index in range(cloud.size):
            writer.writerow(
                [
                    _format_float(cloud.x_mm[index]),
                    _format_float(cloud.y_mm[index]),
                    _format_float(cloud.z_mm[index]),
                    _format_float(cloud.intensity[index]),
                    _format_float(cloud.confidence[index]),
                    int(cloud.valid[index]),
                ]
            )
    return output_path


def write_ply(path: str | Path, cloud: PointCloud) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="\n", encoding="ascii") as stream:
        stream.write("ply\n")
        stream.write("format ascii 1.0\n")
        stream.write(f"element vertex {cloud.size}\n")
        stream.write("property double x\nproperty double y\nproperty double z\n")
        stream.write("property double intensity\nproperty double confidence\n")
        stream.write("property uchar valid\nend_header\n")
        for index in range(cloud.size):
            values = (
                _format_float(cloud.x_mm[index]),
                _format_float(cloud.y_mm[index]),
                _format_float(cloud.z_mm[index]),
                _format_float(cloud.intensity[index]),
                _format_float(cloud.confidence[index]),
                str(int(cloud.valid[index])),
            )
            stream.write(" ".join(values) + "\n")
    return output_path


def _format_float(value: np.floating | float) -> str:
    numeric = float(value)
    return f"{numeric:.9g}" if np.isfinite(numeric) else "nan"
