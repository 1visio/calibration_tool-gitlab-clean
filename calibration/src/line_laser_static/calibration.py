from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LaserCalibration:
    version: str
    image_size: tuple[int, int]
    camera_matrix: NDArray[np.float64]
    distortion: NDArray[np.float64]
    plane_normal: NDArray[np.float64]
    plane_offset: float
    coordinate_system: str
    unit: str
    mechanical_config_id: str
    dataset_id: str
    status: str

    def __post_init__(self) -> None:
        if self.camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix 必须是 3x3")
        if self.plane_normal.shape != (3,):
            raise ValueError("laser_plane.normal 必须包含三个分量")
        if np.linalg.norm(self.plane_normal) == 0:
            raise ValueError("激光平面法向量不能为零")
        if self.unit != "mm":
            raise ValueError("当前三维结果接口仅接受毫米单位")


def load_calibration(path: str | Path) -> LaserCalibration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("仅支持 schema_version=1 的标定文件")
    plane = raw["laser_plane"]
    return LaserCalibration(
        version=str(raw["calibration_version"]),
        image_size=(int(raw["image_size"][0]), int(raw["image_size"][1])),
        camera_matrix=np.asarray(raw["camera_matrix"], dtype=np.float64),
        distortion=np.asarray(raw["distortion"], dtype=np.float64),
        plane_normal=np.asarray(plane["normal"], dtype=np.float64),
        plane_offset=float(plane["offset"]),
        coordinate_system=str(raw["coordinate_system"]),
        unit=str(raw["unit"]),
        mechanical_config_id=str(raw["mechanical_config_id"]),
        dataset_id=str(raw["dataset_id"]),
        status=str(raw.get("status", "UNSPECIFIED")),
    )
