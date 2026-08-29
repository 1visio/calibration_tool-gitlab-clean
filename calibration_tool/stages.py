from __future__ import annotations

import importlib
import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .errors import StageExecutionError
from .io_utils import dump_yaml, load_document


@dataclass(frozen=True)
class StageSpec:
    name: str
    module: str
    output_option: str
    result_file: str | None
    description: str
    forced_args: tuple[str, ...] = ()


STAGES: dict[str, StageSpec] = {
    "intrinsics": StageSpec(
        "intrinsics",
        "calibrate_chessboard_opencv_reusable",
        "--output",
        "calibration_result.yaml",
        "OpenCV 棋盘格内参与独立测试评估",
    ),
    "laser_plane_shared_steger": StageSpec(
        "laser_plane_shared_steger",
        "calibrate_laser_plane_core_v2",
        "--output-dir",
        "laser_plane.yaml",
        "旧 global_plane 激光平面标定（stage 名称和产物保留兼容）",
    ),
    "laser_surface_models": StageSpec(
        "laser_surface_models",
        "calibrate_laser_surface_models",
        "--output-dir",
        "laser_model.yaml",
        "三模型激光面标定；默认 circular_cone，保留旧平面 stage 兼容",
    ),
    "ground_extrinsics_board_only": StageSpec(
        "ground_extrinsics_board_only",
        "calibrate_ground_extrinsics_board_only",
        "--output-dir",
        "camera_ground_extrinsics.yaml",
        "棋盘基准面外参标定",
    ),
    "ground_extrinsics_shared_steger": StageSpec(
        "ground_extrinsics_shared_steger",
        "calibrate_ground_extrinsics_steger_v2",
        "--output-dir",
        "camera_ground_extrinsics.yaml",
        "棋盘法向与地面激光联合外参标定（统一实时 Steger）",
    ),
    "ground_bias": StageSpec(
        "ground_bias",
        "generate_ground_bias_compensation",
        "--output-dir",
        "compensation_metrics.json",
        "地面逐列偏差表生成与独立验证",
    ),
    "reconstruct_shared_steger": StageSpec(
        "reconstruct_shared_steger",
        "reconstruct_ground_pointcloud_cloudcompare_v4",
        "--output-dir",
        None,
        "使用统一实时 Steger 的三维恢复验证（旧 stage 名称保留兼容）",
        forced_args=("--steger-extractor", "shared"),
    ),
}


class ComputationService:
    """通过统一入口在当前进程调用经过验证的旧算法模块。"""

    def __init__(self, calibration_src: str | Path) -> None:
        self.calibration_src = Path(calibration_src).expanduser().resolve()
        if not self.calibration_src.is_dir():
            raise StageExecutionError(f"calibration/src 不存在：{self.calibration_src}")

    def run(
        self,
        stage_name: str,
        argv: Sequence[str],
        *,
        allow_quality_failure: bool = False,
    ) -> dict[str, Any]:
        if stage_name not in STAGES:
            raise StageExecutionError(f"未知阶段：{stage_name}")
        spec = STAGES[stage_name]
        arguments = list(argv)
        for index in range(0, len(spec.forced_args), 2):
            option, value = spec.forced_args[index : index + 2]
            if option in arguments:
                existing = arguments[arguments.index(option) + 1]
                if existing != value:
                    raise StageExecutionError(f"{stage_name} 强制要求 {option} {value}，不能使用 {existing}")
            else:
                arguments.extend((option, value))

        output_dir = _option_path(arguments, spec.output_option)
        started = datetime.now(timezone.utc)
        if str(self.calibration_src) not in sys.path:
            sys.path.insert(0, str(self.calibration_src))
        module = importlib.import_module(spec.module)
        main = getattr(module, "main", None)
        if not callable(main):
            raise StageExecutionError(f"{spec.module} 缺少 main(argv) 入口")
        try:
            exit_code = int(main(arguments))
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
        if exit_code != 0:
            raise StageExecutionError(f"阶段 {stage_name} 失败，退出码 {exit_code}")

        result_path = output_dir / spec.result_file if output_dir and spec.result_file else None
        metrics = _inspect_result(stage_name, result_path)
        gates = _evaluate_stage(stage_name, metrics, output_dir)
        quality_failed = any(gate["status"] == "fail" for gate in gates)
        record = {
            "schema_version": 1,
            "stage": stage_name,
            "module": spec.module,
            "started_utc": started.isoformat(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": arguments,
            "output_dir": str(output_dir) if output_dir else None,
            "result_file": str(result_path) if result_path else None,
            "metrics": metrics,
            "quality_gates": gates,
            "status": "quality_failed" if quality_failed else "completed",
        }
        if output_dir and output_dir.is_dir():
            dump_yaml(output_dir / "stage_run.yaml", record)
        if quality_failed and not allow_quality_failure:
            failed_ids = ", ".join(gate["id"] for gate in gates if gate["status"] == "fail")
            raise StageExecutionError(f"阶段 {stage_name} 计算完成但质量门禁失败：{failed_ids}")
        return record


def options_to_argv(options: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for name, value in options.items():
        option = name if name.startswith("--") else "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(option)
        elif value is None:
            continue
        elif isinstance(value, list):
            argv.append(option)
            argv.extend(str(item) for item in value)
        else:
            argv.extend((option, str(value)))
    return argv


def _option_path(argv: list[str], option: str) -> Path | None:
    if option not in argv:
        return None
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise StageExecutionError(f"{option} 缺少值")
    return Path(argv[index + 1]).expanduser().resolve()


def _inspect_result(stage_name: str, result_path: Path | None) -> dict[str, Any]:
    if result_path is None:
        return {}
    if not result_path.is_file():
        raise StageExecutionError(f"阶段完成但缺少结果文件：{result_path}")
    document = load_document(result_path)
    if stage_name == "intrinsics":
        return {
            "fit_image_count": document.get("fit_metrics", {}).get("image_count"),
            "test_image_count": document.get("test_metrics", {}).get("image_count"),
            "fit_rmse_px": document.get("fit_metrics", {}).get("overall_reprojection_rmse"),
            "test_rmse_px": document.get("test_metrics", {}).get("overall_reprojection_rmse"),
        }
    if stage_name == "laser_plane_shared_steger":
        return {
            "train_rmse_mm": document.get("metrics", {}).get("train_fit_inliers", {}).get("rmse_mm"),
            "validation_rmse_mm": document.get("metrics", {}).get("validation", {}).get("rmse_mm"),
            "validation_p95_mm": document.get("metrics", {}).get("validation", {}).get("p95_mm"),
        }
    if stage_name == "laser_surface_models":
        validation = document.get("metrics", {}).get("validation", {})
        train = document.get("metrics", {}).get("train", {})
        return {
            "model_type": document.get("model_type"),
            "supported_models": document.get("model_selection", {}).get("supported_models"),
            "train_rmse_mm": train.get("board_rmse_mm", train.get("surface_rmse_mm")),
            "validation_rmse_mm": validation.get("board_rmse_mm", validation.get("surface_rmse_mm")),
            "validation_p95_mm": validation.get("board_p95_abs_mm", validation.get("surface_p95_abs_mm")),
            "validation_valid_rate": validation.get("valid_rate"),
        }
    if stage_name.startswith("ground_extrinsics"):
        quality = document.get("quality_checks", {})
        validation = document.get("validation", {}).get("metrics", {})
        return {
            "ground_z_std_passed": quality.get("ground_z_std_passed"),
            "ground_z_std_mm": quality.get("ground_z_std_mm"),
            "validation_rmse_mm": validation.get("rmse_mm"),
            "validation_p95_mm": validation.get("p95_abs_mm"),
        }
    if stage_name == "ground_bias":
        loaded = int(document.get("loaded_frame_count", 0))
        build = int(document.get("build_frame_count", 0))
        evaluation = int(document.get("evaluation_frame_count", 0))
        return {
            "loaded_frame_count": loaded,
            "independent_validation_frame_count": evaluation if build + evaluation == loaded else 0,
            **dict(document.get("metrics", {})),
        }
    return {}


def _evaluate_stage(
    stage_name: str, metrics: dict[str, Any], output_dir: Path | None
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []

    def add(gate_id: str, passed: bool, actual: Any, expectation: str) -> None:
        gates.append(
            {
                "id": gate_id,
                "status": "pass" if passed else "fail",
                "actual": actual,
                "expected": expectation,
            }
        )

    if stage_name == "intrinsics":
        add("intrinsics.fit_rmse", _number_le(metrics.get("fit_rmse_px"), 0.30), metrics.get("fit_rmse_px"), "<= 0.30 px")
        add("intrinsics.independent_test", int(metrics.get("test_image_count") or 0) >= 1, metrics.get("test_image_count"), ">= 1 image")
        add("intrinsics.test_rmse", _number_le(metrics.get("test_rmse_px"), 0.50), metrics.get("test_rmse_px"), "<= 0.50 px")
    elif stage_name == "laser_plane_shared_steger":
        add("laser_plane.validation_rmse", _number_le(metrics.get("validation_rmse_mm"), 0.25), metrics.get("validation_rmse_mm"), "<= 0.25 mm")
        add("laser_plane.validation_p95", _number_le(metrics.get("validation_p95_mm"), 0.50), metrics.get("validation_p95_mm"), "<= 0.50 mm")
        motion = _pair_motion_quality(output_dir)
        metrics.update(motion)
        add("laser_plane.pair_motion_diagnostics", motion["diagnostic_exists"] is True, motion["diagnostic_exists"], "true")
        add("laser_plane.no_unexcluded_pair_motion", motion["unexcluded_moved_count"] == 0, motion["unexcluded_moved_count"], "0")
        if motion["unexcluded_unresolved_count"] == 0:
            add(
                "laser_plane.motion_resolved_or_excluded",
                True,
                0,
                "0",
            )
        else:
            gates.append(
                {
                    "id": "laser_plane.motion_resolved_or_excluded",
                    "status": "warn",
                    "actual": motion["unexcluded_unresolved_count"],
                    "expected": "0（纹理不足时需人工复核）",
                }
            )
    elif stage_name == "laser_surface_models":
        supported = {"global_plane", "quadratic_graph", "circular_cone"}
        model_type = metrics.get("model_type")
        add("laser_model.supported", model_type in supported, model_type, "/".join(sorted(supported)))
        add(
            "laser_model.validation_rmse",
            _number_le(metrics.get("validation_rmse_mm"), 0.25),
            metrics.get("validation_rmse_mm"),
            "<= 0.25 mm",
        )
        add(
            "laser_model.validation_p95",
            _number_le(metrics.get("validation_p95_mm"), 0.50),
            metrics.get("validation_p95_mm"),
            "<= 0.50 mm",
        )
    elif stage_name == "ground_extrinsics_board_only":
        add("ground.board_validation_rmse", _number_le(metrics.get("validation_rmse_mm"), 0.20), metrics.get("validation_rmse_mm"), "<= 0.20 mm")
    elif stage_name == "ground_extrinsics_shared_steger":
        add("ground.hybrid_flatness", metrics.get("ground_z_std_passed") is True, metrics.get("ground_z_std_passed"), "true")
    elif stage_name == "ground_bias":
        count = int(metrics.get("independent_validation_frame_count") or 0)
        add("compensation.independent_validation", count >= 1, count, ">= 1 frame")
    return gates


def _pair_motion_quality(output_dir: Path | None) -> dict[str, Any]:
    path = output_dir / "pair_motion_diagnostics.csv" if output_dir else None
    if path is None or not path.is_file():
        return {
            "diagnostic_exists": False,
            "unexcluded_moved_count": 0,
            "unexcluded_unresolved_count": 0,
            "unexcluded_not_assessed_count": 0,
        }
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    training = [row for row in rows if row.get("split") == "train"]
    unexcluded = [
        row
        for row in training
        if row.get("excluded_from_fit", "").lower() != "true"
    ]
    moved: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    not_assessed: list[dict[str, str]] = []
    for row in unexcluded:
        method = row.get("movement_method", "unresolved")
        tracking_ok = row.get("tracking_ok", "").lower() == "true"
        raw_displacement = row.get("median_displacement_px", "")
        try:
            displacement = float(raw_displacement)
        except (TypeError, ValueError):
            displacement = math.nan
        if tracking_ok and math.isfinite(displacement) and displacement > 1.0:
            moved.append(row)
        if method == "disabled" or tracking_ok:
            continue
        if method == "unresolved_low_texture":
            not_assessed.append(row)
        else:
            unresolved.append(row)
    return {
        "diagnostic_exists": True,
        "unexcluded_moved_count": len(moved),
        # Keep the historical metric name as the total unresolved/not-assessed
        # count so older reports and policies remain comparable.
        "unexcluded_unresolved_count": len(unresolved) + len(not_assessed),
        "unexcluded_tracking_unresolved_count": len(unresolved),
        "unexcluded_not_assessed_count": len(not_assessed),
    }


def _number_le(value: Any, limit: float) -> bool:
    try:
        return value is not None and float(value) <= limit
    except (TypeError, ValueError):
        return False
