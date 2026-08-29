from __future__ import annotations

import argparse
from pathlib import Path

from .bootstrap import build_pipeline
from .config import load_run_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行一次线激光静态轮廓流水线")
    parser.add_argument("--config", type=Path, required=True, help="运行配置 JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_run_config(args.config)
    pipeline, calibration = build_pipeline(config)
    context = {
        "config_version": config.config_version,
        "config_path": str(config.source_path),
        "config_sha256": config.sha256,
        "calibration_version": calibration.version,
        "calibration_path": str(config.calibration_file),
        "calibration_status": calibration.status,
        "mechanical_config_id": calibration.mechanical_config_id,
        "dataset_id": calibration.dataset_id,
    }
    artifacts = pipeline.run_once(config.output_dir, context)
    print(f"运行完成：{artifacts.run_dir}")
    print(f"CSV：{artifacts.csv_path}")
    print(f"PLY：{artifacts.ply_path}")
    print(f"摘要：{artifacts.summary_path}")
    return 0
