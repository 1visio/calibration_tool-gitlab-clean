from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import CalibrationToolError
from .io_utils import dump_yaml, load_document, sha256_file
from .profiles import load_runtime_profile


def build_calibration_bundle(
    config_path: str | Path,
    output_dir: str | Path,
    package_id: str,
    *,
    expected_extractor: str | None = None,
    quality_report: str | Path | None = None,
    allow_failed: bool = False,
) -> dict[str, Any]:
    target = Path(output_dir).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise CalibrationToolError(f"输出目录非空，拒绝覆盖：{target}")
    profile = load_runtime_profile(config_path, expected_extractor=expected_extractor)
    failures = [item for item in profile["findings"] if item.get("severity") == "fail"]
    quality: dict[str, Any] | None = None
    quality_files: dict[str, Any] = {}
    if quality_report is not None:
        quality_source = Path(quality_report).expanduser().resolve()
        quality = load_document(quality_source)
        if quality.get("overall") == "fail" and not allow_failed:
            raise CalibrationToolError("质量报告未通过，拒绝发布标定包")
    if failures and not allow_failed:
        raise CalibrationToolError("运行 profile 与标定链不兼容，拒绝发布标定包")

    target.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for name, record in profile["calibration_files"].items():
        if not record.get("exists"):
            continue
        source = Path(record["path"])
        destination = target / source.name
        shutil.copy2(source, destination)
        files[name] = {"path": destination.name, "sha256": sha256_file(destination)}

    if quality_report is not None:
        quality_source = Path(quality_report).expanduser().resolve()
        quality_destination = target / "acceptance_report.yaml"
        shutil.copy2(quality_source, quality_destination)
        quality_files["yaml"] = {
            "path": quality_destination.name,
            "sha256": sha256_file(quality_destination),
        }
        html_source = quality_source.with_suffix(".html")
        if html_source.is_file():
            html_destination = target / "acceptance_report.html"
            shutil.copy2(html_source, html_destination)
            quality_files["html"] = {
                "path": html_destination.name,
                "sha256": sha256_file(html_destination),
            }

    manifest = {
        "schema_version": 1,
        "package_id": package_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": str(Path(config_path).resolve()),
        "camera": profile["camera"],
        "algorithm_profile": profile["selected_extractor"],
        "laser_model": profile.get("laser_model"),
        "expected_extractor": expected_extractor,
        "files": files,
        "quality": None if quality is None else {
            "overall": quality.get("overall"),
            "decision": quality.get("decision"),
            "counts": quality.get("counts"),
            "reports": quality_files,
        },
        "release_override": bool(allow_failed),
        "findings": profile["findings"],
    }
    dump_yaml(target / "calibration_bundle.yaml", manifest)
    return manifest
