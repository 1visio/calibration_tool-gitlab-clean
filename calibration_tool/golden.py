from __future__ import annotations

import csv
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError, GoldenBaselineError
from .io_utils import dotted_value, dump_yaml, load_document, resolve_relative, sha256_file
from .profiles import load_runtime_profile


def build_golden_baseline(registry_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    registry_file = Path(registry_path).expanduser().resolve()
    registry = load_document(registry_file)
    if registry.get("schema_version") != 1:
        raise GoldenBaselineError("golden registry schema_version 必须为 1")
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    sources: dict[str, Any] = {}
    for name, raw in _mapping(registry.get("sources"), "sources").items():
        entry = _mapping(raw, f"sources.{name}")
        config_path = resolve_relative(registry_file, _text(entry.get("config"), f"sources.{name}.config"))
        expected = entry.get("expected_extractor")
        profile = load_runtime_profile(config_path, expected_extractor=str(expected) if expected else None)
        snapshot_dir = target / "snapshots" / str(name)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_config = snapshot_dir / config_path.name
        _copy_runtime_config_with_profile(config_path, snapshot_config)
        copied: dict[str, str] = {"config": str(snapshot_config.relative_to(target))}
        calibration_dir = snapshot_dir / "calibration"
        manifest = profile.get("manifest")
        if isinstance(manifest, Mapping) and manifest.get("exists"):
            manifest_path = Path(str(manifest["path"]))
            calibration_dir.mkdir(parents=True, exist_ok=True)
            manifest_destination = calibration_dir / manifest_path.name
            shutil.copy2(manifest_path, manifest_destination)
            copied["manifest"] = str(manifest_destination.relative_to(target))
        for key, record in profile["calibration_files"].items():
            if record.get("exists"):
                source_path = Path(record["path"])
                calibration_dir.mkdir(parents=True, exist_ok=True)
                destination = calibration_dir / source_path.name
                shutil.copy2(source_path, destination)
                copied[key] = str(destination.relative_to(target))
        profile["snapshot_files"] = copied
        sources[str(name)] = profile

    regressions: dict[str, Any] = {}
    for name, raw in _mapping(registry.get("regressions", {}), "regressions").items():
        entry = _mapping(raw, f"regressions.{name}")
        source_path = resolve_relative(registry_file, _text(entry.get("path"), f"regressions.{name}.path"))
        kind = str(entry.get("kind", "mapping"))
        regression = _read_regression(source_path, kind, entry)
        regression_snapshot = target / "snapshots" / "regressions" / str(name) / source_path.name
        regression_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, regression_snapshot)
        regression["snapshot_file"] = str(regression_snapshot.relative_to(target))
        regressions[str(name)] = regression

    baseline = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(registry_file),
        "sources": sources,
        "regressions": regressions,
    }
    dump_yaml(target / "baseline.yaml", baseline)
    return baseline


def _copy_runtime_config_with_profile(source: Path, destination: Path) -> None:
    """复制运行配置及其中央 extractor profile，保持快照可独立加载。"""

    document = load_document(source)
    extraction = document.get("extraction")
    profile_value = extraction.get("profile") if isinstance(extraction, Mapping) else None
    if not isinstance(profile_value, str) or not profile_value.strip():
        shutil.copy2(source, destination)
        return

    profile_source = resolve_relative(source, profile_value)
    profile_destination = destination.parent / profile_source.name
    shutil.copy2(profile_source, profile_destination)
    rewritten_extraction = dict(extraction)
    rewritten_extraction["profile"] = profile_destination.name
    rewritten_document = dict(document)
    rewritten_document["extraction"] = rewritten_extraction
    dump_yaml(destination, rewritten_document)


def check_golden_baseline(baseline_path: str | Path) -> dict[str, Any]:
    baseline = load_document(baseline_path)
    changes: list[dict[str, str]] = []
    for source_name, source in _mapping(baseline.get("sources"), "sources").items():
        config_path = Path(str(source["config_path"]))
        _compare_hash(changes, source_name, "config", config_path, source.get("config_sha256"))
        for key, record in _mapping(source.get("calibration_files", {}), "calibration_files").items():
            _compare_hash(changes, source_name, str(key), Path(str(record["path"])), record.get("sha256"))
        manifest = source.get("manifest")
        if isinstance(manifest, Mapping) and manifest.get("exists"):
            _compare_hash(changes, source_name, "manifest", Path(str(manifest["path"])), manifest.get("sha256"))
    for name, regression in _mapping(baseline.get("regressions", {}), "regressions").items():
        _compare_hash(
            changes,
            "regressions",
            str(name),
            Path(str(regression["path"])),
            regression.get("sha256"),
        )
    return {"matches": not changes, "change_count": len(changes), "changes": changes}


def _read_regression(path: Path, kind: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise GoldenBaselineError(f"回归指标源不存在：{path}")
    if kind == "mapping":
        document = load_document(path)
        selectors = _mapping(entry.get("selectors", {}), "selectors")
        values = {str(name): dotted_value(document, str(selector)) for name, selector in selectors.items()}
    elif kind == "pair_motion_csv":
        values = _pair_motion_metrics(path)
    elif kind == "compensation_metrics":
        document = load_document(path)
        selectors = _mapping(entry.get("selectors", {}), "selectors")
        values = {str(name): dotted_value(document, str(selector)) for name, selector in selectors.items()}
        loaded = int(document.get("loaded_frame_count", 0))
        build = int(document.get("build_frame_count", 0))
        evaluation = int(document.get("evaluation_frame_count", 0))
        values["independent_validation_frame_count"] = evaluation if build + evaluation == loaded else 0
    else:
        raise GoldenBaselineError(f"未知 regression kind：{kind}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "kind": kind,
        "values": values,
    }


def _pair_motion_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    training = [row for row in rows if row.get("split") == "train"]
    moved: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    displacements: list[float] = []
    for row in training:
        tracking_ok = row.get("tracking_ok", "").lower() == "true"
        method = row.get("movement_method", "unresolved")
        try:
            displacement = float(row.get("median_displacement_px", ""))
        except (TypeError, ValueError):
            displacement = math.nan
        if math.isfinite(displacement):
            displacements.append(displacement)
        if tracking_ok and math.isfinite(displacement) and displacement > 1.0:
            moved.append(row)
        if method != "disabled" and not tracking_ok:
            unresolved.append(row)
    return {
        "training_frame_count": len(training),
        "training_moved_count": len(moved),
        "training_unresolved_count": len(unresolved),
        "training_excluded_count": sum(row.get("excluded_from_fit", "").lower() == "true" for row in training),
        "max_training_displacement_px": max(displacements, default=None),
    }


def _compare_hash(
    changes: list[dict[str, str]], source: str, item: str, path: Path, expected: Any
) -> None:
    if not path.is_file():
        changes.append({"source": source, "item": item, "message": f"文件不存在：{path}"})
        return
    actual = sha256_file(path)
    if actual != expected:
        changes.append(
            {"source": source, "item": item, "message": f"哈希变化：期望 {expected}，实际 {actual}"}
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} 必须是映射")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} 必须是非空字符串")
    return value.strip()
