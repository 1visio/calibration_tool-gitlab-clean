from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdapterConfig:
    name: str
    options: dict[str, Any]


@dataclass(frozen=True)
class RunConfig:
    schema_version: int
    config_version: str
    output_dir: Path
    calibration_file: Path
    source: AdapterConfig
    extraction: AdapterConfig
    reconstruction: AdapterConfig
    source_path: Path
    sha256: str


def load_run_config(path: str | Path) -> RunConfig:
    source_path = Path(path).resolve()
    raw_bytes = source_path.read_bytes()
    raw = json.loads(raw_bytes.decode("utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("仅支持 schema_version=1 的运行配置")

    base_dir = source_path.parent
    return RunConfig(
        schema_version=1,
        config_version=str(raw["config_version"]),
        output_dir=(base_dir / raw["output_dir"]).resolve(),
        calibration_file=(base_dir / raw["calibration_file"]).resolve(),
        source=_load_adapter(raw, "source"),
        extraction=_load_adapter(raw, "extraction"),
        reconstruction=_load_adapter(raw, "reconstruction"),
        source_path=source_path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _load_adapter(raw: dict[str, Any], key: str) -> AdapterConfig:
    item = raw[key]
    return AdapterConfig(name=str(item["name"]), options=dict(item.get("options", {})))
