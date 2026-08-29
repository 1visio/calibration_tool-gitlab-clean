from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError


_HASH_CHUNK_SIZE = 1024 * 1024


def load_document(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"配置文件不存在：{source}")
    try:
        text = source.read_text(encoding="utf-8-sig")
        data = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"无法读取配置 {source}：{exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError(f"配置顶层必须是映射：{source}")
    return dict(data)


def dump_yaml(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return target


def normalized_bytes(path: str | Path) -> bytes:
    data = Path(path).read_bytes()
    return data.replace(b"\r\n", b"\n")


def sha256_file(path: str | Path, *, normalize_newlines: bool = True) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    pending_carriage_return = False
    with source.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            if not normalize_newlines:
                digest.update(chunk)
                continue
            if pending_carriage_return:
                chunk = b"\r" + chunk
                pending_carriage_return = False
            if chunk.endswith(b"\r"):
                chunk = chunk[:-1]
                pending_carriage_return = True
            digest.update(chunk.replace(b"\r\n", b"\n"))
    if pending_carriage_return:
        digest.update(b"\r")
    return digest.hexdigest()


def canonical_mapping_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_relative(base_file: str | Path, value: str | Path) -> Path:
    raw = Path(value)
    return raw.expanduser().resolve() if raw.is_absolute() else (Path(base_file).resolve().parent / raw).resolve()


def dotted_value(document: Mapping[str, Any], selector: str) -> Any:
    current: Any = document
    for part in selector.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"指标路径不存在：{selector}")
        current = current[part]
    return current
