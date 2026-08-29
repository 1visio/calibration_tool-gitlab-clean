"""把最近采集结果安全地映射到已有 workflow 输入路径。"""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from ..io_utils import dump_yaml, load_document


FIT_KEYS = frozenset({"fit_dir", "train_dir", "fit_root", "train_root"})
VALIDATION_KEYS = frozenset({"test_dir", "validation_dir", "test_root", "validation_root"})


@dataclass(frozen=True, slots=True)
class WorkflowInputChange:
    stage: str
    option: str
    old_value: str
    new_value: str


@dataclass(frozen=True, slots=True)
class WorkflowUpdatePreview:
    workflow_path: Path
    original: Mapping[str, Any]
    updated: Mapping[str, Any]
    changes: tuple[WorkflowInputChange, ...]

    @property
    def changed(self) -> bool:
        return bool(self.changes)

    def text(self) -> str:
        if not self.changes:
            return "没有发现可由最近采集结果更新的 fit/validation 输入路径。"
        lines = [f"Workflow：{self.workflow_path}", "将更新以下路径（算法参数保持不变）："]
        lines.extend(
            f"- {change.stage}.{change.option}\n  {change.old_value}\n  → {change.new_value}"
            for change in self.changes
        )
        return "\n".join(lines)


def build_workflow_update_preview(
    workflow_path: str | Path,
    capture_artifacts: Mapping[str, Any],
) -> WorkflowUpdatePreview:
    """只修改已有 workflow stage options 中的 fit/validation 路径。"""

    source = Path(workflow_path).expanduser().resolve()
    document = load_document(source)
    updated = copy.deepcopy(document)
    stages = updated.get("stages", [])
    if not isinstance(stages, list):
        raise ValueError("workflow.stages 必须是列表")
    fit_dir = _path_value(capture_artifacts, "fit_dir")
    validation_dir = _optional_path_value(capture_artifacts, "validation_dir")
    changes: list[WorkflowInputChange] = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        options = stage.get("options")
        if not isinstance(options, dict):
            continue
        stage_name = str(stage.get("name", "unknown"))
        for option, old_value in list(options.items()):
            if not isinstance(old_value, str):
                continue
            target = fit_dir if option in FIT_KEYS else validation_dir if option in VALIDATION_KEYS else None
            if target is None:
                continue
            new_value = _relative_path(source.parent, target)
            if old_value != new_value:
                options[option] = new_value
                changes.append(WorkflowInputChange(stage_name, str(option), old_value, new_value))
    return WorkflowUpdatePreview(source, document, updated, tuple(changes))


def save_workflow_update(
    preview: WorkflowUpdatePreview,
    *,
    backup: bool = True,
) -> Path | None:
    """备份原 workflow 后原子写入预览内容，返回备份路径。"""

    if not preview.changed:
        return None
    target = preview.workflow_path
    backup_path: Path | None = None
    if backup:
        backup_path = _backup_path(target)
        shutil.copy2(target, backup_path)
    temporary_path: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
        os.close(fd)
        temporary_path = Path(temporary_name)
        dump_yaml(temporary_path, preview.updated)
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return backup_path


def _path_value(artifacts: Mapping[str, Any], key: str) -> Path:
    value = artifacts.get(key)
    if not value:
        raise ValueError(f"采集结果缺少 {key}")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"采集结果目录不存在：{path}")
    return path


def _optional_path_value(artifacts: Mapping[str, Any], key: str) -> Path | None:
    value = artifacts.get(key)
    if not value:
        return None
    path = Path(str(value)).expanduser().resolve()
    return path if path.is_dir() else None


def _relative_path(base: Path, target: Path) -> str:
    try:
        return target.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()


def _backup_path(target: Path) -> Path:
    base = target.with_name(f"{target.name}.bak")
    if not base.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = target.with_name(f"{target.name}.bak-{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = target.with_name(f"{target.name}.bak-{stamp}-{suffix}")
        suffix += 1
    return candidate


__all__ = [
    "WorkflowInputChange",
    "WorkflowUpdatePreview",
    "build_workflow_update_preview",
    "save_workflow_update",
]
