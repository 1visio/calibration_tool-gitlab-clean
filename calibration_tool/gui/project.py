from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from ..io_utils import dump_yaml, load_document, resolve_relative
from ..laser import LaserConfig, parse_laser_config
from ..camera.config import load_camera_config


@dataclass(slots=True)
class WizardProject:
    project_id: str
    workspace: Path
    camera_config: Path
    camera_channel: str | None = None
    workflow_plan: Path | None = None
    acceptance_plan: Path | None = None
    capture_output: Path | None = None
    pattern_cols: int = 11
    pattern_rows: int = 8
    square_size_mm: float = 20.0
    source_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    laser: LaserConfig = field(default_factory=LaserConfig)

    def __post_init__(self) -> None:
        if not self.project_id.strip():
            raise ConfigError("project_id 不能为空")
        if min(self.pattern_cols, self.pattern_rows) <= 0 or self.square_size_mm <= 0:
            raise ConfigError("棋盘格参数必须为正数")
        self.workspace = self.workspace.expanduser().resolve()
        self.camera_config = self.camera_config.expanduser().resolve()
        if self.camera_channel is not None:
            self.camera_channel = self.camera_channel.strip() or None
        if self.workflow_plan is not None:
            self.workflow_plan = self.workflow_plan.expanduser().resolve()
        if self.acceptance_plan is not None:
            self.acceptance_plan = self.acceptance_plan.expanduser().resolve()
        if self.capture_output is not None:
            self.capture_output = self.capture_output.expanduser().resolve()

    @classmethod
    def load(cls, path: str | Path) -> "WizardProject":
        source = Path(path).expanduser().resolve()
        document = load_document(source)
        if document.get("schema_version", 1) != 1:
            raise ConfigError("wizard project schema_version 必须为 1")
        board = document.get("board", {})
        if not isinstance(board, dict):
            raise ConfigError("wizard project.board 必须是映射")
        workspace_value = document.get("workspace", ".")
        camera_value = document.get("camera_config", "camera.mvs.example.yaml")
        camera_config = resolve_relative(source, camera_value)
        channel_value = document.get("camera_channel")
        camera_channel = str(channel_value).strip() if channel_value is not None else None
        camera_runtime = load_camera_config(camera_config, channel=camera_channel)
        workflow_value = document.get("workflow_plan") or camera_runtime.get("workflow_plan")
        acceptance_value = document.get("acceptance_plan")
        capture_value = document.get("capture_output")
        laser = (
            parse_laser_config(document["laser"], field_name="wizard project.laser")
            if "laser" in document
            else camera_runtime["laser"]
        )
        return cls(
            project_id=str(document.get("project_id", "")),
            workspace=resolve_relative(source, workspace_value),
            camera_config=camera_config,
            camera_channel=str(camera_runtime.get("channel") or "") or None,
            laser=laser,
            workflow_plan=resolve_relative(source, workflow_value) if workflow_value else None,
            acceptance_plan=resolve_relative(source, acceptance_value) if acceptance_value else None,
            capture_output=resolve_relative(source, capture_value) if capture_value else None,
            pattern_cols=int(board.get("pattern_cols", 11)),
            pattern_rows=int(board.get("pattern_rows", 8)),
            square_size_mm=float(board.get("square_size_mm", 20.0)),
            source_path=source,
            extra=dict(document),
        )

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or self.source_path or (self.workspace / "wizard_project.yaml")).expanduser().resolve()

        def relative_or_absolute(value: Path | None) -> str | None:
            if value is None:
                return None
            try:
                return value.resolve().relative_to(target.parent).as_posix()
            except ValueError:
                return str(value.resolve())

        document = dict(self.extra)
        document.update({
            "schema_version": 1,
            "project_id": self.project_id,
            "workspace": relative_or_absolute(self.workspace),
            "camera_config": relative_or_absolute(self.camera_config),
            "camera_channel": self.camera_channel,
            "laser": {"orientation": self.laser.orientation},
            "workflow_plan": relative_or_absolute(self.workflow_plan),
            "acceptance_plan": relative_or_absolute(self.acceptance_plan),
            "capture_output": relative_or_absolute(self.capture_output),
            "board": {
                "pattern_cols": self.pattern_cols,
                "pattern_rows": self.pattern_rows,
                "square_size_mm": self.square_size_mm,
            },
        })
        dump_yaml(target, document)
        self.source_path = target
        self.extra = document
        return target

    def record_capture_artifacts(self, artifacts: dict[str, Any], *, persist: bool = True) -> Path | None:
        """记录最近采集路径；旧项目字段保持不变，存在源文件时自动备份后保存。"""

        self.extra = dict(self.extra)
        self.extra["capture_artifacts"] = dict(artifacts)
        if not persist or self.source_path is None:
            return None
        source = self.source_path.resolve()
        backup = source.with_name(f"{source.name}.bak")
        if backup.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = source.with_name(f"{source.name}.bak-{stamp}")
            suffix = 1
            while backup.exists():
                backup = source.with_name(f"{source.name}.bak-{stamp}-{suffix}")
                suffix += 1
        shutil.copy2(source, backup)
        self.save(source)
        return backup
