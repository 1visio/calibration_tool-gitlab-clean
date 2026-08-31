"""CalibrationRun 的 YAML 持久化与旧 workflow report 读取适配。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

from .calibration_run import CalibrationRun
from .errors import ConfigError
from .io_utils import dump_yaml, load_document


DEFAULT_CALIBRATION_RUN_FILENAME = "calibration_run.yaml"


def _report_path(
    value: Any,
    field_name: str,
    *,
    base_dir: Path | None,
) -> Path | None:
    if value is None or value == "":
        return None
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"{field_name} 必须是路径")
    path = Path(value).expanduser()
    is_absolute = path.is_absolute() or PureWindowsPath(str(value)).is_absolute()
    if not is_absolute:
        if base_dir is None:
            return None
        path = base_dir / path
    # POSIX 环境读取 Windows 生成的 report 时，不能把盘符路径拼到当前目录。
    if base_dir is not None and is_absolute and not path.is_absolute():
        return path
    return path.resolve() if base_dir is not None else path


def infer_calibration_run_root(
    report: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> Path | None:
    """从已完成 stage 的产物目录推断共同 run 根目录。

    只有至少两个不同的已完成 stage 输出目录、且它们是同一父目录的直接
    子目录时才返回结果；信息不足或 stage 跨 run 时返回 ``None``，避免把
    不同运行的最长公共祖先误当成 run 根目录。
    """

    if not isinstance(report, Mapping):
        raise ConfigError("workflow report 顶层必须是映射")
    stages = report.get("stages", [])
    if stages is None:
        return None
    if isinstance(stages, (str, bytes)) or not isinstance(stages, Sequence):
        raise ConfigError("stages 必须是列表")
    base_dir = Path(source_path).expanduser().resolve().parent if source_path else None
    output_dirs: list[Path] = []
    for index, raw_stage in enumerate(stages, start=1):
        if not isinstance(raw_stage, Mapping):
            raise ConfigError(f"stages[{index}] 必须是映射")
        if raw_stage.get("status") != "completed":
            continue
        output_dir = _report_path(
            raw_stage.get("output_dir"),
            f"stages[{index}].output_dir",
            base_dir=base_dir,
        )
        result_file = _report_path(
            raw_stage.get("result_file"),
            f"stages[{index}].result_file",
            base_dir=base_dir,
        )
        if output_dir is not None:
            if result_file is not None:
                try:
                    result_file.relative_to(output_dir)
                except ValueError as exc:
                    raise ConfigError(
                        f"stages[{index}].result_file 不在 output_dir 内"
                    ) from exc
            output_dirs.append(output_dir)
        elif result_file is not None:
            output_dirs.append(result_file.parent)

    unique_dirs = list(dict.fromkeys(output_dirs))
    if len(unique_dirs) < 2:
        return None
    parents = {path.parent for path in unique_dirs}
    if len(parents) != 1:
        return None
    return next(iter(parents))


# 兼容更短/更直观的调用名称；实现保持单一。
infer_run_root = infer_calibration_run_root
infer_run_root_from_report = infer_calibration_run_root


def calibration_run_from_report(
    report: Mapping[str, Any],
    *,
    run_id: str | None = None,
    project_id: str | None = None,
    workflow_path: str | Path | None = None,
    source_path: str | Path | None = None,
) -> CalibrationRun:
    """把 ``run_workflow`` 返回的 report 转成 CalibrationRun。"""

    return CalibrationRun.from_report(
        report,
        run_id=run_id,
        project_id=project_id,
        workflow_path=workflow_path,
        source_path=source_path,
    )


def load_calibration_run(
    path: str | Path,
    *,
    run_id: str | None = None,
    project_id: str | None = None,
) -> CalibrationRun:
    """读取新的 ``calibration_run.yaml`` 或旧的 ``workflow_*.yaml``。"""

    source = Path(path).expanduser().resolve()
    document = load_document(source)
    return calibration_run_from_report(
        document,
        run_id=run_id,
        project_id=project_id,
        source_path=source,
    )


def save_calibration_run(run: CalibrationRun, path: str | Path) -> Path:
    """将 CalibrationRun 按稳定 schema 写入 YAML。"""

    if not isinstance(run, CalibrationRun):
        raise TypeError("run 必须是 CalibrationRun")
    target = Path(path).expanduser().resolve()
    dump_yaml(target, run.to_dict())
    return target


# 简短别名便于页面层使用；实际实现只有一套。
load = load_calibration_run
save = save_calibration_run
from_workflow_report = calibration_run_from_report
