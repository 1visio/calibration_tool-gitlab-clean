from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import yaml

from .bundle import build_calibration_bundle
from .acceptance import build_acceptance_report
from .camera import (
    build_camera_provider,
    load_camera_config,
    load_capture_plan,
    preview_camera,
    run_capture_plan,
)
from .camera.models import CapturePlan, CaptureTask
from .errors import CalibrationToolError, CameraError
from .golden import build_golden_baseline, check_golden_baseline
from .profiles import load_runtime_profile
from .quality import audit_baseline
from .stages import ComputationService, STAGES
from .workflow import run_workflow


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PACKAGE_ROOT / "configs" / "golden_sources.example.yaml"
DEFAULT_POLICY = PACKAGE_ROOT / "configs" / "quality_policy.yaml"
DEFAULT_GOLDEN = PACKAGE_ROOT / "runs" / "golden"
DEFAULT_CALIBRATION_SRC = PACKAGE_ROOT / "calibration" / "src"
DEFAULT_CAMERA_CONFIG = PACKAGE_ROOT / "configs" / "camera.example.yaml"
DEFAULT_CAMERA_CHANNELS = PACKAGE_ROOT / "configs" / "camera_channels.example.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="线激光统一标定工具（阶段 0/1/2/3/4）")
    sub = parser.add_subparsers(dest="command", required=True)

    golden = sub.add_parser("golden-build", help="从当前离线/在线 config 创建 golden baseline")
    golden.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    golden.add_argument("--output", type=Path, default=DEFAULT_GOLDEN)

    check = sub.add_parser("golden-check", help="检查当前 config/标定文件是否偏离 baseline")
    check.add_argument("--baseline", type=Path, default=DEFAULT_GOLDEN / "baseline.yaml")

    audit = sub.add_parser("audit", help="对 golden baseline 执行质量门禁")
    audit.add_argument("--baseline", type=Path, default=DEFAULT_GOLDEN / "baseline.yaml")
    audit.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    audit.add_argument("--output", type=Path, default=PACKAGE_ROOT / "reports" / "golden_audit.yaml")

    profile = sub.add_parser("profile", help="解析并审计一个运行 config")
    profile.add_argument("config", type=Path)
    profile.add_argument("--expected-extractor")

    sub.add_parser("list-stages", help="列出统一计算阶段")

    run = sub.add_parser("run", help="运行一个统一计算阶段；其余参数原样传给算法")
    run.add_argument("stage", choices=sorted(STAGES))
    run.add_argument("--calibration-src", type=Path, default=DEFAULT_CALIBRATION_SRC)
    run.add_argument("--allow-quality-failure", action="store_true")
    run.add_argument("args", nargs=argparse.REMAINDER)

    workflow = sub.add_parser("workflow", help="按一个 YAML 计划顺序运行多个阶段")
    workflow.add_argument("plan", type=Path, nargs="?", help="省略时读取 --project 中的 workflow_plan")
    workflow.add_argument(
        "--project",
        type=Path,
        help="读取项目所选通道的 workflow_plan 与 laser.orientation",
    )

    bundle = sub.add_parser("bundle-build", help="发布不可拆分的标定包")
    bundle.add_argument("--config", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    bundle.add_argument("--package-id", required=True)
    bundle.add_argument("--expected-extractor")
    bundle.add_argument("--quality-report", type=Path)
    bundle.add_argument("--allow-failed", action="store_true")

    camera_list = sub.add_parser("camera-list", help="枚举可用相机")
    camera_list.add_argument("--config", type=Path)
    camera_list.add_argument("--channel", help="从统一相机配置中选择通道")
    camera_list.add_argument("--backend", choices=("mvs", "daheng", "synthetic"), default="mvs")
    camera_list.add_argument("--calibration-src", type=Path, default=DEFAULT_CALIBRATION_SRC)

    preview = sub.add_parser("camera-preview", help="取流并输出曝光/清晰度/激光覆盖等质量指标")
    preview.add_argument("--config", type=Path)
    preview.add_argument("--channel", help="从统一相机配置中选择通道")
    preview.add_argument("--frames", type=int, default=20)
    preview.add_argument("--warmup-frames", type=int, default=5)
    preview.add_argument("--quality-mode", choices=("generic", "chessboard", "laser"), default="generic")
    preview.add_argument("--snapshot", type=Path)

    replay = sub.add_parser(
        "search-region-replay",
        help="只读回放单张激光图并输出 Steger search-region health",
    )
    replay.add_argument("image", type=Path)
    replay.add_argument("--calibration-src", type=Path, default=DEFAULT_CALIBRATION_SRC)
    replay.add_argument(
        "--laser-orientation",
        choices=("horizontal", "vertical"),
        default="horizontal",
    )

    capture = sub.add_parser("capture-plan", help="执行 YAML 批量采集计划")
    capture.add_argument("plan", type=Path)
    capture.add_argument("--resume", action="store_true")
    capture.add_argument("--interactive", action="store_true", help="每项任务按提示确认后再采集")
    capture.add_argument(
        "--preview-window",
        action="store_true",
        help="交互采集时打开实时 OpenCV 窗口；Enter/Space 确认任务，Esc/Q 取消",
    )
    capture.add_argument("--calibration-src", type=Path, default=DEFAULT_CALIBRATION_SRC)
    capture.add_argument("--config", type=Path, help="统一相机配置，用于校验计划通道")
    capture.add_argument("--channel", help="与 --config 一起选择相机通道")

    exposure = sub.add_parser("capture-exposure-series", help="按多个曝光值批量采集同一姿态")
    exposure.add_argument("--config", type=Path)
    exposure.add_argument("--channel", help="从统一相机配置中选择通道")
    exposure.add_argument("--output", type=Path, required=True)
    exposure.add_argument("--dataset-id", required=True)
    exposure.add_argument("--pose-id", required=True)
    exposure.add_argument("--exposures-us", type=float, nargs="+", required=True)
    exposure.add_argument("--frames-per-exposure", type=int, default=1)
    exposure.add_argument("--settle-frames", type=int, default=5)
    exposure.add_argument("--quality-mode", choices=("generic", "chessboard", "laser"), default="chessboard")
    exposure.add_argument("--resume", action="store_true")

    gui = sub.add_parser("gui", help="启动 PySide6 标定向导 MVP")
    gui.add_argument("--project", type=Path)
    gui.add_argument("--simulate", action="store_true", help="默认加载 synthetic 相机配置")
    gui.add_argument("--channel", help="默认相机通道，例如 hikrobot 或 daheng")

    acceptance = sub.add_parser("acceptance-report", help="生成补偿前后对比与正式验收报告")
    acceptance.add_argument("plan", type=Path)
    acceptance.add_argument("--output", type=Path)
    acceptance.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "gui":
        from .gui import launch_gui

        return launch_gui(
            project=args.project,
            simulate=args.simulate,
            camera_channel=args.channel,
        )
    try:
        if args.command == "golden-build":
            result = build_golden_baseline(args.registry, args.output)
        elif args.command == "golden-check":
            result = check_golden_baseline(args.baseline)
        elif args.command == "audit":
            result = audit_baseline(args.baseline, args.policy, args.output)
        elif args.command == "profile":
            result = load_runtime_profile(args.config, expected_extractor=args.expected_extractor)
        elif args.command == "list-stages":
            result = {name: spec.description for name, spec in STAGES.items()}
        elif args.command == "run":
            passthrough = list(args.args)
            if passthrough and passthrough[0] == "--":
                passthrough = passthrough[1:]
            result = ComputationService(args.calibration_src).run(
                args.stage,
                passthrough,
                allow_quality_failure=args.allow_quality_failure,
            )
        elif args.command == "workflow":
            if args.project is not None:
                from .gui.project import WizardProject

                project = WizardProject.load(args.project)
                orientation = project.laser.orientation
                workflow_plan = args.plan or project.workflow_plan
            else:
                orientation = "horizontal"
                workflow_plan = args.plan
            if workflow_plan is None:
                raise ValueError("workflow 需要 plan 参数，或通过 --project 提供 workflow_plan")
            result = run_workflow(workflow_plan, laser_orientation=orientation)
        elif args.command == "bundle-build":
            result = build_calibration_bundle(
                args.config,
                args.output,
                args.package_id,
                expected_extractor=args.expected_extractor,
                quality_report=args.quality_report,
                allow_failed=args.allow_failed,
            )
        elif args.command == "camera-list":
            camera_config = args.config or (DEFAULT_CAMERA_CHANNELS if args.channel else None)
            if camera_config:
                camera_runtime = load_camera_config(camera_config, channel=args.channel)
                provider = build_camera_provider(
                    camera_runtime["backend"],
                    calibration_src=camera_runtime["calibration_src"],
                    backend_options=camera_runtime["backend_options"],
                )
            else:
                provider = build_camera_provider(
                    args.backend,
                    calibration_src=args.calibration_src,
                )
            try:
                result = {"devices": [asdict(item) for item in provider.list_devices()]}
            except Exception as exc:
                raise CameraError(f"相机枚举失败：{exc}") from exc
        elif args.command == "camera-preview":
            camera_config = args.config or (
                DEFAULT_CAMERA_CHANNELS if args.channel else DEFAULT_CAMERA_CONFIG
            )
            camera_runtime = load_camera_config(camera_config, channel=args.channel)
            provider = build_camera_provider(
                camera_runtime["backend"],
                calibration_src=camera_runtime["calibration_src"],
                backend_options=camera_runtime["backend_options"],
            )
            result = preview_camera(
                provider,
                camera_runtime["serial_number"],
                camera_runtime["camera"],
                frames=args.frames,
                warmup_frames=args.warmup_frames,
                quality_mode=args.quality_mode,
                quality_thresholds=camera_runtime["quality_thresholds"],
                board_pattern=camera_runtime["board_pattern"],
                laser_orientation=camera_runtime["laser"].orientation,
                snapshot=args.snapshot,
            )
        elif args.command == "search-region-replay":
            import cv2

            from .camera.steger_quality import RealtimeStegerQualityAnalyzer

            image_path = args.image.expanduser().resolve()
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise CameraError(f"无法读取回放图像：{image_path}")
            result = {
                "image": str(image_path),
                "search_region_health": RealtimeStegerQualityAnalyzer(
                    args.calibration_src,
                    args.laser_orientation,
                ).analyze(image),
            }
        elif args.command == "capture-plan":
            plan = load_capture_plan(args.plan)
            calibration_src = args.calibration_src
            if args.config is not None or args.channel is not None:
                camera_config = args.config or DEFAULT_CAMERA_CHANNELS
                camera_runtime = load_camera_config(camera_config, channel=args.channel)
                if camera_runtime["backend"] != plan.backend:
                    raise CameraError(
                        "采集计划 backend 与所选相机通道不一致："
                        f"plan={plan.backend}, channel={camera_runtime['backend']}"
                    )
                plan_channel = plan.metadata.get("camera_channel")
                if (
                    plan_channel
                    and camera_runtime.get("channel")
                    and str(plan_channel) != str(camera_runtime["channel"])
                ):
                    raise CameraError(
                        "采集计划通道与所选通道不一致："
                        f"plan={plan_channel}, channel={camera_runtime['channel']}"
                    )
                configured_serial = str(camera_runtime.get("serial_number", ""))
                if configured_serial and plan.serial_number and configured_serial != plan.serial_number:
                    raise CameraError(
                        "采集计划序列号与所选通道配置不一致："
                        f"plan={plan.serial_number}, channel={configured_serial}"
                    )
                calibration_src = camera_runtime["calibration_src"]
            provider = build_camera_provider(
                plan.backend,
                calibration_src=calibration_src,
                backend_options=plan.backend_options,
            )
            result = _capture_result_dict(run_capture_plan(
                plan,
                provider,
                resume=args.resume,
                interactive=args.interactive,
                interactive_preview=args.preview_window,
                progress=_capture_progress,
            ))
        elif args.command == "capture-exposure-series":
            camera_config = args.config or (
                DEFAULT_CAMERA_CHANNELS if args.channel else DEFAULT_CAMERA_CONFIG
            )
            camera_runtime = load_camera_config(camera_config, channel=args.channel)
            if len(set(args.exposures_us)) != len(args.exposures_us):
                raise CameraError("exposures-us 不能包含重复值")
            if args.frames_per_exposure <= 0 or args.settle_frames < 0:
                raise CameraError("frames-per-exposure 必须为正数，settle-frames 不能为负数")
            tasks = tuple(
                CaptureTask(
                    task_id=f"exposure_{index:02d}",
                    frames=args.frames_per_exposure,
                    filename_template="{exposure_folder}/{pose_id}{index_suffix}{suffix}",
                    config=replace(camera_runtime["camera"], exposure_us=exposure),
                    pose_id=args.pose_id,
                    role="exposure_series",
                    settle_frames=args.settle_frames,
                    image_format="tif" if camera_runtime["camera"].pixel_format == "Mono12" else "png",
                    quality_mode=args.quality_mode,
                    tags={"requested_exposure_us": exposure},
                )
                for index, exposure in enumerate(args.exposures_us, start=1)
            )
            plan = CapturePlan(
                dataset_id=args.dataset_id,
                output_dir=args.output,
                backend=camera_runtime["backend"],
                serial_number=camera_runtime["serial_number"],
                base_config=camera_runtime["camera"],
                tasks=tasks,
                quality_thresholds=camera_runtime["quality_thresholds"],
                board_pattern=camera_runtime["board_pattern"],
                metadata={
                    "kind": "exposure_series",
                    "camera_channel": camera_runtime.get("channel"),
                    "camera_config": str(camera_runtime["camera_config_source"]),
                },
                backend_options=camera_runtime["backend_options"],
                laser=camera_runtime["laser"],
            )
            provider = build_camera_provider(
                plan.backend,
                calibration_src=camera_runtime["calibration_src"],
                backend_options=plan.backend_options,
            )
            result = _capture_result_dict(run_capture_plan(
                plan,
                provider,
                resume=args.resume,
                progress=_capture_progress,
            ))
        elif args.command == "acceptance-report":
            result = build_acceptance_report(
                args.plan,
                output_dir=args.output,
                overwrite=args.overwrite,
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except CalibrationToolError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (TypeError, ValueError) as exc:
        print(f"ERROR: 参数无效：{exc}", file=sys.stderr)
        return 2
    _print_result(result)
    if args.command == "golden-check" and not result.get("matches", False):
        return 1
    if args.command == "audit" and result.get("overall") == "fail":
        return 1
    if args.command == "acceptance-report" and result.get("overall") == "fail":
        return 1
    return 0


def _print_result(value: Any) -> None:
    try:
        print(yaml.safe_dump(value, allow_unicode=True, sort_keys=False).rstrip())
    except yaml.YAMLError:
        print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _capture_progress(event: dict[str, Any]) -> None:
    if event["event"] == "frame":
        warnings = event["quality"]["warnings"]
        suffix = f"，告警={','.join(warnings)}" if warnings else ""
        print(
            f"[{event['task_id']}] {event['index']}/{event['frames']}{suffix}",
            file=sys.stderr,
        )
    elif event["event"] == "task_completed":
        print(f"[{event['task_id']}] 已提交", file=sys.stderr)


def _capture_result_dict(value: Any) -> dict[str, Any]:
    result = asdict(value)
    result["output_dir"] = str(result["output_dir"])
    return result
