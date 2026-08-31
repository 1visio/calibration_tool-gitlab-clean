from __future__ import annotations

import csv
import math
import threading
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
from PySide6.QtCore import QSignalBlocker, QThreadPool, Qt, Signal, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..camera import build_camera_provider, load_camera_config, load_capture_plan, run_capture_plan
from ..camera.config import capture_plan_hash, capture_plan_payload
from ..camera.capture import _new_manifest, _save_state, _write_image
from ..camera.plan_builder import (
    CaptureRecipe,
    build_capture_plan_from_recipe,
    capture_plan_summary,
    save_generated_capture_plan,
)
from ..acceptance import build_acceptance_report
from ..calibration_run import CalibrationRun
from ..calibration_run_io import DEFAULT_CALIBRATION_RUN_FILENAME, load_calibration_run
from ..calibration_results import (
    CalibrationResultsSummary,
    GroundExtrinsicsDetails,
    IntrinsicsDetails,
    LaserSurfaceDetails,
    NOT_EXECUTED,
    load_ground_extrinsics_details,
    load_intrinsics_details,
    load_laser_surface_details,
    summarize_calibration_run,
)
from ..camera.models import CameraConfig, CapturePlan, CaptureTask
from ..camera.quality import analyze_frame, quality_to_dict
from ..camera.steger_quality import RealtimeStegerQualityAnalyzer
from ..io_utils import load_document, resolve_relative
from ..io_utils import sha256_file
from ..laser import LaserConfig
from ..laser_models import SUPPORTED_LASER_MODEL_TYPES
from ..workflow import run_workflow
from .project import WizardProject
from .result_artifacts import (
    ResultArtifact,
    capture_artifacts_record,
    discover_result_artifacts as discover_artifact_records,
)
from .workflow_inputs import build_workflow_update_preview, save_workflow_update
from .acceptance_plan import (
    default_acceptance_plan_path,
    ensure_default_acceptance_plan,
    update_acceptance_plan_from_workflow,
)
from .widgets import ImagePreview, ResidualPlot
from .workers import FunctionWorker, PreviewThread
from .capture_controller import CaptureTaskGate
from .capture_recipe_widget import CaptureRecipeTable


# 临时关闭 Stage 2A 的 GUI 旁路：保留 analyzer/replay 实现，便于后续在质量
# 显示完整且预览性能问题解决后恢复。关闭时既不显示控件，也不逐帧运行 Steger。
_SEARCH_REGION_QUALITY_ENABLED = False


def _preview_settling_info(quality: Mapping[str, Any]) -> tuple[bool, int]:
    """读取 PreviewThread 的稳定帧元数据，兼容旧的质量字典。"""

    settling = bool(quality.get("settling", False))
    try:
        remaining = max(0, int(quality.get("settle_frames_remaining", 0)))
    except (TypeError, ValueError):
        remaining = 0
    return settling, remaining


class LaserPlotPreview(QLabel):
    """随内容宽度变化、保持原始比例的验证误差图预览。"""

    MINIMUM_PLOT_HEIGHT = 280

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._source_pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(self.MINIMUM_PLOT_HEIGHT)

    def set_plot_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._refresh_plot()

    def clear_plot(self) -> None:
        self._source_pixmap = QPixmap()
        self.clear()
        self.setFixedHeight(self.MINIMUM_PLOT_HEIGHT)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        if self._source_pixmap.isNull():
            return
        target_width = self.width()
        if target_width <= 0:
            target_width = self._source_pixmap.width()
        scaled = self._source_pixmap.scaled(
            target_width,
            self._source_pixmap.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self.setFixedHeight(max(self.MINIMUM_PLOT_HEIGHT, scaled.height()))


_RESULT_PAGE_MARGINS = (12, 12, 12, 12)
_RESULT_PAGE_SPACING = 10


def _configure_result_page_layout(layout: QLayout) -> None:
    """保持四个结果页签的外层间距一致。"""

    layout.setContentsMargins(*_RESULT_PAGE_MARGINS)
    layout.setSpacing(_RESULT_PAGE_SPACING)


class ProjectPage(QWidget):
    project_changed = Signal(object)

    def __init__(
        self,
        default_camera_config: Path,
        default_camera_channel: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.default_camera_config = default_camera_config
        self.default_camera_channel = default_camera_channel
        self.project: WizardProject | None = None
        self._camera_config_error: str | None = None
        layout = QVBoxLayout(self)
        title = QLabel("1. 创建或打开标定项目")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        form = QFormLayout()
        self.project_id = QLineEdit("line-laser-calibration")
        self.workspace = QLineEdit(str(default_camera_config.parent.parent / "projects" / "default"))
        self.camera_config = QLineEdit(str(default_camera_config))
        self.camera_channel = QComboBox()
        self.laser_orientation = QComboBox(); self.laser_orientation.addItems(["horizontal", "vertical"])
        self.workflow_plan = QLineEdit("")
        self.acceptance_plan = QLineEdit("")
        self.pattern_cols = QSpinBox(); self.pattern_cols.setRange(2, 50); self.pattern_cols.setValue(11)
        self.pattern_rows = QSpinBox(); self.pattern_rows.setRange(2, 50); self.pattern_rows.setValue(8)
        self.square_size = QDoubleSpinBox(); self.square_size.setRange(0.01, 1000); self.square_size.setValue(20); self.square_size.setSuffix(" mm")
        form.addRow("项目 ID", self.project_id)
        form.addRow("项目工作目录", _path_row(self.workspace, self, directory=True))
        form.addRow("相机通道", self.camera_channel)
        form.addRow("通道注册表/相机配置", _path_row(self.camera_config, self, file_filter="YAML (*.yaml *.yml)"))
        form.addRow("激光线方向", self.laser_orientation)
        form.addRow("标定 workflow", _path_row(self.workflow_plan, self, file_filter="YAML (*.yaml *.yml)"))
        form.addRow("验收计划", _path_row(self.acceptance_plan, self, file_filter="YAML (*.yaml *.yml)"))
        form.addRow("棋盘内角点列数", self.pattern_cols)
        form.addRow("棋盘内角点行数", self.pattern_rows)
        form.addRow("方格尺寸", self.square_size)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.open_button = QPushButton("打开项目…")
        self.save_button = QPushButton("保存项目…")
        self.apply_button = QPushButton("应用到向导")
        self.apply_button.setDefault(True)
        buttons.addWidget(self.open_button); buttons.addWidget(self.save_button); buttons.addStretch(); buttons.addWidget(self.apply_button)
        layout.addLayout(buttons)
        layout.addStretch()
        self.open_button.clicked.connect(self._open)
        self.save_button.clicked.connect(self._save)
        self.apply_button.clicked.connect(self.apply)
        self.camera_config.textChanged.connect(self._camera_config_path_changed)
        self.camera_channel.currentIndexChanged.connect(self._camera_channel_changed)
        self._reload_camera_channels(
            str(default_camera_config),
            preferred=default_camera_channel,
            update_workflow=True,
        )

    def apply(self) -> WizardProject | None:
        try:
            project = self._from_fields()
        except Exception as exc:
            QMessageBox.critical(self, "项目配置无效", str(exc))
            return None
        self.project = project
        self.project_changed.emit(project)
        return project

    def _from_fields(self) -> WizardProject:
        if self._camera_config_error:
            raise ValueError(self._camera_config_error)
        workspace = Path(self.workspace.text()).expanduser().resolve()
        workflow = self.workflow_plan.text().strip()
        capture_output = (
            self.project.capture_output
            if self.project is not None and self.project.capture_output is not None
            else workspace / "data"
        )
        # 默认工作区为 projects/default，因此默认数据集位于
        # projects/default/data；已加载项目仍保留 YAML 中的 capture_output。
        return WizardProject(
            project_id=self.project_id.text().strip(),
            workspace=workspace,
            camera_config=Path(self.camera_config.text()),
            camera_channel=self._selected_camera_channel(),
            laser=LaserConfig(self.laser_orientation.currentText()),
            workflow_plan=Path(workflow) if workflow else None,
            acceptance_plan=Path(self.acceptance_plan.text().strip()) if self.acceptance_plan.text().strip() else None,
            capture_output=capture_output,
            pattern_cols=self.pattern_cols.value(),
            pattern_rows=self.pattern_rows.value(),
            square_size_mm=self.square_size.value(),
            source_path=self.project.source_path if self.project else None,
            extra=self.project.extra if self.project else {},
            last_calibration_run=self.project.last_calibration_run if self.project else None,
        )

    def _selected_camera_channel(self) -> str | None:
        value = self.camera_channel.currentData()
        return str(value) if value not in (None, "") else None

    def _camera_config_path_changed(self, path_text: str) -> None:
        self._reload_camera_channels(path_text, update_workflow=True)

    def _reload_camera_channels(
        self,
        path_text: str,
        *,
        preferred: str | None = None,
        update_workflow: bool = False,
    ) -> None:
        """从统一注册表填充通道；旧版单相机 YAML 显示为一个直连通道。"""

        try:
            path = Path(path_text).expanduser()
            if not path.is_file():
                self._camera_config_error = f"相机通道注册表/配置不存在：{path}"
                return
            runtime = load_camera_config(path, channel=preferred)
        except Exception as exc:
            # 输入路径过程中可能暂时无效；最终校验仍由 apply() 负责。
            self._camera_config_error = str(exc)
            return
        self._camera_config_error = None

        registry = runtime.get("channel_registry")
        with QSignalBlocker(self.camera_channel):
            self.camera_channel.clear()
            if registry is None:
                self.camera_channel.addItem(
                    f"直接配置（{runtime['backend']}）",
                    None,
                )
                self.camera_channel.setEnabled(False)
            else:
                for definition in registry.channels:
                    self.camera_channel.addItem(
                        f"{definition.label}  [{definition.name}]",
                        definition.name,
                    )
                selected = self.camera_channel.findData(runtime["channel"])
                self.camera_channel.setCurrentIndex(max(0, selected))
                self.camera_channel.setEnabled(len(registry.channels) > 1)
        self._apply_camera_runtime(runtime, update_workflow=update_workflow)

    def _camera_channel_changed(self, _index: int) -> None:
        try:
            runtime = load_camera_config(
                Path(self.camera_config.text()),
                channel=self._selected_camera_channel(),
            )
        except Exception as exc:
            self._camera_config_error = str(exc)
            return
        self._camera_config_error = None
        self._apply_camera_runtime(runtime, update_workflow=True)

    def _apply_camera_runtime(
        self,
        runtime: Mapping[str, Any],
        *,
        update_workflow: bool,
    ) -> None:
        self.laser_orientation.setCurrentText(runtime["laser"].orientation)
        workflow = runtime.get("workflow_plan")
        if update_workflow and workflow is not None:
            self.workflow_plan.setText(str(workflow))

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开标定项目", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            self.set_project(WizardProject.load(path))
            self.apply()
        except Exception as exc:
            QMessageBox.critical(self, "无法打开项目", str(exc))

    def _save(self) -> None:
        project = self.apply()
        if project is None:
            return
        initial = str(project.source_path or (project.workspace / "wizard_project.yaml"))
        path, _ = QFileDialog.getSaveFileName(self, "保存标定项目", initial, "YAML (*.yaml)")
        if path:
            try:
                project.workspace.mkdir(parents=True, exist_ok=True)
                project.save(path)
            except Exception as exc:
                QMessageBox.critical(self, "无法保存项目", str(exc))

    def set_project(self, project: WizardProject) -> None:
        self.project = project
        self.project_id.setText(project.project_id)
        self.workspace.setText(str(project.workspace))
        self.camera_config.setText(str(project.camera_config))
        self._reload_camera_channels(
            str(project.camera_config),
            preferred=project.camera_channel,
            update_workflow=False,
        )
        self.laser_orientation.setCurrentText(project.laser.orientation)
        self.workflow_plan.setText(str(project.workflow_plan or ""))
        self.acceptance_plan.setText(str(project.acceptance_plan or ""))
        self.pattern_cols.setValue(project.pattern_cols)
        self.pattern_rows.setValue(project.pattern_rows)
        self.square_size.setValue(project.square_size_mm)


class CameraPage(QWidget):
    status_changed = Signal(str)
    frame_ready = Signal(object, object)

    def __init__(self, thread_pool: QThreadPool, parent=None) -> None:
        super().__init__(parent)
        self.thread_pool = thread_pool
        self.runtime: dict[str, Any] | None = None
        self.preview_thread: PreviewThread | None = None
        self.last_frame = None
        self.last_quality: dict[str, Any] | None = None
        self._pending_capture: tuple[int, Callable[[Any, dict[str, Any]], None]] | None = None
        self._workers: set[FunctionWorker] = set()
        self._enumeration_token = 0
        layout = QVBoxLayout(self)
        title = QLabel("2. 连接相机并调整采集参数"); title.setObjectName("pageTitle"); layout.addWidget(title)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        controls = QWidget(); form = QFormLayout(controls)
        self.config_path = QLineEdit()
        self.config_path.setReadOnly(True)
        self.reload_config_button = QPushButton("重新加载")
        row = QHBoxLayout(); row.addWidget(self.config_path, 1); row.addWidget(self.reload_config_button)
        form.addRow("通道配置", row)
        self.backend_label = QLabel("--")
        self.devices = QComboBox()
        self.refresh_button = QPushButton("枚举相机")
        device_row = QHBoxLayout(); device_row.addWidget(self.devices, 1); device_row.addWidget(self.refresh_button)
        form.addRow("当前通道 / 后端", self.backend_label); form.addRow("设备", device_row)
        self.exposure = QDoubleSpinBox(); self.exposure.setRange(1, 10_000_000); self.exposure.setDecimals(1); self.exposure.setSuffix(" μs")
        self.gain = QDoubleSpinBox(); self.gain.setRange(-100, 100); self.gain.setDecimals(2); self.gain.setSuffix(" dB")
        self.pixel_format = QComboBox(); self.pixel_format.addItems(["Mono8", "Mono12"])
        self.offset_x = QSpinBox(); self.offset_x.setRange(0, 10000)
        self.offset_y = QSpinBox(); self.offset_y.setRange(0, 10000)
        self.width = QSpinBox(); self.width.setRange(1, 20000)
        self.height = QSpinBox(); self.height.setRange(1, 20000)
        self.quality_mode = QComboBox()
        self.quality_mode.addItem("激光线（激光开启）", "laser")
        self.quality_mode.addItem("棋盘格（内参时关闭激光）", "chessboard")
        self.quality_mode.addItem("通用曝光", "generic")
        for label, widget in (("曝光", self.exposure), ("增益", self.gain), ("像素格式", self.pixel_format),
                              ("Offset X", self.offset_x), ("Offset Y", self.offset_y), ("宽度", self.width),
                              ("高度", self.height), ("质量模式", self.quality_mode)):
            form.addRow(label, widget)
        self.apply_live_button = QPushButton("应用曝光/增益")
        self.auto_stretch = QCheckBox("自动拉伸预览（仅改变显示，不代表实际曝光）")
        self.auto_stretch.setChecked(False)
        form.addRow("在线参数", self.apply_live_button)
        form.addRow("显示映射", self.auto_stretch)
        self.quality_help = QLabel()
        self.quality_help.setWordWrap(True)
        form.addRow("筛查内容", self.quality_help)
        action_row = QHBoxLayout()
        self.preview_button = QPushButton("开始取流")
        self.stop_button = QPushButton("停止")
        self.snapshot_button = QPushButton("保存当前帧…")
        action_row.addWidget(self.preview_button); action_row.addWidget(self.stop_button); action_row.addWidget(self.snapshot_button)
        form.addRow(action_row)
        self.status = QLabel("尚未加载相机配置"); self.status.setWordWrap(True); form.addRow("状态", self.status)
        self.quality = QLabel("--"); self.quality.setWordWrap(True); form.addRow("质量", self.quality)
        self.search_region_quality = QLabel("Search region: --", controls)
        self.search_region_quality.setWordWrap(True)
        if _SEARCH_REGION_QUALITY_ENABLED:
            form.addRow("Search Region Quality", self.search_region_quality)
        else:
            self.search_region_quality.hide()
        splitter.addWidget(controls)
        self.preview = ImagePreview(); splitter.addWidget(self.preview); splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        self.reload_config_button.clicked.connect(
            lambda: self.load_config(
                Path(self.config_path.text()),
                channel=(self.runtime or {}).get("channel"),
            )
        )
        self.refresh_button.clicked.connect(lambda: self.enumerate_devices())
        self.preview_button.clicked.connect(self.start_preview)
        self.stop_button.clicked.connect(self.stop_preview)
        self.snapshot_button.clicked.connect(self.save_snapshot)
        self.apply_live_button.clicked.connect(self.apply_live_parameters)
        self.exposure.editingFinished.connect(self.apply_live_parameters)
        self.gain.editingFinished.connect(self.apply_live_parameters)
        self.auto_stretch.toggled.connect(
            lambda checked: self.preview.refresh_display(auto_stretch=checked)
        )
        self.quality_mode.currentIndexChanged.connect(self._update_quality_help)
        self._update_quality_help()

    def load_config(self, path: Path, *, channel: str | None = None) -> bool:
        try:
            runtime = load_camera_config(path, channel=channel)
        except Exception as exc:
            QMessageBox.critical(self, "相机配置无效", str(exc)); return False
        # 使旧通道尚未结束的枚举结果失效，避免快速切换时覆盖当前设备列表。
        self._enumeration_token += 1
        self.refresh_button.setEnabled(True)
        self.devices.clear()
        self.runtime = runtime
        self.config_path.setText(str(runtime["source"]))
        self.config_path.setToolTip(
            f"当前通道实际配置：{runtime['camera_config_source']}"
        )
        self.backend_label.setText(
            f"{runtime['channel_label']}  /  {runtime['backend']}"
        )
        config = runtime["camera"]
        self.exposure.setValue(config.exposure_us); self.gain.setValue(config.gain_db)
        self.pixel_format.setCurrentText(config.pixel_format)
        self.offset_x.setValue(config.offset_x); self.offset_y.setValue(config.offset_y)
        self.width.setValue(config.width); self.height.setValue(config.height)
        self.status.setText("配置已加载，请枚举相机")
        return True

    def current_config(self):
        if self.runtime is None:
            raise RuntimeError("请先加载相机配置")
        return replace(
            self.runtime["camera"], exposure_us=self.exposure.value(), gain_db=self.gain.value(),
            pixel_format=self.pixel_format.currentText(), offset_x=self.offset_x.value(), offset_y=self.offset_y.value(),
            width=self.width.value(), height=self.height.value(),
        )

    def selected_serial(self) -> str:
        return str(self.devices.currentData() or (self.runtime or {}).get("serial_number", ""))

    def enumerate_devices(self, *, silent: bool = False) -> None:
        if self.runtime is None:
            QMessageBox.warning(self, "尚未配置", "请先加载相机配置"); return
        self.refresh_button.setEnabled(False); self.status.setText("正在枚举相机…")
        runtime = self.runtime
        self._enumeration_token += 1
        token = self._enumeration_token
        worker = FunctionWorker(lambda _progress: build_camera_provider(
            runtime["backend"], calibration_src=runtime["calibration_src"], backend_options=runtime["backend_options"]
        ).list_devices())
        worker.signals.result.connect(
            lambda devices: self._set_devices(devices)
            if token == self._enumeration_token
            else None
        )
        if silent:
            worker.signals.error.connect(
                lambda message: self.status.setText(f"相机枚举失败：{message}")
                if token == self._enumeration_token
                else None
            )
        else:
            worker.signals.error.connect(
                lambda message: self._show_error("相机枚举失败", message)
                if token == self._enumeration_token
                else None
            )
        worker.signals.finished.connect(
            lambda: self.refresh_button.setEnabled(True)
            if token == self._enumeration_token
            else None
        )
        self._start_worker(worker)

    def _set_devices(self, devices: list[Any]) -> None:
        self.devices.clear()
        for device in devices:
            self.devices.addItem(device.display_name, device.serial_number)
        configured_serial = str((self.runtime or {}).get("serial_number", ""))
        if configured_serial:
            selected = self.devices.findData(configured_serial)
            if selected >= 0:
                self.devices.setCurrentIndex(selected)
        self.status.setText(f"找到 {len(devices)} 台相机")

    def start_preview(self, initial_discard_frames: int = 3) -> None:
        if self.runtime is None:
            QMessageBox.warning(self, "尚未配置", "请先加载相机配置"); return
        self.stop_preview()
        try:
            provider = build_camera_provider(
                self.runtime["backend"], calibration_src=self.runtime["calibration_src"], backend_options=self.runtime["backend_options"]
            )
            steger_quality_analyzer = (
                RealtimeStegerQualityAnalyzer(
                    self.runtime["calibration_src"],
                    self.runtime["laser"].orientation,
                )
                if _SEARCH_REGION_QUALITY_ENABLED
                else None
            )
            thread = PreviewThread(
                provider, self.selected_serial(), self.current_config(), str(self.quality_mode.currentData()),
                self.runtime["quality_thresholds"], self.runtime["board_pattern"],
                laser_orientation=self.runtime["laser"].orientation,
                steger_quality_analyzer=steger_quality_analyzer,
                initial_discard_frames=initial_discard_frames, parent=self,
            )
        except Exception as exc:
            self._show_error("无法开始取流", str(exc)); return
        self.preview_thread = thread
        thread.opened.connect(lambda device, config: self.status.setText(
            f"取流中 · {device.model} · SN {device.serial_number} · {config.width}×{config.height}"
        ))
        thread.frame_ready.connect(self._on_frame)
        thread.failed.connect(lambda message: self._show_error("取流失败", message))
        thread.settings_applied.connect(self._settings_applied)
        thread.parameter_update_failed.connect(
            lambda message: self._show_error("在线参数更新失败", message)
        )
        thread.finished.connect(lambda thread=thread: self._preview_finished(thread))
        self.preview_button.setEnabled(False); self.stop_button.setEnabled(True)
        self._set_restart_controls_enabled(False)
        thread.start()

    def _on_frame(self, frame, quality: dict[str, Any]) -> None:
        self.last_frame = frame
        self.last_quality = quality
        sensor_max = self.preview_thread.config.sensor_max_value if self.preview_thread else self.current_config().sensor_max_value
        # CameraPage 与 CapturePage 都订阅同一帧。只绘制当前可见页面，避免
        # 每帧为隐藏的 QPixmap 再做一次全分辨率拷贝和 smooth scaling。
        if self.preview.isVisible():
            self.preview.set_array(
                frame.image,
                auto_stretch=self.auto_stretch.isChecked(),
                sensor_max_value=sensor_max,
            )
        warnings = "、".join(_quality_warning_text(item) for item in quality["warnings"]) or "通过"
        thresholds = self.runtime["quality_thresholds"] if self.runtime else None
        coverage_text = _laser_quality_metrics_text(quality, thresholds)
        chess_hint = quality.get("chessboard_hint")
        chess_text = f" · {chess_hint}" if chess_hint else ""
        settling, settle_remaining = _preview_settling_info(quality)
        settling_text = f" · 正在稳定，剩余 {settle_remaining} 帧" if settling else ""
        self.quality.setText(
            f"{warnings} · 动态范围 {quality['dynamic_range_u8']:.1f} DN8 · "
            f"清晰度 {quality['focus_laplacian']:.1f}{coverage_text}{chess_text}{settling_text}"
        )
        self.search_region_quality.setText(_search_region_quality_text(quality))
        self.frame_ready.emit(frame, quality)
        pending = self._pending_capture
        if pending is not None:
            remaining, callback = pending
            if remaining > 0:
                self._pending_capture = (remaining - 1, callback)
            else:
                self._pending_capture = None
                callback(frame, quality)

    def _update_quality_help(self) -> None:
        descriptions = {
            "generic": "检查过曝、欠曝、全局动态范围；清晰度只显示数值，暂不设统一阈值。",
            "laser": "允许暗背景，按配置方向检查激光线覆盖率、过曝和动态范围。",
            "chessboard": "检查曝光、动态范围和完整内角点检测。内参棋盘图应关闭激光。",
        }
        self.quality_help.setText(descriptions[str(self.quality_mode.currentData())])

    def stop_preview(self) -> None:
        self.cancel_pending_capture()
        thread = self.preview_thread
        if thread is not None and thread.isRunning():
            self.status.setText("正在停止取流…")
            if not thread.stop():
                self.status.setText("相机停止超时，请检查连接")
                return
            self._preview_finished(thread)

    def capture_after_settle(
        self,
        discard_frames: int,
        callback: Callable[[Any, dict[str, Any]], None],
    ) -> bool:
        """在当前预览流中丢弃指定帧后，把下一帧交给回调。"""
        thread = self.preview_thread
        if thread is None or thread.isFinished() or self._pending_capture is not None:
            return False
        self._pending_capture = (max(0, int(discard_frames)), callback)
        thread.request_fresh_quality_after_frames(discard_frames)
        return True

    def request_preview_task(
        self,
        config: CameraConfig,
        quality_mode: str,
        settle_frames: int = 0,
    ) -> bool:
        """让现有 PreviewThread 在线切换任务配置并继续发帧。"""

        thread = self.preview_thread
        if thread is None or not thread.isRunning():
            return False
        return thread.request_task_config(config, quality_mode, settle_frames)

    def cancel_pending_capture(self) -> None:
        self._pending_capture = None
        thread = self.preview_thread
        if thread is not None:
            thread.cancel_fresh_quality_request()

    def apply_live_parameters(self) -> None:
        thread = self.preview_thread
        if thread is None or not thread.isRunning():
            self.status.setText("曝光/增益将在下次开始取流时应用")
            return
        self.status.setText(
            f"正在应用曝光 {self.exposure.value():g} μs、增益 {self.gain.value():g} dB…"
        )
        thread.request_exposure_gain(self.exposure.value(), self.gain.value())

    def _settings_applied(self, config) -> None:
        with QSignalBlocker(self.exposure):
            self.exposure.setValue(config.exposure_us)
        with QSignalBlocker(self.gain):
            self.gain.setValue(config.gain_db)
        self.status.setText(
            f"取流中 · 相机回读：曝光 {config.exposure_us:g} μs，增益 {config.gain_db:g} dB"
        )

    def _set_restart_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.reload_config_button, self.devices, self.refresh_button, self.pixel_format, self.offset_x,
            self.offset_y, self.width, self.height, self.quality_mode,
        ):
            widget.setEnabled(enabled)

    def _preview_finished(self, thread: PreviewThread | None = None) -> None:
        if thread is not None and self.preview_thread is not thread:
            return
        self.cancel_pending_capture()
        self.preview_button.setEnabled(True); self.stop_button.setEnabled(False)
        self._set_restart_controls_enabled(True)
        if not self.status.text().startswith("取流失败"):
            self.status.setText("取流已停止")
        self.preview_thread = None

    def save_snapshot(self) -> None:
        if self.last_frame is None:
            QMessageBox.information(self, "没有图像", "请先开始取流"); return
        default_suffix = ".tif" if self.last_frame.image.dtype.itemsize > 1 else ".png"
        path, _ = QFileDialog.getSaveFileName(self, "保存当前帧", f"snapshot{default_suffix}", "TIFF (*.tif);;PNG (*.png)")
        if not path:
            return
        ok, encoded = cv2.imencode(Path(path).suffix, self.last_frame.image)
        if not ok:
            QMessageBox.critical(self, "保存失败", "OpenCV 无法编码该图像"); return
        Path(path).write_bytes(encoded.tobytes())

    def _show_error(self, title: str, message: str) -> None:
        self.status.setText(f"{title}：{message}")
        QMessageBox.critical(self, title, message)

    def _start_worker(self, worker: FunctionWorker) -> None:
        self._workers.add(worker)
        worker.signals.finished.connect(lambda: self._workers.discard(worker))
        self.thread_pool.start(worker)


class CapturePage(QWidget):
    capture_finished = Signal(object)
    request_camera_page = Signal()

    def __init__(self, thread_pool: QThreadPool, camera_page: CameraPage, parent=None) -> None:
        super().__init__(parent)
        self.thread_pool = thread_pool; self.camera_page = camera_page; self._workers: set[FunctionWorker] = set()
        self.loaded_plan: CapturePlan | None = None
        self.loaded_plan_path: Path | None = None
        self._generated_recipe: CaptureRecipe | None = None
        self._plan_dirty = True
        self._capture_gate: CaptureTaskGate | None = None
        self._capture_cancel_event: threading.Event | None = None
        self._capture_worker: FunctionWorker | None = None
        self._guided_capture_active = False
        self._last_completed_pose: str | None = None
        self.last_capture_artifacts: dict[str, str] | None = None
        self.guided_preview_index: int | None = None
        self._preview_request_token = 0
        self._pending_preview_callback: Callable[[], None] | None = None
        self._capture_in_progress = False
        self.laser = LaserConfig()
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_content = QWidget()
        self.scroll_content_layout = QVBoxLayout(self.scroll_content)
        self.scroll_content_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidget(self.scroll_content)
        page_layout.addWidget(self.scroll_area)
        layout = self.scroll_content_layout
        title = QLabel("3. 批量采集标定图像"); title.setObjectName("pageTitle"); layout.addWidget(title)
        form = QFormLayout()
        self.plan_path = QLineEdit()
        self.load_plan_button = QPushButton("加载计划")
        plan_row = QHBoxLayout(); plan_row.addWidget(_path_row(self.plan_path, self, file_filter="YAML (*.yaml *.yml)"), 1); plan_row.addWidget(self.load_plan_button)
        self.output = QLineEdit(); self.dataset_id = QLineEdit("laser_plane")
        self.image_format = QComboBox(); self.image_format.addItem("TIFF（正式标定默认）", "tif"); self.image_format.addItem("PNG", "png")
        self.fit_groups = QSpinBox(); self.fit_groups.setRange(1, 10000); self.fit_groups.setValue(18)
        self.start_index = QSpinBox(); self.start_index.setRange(0, 1_000_000); self.start_index.setValue(1)
        self.index_digits = QSpinBox(); self.index_digits.setRange(1, 8); self.index_digits.setValue(3)
        self.include_validation = QCheckBox("包含独立验证集"); self.include_validation.setChecked(True)
        self.validation_groups = QSpinBox(); self.validation_groups.setRange(1, 10000); self.validation_groups.setValue(6)
        self.resume = QCheckBox("续采对应的 .inprogress 数据集")
        # 保留语义化别名，便于项目内其它页面/测试读取而不依赖控件布局。
        self.output_dir = self.output
        self.plan_output_path = self.plan_path
        self.fit_group_count = self.fit_groups
        self.validation_group_count = self.validation_groups
        form.addRow("计划 YAML（生成/加载）", plan_row)
        form.addRow("输出数据集", _path_row(self.output, self, directory=True))
        form.addRow("数据集 ID", self.dataset_id); form.addRow("图像格式", self.image_format)
        form.addRow("拟合集组数", self.fit_groups); form.addRow("起始编号", self.start_index)
        form.addRow("编号位数", self.index_digits); form.addRow("验证集", self.include_validation)
        form.addRow("验证集组数", self.validation_groups); form.addRow("异常恢复", self.resume)
        layout.addLayout(form)

        self.recipe_table = CaptureRecipeTable()
        layout.addWidget(QLabel("每组图像配方")); layout.addWidget(self.recipe_table)
        plan_action_row = QHBoxLayout()
        self.generate_plan_button = QPushButton("生成并检查计划")
        self.restore_recipe_button = QPushButton("恢复三联图默认值")
        plan_action_row.addWidget(self.generate_plan_button); plan_action_row.addWidget(self.restore_recipe_button); plan_action_row.addStretch()
        layout.addLayout(plan_action_row)
        self.plan_summary = QLabel("尚未生成计划")
        self.plan_summary.setWordWrap(True)
        layout.addWidget(self.plan_summary)
        self.plan_table = QTableWidget(0, 9)
        self.plan_table.setHorizontalHeaderLabels([
            "split", "pose_id", "task_id", "role", "曝光 μs", "激光状态", "质量模式", "输出文件", "frames",
        ])
        self.plan_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.plan_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.plan_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.plan_table.setMaximumHeight(220)
        self.plan_table.horizontalHeaderItem(4).setToolTip("双击可单独修改任务曝光；修改后自动保存到计划 YAML。")
        self.plan_preview = self.plan_table
        layout.addWidget(self.plan_table)

        self.plan_tasks = QListWidget()
        self.plan_tasks.setMaximumHeight(150)
        self.plan_status = QLabel("请先生成并检查计划；也可加载旧版 capture-plan YAML。")
        self.plan_status.setWordWrap(True)
        plan_buttons = QHBoxLayout()
        self.preview_task_button = QPushButton("预览选中任务")
        self.capture_task_button = QPushButton("采集当前任务全部帧")
        self.next_task_button = QPushButton("下一任务")
        plan_buttons.addWidget(self.preview_task_button); plan_buttons.addWidget(self.capture_task_button); plan_buttons.addWidget(self.next_task_button)
        self.start_button = QPushButton("开始采集")
        self.resume_button = QPushButton("续采")
        self.cancel_button = QPushButton("取消")
        # 引导采集期间同一个入口负责采满当前任务并切换下一任务。
        # 旧的后台 run_capture_plan API 仍保留，但不再让实时画面承担 gate 等待。
        self.capture_task_button.setToolTip(
            "点击后先丢弃当前任务的稳定帧，再一次采满全部剩余帧；完成后自动切换下一任务。"
        )
        # 兼容旧测试/插件属性，但不再创建第二个可见按钮。
        self.ready_button = self.capture_task_button
        self.resume_button.setEnabled(False); self.cancel_button.setEnabled(False); self.capture_task_button.setEnabled(False)
        capture_buttons = QHBoxLayout()
        capture_buttons.addWidget(self.start_button); capture_buttons.addWidget(self.resume_button)
        capture_buttons.addWidget(self.cancel_button)
        self.progress = QTextEdit(); self.progress.setReadOnly(True)
        self.current_task = QLabel("当前任务：--")
        self.current_task.setWordWrap(True)

        live_panel = QGroupBox("任务实时画面")
        live_layout = QVBoxLayout(live_panel)
        self.live_preview = ImagePreview()
        self.live_auto_stretch = QCheckBox("自动拉伸预览（仅改变显示）")
        self.live_quality = QLabel("尚未取流")
        self.live_quality.setWordWrap(True)
        self.live_search_region_quality = QLabel("Search region: --", live_panel)
        self.live_search_region_quality.setWordWrap(True)
        self.live_camera = QLabel("当前任务：--")
        self.live_camera.setWordWrap(True)
        live_layout.addWidget(self.live_preview, 1)
        live_layout.addWidget(self.live_auto_stretch)
        live_layout.addWidget(self.live_camera)
        live_layout.addWidget(self.live_quality)
        if _SEARCH_REGION_QUALITY_ENABLED:
            live_layout.addWidget(self.live_search_region_quality)
        else:
            self.live_search_region_quality.hide()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.plan_tasks)
        left_layout.addWidget(self.plan_status)
        left_layout.addLayout(plan_buttons)
        left_layout.addWidget(self.current_task)
        left_layout.addLayout(capture_buttons)
        left_layout.addWidget(self.progress, 1)
        content = QSplitter(Qt.Orientation.Horizontal)
        content.addWidget(left)
        content.addWidget(live_panel)
        content.setStretchFactor(0, 1)
        content.setStretchFactor(1, 1)
        layout.addWidget(content, 1)

        # 可见按钮进入 GUI 引导采集；后台 run_capture_plan 入口保留在
        # start_capture()，供旧调用方和兼容测试使用。
        self.start_button.clicked.connect(lambda _checked=False: self.start_guided_capture())
        self.resume_button.clicked.connect(lambda: self.start_guided_capture(resume=True))
        self.cancel_button.clicked.connect(self.cancel_capture)
        self.capture_task_button.clicked.connect(self._capture_task_button_clicked)
        self.generate_plan_button.clicked.connect(self.generate_plan)
        self.restore_recipe_button.clicked.connect(self.recipe_table.reset_defaults)
        self.load_plan_button.clicked.connect(self.load_guided_plan)
        self.preview_task_button.clicked.connect(self.preview_selected_task)
        self.next_task_button.clicked.connect(self.select_next_task)
        self.plan_table.itemChanged.connect(self._on_plan_item_changed)
        self.plan_table.cellClicked.connect(lambda row, _column: self.plan_tasks.setCurrentRow(row))
        self.plan_tasks.currentRowChanged.connect(self.plan_table.selectRow)
        self.live_auto_stretch.toggled.connect(
            lambda checked: self.live_preview.refresh_display(auto_stretch=checked)
        )
        self.camera_page.frame_ready.connect(self._on_camera_frame)
        for signal in (
            self.plan_path.textChanged,
            self.output.textChanged,
            self.dataset_id.textChanged,
            self.image_format.currentIndexChanged,
            self.fit_groups.valueChanged,
            self.start_index.valueChanged,
            self.index_digits.valueChanged,
            self.include_validation.toggled,
            self.validation_groups.valueChanged,
        ):
            signal.connect(self._mark_plan_dirty)
        self.recipe_table.changed.connect(self._mark_plan_dirty)

    def set_project(self, project: WizardProject) -> None:
        self.laser = project.laser
        self.output.setText(str((project.capture_output or project.workspace / "data") / self.dataset_id.text()))
        stored_artifacts = project.extra.get("capture_artifacts") if isinstance(project.extra, dict) else None
        self.last_capture_artifacts = dict(stored_artifacts) if isinstance(stored_artifacts, dict) else None
        if project.workflow_plan and project.workflow_plan.name.startswith("capture_"):
            self.plan_path.setText(str(resolve_relative(project.source_path or project.workspace / "wizard_project.yaml", project.workflow_plan)))

        if not self.plan_path.text().strip():
            self.plan_path.setText(str(project.workspace / "plans" / f"{self.dataset_id.text().strip()}.yaml"))
        if self.loaded_plan is not None:
            mismatch = self._plan_camera_mismatch(self.loaded_plan)
            if mismatch:
                self._mark_plan_dirty()
                self.plan_status.setText(f"相机通道已改变：{mismatch}；请重新生成计划。")

    def _plan_camera_mismatch(self, plan: CapturePlan) -> str | None:
        runtime = self.camera_page.runtime
        if runtime is None:
            return "尚未加载相机通道"
        if plan.backend != runtime["backend"]:
            return f"计划 backend={plan.backend}，当前通道 backend={runtime['backend']}"
        plan_channel = plan.metadata.get("camera_channel")
        runtime_channel = runtime.get("channel")
        if plan_channel and runtime_channel and str(plan_channel) != str(runtime_channel):
            return f"计划通道={plan_channel}，当前通道={runtime_channel}"
        configured_serial = str(runtime.get("serial_number", ""))
        if configured_serial and plan.serial_number and configured_serial != plan.serial_number:
            return f"计划序列号={plan.serial_number}，当前配置序列号={configured_serial}"
        return None

    def _mark_plan_dirty(self, *_args: Any) -> None:
        if self._capture_worker is not None or self._guided_capture_active:
            return
        self._plan_dirty = True
        self.start_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.capture_task_button.setEnabled(False)
        self.plan_status.setText("配置已改变，需要重新生成并检查计划。")

    def _recipe_from_fields(self) -> CaptureRecipe:
        runtime = self.camera_page.runtime
        if runtime is None:
            raise ValueError("请先在第 2 页加载相机配置")
        image_format = str(self.image_format.currentData())
        return CaptureRecipe(
            dataset_id=self.dataset_id.text().strip(),
            output_dir=Path(self.output.text().strip()),
            plan_output_path=Path(self.plan_path.text().strip()),
            fit_group_count=self.fit_groups.value(),
            include_validation=self.include_validation.isChecked(),
            validation_group_count=self.validation_groups.value(),
            start_index=self.start_index.value(),
            index_digits=self.index_digits.value(),
            camera=self.camera_page.current_config(),
            serial_number=self.camera_page.selected_serial() or str(runtime.get("serial_number", "")),
            backend=str(runtime["backend"]),
            backend_options=dict(runtime.get("backend_options", {})),
            metadata={
                "created_by": "pyside6_wizard",
                "camera_channel": runtime.get("channel"),
                "camera_config": str(
                    runtime.get("camera_config_source", runtime.get("source", ""))
                ),
            },
            items=self.recipe_table.recipe_items(image_format),
            board_pattern=runtime.get("board_pattern") or (11, 8),
            quality_thresholds=runtime["quality_thresholds"],
            laser=self.laser,
        )

    def generate_plan(self) -> bool:
        try:
            recipe = self._recipe_from_fields()
            plan = build_capture_plan_from_recipe(recipe)
            # “生成并检查”是用户明确的重生成动作；底层 API 仍默认拒绝静默覆盖。
            saved = save_generated_capture_plan(plan, recipe.plan_output_path, overwrite=True)
            loaded = load_capture_plan(saved)
        except Exception as exc:
            self._plan_dirty = True
            self.start_button.setEnabled(False); self.resume_button.setEnabled(False)
            QMessageBox.critical(self, "采集计划无效", str(exc))
            return False
        self.loaded_plan = loaded
        self.loaded_plan_path = saved.resolve()
        self._generated_recipe = recipe
        self._show_plan(loaded)
        self.plan_path.setText(str(saved.resolve()))
        self._plan_dirty = False
        self._set_capture_buttons_ready()
        self.plan_status.setText(f"计划已保存并 round-trip 校验：{saved}")
        return True

    def _show_plan(self, plan: CapturePlan) -> None:
        summary = capture_plan_summary(plan)
        self.plan_summary.setText(
            f"拟合组 {summary['fit_group_count']} · 验证组 {summary['validation_group_count']} · "
            f"任务 {summary['task_count']} · 图像 {summary['image_count']} · 双击曝光列可单独调节"
        )
        with QSignalBlocker(self.plan_table):
            self.plan_table.setRowCount(len(summary["tasks"]))
            for row, record in enumerate(summary["tasks"]):
                values = (
                    record["split"], record["pose_id"], record["task_id"], record["role"],
                    f"{record['exposure_us']:g}", record["laser_state"], record["quality_mode"],
                    record["relative_output_path"], str(record["frames"]),
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    if column == 4:
                        item.setToolTip("双击输入该任务的曝光时间（μs），按 Enter 保存。")
                    else:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.plan_table.setItem(row, column, item)
        self.plan_tasks.clear()
        for index, task in enumerate(plan.tasks, start=1):
            self.plan_tasks.addItem(self._plan_task_text(index, task))
        if plan.tasks:
            self.plan_tasks.setCurrentRow(0)
            self.plan_table.selectRow(0)
        self.guided_preview_index = None

    @staticmethod
    def _plan_task_text(index: int, task: CaptureTask) -> str:
        return (
            f"{index:02d}. {task.task_id} · pose {task.pose_id} · {task.quality_mode} · "
            f"{task.config.exposure_us:g} μs · {task.instruction}"
        )

    def _on_plan_item_changed(self, item: QTableWidgetItem) -> None:
        """把计划表中的单任务曝光修改回写到模型、YAML 和续采 manifest。"""

        if item.column() != 4 or self.loaded_plan is None:
            return
        row = item.row()
        if row < 0 or row >= len(self.loaded_plan.tasks):
            return
        old_plan = self.loaded_plan
        old_task = old_plan.tasks[row]

        def restore_cell() -> None:
            with QSignalBlocker(self.plan_table):
                item.setText(f"{old_task.config.exposure_us:g}")

        try:
            if self._plan_dirty:
                raise ValueError("配置已改变，请先重新生成并检查计划")
            if self._capture_worker is not None:
                raise ValueError("后台采集正在运行，不能修改任务曝光")
            if self._capture_in_progress:
                raise ValueError("当前任务正在采集，完成当前任务后再调整下一任务曝光")
            exposure_us = float(item.text().strip())
            updated_task = replace(
                old_task,
                config=old_task.config.updated({"exposure_us": exposure_us}),
            )
            if updated_task.config.exposure_us == old_task.config.exposure_us:
                restore_cell()
                return
            tasks = list(old_plan.tasks)
            tasks[row] = updated_task
            updated_plan = replace(old_plan, tasks=tuple(tasks))
            loaded = self._save_task_exposure_update(old_plan, updated_plan, old_task.task_id)
        except Exception as exc:
            restore_cell()
            QMessageBox.warning(self, "无法修改任务曝光", str(exc))
            return

        self.loaded_plan = loaded
        self._generated_recipe = None
        self._plan_dirty = False
        task = loaded.tasks[row]
        with QSignalBlocker(self.plan_table):
            item.setText(f"{task.config.exposure_us:g}")
            output_item = self.plan_table.item(row, 7)
            if output_item is not None:
                output_item.setText(task.relative_path(1).as_posix())
        list_item = self.plan_tasks.item(row)
        if list_item is not None:
            completed_prefix = "✓ " if list_item.text().startswith("✓ ") else ""
            list_item.setText(completed_prefix + self._plan_task_text(row + 1, task))
        self.plan_tasks.setCurrentRow(row)
        self.plan_table.selectRow(row)
        self.plan_status.setText(
            f"{task.task_id} 曝光已更新为 {task.config.exposure_us:g} μs，并保存到 {self.loaded_plan_path}"
        )
        preview = self.camera_page.preview_thread
        if self.guided_preview_index == row and preview is not None and preview.isRunning():
            self.preview_selected_task()

    def _save_task_exposure_update(
        self,
        old_plan: CapturePlan,
        updated_plan: CapturePlan,
        task_id: str,
    ) -> CapturePlan:
        """保存单任务曝光；存在续采现场时同步其计划快照与 hash。"""

        if self.loaded_plan_path is None:
            raise ValueError("当前计划没有可写入的 YAML 路径")
        work_dir = old_plan.output_dir.expanduser().resolve().parent / (
            f".{old_plan.output_dir.expanduser().resolve().name}.inprogress"
        )
        manifest_path = work_dir / "dataset_manifest.yaml"
        manifest: dict[str, Any] | None = None
        if manifest_path.is_file():
            candidate = load_document(manifest_path)
            if candidate.get("plan_sha256") != capture_plan_hash(old_plan):
                # 输出目录可能保留上一次异常退出/失败采集的现场。未勾选
                # “续采”时用户是在编辑新计划，不应被旧现场阻塞；一旦明确
                # 续采或当前确实正在写帧，仍严格拒绝，避免混用旧帧。
                capture_active = (
                    self._capture_worker is not None
                    or self._capture_in_progress
                )
                if self.resume.isChecked() or capture_active:
                    raise ValueError("当前计划与未完成数据集不一致，拒绝修改曝光")
            else:
                manifest = candidate
                task_state = manifest.get("tasks", {}).get(task_id, {})
                if (
                    int(task_state.get("frames_captured") or 0) > 0
                    or task_state.get("status") == "completed"
                ):
                    raise ValueError(f"任务 {task_id} 已开始采集，不能再修改曝光")

        saved = save_generated_capture_plan(
            updated_plan,
            self.loaded_plan_path,
            overwrite=True,
        )
        try:
            if manifest is not None:
                manifest["plan_sha256"] = capture_plan_hash(updated_plan)
                manifest["plan"] = capture_plan_payload(updated_plan)
                _save_state(work_dir, manifest, write_frames_csv=False)
        except Exception:
            # YAML 与续采 manifest 必须保持同一个计划；同步失败时恢复旧 YAML。
            save_generated_capture_plan(old_plan, self.loaded_plan_path, overwrite=True)
            raise
        self.loaded_plan_path = saved.resolve()
        return load_capture_plan(saved)

    def _set_capture_buttons_ready(self) -> None:
        ready = (
            self.loaded_plan is not None
            and not self._plan_dirty
            and self._capture_worker is None
            and not self._guided_capture_active
        )
        self.start_button.setEnabled(ready)
        self.resume_button.setEnabled(ready)
        self.capture_task_button.setEnabled(ready)

    def _set_guided_running(self, running: bool) -> None:
        """设置 GUI 引导采集状态；引导模式保持 PreviewThread 持续取流。"""

        self._guided_capture_active = running
        for widget in (
            self.plan_path, self.output, self.dataset_id, self.image_format,
            self.fit_groups, self.start_index, self.index_digits,
            self.include_validation, self.validation_groups, self.resume,
            self.recipe_table, self.load_plan_button,
        ):
            widget.setEnabled(not running)
        self.generate_plan_button.setEnabled(not running)
        self.restore_recipe_button.setEnabled(not running)
        self.preview_task_button.setEnabled(not running)
        self.next_task_button.setEnabled(not running)
        self.start_button.setEnabled(not running and self.loaded_plan is not None and not self._plan_dirty)
        self.resume_button.setEnabled(not running and self.loaded_plan is not None and not self._plan_dirty)
        self.cancel_button.setEnabled(running)
        self.capture_task_button.setEnabled(
            running and not self._capture_in_progress and self.camera_page.preview_thread is not None
        )

    def _set_capture_running(self, running: bool) -> None:
        for widget in (
            self.plan_path, self.output, self.dataset_id, self.image_format,
            self.fit_groups, self.start_index, self.index_digits,
            self.include_validation, self.validation_groups, self.resume, self.plan_table,
            self.recipe_table, self.load_plan_button,
        ):
            widget.setEnabled(not running)
        self.generate_plan_button.setEnabled(not running)
        self.restore_recipe_button.setEnabled(not running)
        self.preview_task_button.setEnabled(not running)
        self.next_task_button.setEnabled(not running)
        self.start_button.setEnabled(not running and self.loaded_plan is not None and not self._plan_dirty)
        self.resume_button.setEnabled(not running and self.loaded_plan is not None and not self._plan_dirty)
        self.cancel_button.setEnabled(running)
        self.capture_task_button.setEnabled(False if running else self.loaded_plan is not None and not self._plan_dirty)

    def _guided_start_row(self, resume: bool) -> int:
        """返回引导采集要预览的首个任务，并校验续采计划 hash。"""

        if self.loaded_plan is None or not self.loaded_plan.tasks:
            raise ValueError("采集计划没有可执行任务")
        output_dir = self.loaded_plan.output_dir.expanduser().resolve()
        if output_dir.exists():
            raise RuntimeError(f"输出数据集已经存在，不会覆盖：{output_dir}")
        work_dir = output_dir.parent / f".{output_dir.name}.inprogress"
        if not resume:
            if work_dir.exists():
                raise RuntimeError(f"发现未完成数据集；确认后勾选续采：{work_dir}")
            return 0

        manifest_path = work_dir / "dataset_manifest.yaml"
        if not manifest_path.is_file():
            raise RuntimeError(f"没有可续采的 manifest：{manifest_path}")
        manifest = load_document(manifest_path)
        if manifest.get("plan_sha256") != capture_plan_hash(self.loaded_plan):
            raise RuntimeError("当前采集计划与未完成数据集不一致，拒绝续采")
        states = manifest.get("tasks", {})
        for row, task in enumerate(self.loaded_plan.tasks):
            if states.get(task.task_id, {}).get("status") == "completed":
                self._mark_task_row(row)
                continue
            return row
        raise RuntimeError("未完成数据集中没有可继续的任务")

    def start_guided_capture(self, resume: bool | None = None) -> None:
        """启动 GUI 逐任务引导采集，保持实时预览并自动切换任务。"""

        if self._capture_worker is not None:
            QMessageBox.information(self, "采集正在运行", "当前正在执行后台采集计划，请先取消后再启动引导采集。")
            return
        if self.loaded_plan is None or self._plan_dirty:
            QMessageBox.warning(self, "计划未就绪", "请先点击“生成并检查计划”，并保持配置不变")
            return
        if self.camera_page.runtime is None:
            QMessageBox.warning(self, "相机未配置", "请先在相机页面加载配置")
            return
        mismatch = self._plan_camera_mismatch(self.loaded_plan)
        if mismatch:
            QMessageBox.warning(self, "相机通道不匹配", mismatch)
            return
        resume_requested = self.resume.isChecked() if resume is None else bool(resume)
        try:
            row = self._guided_start_row(resume_requested)
        except Exception as exc:
            QMessageBox.critical(self, "无法开始引导采集", str(exc))
            return

        self.camera_page.stop_preview()
        self._capture_in_progress = False
        self._set_guided_running(True)
        self.progress.clear()
        self.progress.append("开始引导采集；实时画面稳定后点击“采集当前任务全部帧”。")
        self.plan_tasks.setCurrentRow(row)
        self.plan_table.selectRow(row)
        self.guided_preview_index = None
        self.plan_status.setText("正在切换到首个未完成任务的相机配置…")
        self.preview_selected_task()

    def start_capture(self, resume: bool | None = None) -> None:
        """兼容旧调用方的后台 run_capture_plan 采集入口。

        GUI 按钮使用 :meth:`start_guided_capture`，以便保存每帧后继续保持
        PreviewThread 取流；该方法保留给旧测试、插件和需要后台计划执行的调用方。
        """

        runtime = self.camera_page.runtime
        if runtime is None:
            QMessageBox.warning(self, "相机未配置", "请先在相机页面加载配置"); return
        if self.loaded_plan is None or self._plan_dirty:
            QMessageBox.warning(self, "计划未就绪", "请先点击“生成并检查计划”，并保持配置不变"); return
        mismatch = self._plan_camera_mismatch(self.loaded_plan)
        if mismatch:
            QMessageBox.warning(self, "相机通道不匹配", mismatch); return
        self.camera_page.stop_preview()
        try:
            provider = build_camera_provider(
                self.loaded_plan.backend,
                calibration_src=runtime["calibration_src"],
                backend_options=self.loaded_plan.backend_options,
            )
        except Exception as exc:
            QMessageBox.critical(self, "相机后端无效", str(exc)); return
        resume_requested = self.resume.isChecked() if resume is None else bool(resume)
        # 先把页面切到本次采集的首个任务；工作线程随后会在同一 session
        # 中应用实际相机参数，并在续采时通过 task_requested 选择首个未完成任务。
        if not resume_requested and self.loaded_plan.tasks:
            self._select_batch_task(self.loaded_plan.tasks[0])
        gate = CaptureTaskGate(self)
        cancel_event = threading.Event()
        self._capture_gate = gate
        self._capture_cancel_event = cancel_event
        self._last_completed_pose = None
        self.progress.clear(); self.progress.append("开始采集；每个任务准备好后点击“采集当前任务全部帧”继续…")
        self._set_capture_running(True)
        worker = FunctionWorker(
            lambda report: run_capture_plan(
                self.loaded_plan,
                provider,
                resume=resume_requested,
                progress=report,
                before_task=lambda task: gate.wait_for_task(task, cancel_event),
                cancel_event=cancel_event,
            )
        )
        self._capture_worker = worker
        gate.task_requested.connect(self._on_task_requested)
        worker.signals.progress.connect(self._capture_progress)
        worker.signals.result.connect(self._capture_done)
        worker.signals.error.connect(self._capture_error)
        worker.signals.finished.connect(self._capture_finished)
        self._workers.add(worker)
        worker.signals.finished.connect(lambda: self._workers.discard(worker))
        self.thread_pool.start(worker)

    def approve_current_task(self) -> None:
        if self._capture_gate is None:
            return
        self.capture_task_button.setEnabled(False)
        self.plan_status.setText("已确认，工作线程正在采集当前任务…")
        self._capture_gate.approve()

    def _capture_task_button_clicked(self) -> None:
        """统一处理引导保存和批量 gate 确认，避免两个按钮保存同一任务。"""

        if self._capture_worker is not None:
            self.approve_current_task()
        else:
            self.capture_current_task_frame()

    def cancel_capture(self) -> None:
        if self._guided_capture_active:
            self._capture_in_progress = False
            self.camera_page.stop_preview()
            self._set_guided_running(False)
            self.plan_status.setText("引导采集已取消；未完成数据保留在 .inprogress，可勾选续采。")
            return
        if self._capture_cancel_event is None:
            return
        self._capture_cancel_event.set()
        if self._capture_gate is not None:
            self._capture_gate.cancel()
        self.capture_task_button.setEnabled(False)
        self.plan_status.setText("正在取消采集并关闭相机会话…")

    def _on_task_requested(self, task: CaptureTask) -> None:
        if self._capture_worker is None or (
            self._capture_cancel_event is not None and self._capture_cancel_event.is_set()
        ):
            return
        self._select_batch_task(task)
        split = str(task.tags.get("split", ""))
        laser_state = task.tags.get("laser_state", "unchanged")
        output = task.relative_path(1).as_posix()
        self.current_task.setText(
            f"当前组：{split or '--'} · 当前任务：{task.task_id} · pose {task.pose_id} · role {task.role} · "
            f"曝光 {task.config.exposure_us:g} μs · 激光 {laser_state}\n"
            f"提示：{task.instruction or '请按现场要求准备'}\n输出：{output}"
        )
        move_hint = ""
        if self._last_completed_pose and self._last_completed_pose != task.pose_id:
            move_hint = f"上一姿态 {self._last_completed_pose} 已完成，请移动到 {task.pose_id}。\n"
        self.plan_status.setText(move_hint + "请完成当前姿态准备后点击“采集当前任务全部帧”。")
        self.capture_task_button.setEnabled(True)

    def _select_batch_task(self, task: CaptureTask) -> None:
        """批量 worker 进入新 task 时，同步任务选择和相机设置显示。

        批量采集期间不能重新打开预览线程（相机会被 worker 独占），但仍应让
        页面立即显示下一任务的参数；真正的相机 configure 由 run_capture_plan
        在同一 session 中完成。
        """

        if self.loaded_plan is None:
            return
        try:
            row = [item.task_id for item in self.loaded_plan.tasks].index(task.task_id)
        except ValueError:
            return
        self.plan_tasks.setCurrentRow(row)
        self.plan_table.selectRow(row)
        config = task.config
        for widget, value in (
            (self.camera_page.exposure, config.exposure_us),
            (self.camera_page.gain, config.gain_db),
            (self.camera_page.offset_x, config.offset_x),
            (self.camera_page.offset_y, config.offset_y),
            (self.camera_page.width, config.width),
            (self.camera_page.height, config.height),
        ):
            with QSignalBlocker(widget):
                widget.setValue(value)
        with QSignalBlocker(self.camera_page.pixel_format):
            self.camera_page.pixel_format.setCurrentText(config.pixel_format)
        with QSignalBlocker(self.camera_page.quality_mode):
            self.camera_page.quality_mode.setCurrentIndex(
                self.camera_page.quality_mode.findData(task.quality_mode)
            )
        self.guided_preview_index = row
        self.live_camera.setText(
            f"当前任务：{task.task_id} · 曝光 {config.exposure_us:g} μs · "
            f"增益 {config.gain_db:g} dB · 稳定帧 {task.settle_frames}"
        )
        self.live_quality.setText("已切换到当前任务参数，等待采集帧…")

    def _capture_finished(self) -> None:
        self._capture_worker = None
        self._capture_gate = None
        self._capture_cancel_event = None
        self._set_capture_running(False)
        self._set_capture_buttons_ready()

    def load_guided_plan(self) -> None:
        path = Path(self.plan_path.text()).expanduser()
        try:
            plan = load_capture_plan(path)
            mismatch = self._plan_camera_mismatch(plan)
            if mismatch:
                raise ValueError(f"采集计划与当前相机通道不一致：{mismatch}")
        except Exception as exc:
            QMessageBox.critical(self, "采集计划无效", str(exc)); return
        self.camera_page.stop_preview()
        self.loaded_plan = plan
        self.loaded_plan_path = path.resolve()
        self.output.setText(str(plan.output_dir))
        self.dataset_id.setText(plan.dataset_id)
        self._generated_recipe = None
        self._plan_dirty = False
        self._show_plan(plan)
        self._set_capture_buttons_ready()
        self.plan_status.setText(
            f"已加载 {plan.dataset_id}：{len(plan.tasks)} 个任务，backend={plan.backend}，输出到 {plan.output_dir}"
        )

    def _selected_plan_task(self, *, silent: bool = False) -> tuple[int, CaptureTask] | None:
        if self.loaded_plan is None:
            if not silent:
                QMessageBox.information(self, "未加载计划", "请先加载 capture-plan YAML")
            return None
        row = self.plan_tasks.currentRow()
        if row < 0 or row >= len(self.loaded_plan.tasks):
            if not silent:
                QMessageBox.information(self, "未选择任务", "请先在任务列表中选择一个 task")
            return None
        return row, self.loaded_plan.tasks[row]

    def _on_camera_frame(self, frame, quality: dict[str, Any]) -> None:
        if self.live_preview.isVisible():
            self.live_preview.set_array(
                frame.image,
                auto_stretch=self.live_auto_stretch.isChecked(),
                sensor_max_value=(
                    self.camera_page.preview_thread.config.sensor_max_value
                    if self.camera_page.preview_thread is not None
                    else None
                ),
            )
        warnings = "、".join(_quality_warning_text(item) for item in quality["warnings"]) or "通过"
        thresholds = (
            self.loaded_plan.quality_thresholds
            if self.loaded_plan is not None
            else (self.camera_page.runtime or {}).get("quality_thresholds")
        )
        coverage_text = _laser_quality_metrics_text(quality, thresholds)
        chess_hint = quality.get("chessboard_hint")
        chess_text = f" · {chess_hint}" if chess_hint else ""
        settling, settle_remaining = _preview_settling_info(quality)
        settling_text = f" · 正在稳定，剩余 {settle_remaining} 帧" if settling else ""
        self.live_quality.setText(
            f"{warnings} · 动态范围 {quality['dynamic_range_u8']:.1f} DN8 · "
            f"清晰度 {quality['focus_laplacian']:.1f}{coverage_text}{chess_text}{settling_text}"
        )
        self.live_search_region_quality.setText(_search_region_quality_text(quality))
        task_text = "当前任务：--"
        selected = self._selected_plan_task(silent=True)
        if selected is not None:
            _, task = selected
            applied = (
                self.camera_page.preview_thread.config
                if self.camera_page.preview_thread is not None
                else task.config
            )
            task_text = (
                f"当前任务：{task.task_id} · 曝光 {applied.exposure_us:g} μs · "
                f"增益 {applied.gain_db:g} dB · 稳定帧 {task.settle_frames}"
        )
        self.live_camera.setText(task_text)
        if self._guided_capture_active and not self._capture_in_progress:
            if settling:
                self.capture_task_button.setEnabled(False)
            elif self._preview_matches_selected_task():
                self.capture_task_button.setEnabled(True)

    def _preview_matches_selected_task(self) -> bool:
        selected = self._selected_plan_task(silent=True)
        thread = self.camera_page.preview_thread
        if selected is None or thread is None or not thread.isRunning():
            return False
        _, task = selected
        config = thread.config
        return (
            thread.quality_mode == task.quality_mode
            and config.pixel_format == task.config.pixel_format
            and config.offset_x == task.config.offset_x
            and config.offset_y == task.config.offset_y
            and config.width == task.config.width
            and config.height == task.config.height
            and abs(config.exposure_us - task.config.exposure_us) <= max(1.0, task.config.exposure_us * 1e-3)
            and abs(config.gain_db - task.config.gain_db) <= 1e-3
        )

    def preview_selected_task(self, on_started: Callable[[], None] | None = None) -> None:
        selected = self._selected_plan_task()
        if selected is None:
            return
        row, task = selected
        assert self.loaded_plan is not None
        runtime = dict(self.camera_page.runtime or {})
        if "calibration_src" not in runtime:
            QMessageBox.warning(self, "相机配置不足", "请先在第 2 页加载相机配置，以确定 calibration_src"); return
        runtime.update(
            source=self.loaded_plan_path,
            backend=self.loaded_plan.backend,
            serial_number=self.loaded_plan.serial_number,
            camera=task.config,
            quality_thresholds=self.loaded_plan.quality_thresholds,
            board_pattern=self.loaded_plan.board_pattern,
            backend_options=self.loaded_plan.backend_options,
            laser=self.loaded_plan.laser,
        )
        self._preview_request_token += 1
        token = self._preview_request_token
        self._pending_preview_callback = on_started
        preview_running = (
            self.camera_page.preview_thread is not None
            and self.camera_page.preview_thread.isRunning()
        )
        self.camera_page.runtime = runtime
        self.camera_page.backend_label.setText(self.loaded_plan.backend)
        serial = self.loaded_plan.serial_number
        if self.loaded_plan.backend == "synthetic":
            serial = serial or "SIM-001"
            runtime["serial_number"] = serial
        if not preview_running:
            self.camera_page.devices.clear()
            if self.loaded_plan.backend == "synthetic":
                self.camera_page.devices.addItem(f"模拟相机 · SN {serial}", serial)
            elif serial:
                self.camera_page.devices.addItem(f"计划指定相机 · SN {serial}", serial)
        self.camera_page.exposure.setValue(task.config.exposure_us)
        self.camera_page.gain.setValue(task.config.gain_db)
        self.camera_page.pixel_format.setCurrentText(task.config.pixel_format)
        self.camera_page.offset_x.setValue(task.config.offset_x)
        self.camera_page.offset_y.setValue(task.config.offset_y)
        self.camera_page.width.setValue(task.config.width)
        self.camera_page.height.setValue(task.config.height)
        self.camera_page.quality_mode.setCurrentIndex(
            self.camera_page.quality_mode.findData(task.quality_mode)
        )
        split = str(task.tags.get("split", ""))
        laser_state = task.tags.get("laser_state", "unchanged")
        self.current_task.setText(
            f"当前组：{split or '--'} · 当前任务：{task.task_id} · pose {task.pose_id} · role {task.role} · "
            f"曝光 {task.config.exposure_us:g} μs · 激光 {laser_state}\n"
            f"提示：{task.instruction or '请按现场要求准备'}\n输出：{task.relative_path(1).as_posix()}"
        )
        self.camera_page.last_quality = None
        self.live_camera.setText(
            f"当前任务：{task.task_id} · 曝光 {task.config.exposure_us:g} μs · "
            f"增益 {task.config.gain_db:g} dB · 正在应用任务配置…"
        )
        self.live_quality.setText("实时画面保持连接，正在等待新配置稳定帧…")
        self.guided_preview_index = row
        self.plan_status.setText(f"正在切换到 {task.task_id} 预览：{task.instruction}")
        if self._guided_capture_active:
            self.capture_task_button.setEnabled(False)

        if self.camera_page.request_preview_task(
            task.config, task.quality_mode, 0
        ):
            callback = self._pending_preview_callback
            self._pending_preview_callback = None
            if callback is not None:
                QTimer.singleShot(0, callback)
            return

        QTimer.singleShot(
            350,
            lambda: self._start_guided_preview(
                token, row, task.task_id, task.instruction
            ),
        )

    def _start_guided_preview(
        self,
        token: int,
        row: int,
        task_id: str,
        instruction: str,
    ) -> None:
        if token != self._preview_request_token:
            return
        if self.loaded_plan is None or row < 0 or row >= len(self.loaded_plan.tasks):
            return
        self.camera_page.start_preview(initial_discard_frames=0)
        self.guided_preview_index = row
        self.plan_status.setText(f"正在预览 {task_id}：{instruction}")
        if self._guided_capture_active:
            self.capture_task_button.setEnabled(False)
        callback = self._pending_preview_callback
        self._pending_preview_callback = None
        if callback is not None:
            QTimer.singleShot(0, callback)

    def capture_current_task_frame(self) -> None:
        if self._capture_in_progress:
            self.plan_status.setText("正在自动采集当前任务的剩余帧，请稍候…")
            return
        selected = self._selected_plan_task()
        if selected is None:
            return
        row, task = selected
        self._capture_in_progress = True
        self.capture_task_button.setEnabled(False)
        self.preview_task_button.setEnabled(False)
        self.next_task_button.setEnabled(False)

        def save_after_settle(frame, quality) -> None:
            try:
                completed = self._save_guided_frame(task, frame, quality)
            except Exception as exc:
                self._capture_in_progress = False
                self.capture_task_button.setEnabled(True)
                self.preview_task_button.setEnabled(not self._guided_capture_active)
                self.next_task_button.setEnabled(not self._guided_capture_active)
                QMessageBox.critical(self, "保存任务失败", str(exc))
                return
            if completed:
                self._capture_in_progress = False
                self._mark_task_row(row)
                if self._guided_capture_active:
                    self.capture_task_button.setEnabled(False)
                else:
                    self.capture_task_button.setEnabled(True)
                    self.preview_task_button.setEnabled(True)
                    self.next_task_button.setEnabled(True)
                self.select_next_task()
            else:
                self.plan_status.setText(
                    f"{task.task_id}：正在连续采集，完成后自动进入下一任务…"
                )
                self._schedule_guided_capture(task, save_after_settle)

        if self.guided_preview_index != row or self.camera_page.preview_thread is None:
            self.preview_selected_task(
                on_started=lambda: self._schedule_guided_capture(
                    task, save_after_settle, discard_frames=task.settle_frames
                )
            )
            return
        self._schedule_guided_capture(
            task, save_after_settle, discard_frames=task.settle_frames
        )

    def _schedule_guided_capture(
        self,
        task: CaptureTask,
        callback: Callable[[Any, dict[str, Any]], None],
        *,
        discard_frames: int = 0,
    ) -> None:
        # 点击采集时为当前 task 执行一次 settle_frames；递归保存后续帧时
        # discard_frames 保持为 0，同一 task 内不重复丢弃预热帧。
        discard_frames = max(0, int(discard_frames))
        if self.camera_page.capture_after_settle(discard_frames, callback):
            if discard_frames:
                self.plan_status.setText(
                    f"{task.task_id}：先丢弃 {discard_frames} 帧，再自动采集全部剩余帧…"
                )
            else:
                self.plan_status.setText(f"{task.task_id}：正在自动采集全部剩余帧…")
            return
        self._capture_in_progress = False
        self.capture_task_button.setEnabled(True)
        self.preview_task_button.setEnabled(not self._guided_capture_active)
        self.next_task_button.setEnabled(not self._guided_capture_active)
        QMessageBox.warning(self, "尚未取流", "任务预览尚未建立，请等待实时画面出现后重试。")

    def _save_guided_frame(
        self,
        task: CaptureTask,
        frame: Any,
        preview_quality: Mapping[str, Any] | None = None,
    ) -> bool:
        assert self.loaded_plan is not None
        output_dir = self.loaded_plan.output_dir.expanduser().resolve()
        work_dir = output_dir.parent / f".{output_dir.name}.inprogress"
        manifest_path = work_dir / "dataset_manifest.yaml"
        if output_dir.exists():
            raise RuntimeError(f"输出数据集已经存在，不会覆盖：{output_dir}")
        if manifest_path.is_file():
            manifest = load_document(manifest_path)
            if manifest.get("plan_sha256") != capture_plan_hash(self.loaded_plan):
                raise RuntimeError("当前采集计划与未完成数据集不一致，拒绝续采")
        elif work_dir.exists() and not self.resume.isChecked():
            raise RuntimeError(f"发现未完成数据集；确认后勾选续采：{work_dir}")
        else:
            work_dir.mkdir(parents=True, exist_ok=True)
            manifest = _new_manifest(self.loaded_plan)
        task_state = manifest["tasks"][task.task_id]
        captured = int(task_state.get("frames_captured") or 0)
        if captured >= task.frames:
            raise RuntimeError(f"{task.task_id} 已采满 {task.frames} 帧")
        index = captured + 1
        relative = task.relative_path(index)
        destination = work_dir / relative
        _write_image(destination, frame)
        quality_source_frame = (
            preview_quality.get("preview_quality_source_frame_number")
            if preview_quality is not None
            else None
        )
        preview_quality_is_exact = (
            preview_quality is not None
            and (
                quality_source_frame is None
                or int(quality_source_frame) == int(frame.camera_frame_number)
            )
        )
        quality = (
            {
                key: value
                for key, value in preview_quality.items()
                if key not in {
                    "settling",
                    "settle_frames_remaining",
                    "preview_quality_processing_ms",
                    "preview_quality_source_frame_number",
                    "preview_quality_fresh",
                    "preview_quality_reused",
                    "preview_quality_age_frames",
                }
            }
            if preview_quality_is_exact
            else quality_to_dict(analyze_frame(
                frame.image,
                sensor_max_value=task.config.sensor_max_value,
                mode=task.quality_mode,
                thresholds=self.loaded_plan.quality_thresholds,
                board_pattern=self.loaded_plan.board_pattern,
                laser_orientation=self.loaded_plan.laser.orientation,
            ))
        )
        record = {
            "task_id": task.task_id,
            "pose_id": task.pose_id,
            "role": task.role,
            "tags": task.tags,
            "index": index,
            "filename": relative.as_posix(),
            "sha256": sha256_file(destination, normalize_newlines=False),
            "camera_frame_number": frame.camera_frame_number,
            "camera_frame_gap": None,
            "transport_warnings": [],
            "camera_timestamp_ticks": frame.camera_timestamp_ticks,
            "host_timestamp_ns": frame.host_timestamp_ns,
            "host_monotonic_ns": frame.host_monotonic_ns,
            "requested_camera": asdict(task.config),
            "applied_camera": asdict(self.camera_page.preview_thread.config if self.camera_page.preview_thread else task.config),
            "quality": quality,
        }
        manifest["frames"] = [
            item for item in manifest["frames"]
            if not (item["task_id"] == task.task_id and int(item["index"]) == index)
        ]
        manifest["frames"].append(record)
        task_state.update(
            status="completed" if index >= task.frames else "capturing",
            frames_captured=index,
            completed_at=_utc_now_text() if index >= task.frames else None,
        )
        task_completed = index >= task.frames
        if all(item.get("status") == "completed" for item in manifest["tasks"].values()):
            manifest["status"] = "completed"
            manifest["completed_at"] = _utc_now_text()
            _save_state(work_dir, manifest)
            work_dir.replace(output_dir)
            self.camera_page.stop_preview()
            self._set_guided_running(False)
            self.progress.append(f"完成：{output_dir}")
            self.last_capture_artifacts = capture_artifacts_record(self.loaded_plan_path, output_dir)
            self.capture_finished.emit(
                type(
                    "Result",
                    (),
                    {"output_dir": output_dir, "capture_artifacts": self.last_capture_artifacts},
                )()
            )
            return task_completed
        else:
            # manifest 是续采的事实来源，每帧持久化；frames.csv 是派生索引，
            # 只在 task 完成时刷新，避免 Windows 下连续替换同一 CSV 触发共享冲突。
            _save_state(work_dir, manifest, write_frames_csv=task_completed)
            self.progress.append(
                f"{task.task_id}  {index}/{task.frames}  {','.join(quality['warnings']) or '通过'}  → {relative.as_posix()}"
            )
            return task_completed

    def _mark_task_row(self, row: int) -> None:
        item = self.plan_tasks.item(row)
        if item and not item.text().startswith("✓ "):
            item.setText("✓ " + item.text())

    def select_next_task(self, *, auto_preview: bool = True) -> None:
        if self.loaded_plan is None:
            return
        row = self.plan_tasks.currentRow()
        for candidate in range(row + 1, len(self.loaded_plan.tasks)):
            item = self.plan_tasks.item(candidate)
            if item and not item.text().startswith("✓ "):
                self.plan_tasks.setCurrentRow(candidate)
                if auto_preview:
                    self.preview_selected_task()
                else:
                    self.plan_status.setText(f"下一任务：{self.loaded_plan.tasks[candidate].instruction}")
                return
        self.plan_status.setText("没有后续未完成任务。")

    def _capture_progress(self, event: dict[str, Any]) -> None:
        event_name = event.get("event")
        if event_name == "task_started":
            self.progress.append(
                f"开始：{event['task_id']} · pose {event.get('pose_id', '')} · "
                f"曝光 {float(event.get('exposure_us', 0)):g} μs · 激光 {event.get('laser_state', 'unchanged')}"
            )
        elif event_name == "frame":
            warnings = ",".join(event["quality"]["warnings"]) or "通过"
            relative = event.get("relative_output_path", "")
            self.progress.append(
                f"{event['task_id']}  {event['index']}/{event['frames']}  {warnings}  → {relative}"
            )
        elif event_name == "quality_warning":
            self.progress.append(
                f"质量告警：{event['task_id']} 帧 {event['index']}：{', '.join(event['warnings'])}"
            )
        elif event_name == "task_completed":
            self.progress.append(f"✓ {event['task_id']} 已提交")
            self._last_completed_pose = str(event.get("pose_id", ""))
            self._display_saved_image(event["task_id"], event.get("relative_output_path", ""))
            if self.loaded_plan is not None:
                task_ids = [task.task_id for task in self.loaded_plan.tasks]
                try:
                    index = task_ids.index(event["task_id"])
                except ValueError:
                    index = -1
                if index >= 0 and index + 1 < len(self.loaded_plan.tasks):
                    next_task = self.loaded_plan.tasks[index + 1]
                    if next_task.pose_id != event.get("pose_id"):
                        self.plan_status.setText(
                            f"姿态 {event.get('pose_id', '')} 的图像已完成，请移动到下一姿态。"
                        )

    def _display_saved_image(self, task_id: str, relative: str) -> None:
        if self.loaded_plan is None or not relative:
            return
        output_dir = self.loaded_plan.output_dir.expanduser().resolve()
        path = output_dir / relative
        if not path.is_file():
            return
        for path in (path,):
            if not path.is_file():
                continue
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            try:
                self.live_preview.set_array(
                    image,
                    auto_stretch=self.live_auto_stretch.isChecked(),
                    sensor_max_value=self.loaded_plan.base_config.sensor_max_value,
                )
            except Exception:
                pass
            return

    def _capture_done(self, result) -> None:
        output_dir = Path(result.output_dir)
        self.last_capture_artifacts = capture_artifacts_record(self.loaded_plan_path, output_dir)
        self.progress.append(f"完成：{result.frame_count} 帧 → {output_dir}")
        self.progress.append(f"manifest：{output_dir / 'dataset_manifest.yaml'}")
        self.progress.append(f"frames：{output_dir / 'frames.csv'}")
        self.capture_finished.emit(result)

    def _capture_error(self, message: str) -> None:
        self.progress.append(f"失败：{message}")
        if "采集已取消" not in message:
            QMessageBox.critical(self, "采集失败", message)


class CalibrationPage(QWidget):
    workflow_finished = Signal(object)

    def __init__(self, thread_pool: QThreadPool, parent=None) -> None:
        super().__init__(parent)
        self.thread_pool = thread_pool; self._workers: set[FunctionWorker] = set()
        self.capture_artifacts: dict[str, Any] | None = None
        self.project: WizardProject | None = None
        layout = QVBoxLayout(self)
        title = QLabel("4. 一键执行标定 workflow"); title.setObjectName("pageTitle"); layout.addWidget(title)
        self.workflow = QLineEdit()
        self.workflow.setReadOnly(True)
        self.workflow.setToolTip("Workflow 由第 1 页项目配置统一选择。")
        layout.addWidget(
            _labeled_path(
                "Workflow YAML",
                self.workflow,
                self,
                "YAML (*.yaml *.yml)",
                browse=False,
            )
        )
        self.refresh_button = QPushButton("检查 Workflow 阶段")
        self.update_workflow_button = QPushButton("从最近采集结果更新 Workflow 输入")
        self.update_workflow_button.setEnabled(False)
        action_row = QHBoxLayout()
        action_row.addWidget(self.refresh_button); action_row.addWidget(self.update_workflow_button); action_row.addStretch()
        layout.addLayout(action_row)
        self.stage_table = QTableWidget(0, 4)
        self.stage_table.setHorizontalHeaderLabels(["启用", "阶段", "输入/配置", "输出"])
        self.stage_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.stage_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.stage_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.stage_table.setMaximumHeight(220)
        layout.addWidget(self.stage_table)
        self.allow_note = QLabel("向导按 workflow 中启用的阶段顺序执行；每个阶段仍使用阶段 1 的质量门禁。")
        self.allow_note.setWordWrap(True); layout.addWidget(self.allow_note)
        self.capture_artifact_status = QLabel("尚未记录最近采集结果")
        self.capture_artifact_status.setWordWrap(True); layout.addWidget(self.capture_artifact_status)
        self.run_button = QPushButton("一键运行完整标定"); layout.addWidget(self.run_button)
        self.log = QTextEdit(); self.log.setReadOnly(True); layout.addWidget(self.log, 1)
        self.run_button.clicked.connect(self.run); self.refresh_button.clicked.connect(lambda: self.refresh_plan())
        self.update_workflow_button.clicked.connect(self.update_workflow_inputs)
        self.workflow.textChanged.connect(
            lambda _text: self.update_workflow_button.setEnabled(
                bool(self.capture_artifacts and self.workflow.text().strip())
            )
        )

    def set_project(self, project: WizardProject) -> None:
        self.project = project
        self.workflow.setText(str(project.workflow_plan or ""))
        stored = project.extra.get("capture_artifacts") if isinstance(project.extra, dict) else None
        self.set_capture_artifacts(stored if isinstance(stored, dict) else None)
        self.refresh_plan(silent=True)

    def set_capture_artifacts(self, artifacts: Mapping[str, Any] | None) -> None:
        self.capture_artifacts = dict(artifacts) if artifacts else None
        enabled = bool(self.capture_artifacts and self.workflow.text().strip())
        self.update_workflow_button.setEnabled(enabled)
        if self.capture_artifacts:
            self.capture_artifact_status.setText(
                "最近采集："
                f"\n数据集：{self.capture_artifacts.get('dataset_root', '暂无')}"
                f"\nfit：{self.capture_artifacts.get('fit_dir', '暂无')}"
                f"\nvalidation：{self.capture_artifacts.get('validation_dir', '暂无')}"
            )
        else:
            self.capture_artifact_status.setText("尚未记录最近采集结果")

    def update_workflow_inputs(self) -> None:
        if not self.capture_artifacts:
            QMessageBox.information(self, "没有采集结果", "请先完成一次批量采集。")
            return
        path = Path(self.workflow.text()).expanduser()
        if not path.is_file():
            QMessageBox.warning(self, "Workflow 不存在", str(path))
            return
        try:
            preview = build_workflow_update_preview(path, self.capture_artifacts)
        except Exception as exc:
            QMessageBox.critical(self, "无法生成更新预览", str(exc))
            return
        if not preview.changed:
            QMessageBox.information(self, "无需更新", preview.text())
            return
        answer = QMessageBox.question(
            self,
            "确认更新 Workflow 输入",
            preview.text() + "\n\n确认后会先备份原文件，再保存更新后的 YAML。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            backup = save_workflow_update(preview, backup=True)
            self.refresh_plan(silent=True)
            self.capture_artifact_status.setText(
                self.capture_artifact_status.text()
                + f"\nWorkflow 已更新；备份：{backup or '未生成'}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Workflow 保存失败", str(exc))

    def refresh_plan(self, *, silent: bool = False) -> bool:
        path = Path(self.workflow.text()).expanduser()
        if not path.is_file():
            self.stage_table.setRowCount(0)
            if not silent:
                QMessageBox.warning(self, "Workflow 不存在", str(path))
            return False
        try:
            document = load_document(path)
            stages = document.get("stages", [])
            if not isinstance(stages, list):
                raise ValueError("stages 必须是列表")
            self.stage_table.setRowCount(len(stages))
            for row, stage in enumerate(stages):
                options = stage.get("options", {}) if isinstance(stage, dict) else {}
                enabled = "是" if isinstance(stage, dict) and stage.get("enabled", True) else "否"
                name = str(stage.get("name", "")) if isinstance(stage, dict) else ""
                inputs = [f"{key}={value}" for key, value in options.items() if key not in {"output", "output_dir"}]
                output = options.get("output", options.get("output_dir", ""))
                for column, value in enumerate((enabled, name, "; ".join(inputs), str(output))):
                    item = QTableWidgetItem(value); item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.stage_table.setItem(row, column, item)
            return True
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "Workflow 无效", str(exc))
            return False

    def run(self) -> None:
        path = Path(self.workflow.text()).expanduser()
        if not self.refresh_plan():
            return
        self.run_button.setEnabled(False); self.log.clear(); self.log.append(f"开始：{path.resolve()}")
        orientation = self.project.laser.orientation if self.project is not None else "horizontal"
        worker = FunctionWorker(
            lambda progress: run_workflow(
                path,
                progress=progress,
                laser_orientation=orientation,
            )
        )
        worker.signals.progress.connect(self._progress)
        worker.signals.result.connect(self._done)
        worker.signals.error.connect(self._error)
        worker.signals.finished.connect(lambda: self.run_button.setEnabled(True))
        self._workers.add(worker); worker.signals.finished.connect(lambda: self._workers.discard(worker)); self.thread_pool.start(worker)

    def _done(self, result: dict[str, Any]) -> None:
        for stage in result.get("stages", []):
            self.log.append(f"{stage.get('stage')}: {stage.get('status')}")
        self.log.append(f"Workflow: {result.get('status')}")
        if self.capture_artifacts:
            result = dict(result)
            result["capture_artifacts"] = dict(self.capture_artifacts)
        self.workflow_finished.emit(result)

    def _progress(self, event: dict[str, Any]) -> None:
        if event.get("event") == "stage_started":
            self.log.append(f"▶ {event['stage']} 开始")
        elif event.get("event") == "stage_finished":
            self.log.append(f"✓ {event['stage']}：{event['status']}")

    def _error(self, message: str) -> None:
        self.log.append(f"失败：{message}"); QMessageBox.critical(self, "标定失败", message)


class ResultsPage(QWidget):
    acceptance_finished = Signal(object)

    def __init__(self, thread_pool: QThreadPool, parent=None) -> None:
        super().__init__(parent)
        self.thread_pool = thread_pool
        self._workers: set[FunctionWorker] = set()
        self.current_run: CalibrationRun | None = None
        self.current_run_path: Path | None = None
        # current_result 保留为展示投影，正式标定结果以 current_run 为准。
        self.current_result: dict[str, Any] | None = None
        self.project: WizardProject | None = None
        self.last_html: Path | None = None
        self.capture_artifacts: dict[str, Any] | None = None
        self._artifact_records: list[ResultArtifact] = []
        self._filtered_artifacts: list[ResultArtifact] = []
        self._current_artifact_index = -1
        self._results_summary: CalibrationResultsSummary | None = None
        self._laser_surface_details: LaserSurfaceDetails | None = None
        self._ground_extrinsics_details: GroundExtrinsicsDetails | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        title = QLabel("5. 标定结果与误差分析"); title.setObjectName("pageTitle")
        self.run_status = QLabel("当前项目暂无标定结果")
        self.run_status.setWordWrap(True)
        self.load_button = QPushButton("打开历史标定结果…")
        title_row.addWidget(title); title_row.addWidget(self.run_status, 1); title_row.addWidget(self.load_button); layout.addLayout(title_row)

        self.result_tabs = QTabWidget()
        overview_page = QWidget()
        overview_page_layout = QVBoxLayout(overview_page)
        _configure_result_page_layout(overview_page_layout)
        self.overview_box = QGroupBox("结果总览")
        overview_layout = QGridLayout(self.overview_box)
        self.overview_run_id = QLabel("暂无")
        self.overview_time = QLabel("暂无")
        self.overview_status = QLabel("暂无")
        self.overview_overall = QLabel("暂无")
        self.overview_orientation = QLabel("暂无")
        for row, (label, widget) in enumerate(
            (
                ("run_id", self.overview_run_id),
                ("标定时间", self.overview_time),
                ("执行状态", self.overview_status),
                ("整体状态", self.overview_overall),
                ("激光方向", self.overview_orientation),
            )
        ):
            overview_layout.addWidget(QLabel(label), row // 2, (row % 2) * 2)
            overview_layout.addWidget(widget, row // 2, (row % 2) * 2 + 1)

        self.intrinsics_summary_box = QGroupBox("相机内参")
        intrinsics_layout = QGridLayout(self.intrinsics_summary_box)
        self.intrinsics_status = QLabel("未执行")
        self.intrinsics_fit_rmse = QLabel("未执行")
        self.intrinsics_test_rmse = QLabel("未执行")
        self.intrinsics_fit_images = QLabel("未执行")
        self.intrinsics_test_images = QLabel("未执行")
        intrinsics_fields = (
            ("状态", self.intrinsics_status),
            ("Fit RMSE", self.intrinsics_fit_rmse),
            ("Test RMSE", self.intrinsics_test_rmse),
            ("Fit 图像数", self.intrinsics_fit_images),
            ("Test 图像数", self.intrinsics_test_images),
        )
        for row, (label, widget) in enumerate(intrinsics_fields):
            intrinsics_layout.addWidget(QLabel(label), row, 0)
            intrinsics_layout.addWidget(widget, row, 1)

        self.laser_summary_box = QGroupBox("激光表面")
        laser_layout = QGridLayout(self.laser_summary_box)
        self.laser_status = QLabel("未执行")
        self.laser_model = QLabel("未执行")
        self.laser_validation_rmse = QLabel("未执行")
        self.laser_validation_p95 = QLabel("未执行")
        self.laser_valid_rate = QLabel("未执行")
        laser_fields = (
            ("状态", self.laser_status),
            ("模型", self.laser_model),
            ("Validation RMSE", self.laser_validation_rmse),
            ("Validation P95", self.laser_validation_p95),
            ("Valid Rate", self.laser_valid_rate),
        )
        for row, (label, widget) in enumerate(laser_fields):
            laser_layout.addWidget(QLabel(label), row, 0)
            laser_layout.addWidget(widget, row, 1)

        self.ground_summary_box = QGroupBox("地面外参")
        ground_layout = QGridLayout(self.ground_summary_box)
        self.ground_status = QLabel("未执行")
        self.ground_validation_rmse = QLabel("未执行")
        self.ground_validation_p95 = QLabel("未执行")
        ground_fields = (
            ("状态", self.ground_status),
            ("Validation RMSE", self.ground_validation_rmse),
            ("Validation P95", self.ground_validation_p95),
        )
        for row, (label, widget) in enumerate(ground_fields):
            ground_layout.addWidget(QLabel(label), row, 0)
            ground_layout.addWidget(widget, row, 1)

        cards = QHBoxLayout()
        cards.addWidget(self.intrinsics_summary_box, 1)
        cards.addWidget(self.laser_summary_box, 1)
        cards.addWidget(self.ground_summary_box, 1)
        overview_layout.addLayout(cards, 3, 0, 1, 4)
        overview_page_layout.addWidget(self.overview_box)
        overview_page_layout.addStretch()
        self.result_tabs.addTab(overview_page, "结果总览")

        intrinsics_page = QWidget()
        intrinsics_page_layout = QVBoxLayout(intrinsics_page)
        _configure_result_page_layout(intrinsics_page_layout)
        self.intrinsics_detail_status = QLabel("未执行")
        self.intrinsics_detail_status.setWordWrap(True)
        intrinsics_page_layout.addWidget(self.intrinsics_detail_status)

        camera_parameters_box = QGroupBox("相机内参")
        camera_parameters_layout = QGridLayout(camera_parameters_box)
        self.intrinsics_fx = QLabel("未执行")
        self.intrinsics_fy = QLabel("未执行")
        self.intrinsics_cx = QLabel("未执行")
        self.intrinsics_cy = QLabel("未执行")
        for column, (label, widget) in enumerate(
            (
                ("fx", self.intrinsics_fx),
                ("fy", self.intrinsics_fy),
                ("cx", self.intrinsics_cx),
                ("cy", self.intrinsics_cy),
            )
        ):
            camera_parameters_layout.addWidget(QLabel(label), 0, column * 2)
            camera_parameters_layout.addWidget(widget, 0, column * 2 + 1)
        self.intrinsics_k1 = QLabel("未执行")
        self.intrinsics_k2 = QLabel("未执行")
        self.intrinsics_p1 = QLabel("未执行")
        self.intrinsics_p2 = QLabel("未执行")
        self.intrinsics_k3 = QLabel("未执行")
        for column, (label, widget) in enumerate(
            (
                ("k1", self.intrinsics_k1),
                ("k2", self.intrinsics_k2),
                ("p1", self.intrinsics_p1),
                ("p2", self.intrinsics_p2),
                ("k3", self.intrinsics_k3),
            )
        ):
            camera_parameters_layout.addWidget(QLabel(label), 1, column * 2)
            camera_parameters_layout.addWidget(widget, 1, column * 2 + 1)
        intrinsics_page_layout.addWidget(camera_parameters_box)

        intrinsics_metrics_box = QGroupBox("重投影误差")
        intrinsics_metrics_layout = QGridLayout(intrinsics_metrics_box)
        self.intrinsics_detail_fit_rmse = QLabel("未执行")
        self.intrinsics_detail_test_rmse = QLabel("未执行")
        self.intrinsics_detail_fit_images = QLabel("未执行")
        self.intrinsics_detail_test_images = QLabel("未执行")
        for row, (label, widget) in enumerate(
            (
                ("Fit RMSE", self.intrinsics_detail_fit_rmse),
                ("Test RMSE", self.intrinsics_detail_test_rmse),
                ("Fit 图像数", self.intrinsics_detail_fit_images),
                ("Test 图像数", self.intrinsics_detail_test_images),
            )
        ):
            intrinsics_metrics_layout.addWidget(QLabel(label), row // 2, (row % 2) * 2)
            intrinsics_metrics_layout.addWidget(widget, row // 2, (row % 2) * 2 + 1)
        intrinsics_page_layout.addWidget(intrinsics_metrics_box)

        reprojection_box = QGroupBox("逐图重投影误差（已有结果）")
        reprojection_layout = QVBoxLayout(reprojection_box)
        self.intrinsics_reprojection_status = QLabel("未执行")
        self.intrinsics_reprojection_status.setWordWrap(True)
        reprojection_layout.addWidget(self.intrinsics_reprojection_status)
        self.intrinsics_reprojection_table = QTableWidget()
        self.intrinsics_reprojection_table.setColumnCount(5)
        self.intrinsics_reprojection_table.setHorizontalHeaderLabels(
            ["数据集", "图像", "状态", "RMSE (px)", "平均误差 (px)"]
        )
        self.intrinsics_reprojection_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.intrinsics_reprojection_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.intrinsics_reprojection_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.intrinsics_reprojection_table.horizontalHeader().setStretchLastSection(True)
        self.intrinsics_reprojection_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        reprojection_layout.addWidget(self.intrinsics_reprojection_table)
        intrinsics_page_layout.addWidget(reprojection_box)
        intrinsics_page_layout.addStretch()
        self.result_tabs.addTab(intrinsics_page, "相机内参")

        self.laser_page = QWidget()
        laser_page_layout = QVBoxLayout(self.laser_page)
        _configure_result_page_layout(laser_page_layout)
        self.laser_scroll_area = QScrollArea(self.laser_page)
        self.laser_scroll_area.setWidgetResizable(True)
        self.laser_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.laser_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.laser_content = QWidget()
        self.laser_content.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.laser_content_layout = QVBoxLayout(self.laser_content)
        _configure_result_page_layout(self.laser_content_layout)
        self.laser_content_layout.setSizeConstraint(
            QLayout.SizeConstraint.SetMinimumSize
        )
        laser_details_box = QGroupBox("当前采用模型")
        laser_details_layout = QGridLayout(laser_details_box)
        self.laser_detail_status = QLabel("未执行")
        self.laser_detail_model = QLabel("未执行")
        self.laser_detail_validation_rmse = QLabel("未执行")
        self.laser_detail_validation_p95 = QLabel("未执行")
        self.laser_detail_valid_rate = QLabel("未执行")
        for row, (label, widget) in enumerate(
            (
                ("阶段状态", self.laser_detail_status),
                ("当前采用模型", self.laser_detail_model),
                ("Validation RMSE", self.laser_detail_validation_rmse),
                ("Validation P95", self.laser_detail_validation_p95),
                ("Valid Rate", self.laser_detail_valid_rate),
            )
        ):
            laser_details_layout.addWidget(QLabel(label), row, 0)
            laser_details_layout.addWidget(widget, row, 1)
        self.laser_content_layout.addWidget(laser_details_box)

        laser_comparison_box = QGroupBox("三模型比较（训练集 / 独立验证集）")
        laser_comparison_layout = QVBoxLayout(laser_comparison_box)
        self.laser_model_comparison_status = QLabel("未执行")
        self.laser_model_comparison_status.setWordWrap(True)
        laser_comparison_layout.addWidget(self.laser_model_comparison_status)
        self.laser_model_comparison_table = QTableWidget()
        # 别名便于后续详细页复用，也不与旧的兼容 table 混淆。
        self.laser_comparison_table = self.laser_model_comparison_table
        self.laser_model_comparison_table.setColumnCount(7)
        self.laser_model_comparison_table.setHorizontalHeaderLabels(
            [
                "模型",
                "Train RMSE (mm)",
                "Validation RMSE (mm)",
                "Validation P95 (mm)",
                "Validation Max (mm)",
                "Valid Rate",
                "状态",
            ]
        )
        self.laser_model_comparison_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.laser_model_comparison_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.laser_model_comparison_table.horizontalHeader().setStretchLastSection(True)
        laser_comparison_layout.addWidget(self.laser_model_comparison_table)
        self.laser_content_layout.addWidget(laser_comparison_box)

        self.laser_error_plots_box = QGroupBox("已有验证误差趋势")
        laser_error_plots_layout = QVBoxLayout(self.laser_error_plots_box)
        self.laser_error_plots_status = QLabel(
            "仅展示已有 validation error 图，不重新计算。"
        )
        self.laser_error_plots_status.setWordWrap(True)
        laser_error_plots_layout.addWidget(self.laser_error_plots_status)
        laser_plot_layout = QVBoxLayout()
        self.laser_error_vs_u = LaserPlotPreview()
        self.laser_error_vs_v = LaserPlotPreview()
        self.laser_error_vs_depth = LaserPlotPreview()
        self.laser_error_vs_u_card = self._build_laser_plot_card(
            "error vs U", self.laser_error_vs_u
        )
        self.laser_error_vs_v_card = self._build_laser_plot_card(
            "error vs V", self.laser_error_vs_v
        )
        self.laser_error_vs_depth_card = self._build_laser_plot_card(
            "error vs Depth", self.laser_error_vs_depth
        )
        for card in (
            (
                self.laser_error_vs_u_card,
                self.laser_error_vs_v_card,
                self.laser_error_vs_depth_card,
            )
        ):
            laser_plot_layout.addWidget(card)
        laser_error_plots_layout.addLayout(laser_plot_layout)
        self.laser_content_layout.addWidget(self.laser_error_plots_box)

        self.laser_parameters_toggle = QPushButton("模型参数（展开）")
        self.laser_parameters_toggle.setCheckable(True)
        self.laser_parameters_toggle.toggled.connect(self._toggle_laser_parameters)
        self.laser_content_layout.addWidget(self.laser_parameters_toggle)
        self.laser_parameters_panel = QWidget()
        laser_parameters_layout = QVBoxLayout(self.laser_parameters_panel)
        laser_parameters_layout.setContentsMargins(0, 0, 0, 0)
        self.laser_model_parameters_table = QTableWidget()
        self.laser_model_parameters_table.setColumnCount(3)
        self.laser_model_parameters_table.setHorizontalHeaderLabels(
            ["模型", "参数", "值"]
        )
        self.laser_model_parameters_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.laser_model_parameters_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.laser_model_parameters_table.horizontalHeader().setStretchLastSection(True)
        laser_parameters_layout.addWidget(self.laser_model_parameters_table)
        self.laser_parameters_panel.setVisible(False)
        self.laser_content_layout.addWidget(self.laser_parameters_panel)
        self.laser_content_layout.addStretch()
        self.laser_scroll_area.setWidget(self.laser_content)
        laser_page_layout.addWidget(self.laser_scroll_area)
        self.result_tabs.addTab(self.laser_page, "激光表面")

        self.ground_page = QWidget()
        ground_page_layout = QVBoxLayout(self.ground_page)
        _configure_result_page_layout(ground_page_layout)
        self.ground_scroll_area = QScrollArea(self.ground_page)
        self.ground_scroll_area.setWidgetResizable(True)
        self.ground_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.ground_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.ground_content = QWidget()
        self.ground_content.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.ground_content_layout = QVBoxLayout(self.ground_content)
        _configure_result_page_layout(self.ground_content_layout)
        self.ground_content_layout.setSizeConstraint(
            QLayout.SizeConstraint.SetMinimumSize
        )

        ground_details_box = QGroupBox("最终结果")
        ground_details_layout = QGridLayout(ground_details_box)
        self.ground_detail_status = QLabel("未执行")
        self.ground_detail_stage = QLabel("未执行")
        self.ground_detail_validation_rmse = QLabel("未执行")
        self.ground_detail_validation_p95 = QLabel("未执行")
        self.ground_detail_fit_frames = QLabel("未执行")
        self.ground_detail_validation_frames = QLabel("未执行")
        for row, (label, widget) in enumerate(
            (
                ("阶段状态", self.ground_detail_status),
                ("阶段", self.ground_detail_stage),
                ("Validation RMSE", self.ground_detail_validation_rmse),
                ("Validation P95", self.ground_detail_validation_p95),
                ("Fit frame 数", self.ground_detail_fit_frames),
                ("Validation frame 数", self.ground_detail_validation_frames),
            )
        ):
            ground_details_layout.addWidget(QLabel(label), row, 0)
            ground_details_layout.addWidget(widget, row, 1)
        self.ground_content_layout.addWidget(ground_details_box)

        ground_pose_box = QGroupBox("最终外参（camera → ground）")
        ground_pose_layout = QGridLayout(ground_pose_box)
        self.ground_pose_status = QLabel("未执行")
        self.ground_pose_status.setWordWrap(True)
        self.ground_transform_name = QLabel("未执行")
        ground_pose_layout.addWidget(QLabel("读取状态"), 0, 0)
        ground_pose_layout.addWidget(self.ground_pose_status, 0, 1, 1, 3)
        ground_pose_layout.addWidget(QLabel("变换"), 1, 0)
        ground_pose_layout.addWidget(self.ground_transform_name, 1, 1, 1, 3)

        ground_pose_layout.addWidget(QLabel("Rotation Matrix R"), 2, 0)
        self.ground_rotation_table = QTableWidget()
        self.ground_rotation_matrix_table = self.ground_rotation_table
        self.ground_rotation_table.setRowCount(3)
        self.ground_rotation_table.setColumnCount(3)
        self.ground_rotation_table.setHorizontalHeaderLabels(["Xc", "Yc", "Zc"])
        self.ground_rotation_table.setVerticalHeaderLabels(["Xg", "Yg", "Zg"])
        self.ground_rotation_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.ground_rotation_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.ground_rotation_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.ground_rotation_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.ground_rotation_table.horizontalHeader().setStretchLastSection(True)
        ground_pose_layout.addWidget(self.ground_rotation_table, 2, 1, 3, 3)

        ground_pose_layout.addWidget(QLabel("Translation t (mm)"), 5, 0)
        self.ground_translation = QLabel("未执行")
        self.ground_translation_values = self.ground_translation
        self.ground_translation.setWordWrap(True)
        ground_pose_layout.addWidget(self.ground_translation, 5, 1, 1, 3)

        ground_pose_layout.addWidget(QLabel("Roll / Pitch / Yaw"), 6, 0)
        self.ground_euler_status = QLabel("未执行")
        self.ground_euler_status.setWordWrap(True)
        ground_pose_layout.addWidget(self.ground_euler_status, 6, 1, 1, 3)
        self.ground_roll = QLabel("未执行")
        self.ground_pitch = QLabel("未执行")
        self.ground_yaw = QLabel("未执行")
        for column, (label, widget) in enumerate(
            (
                ("Roll", self.ground_roll),
                ("Pitch", self.ground_pitch),
                ("Yaw", self.ground_yaw),
            )
        ):
            ground_pose_layout.addWidget(QLabel(label), 7, column * 2)
            ground_pose_layout.addWidget(widget, 7, column * 2 + 1)
        self.ground_content_layout.addWidget(ground_pose_box)

        ground_validation_box = QGroupBox("独立验证误差（已有结果）")
        ground_validation_layout = QVBoxLayout(ground_validation_box)
        self.ground_validation_status = QLabel("未执行")
        self.ground_validation_status.setWordWrap(True)
        ground_validation_layout.addWidget(self.ground_validation_status)
        self.ground_validation_table = QTableWidget()
        self.ground_validation_errors_table = self.ground_validation_table
        self.ground_validation_table.setColumnCount(8)
        self.ground_validation_table.setHorizontalHeaderLabels(
            [
                "Frame",
                "状态",
                "PnP RMSE (px)",
                "RMSE (mm)",
                "P95 (mm)",
                "Max (mm)",
                "Std (mm)",
                "Points",
            ]
        )
        self.ground_validation_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.ground_validation_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.ground_validation_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.ground_validation_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.ground_validation_table.horizontalHeader().setStretchLastSection(True)
        ground_validation_layout.addWidget(self.ground_validation_table)
        self.ground_validation_error_plot = LaserPlotPreview()
        self.ground_validation_plot = self.ground_validation_error_plot
        self.ground_validation_error_plot_card = self._build_laser_plot_card(
            "Validation Zg residual", self.ground_validation_error_plot
        )
        self.ground_validation_plot_card = self.ground_validation_error_plot_card
        self.ground_validation_error_plot_card.setVisible(False)
        ground_validation_layout.addWidget(self.ground_validation_error_plot_card)
        self.ground_content_layout.addWidget(ground_validation_box)
        self.ground_content_layout.addStretch()
        self.ground_scroll_area.setWidget(self.ground_content)
        ground_page_layout.addWidget(self.ground_scroll_area)
        self.result_tabs.addTab(self.ground_page, "地面外参")
        layout.addWidget(self.result_tabs, 1)

        # acceptance/result_artifacts 仍由底层模块和旧脚本使用；这里只保留不挂到
        # 页面布局的兼容对象，不向用户暴露验收、逐图、CSV 或原始图像控件。
        self._legacy_compat_host = QWidget(self)
        self._legacy_compat_host.hide()
        self.acceptance_plan = QLineEdit(self._legacy_compat_host)
        self.acceptance_overwrite = QCheckBox("覆盖已有报告", self._legacy_compat_host)
        self.acceptance_button = QPushButton("生成并判定", self._legacy_compat_host)
        self.open_html_button = QPushButton("打开 HTML 报告", self._legacy_compat_host)
        self.open_html_button.setEnabled(False)
        self.acceptance_status = QLabel("尚未执行验收", self._legacy_compat_host)
        self.stage_filter = QComboBox(self._legacy_compat_host)
        self.split_filter = QComboBox(self._legacy_compat_host)
        self.pose_filter = QComboBox(self._legacy_compat_host)
        self.status_filter = QComboBox(self._legacy_compat_host)
        self.previous_artifact_button = QPushButton("上一张", self._legacy_compat_host)
        self.next_artifact_button = QPushButton("下一张", self._legacy_compat_host)
        self.open_artifact_button = QPushButton("在文件夹中打开", self._legacy_compat_host)
        self.tree = QTreeWidget(self._legacy_compat_host)
        self.tree.setHeaderLabels(["项目", "值"])
        self.artifact_tree = QTreeWidget(self._legacy_compat_host)
        self.artifact_tree.setHeaderLabels(["结果产物", "状态"])
        self.artifacts = QListWidget(self._legacy_compat_host)
        self.image_preview = ImagePreview(self._legacy_compat_host)
        self.image_status = QLabel("请选择结果图", self._legacy_compat_host)
        self.artifact_info = QLabel("暂无文件信息", self._legacy_compat_host)
        self.plot = ResidualPlot(self._legacy_compat_host)
        self.table = QTableWidget(self._legacy_compat_host)
        self.load_button.clicked.connect(self._open)
        self.acceptance_button.clicked.connect(self.generate_acceptance)
        self.open_html_button.clicked.connect(self.open_html)
        self.artifacts.itemSelectionChanged.connect(self._artifact_selected)
        self.artifact_tree.itemClicked.connect(self._artifact_tree_selected)
        for combo in (self.stage_filter, self.split_filter, self.pose_filter, self.status_filter):
            combo.currentIndexChanged.connect(self._apply_artifact_filters)
        self.previous_artifact_button.clicked.connect(self._previous_artifact)
        self.next_artifact_button.clicked.connect(self._next_artifact)
        self.open_artifact_button.clicked.connect(self._open_artifact_folder)

    def set_project(self, project: WizardProject) -> None:
        self.project = project
        existing = project.acceptance_plan
        plan_path = default_acceptance_plan_path(project.workspace, existing)
        if existing is None:
            project.acceptance_plan = plan_path
        self.acceptance_status.setText(f"当前项目验收计划：{plan_path}")
        self.acceptance_plan.setText(str(plan_path or ""))
        stored = project.extra.get("capture_artifacts") if isinstance(project.extra, dict) else None
        self.set_capture_artifacts(stored if isinstance(stored, dict) else None)
        run_path = project.last_calibration_run
        if run_path is None:
            self.clear_calibration_run()
        elif run_path.is_file():
            self.load_calibration_run_path(run_path, silent=True)
        else:
            self.clear_calibration_run(
                f"最近一次标定结果文件不存在：{run_path}",
                path=run_path,
            )

    def clear_calibration_run(
        self,
        message: str = "当前项目暂无标定结果",
        *,
        path: str | Path | None = None,
    ) -> None:
        """清空 Calibration Run 展示，但保留项目和采集状态。"""

        self.current_run = None
        self.current_run_path = Path(path).expanduser().resolve() if path is not None else None
        self.current_result = None
        self._results_summary = None
        self.last_html = None
        self.open_html_button.setEnabled(False)
        self.tree.clear()
        self._set_artifact_records([])
        self.run_status.setText(message)
        self.result_tabs.setCurrentIndex(0)
        self._update_result_overview(None)
        self.image_preview.clear_image(message)
        self.image_status.setText(message)
        self.artifact_info.setText("暂无文件信息")

    def load_calibration_run_path(
        self,
        path: str | Path,
        *,
        silent: bool = False,
    ) -> CalibrationRun | None:
        """从 Calibration Run 或旧 workflow report 路径加载统一结果。"""

        source = Path(path).expanduser().resolve()
        try:
            run = load_calibration_run(source)
        except Exception as exc:
            message = f"标定结果读取失败：{source}\n{exc}"
            self.clear_calibration_run(message, path=source)
            if not silent:
                QMessageBox.critical(self, "标定结果读取失败", message)
            return None
        self.show_calibration_run(run, path=source)
        return run

    def show_calibration_run(
        self,
        run: CalibrationRun,
        *,
        path: str | Path | None = None,
    ) -> None:
        """设置当前 CalibrationRun，并更新总览与结构化详细结果。"""

        if not isinstance(run, CalibrationRun):
            raise TypeError("run 必须是 CalibrationRun")
        self.current_run = run
        self.current_run_path = Path(path).expanduser().resolve() if path is not None else None
        self.current_result = None
        self._set_artifact_records([])
        source = f"\n来源：{self.current_run_path}" if self.current_run_path is not None else ""
        self.run_status.setText(
            f"当前 Calibration Run：{run.run_id} · overall：{_result_status_text(run.overall_status)}{source}"
        )
        self.result_tabs.setCurrentIndex(0)
        self._update_result_overview(summarize_calibration_run(run))

    def _update_result_overview(self, summary: CalibrationResultsSummary | None) -> None:
        self._results_summary = summary
        if summary is None:
            self.overview_run_id.setText("暂无")
            self.overview_time.setText("暂无")
            self.overview_status.setText("暂无")
            self.overview_overall.setText("暂无")
            self.overview_orientation.setText("暂无")
            for widget in (
                self.intrinsics_status,
                self.intrinsics_fit_rmse,
                self.intrinsics_test_rmse,
                self.intrinsics_fit_images,
                self.intrinsics_test_images,
                self.laser_status,
                self.laser_model,
                self.laser_validation_rmse,
                self.laser_validation_p95,
                self.laser_valid_rate,
                self.ground_status,
                self.ground_validation_rmse,
                self.ground_validation_p95,
                self.laser_detail_status,
                self.laser_detail_model,
                self.laser_detail_validation_rmse,
                self.laser_detail_validation_p95,
                self.laser_detail_valid_rate,
                self.ground_detail_status,
                self.ground_detail_stage,
                self.ground_detail_validation_rmse,
                self.ground_detail_validation_p95,
                self.ground_detail_fit_frames,
                self.ground_detail_validation_frames,
                self.ground_pose_status,
                self.ground_transform_name,
                self.ground_translation,
                self.ground_euler_status,
                self.ground_roll,
                self.ground_pitch,
                self.ground_yaw,
                self.ground_validation_status,
            ):
                widget.setText("未执行")
            self._update_intrinsics_details(None)
            self._update_laser_surface_details(None)
            self._update_ground_extrinsics_details(None)
            return

        self.overview_run_id.setText(summary.run_id)
        self.overview_time.setText(_overview_time_text(summary.started_utc, summary.completed_utc))
        self.overview_status.setText(_result_status_text(summary.status))
        self.overview_overall.setText(_result_status_text(summary.overall))
        self.overview_orientation.setText(summary.laser_orientation or "暂无")

        self.intrinsics_status.setText(_stage_status_text(summary.intrinsics.status))
        self.intrinsics_fit_rmse.setText(
            _overview_metric_text(summary.intrinsics.status, summary.intrinsics.fit_rmse_px, unit=" px")
        )
        self.intrinsics_test_rmse.setText(
            _overview_metric_text(summary.intrinsics.status, summary.intrinsics.test_rmse_px, unit=" px")
        )
        self.intrinsics_fit_images.setText(
            _overview_metric_text(summary.intrinsics.status, summary.intrinsics.fit_image_count)
        )
        self.intrinsics_test_images.setText(
            _overview_metric_text(summary.intrinsics.status, summary.intrinsics.test_image_count)
        )

        self.laser_status.setText(_stage_status_text(summary.laser_surface.status))
        self.laser_model.setText(
            _overview_metric_text(summary.laser_surface.status, summary.laser_surface.model_type)
        )
        self.laser_validation_rmse.setText(
            _overview_metric_text(
                summary.laser_surface.status,
                summary.laser_surface.validation_rmse_mm,
                unit=" mm",
            )
        )
        self.laser_validation_p95.setText(
            _overview_metric_text(
                summary.laser_surface.status,
                summary.laser_surface.validation_p95_mm,
                unit=" mm",
            )
        )
        self.laser_valid_rate.setText(
            _overview_metric_text(
                summary.laser_surface.status,
                summary.laser_surface.validation_valid_rate,
                percent=True,
            )
        )

        self.ground_status.setText(_stage_status_text(summary.ground_extrinsics.status))
        self.ground_validation_rmse.setText(
            _overview_metric_text(
                summary.ground_extrinsics.status,
                summary.ground_extrinsics.validation_rmse_mm,
                unit=" mm",
            )
        )
        self.ground_validation_p95.setText(
            _overview_metric_text(
                summary.ground_extrinsics.status,
                summary.ground_extrinsics.validation_p95_mm,
                unit=" mm",
            )
        )

        self.laser_detail_status.setText(_stage_status_text(summary.laser_surface.status))
        self.laser_detail_model.setText(
            _overview_metric_text(summary.laser_surface.status, summary.laser_surface.model_type)
        )
        self.laser_detail_validation_rmse.setText(
            _overview_metric_text(
                summary.laser_surface.status,
                summary.laser_surface.validation_rmse_mm,
                unit=" mm",
            )
        )
        self.laser_detail_validation_p95.setText(
            _overview_metric_text(
                summary.laser_surface.status,
                summary.laser_surface.validation_p95_mm,
                unit=" mm",
            )
        )
        self.laser_detail_valid_rate.setText(
            _overview_metric_text(
                summary.laser_surface.status,
                summary.laser_surface.validation_valid_rate,
                percent=True,
            )
        )

        self.ground_detail_status.setText(_stage_status_text(summary.ground_extrinsics.status))
        self.ground_detail_validation_rmse.setText(
            _overview_metric_text(
                summary.ground_extrinsics.status,
                summary.ground_extrinsics.validation_rmse_mm,
                unit=" mm",
            )
        )
        self.ground_detail_validation_p95.setText(
            _overview_metric_text(
                summary.ground_extrinsics.status,
                summary.ground_extrinsics.validation_p95_mm,
                unit=" mm",
            )
        )
        self._update_intrinsics_details(summary)
        self._update_laser_surface_details(summary)
        self._update_ground_extrinsics_details(summary)

    @staticmethod
    def _build_laser_plot_card(title: str, preview: QLabel) -> QWidget:
        card = QWidget()
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        caption = QLabel(title)
        caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(caption)
        card_layout.addWidget(preview, 1)
        return card

    def _toggle_laser_parameters(self, expanded: bool) -> None:
        self.laser_parameters_panel.setVisible(expanded)
        self.laser_parameters_toggle.setText(
            "模型参数（收起）" if expanded else "模型参数（展开）"
        )

    @staticmethod
    def _laser_table_metric(
        stage_status: str,
        comparison: Any,
        value: Any,
        *,
        percent: bool = False,
    ) -> str:
        if stage_status == NOT_EXECUTED:
            return "未执行"
        if comparison is None or not comparison.available:
            return "未找到"
        return _overview_metric_text(
            "completed",
            value,
            percent=percent,
        )

    def _update_laser_comparison_table(
        self,
        stage_status: str,
        details: LaserSurfaceDetails | None,
    ) -> None:
        comparisons = {
            comparison.model: comparison
            for comparison in (details.model_comparisons if details is not None else ())
        }
        selected_model = details.selected_model if details is not None else None
        table = self.laser_model_comparison_table
        table.setRowCount(len(SUPPORTED_LASER_MODEL_TYPES))
        for row_index, model in enumerate(SUPPORTED_LASER_MODEL_TYPES):
            comparison = comparisons.get(model)
            values = (
                model,
                self._laser_table_metric(
                    stage_status,
                    comparison,
                    comparison.train_rmse_mm if comparison is not None else None,
                ),
                self._laser_table_metric(
                    stage_status,
                    comparison,
                    comparison.validation_rmse_mm if comparison is not None else None,
                ),
                self._laser_table_metric(
                    stage_status,
                    comparison,
                    comparison.validation_p95_mm if comparison is not None else None,
                ),
                self._laser_table_metric(
                    stage_status,
                    comparison,
                    comparison.validation_max_mm if comparison is not None else None,
                ),
                self._laser_table_metric(
                    stage_status,
                    comparison,
                    comparison.validation_valid_rate if comparison is not None else None,
                    percent=True,
                ),
                (
                    "当前采用"
                    if model == selected_model
                    else "已读取"
                    if comparison is not None and comparison.available
                    else "未找到"
                    if stage_status != NOT_EXECUTED
                    else "未执行"
                ),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if model == selected_model:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                table.setItem(row_index, column, item)

    @staticmethod
    def _laser_parameter_value(value: Any) -> str:
        if isinstance(value, tuple):
            return "[" + ", ".join(_detail_value(item) for item in value) + "]"
        return _detail_value(value)

    def _update_laser_parameter_table(self, details: LaserSurfaceDetails | None) -> None:
        rows: list[tuple[str, str, str]] = []
        if details is not None and details.status != NOT_EXECUTED:
            for comparison in details.model_comparisons:
                parameters = comparison.parameters
                if parameters is None:
                    continue
                if comparison.model == "global_plane":
                    for name, value in zip(
                        ("a", "b", "c", "d"), parameters.plane_abcd
                    ):
                        rows.append(
                            (
                                comparison.model,
                                name,
                                self._laser_parameter_value(value),
                            )
                        )
                elif comparison.model == "circular_cone":
                    if parameters.apex_camera_mm:
                        rows.append(
                            (
                                comparison.model,
                                "apex",
                                self._laser_parameter_value(parameters.apex_camera_mm),
                            )
                        )
                    if parameters.axis_unit_camera:
                        rows.append(
                            (
                                comparison.model,
                                "axis",
                                self._laser_parameter_value(parameters.axis_unit_camera),
                            )
                        )
                    if parameters.half_apex_angle_deg is not None:
                        rows.append(
                            (
                                comparison.model,
                                "half-angle",
                                f"{_detail_value(parameters.half_apex_angle_deg)}°",
                            )
                        )
                elif comparison.model == "quadratic_graph":
                    for index, value in enumerate(parameters.coefficients):
                        rows.append(
                            (
                                comparison.model,
                                f"coefficient β{index}",
                                self._laser_parameter_value(value),
                            )
                        )
        table = self.laser_model_parameters_table
        table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column, value in enumerate(values):
                table.setItem(row_index, column, QTableWidgetItem(value))
        table.resizeRowsToContents()
        header_height = table.horizontalHeader().sizeHint().height()
        row_height = sum(table.rowHeight(index) for index in range(table.rowCount()))
        table.setFixedHeight(
            max(1, header_height + row_height + table.frameWidth() * 2 + 4)
        )
        table.updateGeometry()

    @staticmethod
    def _set_laser_plot_preview(
        preview: LaserPlotPreview,
        card: QWidget,
        path: Path | None,
    ) -> bool:
        if path is None or not path.is_file():
            preview.clear_plot()
            card.setVisible(False)
            return False
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            preview.clear_plot()
            card.setVisible(False)
            return False
        preview.set_plot_pixmap(pixmap)
        card.setVisible(True)
        return True

    def _update_laser_surface_details(
        self,
        summary: CalibrationResultsSummary | None,
    ) -> None:
        details = (
            load_laser_surface_details(summary.laser_surface)
            if summary is not None
            else None
        )
        self._laser_surface_details = details
        stage_status = summary.laser_surface.status if summary is not None else NOT_EXECUTED
        if details is not None and details.status == NOT_EXECUTED:
            stage_status = NOT_EXECUTED

        if summary is None or details is None or stage_status == NOT_EXECUTED:
            for widget in (
                self.laser_detail_status,
                self.laser_detail_model,
                self.laser_detail_validation_rmse,
                self.laser_detail_validation_p95,
                self.laser_detail_valid_rate,
            ):
                widget.setText("未执行")
            self.laser_model_comparison_status.setText("未执行")
            self.laser_error_plots_status.setText("暂无已有 validation error 图")
        else:
            selected_model = details.selected_model or summary.laser_surface.model_type
            self.laser_detail_status.setText(_stage_status_text(stage_status))
            self.laser_detail_model.setText(
                _overview_metric_text(stage_status, selected_model)
            )
            self.laser_detail_validation_rmse.setText(
                _overview_metric_text(
                    stage_status,
                    details.validation_rmse_mm,
                    unit=" mm",
                )
            )
            self.laser_detail_validation_p95.setText(
                _overview_metric_text(
                    stage_status,
                    details.validation_p95_mm,
                    unit=" mm",
                )
            )
            self.laser_detail_valid_rate.setText(
                _overview_metric_text(
                    stage_status,
                    details.validation_valid_rate,
                    percent=True,
                )
            )
            available_count = sum(
                1 for comparison in details.model_comparisons if comparison.available
            )
            status_parts = []
            if details.error:
                status_parts.append(details.error)
            else:
                status_parts.append(f"已读取 {available_count}/3 个模型结果")
            status_parts.extend(details.notes)
            self.laser_model_comparison_status.setText("；".join(status_parts))
            plot_count = sum(
                self._set_laser_plot_preview(preview, card, path)
                for preview, card, path in (
                    (
                        self.laser_error_vs_u,
                        self.laser_error_vs_u_card,
                        details.error_vs_u,
                    ),
                    (
                        self.laser_error_vs_v,
                        self.laser_error_vs_v_card,
                        details.error_vs_v,
                    ),
                    (
                        self.laser_error_vs_depth,
                        self.laser_error_vs_depth_card,
                        details.error_vs_depth,
                    ),
                )
            )
            self.laser_error_plots_status.setText(
                f"已展示 {plot_count} 张已有 validation error 图（未重新计算）"
                if plot_count
                else "未找到已有 validation error 图"
            )

        self._update_laser_comparison_table(stage_status, details)
        self._update_laser_parameter_table(details)
        if summary is None or details is None or stage_status == NOT_EXECUTED:
            for preview, card in (
                (self.laser_error_vs_u, self.laser_error_vs_u_card),
                (self.laser_error_vs_v, self.laser_error_vs_v_card),
                (self.laser_error_vs_depth, self.laser_error_vs_depth_card),
            ):
                self._set_laser_plot_preview(preview, card, None)

    @staticmethod
    def _resize_result_table(table: QTableWidget) -> None:
        table.resizeRowsToContents()
        header_height = table.horizontalHeader().sizeHint().height()
        row_height = sum(table.rowHeight(index) for index in range(table.rowCount()))
        table.setFixedHeight(
            max(1, header_height + row_height + table.frameWidth() * 2 + 4)
        )
        table.updateGeometry()

    def _resize_intrinsics_reprojection_table(self) -> None:
        """按逐图结果数量收紧内参表格，避免空状态占满整页。"""

        table = self.intrinsics_reprojection_table
        table.resizeRowsToContents()
        header_height = table.horizontalHeader().sizeHint().height()
        row_height = sum(table.rowHeight(index) for index in range(table.rowCount()))
        content_height = header_height + row_height + table.frameWidth() * 2 + 4
        # 逐图结果较多时保留表格自身的按需滚动，避免把内参页撑出窗口。
        table.setFixedHeight(max(36, min(content_height, 420)))
        table.updateGeometry()

    def _update_ground_validation_table(self, rows: tuple[Any, ...]) -> None:
        table = self.ground_validation_table
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                row.frame,
                _result_status_text(row.status or "recorded"),
                _detail_value(row.pnp_rmse_px),
                _detail_value(row.rmse_mm),
                _detail_value(row.p95_mm),
                _detail_value(row.max_mm),
                _detail_value(row.std_mm),
                _detail_value(row.point_count),
            )
            for column, value in enumerate(values):
                table.setItem(row_index, column, QTableWidgetItem(value))
        self._resize_result_table(table)

    def _update_ground_extrinsics_details(
        self,
        summary: CalibrationResultsSummary | None,
    ) -> None:
        details = (
            load_ground_extrinsics_details(summary.ground_extrinsics)
            if summary is not None
            else None
        )
        self._ground_extrinsics_details = details
        stage_status = (
            summary.ground_extrinsics.status if summary is not None else NOT_EXECUTED
        )
        if details is not None and details.status == NOT_EXECUTED:
            stage_status = NOT_EXECUTED

        if summary is None or details is None or stage_status == NOT_EXECUTED:
            for widget in (
                self.ground_detail_status,
                self.ground_detail_stage,
                self.ground_detail_validation_rmse,
                self.ground_detail_validation_p95,
                self.ground_detail_fit_frames,
                self.ground_detail_validation_frames,
                self.ground_pose_status,
                self.ground_transform_name,
                self.ground_translation,
                self.ground_euler_status,
                self.ground_roll,
                self.ground_pitch,
                self.ground_yaw,
                self.ground_validation_status,
            ):
                widget.setText("未执行")
            self._set_ground_rotation((), "未执行")
            self._update_ground_validation_table(())
            self._set_laser_plot_preview(
                self.ground_validation_error_plot,
                self.ground_validation_error_plot_card,
                None,
            )
            return

        validation_rmse = summary.ground_extrinsics.validation_rmse_mm
        if validation_rmse is None:
            validation_rmse = details.validation_rmse_mm
        validation_p95 = summary.ground_extrinsics.validation_p95_mm
        if validation_p95 is None:
            validation_p95 = details.validation_p95_mm
        self.ground_status.setText(_stage_status_text(stage_status))
        self.ground_validation_rmse.setText(
            _overview_metric_text(stage_status, validation_rmse, unit=" mm")
        )
        self.ground_validation_p95.setText(
            _overview_metric_text(stage_status, validation_p95, unit=" mm")
        )
        self.ground_detail_status.setText(_stage_status_text(stage_status))
        self.ground_detail_stage.setText(details.stage)
        self.ground_detail_validation_rmse.setText(
            _overview_metric_text(stage_status, validation_rmse, unit=" mm")
        )
        self.ground_detail_validation_p95.setText(
            _overview_metric_text(stage_status, validation_p95, unit=" mm")
        )
        self.ground_detail_fit_frames.setText(
            _overview_metric_text(stage_status, details.fit_frame_count)
        )
        self.ground_detail_validation_frames.setText(
            _overview_metric_text(stage_status, details.validation_frame_count)
        )

        self.ground_transform_name.setText(details.transform_name or "暂无")
        self._set_ground_rotation(details.rotation_matrix, "暂无")
        if len(details.translation_mm) >= 3:
            translation = "[" + ", ".join(
                _detail_value(value) for value in details.translation_mm[:3]
            ) + "] mm"
        else:
            translation = "暂无"
        self.ground_translation.setText(translation)
        if (
            details.roll_deg is not None
            and details.pitch_deg is not None
            and details.yaw_deg is not None
        ):
            self.ground_euler_status.setText("结果文件已提供（单位 °）")
            self.ground_roll.setText(f"{details.roll_deg:.6f}°")
            self.ground_pitch.setText(f"{details.pitch_deg:.6f}°")
            self.ground_yaw.setText(f"{details.yaw_deg:.6f}°")
        else:
            self.ground_euler_status.setText(
                "暂无（项目未定义可靠的 Euler 轴顺序，未自行换算）"
            )
            self.ground_roll.setText("暂无")
            self.ground_pitch.setText("暂无")
            self.ground_yaw.setText("暂无")

        pose_messages: list[str] = []
        if details.error:
            pose_messages.append(details.error)
        if details.rotation_matrix:
            pose_messages.append("已读取 T_ground_from_camera 的 R/t")
        else:
            pose_messages.append("结果文件未提供有效的 camera → ground R/t")
        self.ground_pose_status.setText("；".join(pose_messages))

        validation_messages: list[str] = []
        if details.error:
            validation_messages.append(details.error)
        if details.validation_rows:
            validation_messages.append(
                f"已读取 {len(details.validation_rows)} 条逐帧 validation 结果（未重新计算）"
            )
        else:
            validation_messages.append("未找到已有独立 validation 逐帧结果")
        self.ground_validation_status.setText("；".join(validation_messages))
        self._update_ground_validation_table(details.validation_rows)
        self._set_laser_plot_preview(
            self.ground_validation_error_plot,
            self.ground_validation_error_plot_card,
            details.validation_error_plot,
        )

    def _set_ground_rotation(
        self,
        matrix: tuple[tuple[float, ...], ...],
        placeholder: str,
    ) -> None:
        table = self.ground_rotation_table
        table.setRowCount(3)
        table.setColumnCount(3)
        for row in range(3):
            for column in range(3):
                value = _matrix_value(matrix, row, column)
                text = _detail_value(value) if value is not None else placeholder
                table.setItem(row, column, QTableWidgetItem(text))
        self._resize_result_table(table)

    def _update_intrinsics_details(
        self,
        summary: CalibrationResultsSummary | None,
    ) -> None:
        details: IntrinsicsDetails | None = (
            load_intrinsics_details(summary.intrinsics) if summary is not None else None
        )

        stage_status = summary.intrinsics.status if summary is not None else NOT_EXECUTED
        fit_rmse = summary.intrinsics.fit_rmse_px if summary is not None else None
        test_rmse = summary.intrinsics.test_rmse_px if summary is not None else None
        fit_images = summary.intrinsics.fit_image_count if summary is not None else None
        test_images = summary.intrinsics.test_image_count if summary is not None else None
        if details is not None:
            fit_rmse = fit_rmse if fit_rmse is not None else details.fit_rmse_px
            test_rmse = test_rmse if test_rmse is not None else details.test_rmse_px
            fit_images = fit_images if fit_images is not None else details.fit_image_count
            test_images = test_images if test_images is not None else details.test_image_count
        self.intrinsics_detail_fit_rmse.setText(
            _overview_metric_text(
                stage_status,
                fit_rmse,
                unit=" px",
            )
        )
        self.intrinsics_detail_test_rmse.setText(
            _overview_metric_text(
                stage_status,
                test_rmse,
                unit=" px",
            )
        )
        self.intrinsics_detail_fit_images.setText(
            _overview_metric_text(
                stage_status,
                fit_images,
            )
        )
        self.intrinsics_detail_test_images.setText(
            _overview_metric_text(
                stage_status,
                test_images,
            )
        )

        if details is None or details.status == NOT_EXECUTED:
            self.intrinsics_detail_status.setText("未执行")
            self.intrinsics_reprojection_status.setText("未执行")
            self.intrinsics_reprojection_table.setRowCount(0)
            self._resize_intrinsics_reprojection_table()
            for widget in (
                self.intrinsics_fx,
                self.intrinsics_fy,
                self.intrinsics_cx,
                self.intrinsics_cy,
                self.intrinsics_k1,
                self.intrinsics_k2,
                self.intrinsics_p1,
                self.intrinsics_p2,
                self.intrinsics_k3,
            ):
                widget.setText("未执行")
            return

        if details.error:
            self.intrinsics_detail_status.setText(details.error)
        else:
            message = "内参结果已加载"
            if details.notes:
                message += "；" + "；".join(details.notes)
            self.intrinsics_detail_status.setText(message)

        matrix = details.camera_matrix
        self.intrinsics_fx.setText(_detail_value(_matrix_value(matrix, 0, 0)))
        self.intrinsics_fy.setText(_detail_value(_matrix_value(matrix, 1, 1)))
        self.intrinsics_cx.setText(_detail_value(_matrix_value(matrix, 0, 2)))
        self.intrinsics_cy.setText(_detail_value(_matrix_value(matrix, 1, 2)))
        distortion = details.dist_coeffs
        for widget, index in (
            (self.intrinsics_k1, 0),
            (self.intrinsics_k2, 1),
            (self.intrinsics_p1, 2),
            (self.intrinsics_p2, 3),
            (self.intrinsics_k3, 4),
        ):
            widget.setText(_detail_value(distortion[index] if index < len(distortion) else None))

        rows = details.reprojection_rows
        self.intrinsics_reprojection_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                row.split,
                row.image,
                _result_status_text(row.status),
                _detail_value(row.rmse_px),
                _detail_value(row.mean_error_px),
            )
            for column, value in enumerate(values):
                self.intrinsics_reprojection_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(value),
                )
        self._resize_intrinsics_reprojection_table()
        if rows:
            self.intrinsics_reprojection_status.setText(
                f"已读取 {len(rows)} 条逐图重投影误差（未重新计算）"
            )
        elif details.error:
            self.intrinsics_reprojection_status.setText(details.error)
        else:
            self.intrinsics_reprojection_status.setText("未找到已有逐图重投影误差")

    def prepare_default_plan(self, *, announce: bool = True) -> Path | None:
        """进入第 5 页时创建默认计划；已有显式计划保持不动。"""

        if self.project is None:
            return None
        try:
            default_path = default_acceptance_plan_path(self.project.workspace)
            field_value = self.acceptance_plan.text().strip()
            existing = Path(field_value).expanduser().resolve() if field_value else self.project.acceptance_plan
            # set_project 会先把默认路径显示到控件；该路径仍属于自动计划，
            # 不应被当成“用户显式计划”而跳过首次创建。
            explicit = existing if existing is not None and existing.resolve() != default_path.resolve() else None
            path = ensure_default_acceptance_plan(
                self.project.workspace,
                self.project.project_id,
                existing_path=explicit,
            )
        except Exception as exc:
            self.acceptance_status.setText(f"默认验收计划准备失败：{exc}")
            return None
        self.project.acceptance_plan = path
        self.acceptance_plan.setText(str(path))
        if announce:
            self.acceptance_status.setText(f"当前项目验收计划：{path}")
        return path

    def update_acceptance_from_workflow(self, result: Mapping[str, Any]) -> None:
        """workflow 完成后自动填充当前项目验收计划的空输入项。"""

        self.prepare_default_plan(announce=False)
        plan_value = self.acceptance_plan.text().strip()
        workflow_value = result.get("workflow")
        if not plan_value or not workflow_value:
            return
        try:
            update = update_acceptance_plan_from_workflow(
                plan_value,
                str(workflow_value),
                result,
            )
        except Exception as exc:
            self.acceptance_status.setText(f"Workflow 已完成，但验收计划路径更新失败：{exc}")
            return
        if update.changed:
            self.acceptance_status.setText(
                "已从最近 workflow 自动更新验收输入："
                f"\nworkflow：{update.workflow_report or '暂无'}"
                f"\n补偿指标：{update.compensation_metrics or '暂无'}"
                "\n请核对后点击“生成并判定”。"
            )
        else:
            self.acceptance_status.setText("Workflow 已完成，验收计划未发现需要更新的空路径。")

    def set_capture_artifacts(self, artifacts: Mapping[str, Any] | None) -> None:
        self.capture_artifacts = dict(artifacts) if artifacts else None
        if self.current_result is not None:
            self._set_artifact_records(
                discover_artifact_records(self.current_result, self.capture_artifacts)
            )

    def generate_acceptance(self) -> None:
        path = Path(self.acceptance_plan.text()).expanduser()
        if not path.is_file():
            QMessageBox.warning(self, "验收计划不存在", str(path)); return
        overwrite = self.acceptance_overwrite.isChecked()
        self.acceptance_button.setEnabled(False)
        self.acceptance_status.setText("正在汇总 workflow、补偿指标、golden 和产物哈希…")
        worker = FunctionWorker(lambda _progress: build_acceptance_report(path, overwrite=overwrite))
        worker.signals.result.connect(self._acceptance_done)
        worker.signals.error.connect(self._acceptance_error)
        worker.signals.finished.connect(lambda: self.acceptance_button.setEnabled(True))
        self._workers.add(worker); worker.signals.finished.connect(lambda: self._workers.discard(worker)); self.thread_pool.start(worker)

    def _acceptance_done(self, report: dict[str, Any]) -> None:
        self.show_result(report)
        files = report.get("report_files", {})
        html_value = files.get("html")
        self.last_html = Path(html_value) if html_value else None
        self.open_html_button.setEnabled(bool(self.last_html and self.last_html.is_file()))
        self.acceptance_status.setText(
            f"验收结论：{report.get('decision')} · PASS {report['counts']['pass']} · "
            f"WARN {report['counts']['warn']} · FAIL {report['counts']['fail']} · "
            f"发布：{report.get('release', {}).get('status')}"
        )
        self.acceptance_finished.emit(report)

    def _acceptance_error(self, message: str) -> None:
        self.acceptance_status.setText(f"验收失败：{message}")
        QMessageBox.critical(self, "无法生成验收报告", message)

    def open_html(self) -> None:
        if self.last_html and self.last_html.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_html.resolve())))

    def show_result(self, result: dict[str, Any]) -> None:
        """展示通用报告投影；标定主结果入口是 ``show_calibration_run``。"""

        self.current_result = dict(result)
        self.tree.clear()
        _add_tree(self.tree.invisibleRootItem(), self.current_result)
        self.tree.expandToDepth(2)
        capture = self.current_result.get("capture_artifacts")
        if not isinstance(capture, Mapping):
            capture = self.capture_artifacts
        self._set_artifact_records(
            discover_artifact_records(self.current_result, capture if isinstance(capture, Mapping) else None)
        )

    def _set_artifact_records(self, records: list[ResultArtifact]) -> None:
        self._artifact_records = list(records)
        self._filtered_artifacts = []
        self._current_artifact_index = -1
        self.artifacts.clear()
        for record in self._artifact_records:
            self.artifacts.addItem(str(record.display_path))
        self._set_filter_values(self.stage_filter, [record.stage for record in records])
        self._set_filter_values(self.split_filter, [record.split for record in records if record.split])
        self._set_filter_values(self.pose_filter, [record.pose_id for record in records if record.pose_id])
        status_values = [record.status or "暂无关联指标" for record in records]
        self._set_filter_values(self.status_filter, status_values)
        self._apply_artifact_filters()

    @staticmethod
    def _set_filter_values(combo: QComboBox, values: list[str | None]) -> None:
        normalized = sorted({str(value) for value in values if value not in (None, "")})
        with QSignalBlocker(combo):
            combo.clear()
            combo.addItem("全部")
            combo.addItems(normalized)
        combo.setCurrentIndex(0)

    def _apply_artifact_filters(self) -> None:
        stage = self.stage_filter.currentText()
        split = self.split_filter.currentText()
        pose = self.pose_filter.currentText()
        status = self.status_filter.currentText()
        self._filtered_artifacts = [
            record
            for record in self._artifact_records
            if (stage in ("", "全部") or record.stage == stage)
            and (split in ("", "全部") or record.split == split)
            and (pose in ("", "全部") or record.pose_id == pose)
            and (
                status in ("", "全部")
                or (record.status or "暂无关联指标") == status
            )
        ]
        self.artifact_tree.clear()
        stage_items: dict[str, QTreeWidgetItem] = {}
        split_items: dict[tuple[str, str], QTreeWidgetItem] = {}
        pose_items: dict[tuple[str, str, str], QTreeWidgetItem] = {}
        index_lookup = {id(record): index for index, record in enumerate(self._artifact_records)}
        for record in self._filtered_artifacts:
            stage_item = stage_items.get(record.stage)
            if stage_item is None:
                stage_item = QTreeWidgetItem([record.stage, ""])
                self.artifact_tree.addTopLevelItem(stage_item)
                stage_items[record.stage] = stage_item
            split_name = (
                "summary_plots"
                if record.pose_id is None and record.artifact_type.startswith("validation_error")
                else record.split or "summary_plots"
            )
            split_key = (record.stage, split_name)
            split_item = split_items.get(split_key)
            if split_item is None:
                split_item = QTreeWidgetItem([split_name, ""])
                stage_item.addChild(split_item)
                split_items[split_key] = split_item
            pose_name = record.pose_id or "汇总"
            pose_key = (record.stage, split_name, pose_name)
            pose_item = pose_items.get(pose_key)
            if pose_item is None:
                pose_item = QTreeWidgetItem([pose_name, ""])
                split_item.addChild(pose_item)
                pose_items[pose_key] = pose_item
            child = QTreeWidgetItem([f"{record.artifact_type} · {record.display_path.name}", record.status or "暂无关联指标"])
            child.setData(0, Qt.ItemDataRole.UserRole, index_lookup[id(record)])
            pose_item.addChild(child)
        if not self._filtered_artifacts:
            self.artifact_tree.addTopLevelItem(QTreeWidgetItem(["没有发现结果图或诊断产物", ""]))
            self.image_preview.clear_image("没有可显示的结果图")
            self.image_status.setText("当前筛选没有结果产物")
            self.artifact_info.setText("暂无文件信息")
        else:
            self.artifact_tree.expandAll()
        self.previous_artifact_button.setEnabled(bool(self._filtered_artifacts))
        self.next_artifact_button.setEnabled(bool(self._filtered_artifacts))
        self.open_artifact_button.setEnabled(bool(self._filtered_artifacts))

    def _artifact_tree_selected(self, item: QTreeWidgetItem, _column: int) -> None:
        value = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(value, int):
            self._display_artifact(value)

    def _display_artifact(self, index: int) -> None:
        if index < 0 or index >= len(self._artifact_records):
            return
        record = self._artifact_records[index]
        self._current_artifact_index = index
        metrics = "暂无关联指标"
        if record.metrics:
            metrics = "\n".join(f"{key}: {value}" for key, value in record.metrics.items())
        self.artifact_info.setText(
            f"stage：{record.stage}\n"
            f"split：{record.split or 'summary'}\n"
            f"pose_id：{record.pose_id or '汇总'}\n"
            f"类型：{record.artifact_type}\n"
            f"状态：{record.status or '暂无关联指标'}\n"
            f"文件：{record.display_path}\n\n{metrics}"
        )
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.plot.set_values([])
        path = record.display_path
        if record.image_path is not None:
            if not path.is_file():
                self.image_preview.clear_image("图像缺失")
                self.image_status.setText(f"图像缺失：{path}")
                return
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                self.image_preview.clear_image("图像损坏或无法读取")
                self.image_status.setText(f"图像损坏或无法读取：{path}")
                return
            try:
                if image.ndim == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                self.image_preview.set_array(image, auto_stretch=True)
                self.image_status.setText(f"已显示：{path}")
            except Exception as exc:
                self.image_preview.clear_image("图像无法显示")
                self.image_status.setText(f"图像无法显示：{exc}")
            return
        self.image_preview.clear_image("当前产物不是图像")
        if path.suffix.lower() == ".csv":
            try:
                headers, rows, values = load_residual_csv(path)
            except Exception as exc:
                self.image_status.setText(f"诊断文件读取失败：{exc}")
                return
            self.table.setColumnCount(len(headers)); self.table.setHorizontalHeaderLabels(headers)
            self.table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                for column, value in enumerate(row):
                    self.table.setItem(row_index, column, QTableWidgetItem(value))
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self.plot.set_values(values)
            self.image_status.setText(f"诊断 CSV：{path}")
        else:
            self.image_status.setText(f"诊断文件：{path}")

    def _previous_artifact(self) -> None:
        if not self._filtered_artifacts:
            return
        current = self._filtered_artifacts.index(self._artifact_records[self._current_artifact_index]) if self._current_artifact_index >= 0 and self._artifact_records[self._current_artifact_index] in self._filtered_artifacts else 0
        record = self._filtered_artifacts[(current - 1) % len(self._filtered_artifacts)]
        self._display_artifact(self._artifact_records.index(record))

    def _next_artifact(self) -> None:
        if not self._filtered_artifacts:
            return
        current = self._filtered_artifacts.index(self._artifact_records[self._current_artifact_index]) if self._current_artifact_index >= 0 and self._artifact_records[self._current_artifact_index] in self._filtered_artifacts else -1
        record = self._filtered_artifacts[(current + 1) % len(self._filtered_artifacts)]
        self._display_artifact(self._artifact_records.index(record))

    def _open_artifact_folder(self) -> None:
        if self._current_artifact_index < 0 or self._current_artifact_index >= len(self._artifact_records):
            return
        path = self._artifact_records[self._current_artifact_index].display_path
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent.resolve())))

    def _open(self) -> None:
        start = ""
        if self.current_run_path is not None:
            start = str(self.current_run_path)
        elif self.project is not None and self.project.last_calibration_run is not None:
            start = str(self.project.last_calibration_run)
        elif self.project is not None:
            start = str(self.project.workspace)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开历史标定结果",
            start,
            "Calibration Run / Workflow (*.yaml *.yml);;JSON (*.json);;All files (*)",
        )
        if not path:
            return
        source = Path(path).expanduser().resolve()
        run = self.load_calibration_run_path(source, silent=True)
        if run is None:
            # 验收报告仍可作为通用报告打开；Calibration Run/旧 workflow 文件
            # 加载失败时不回退到原始 dict，避免绕过统一结果入口。
            is_calibration_path = (
                source.stem.lower() == Path(DEFAULT_CALIBRATION_RUN_FILENAME).stem.lower()
                or source.stem.lower().startswith("workflow_")
            )
            if is_calibration_path:
                QMessageBox.critical(self, "历史标定结果读取失败", self.run_status.text())
                return
            try:
                report = load_document(source)
            except Exception as exc:
                QMessageBox.critical(self, "报告读取失败", str(exc))
                return
            self.current_run = None
            self.current_run_path = None
            self._update_result_overview(None)
            self.show_result(report)
            self.run_status.setText("当前显示历史验收报告；尚未加载 Calibration Run")
        html_path = Path(path).with_suffix(".html")
        self.last_html = html_path if html_path.is_file() else None
        self.open_html_button.setEnabled(self.last_html is not None)

    def _artifact_selected(self) -> None:
        items = self.artifacts.selectedItems()
        if not items:
            return
        path = Path(items[0].text())
        for index, record in enumerate(self._artifact_records):
            if record.display_path.resolve() == path.resolve():
                self._display_artifact(index)
                return


def discover_result_artifacts(result: dict[str, Any]) -> list[Path]:
    """兼容旧页面/插件的 Path 列表 API。"""

    records = discover_artifact_records(result)
    return sorted({record.display_path.resolve() for record in records}, key=str)


def load_residual_csv(path: Path, limit: int = 500) -> tuple[list[str], list[list[str]], list[float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        dictionaries = [row for _, row in zip(range(limit), reader)]
    rows = [[str(row.get(header, "")) for header in headers] for row in dictionaries]
    preferred = [header for header in headers if any(key in header.lower() for key in ("residual", "error", "distance", "rmse"))]
    for header in preferred + headers:
        values: list[float] = []
        try:
            values = [float(row[header]) for row in dictionaries if row.get(header, "") not in ("", None)]
        except ValueError:
            continue
        if len(values) >= 2:
            return headers, rows, values
    return headers, rows, []


_RESULT_STATUS_LABELS = {
    "completed": "已完成",
    "complete": "已完成",
    "pass": "通过",
    "passed": "通过",
    "warn": "警告",
    "warning": "警告",
    "fail": "失败",
    "failed": "失败",
    "error": "错误",
    "unknown": "未知",
    "not_executed": "未执行",
    "skipped": "已跳过",
    "unavailable": "不可用",
    "loaded": "已加载",
    "used": "已使用",
    "evaluated": "已评估",
    "recorded": "已记录",
    "valid": "有效",
    "invalid": "无效",
}


def _result_status_text(status: Any) -> str:
    if status is None:
        return "暂无"
    value = str(status).strip()
    if not value:
        return "暂无"
    return _RESULT_STATUS_LABELS.get(value.lower(), value)


def _stage_status_text(status: str) -> str:
    return _result_status_text(status)


def _overview_metric_text(
    status: str,
    value: Any,
    *,
    unit: str = "",
    percent: bool = False,
) -> str:
    if status == NOT_EXECUTED:
        return "未执行"
    if value is None:
        return "暂无"
    if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
        return "暂无"
    if percent and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.1%}"
    if isinstance(value, float):
        return f"{value:.4f}{unit}"
    return f"{value}{unit}"


def _overview_time_text(
    started_utc: datetime | None,
    completed_utc: datetime | None,
) -> str:
    def format_time(value: datetime | None) -> str | None:
        return value.isoformat(timespec="seconds") if value is not None else None

    started = format_time(started_utc)
    completed = format_time(completed_utc)
    if started and completed:
        return f"{started} ~ {completed}"
    return started or completed or "暂无"


def _matrix_value(
    matrix: tuple[tuple[float, ...], ...],
    row: int,
    column: int,
) -> float | None:
    if row >= len(matrix) or column >= len(matrix[row]):
        return None
    return matrix[row][column]


def _detail_value(value: Any) -> str:
    if value is None:
        return "暂无"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(float(value)):
            return "暂无"
        return f"{float(value):.6f}"
    return str(value)


def _add_tree(parent: QTreeWidgetItem, value: Any, name: str = "root") -> None:
    if isinstance(value, dict):
        item = QTreeWidgetItem([name, ""]); parent.addChild(item)
        for key, child in value.items():
            _add_tree(item, child, str(key))
    elif isinstance(value, list):
        item = QTreeWidgetItem([name, f"{len(value)} 项"]); parent.addChild(item)
        for index, child in enumerate(value):
            _add_tree(item, child, str(index + 1))
    else:
        parent.addChild(QTreeWidgetItem([name, "" if value is None else str(value)]))


def _path_row(line_edit: QLineEdit, parent: QWidget, *, directory: bool = False, file_filter: str = "All files (*)") -> QWidget:
    widget = QWidget(parent); layout = QHBoxLayout(widget); layout.setContentsMargins(0, 0, 0, 0)
    button = QPushButton("浏览…", widget); layout.addWidget(line_edit, 1); layout.addWidget(button)
    def browse() -> None:
        if directory:
            selected = QFileDialog.getExistingDirectory(parent, "选择目录", line_edit.text())
        else:
            selected, _ = QFileDialog.getOpenFileName(parent, "选择文件", line_edit.text(), file_filter)
        if selected:
            line_edit.setText(selected)
    button.clicked.connect(browse)
    return widget


def _labeled_path(
    label: str,
    line_edit: QLineEdit,
    parent: QWidget,
    file_filter: str,
    *,
    browse: bool = True,
) -> QWidget:
    widget = QWidget(parent); layout = QHBoxLayout(widget); layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QLabel(label))
    layout.addWidget(
        _path_row(line_edit, parent, file_filter=file_filter) if browse else line_edit,
        1,
    )
    return widget


def _quality_warning_text(code: str) -> str:
    return {
        "saturation_high": "过曝像素过多",
        "chessboard_saturation_high": "棋盘亮区过曝",
        "chessboard_near_saturation": "棋盘白格接近饱和",
        "laser_saturation_high": "激光饱和像素过多",
        "laser_peak_saturated": "激光峰值已饱和",
        "laser_peak_near_saturation": "激光峰值接近饱和",
        "laser_saturated_line_wide": "激光饱和线宽过大",
        "image_too_dark": "图像整体过暗",
        "dynamic_range_low": "动态范围不足",
        "laser_coverage_low": "激光方向覆盖不足",
        "chessboard_not_found": "未找到完整棋盘",
        "chessboard_pattern_mismatch": "棋盘内角点配置不匹配",
        "chessboard_obscured_by_laser": "激光线遮挡棋盘",
    }.get(code, code)


def _laser_quality_metrics_text(quality: dict[str, Any], thresholds: Any | None) -> str:
    coverage = quality.get("laser_coverage")
    if coverage is None:
        return ""
    parts = [f"覆盖 {coverage:.1%}"]
    saturated = quality.get("laser_peak_saturation_fraction")
    near = quality.get("laser_peak_near_saturation_fraction")
    saturated_width = quality.get("laser_saturated_width_p95_px")
    fwhm_p50 = quality.get("laser_fwhm_p50_px")
    fwhm_p95 = quality.get("laser_fwhm_p95_px")
    if saturated is not None:
        limit = getattr(thresholds, "max_laser_peak_saturation_fraction", None)
        suffix = f"≤{limit:.1%}" if isinstance(limit, (int, float)) else ""
        parts.append(f"峰饱 {saturated:.1%}{suffix}")
    if near is not None:
        limit = getattr(thresholds, "max_laser_peak_near_saturation_fraction", None)
        suffix = f"≤{limit:.1%}" if isinstance(limit, (int, float)) else ""
        parts.append(f"近饱 {near:.1%}{suffix}")
    if saturated_width is not None:
        limit = getattr(thresholds, "max_laser_saturated_width_px", None)
        suffix = f"≤{limit:.1f}px" if isinstance(limit, (int, float)) else ""
        parts.append(f"饱和宽P95 {saturated_width:.1f}px{suffix}")
    if fwhm_p50 is not None and fwhm_p95 is not None:
        parts.append(f"FWHM P50/P95 {fwhm_p50:.1f}/{fwhm_p95:.1f}px")
    return " · 激光" + "，".join(parts)


def _search_region_quality_text(quality: Mapping[str, Any]) -> str:
    health = quality.get("search_region_health")
    if not isinstance(health, Mapping):
        return "Search region: --"
    status = str(health.get("status", "WARNING"))
    p05 = health.get("boundary_clearance_p05_px")
    support = health.get("kernel_support_px")
    outside = health.get("outside_search_region_peak_fraction")
    p05_text = f"{float(p05):.1f} px" if isinstance(p05, (int, float)) else "--"
    support_text = f"{int(support)} px" if isinstance(support, (int, float)) else "--"
    outside_text = f"{float(outside):.1%}" if isinstance(outside, (int, float)) else "--"
    lines = [
        f"Search region: {status}",
        f"Boundary P05: {p05_text} · Kernel support: {support_text} · "
        f"Outside-band risk: {outside_text}",
    ]
    reasons = health.get("warning_reasons")
    if isinstance(reasons, (list, tuple)) and reasons:
        lines.append("Reasons: " + ", ".join(str(item) for item in reasons))
    error = health.get("error")
    if error:
        lines.append(f"Detail: {error}")
    return "\n".join(lines)


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
