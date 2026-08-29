"""批量采集配方表格控件。"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..camera.plan_builder import CaptureRecipeItem


class CaptureRecipeTable(QWidget):
    """默认 chess/laser/nolaser 三联图，字段变化通过 ``changed`` 通知页面。"""

    changed = Signal()
    HEADERS = ("启用", "角色", "文件前缀", "曝光 μs", "激光提示", "质量模式", "帧数", "稳定帧")
    _DEFAULTS = (
        {
            "role": "chess",
            "prefix": "chess",
            "exposure": 35000.0,
            "laser": "off",
            "quality": "chessboard",
            "frames": 1,
            "settle": 5,
        },
        {
            "role": "laser",
            "prefix": "laser",
            "exposure": 500.0,
            "laser": "on",
            "quality": "laser",
            "frames": 1,
            "settle": 5,
        },
        {
            "role": "nolaser",
            "prefix": "nolaser",
            "exposure": 500.0,
            "laser": "off",
            "quality": "generic",
            "frames": 1,
            "settle": 5,
        },
    )
    _INSTRUCTIONS = {
        "chess": "关闭激光，摆放棋盘；准备好后采集姿态 {pose_id}。",
        "nolaser": "保持棋盘完全不动、关闭激光；准备好后采集姿态 {pose_id}。",
        "laser": "保持棋盘完全不动、只打开激光；准备好后采集姿态 {pose_id}。",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._building = False
        self.table = QTableWidget(0, len(self.HEADERS), self)
        self.table.setHorizontalHeaderLabels(list(self.HEADERS))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.EditKeyPressed)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(150)
        reset_button = QPushButton("恢复三联图默认值")
        reset_button.clicked.connect(self.reset_defaults)
        hint = QLabel("激光状态仅作为人工提示，不会自动控制激光器。")
        hint.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        buttons = QHBoxLayout(); buttons.addWidget(reset_button); buttons.addStretch()
        layout.addLayout(buttons); layout.addWidget(hint)
        self.table.itemChanged.connect(self._changed)
        self.reset_defaults()

    def reset_defaults(self) -> None:
        self._building = True
        try:
            self.table.setRowCount(0)
            for row, values in enumerate(self._DEFAULTS):
                self.table.insertRow(row)
                enabled = QCheckBox(); enabled.setChecked(True); enabled.stateChanged.connect(self._changed)
                self.table.setCellWidget(row, 0, enabled)
                role_item = QTableWidgetItem(values["role"])
                role_item.setFlags(role_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, 1, role_item)
                prefix = QTableWidgetItem(values["prefix"])
                self.table.setItem(row, 2, prefix)
                exposure = QDoubleSpinBox(); exposure.setRange(0.1, 10_000_000); exposure.setDecimals(1)
                exposure.setValue(values["exposure"]); exposure.valueChanged.connect(self._changed)
                self.table.setCellWidget(row, 3, exposure)
                laser = self._combo(
                    (("开启", "on"), ("关闭", "off"), ("保持不变", "unchanged")), values["laser"]
                )
                quality = self._combo(
                    (("棋盘格", "chessboard"), ("通用", "generic"), ("激光线", "laser")), values["quality"]
                )
                self.table.setCellWidget(row, 4, laser); self.table.setCellWidget(row, 5, quality)
                frames = self._spin(1, 1000, values["frames"])
                settle = self._spin(0, 1000, values["settle"])
                self.table.setCellWidget(row, 6, frames); self.table.setCellWidget(row, 7, settle)
        finally:
            self._building = False
        self.changed.emit()

    def _combo(self, values: tuple[tuple[str, str], ...], current: str) -> QComboBox:
        combo = QComboBox()
        for label, data in values:
            combo.addItem(label, data)
        combo.setCurrentIndex(max(0, combo.findData(current)))
        combo.currentIndexChanged.connect(self._changed)
        return combo

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox(); spin.setRange(minimum, maximum); spin.setValue(value)
        spin.valueChanged.connect(self._changed)
        return spin

    def _changed(self, *_args: Any) -> None:
        if not self._building:
            self.changed.emit()

    def recipe_items(self, image_format: str) -> tuple[CaptureRecipeItem, ...]:
        result: list[CaptureRecipeItem] = []
        for row in range(self.table.rowCount()):
            enabled = self.table.cellWidget(row, 0)
            role_item = self.table.item(row, 1)
            prefix_item = self.table.item(row, 2)
            exposure = self.table.cellWidget(row, 3)
            laser = self.table.cellWidget(row, 4)
            quality = self.table.cellWidget(row, 5)
            frames = self.table.cellWidget(row, 6)
            settle = self.table.cellWidget(row, 7)
            assert isinstance(enabled, QCheckBox)
            assert isinstance(role_item, QTableWidgetItem)
            assert isinstance(prefix_item, QTableWidgetItem)
            assert isinstance(exposure, QDoubleSpinBox)
            assert isinstance(laser, QComboBox)
            assert isinstance(quality, QComboBox)
            assert isinstance(frames, QSpinBox)
            assert isinstance(settle, QSpinBox)
            role = role_item.text().strip()
            result.append(
                CaptureRecipeItem(
                    enabled=enabled.isChecked(),
                    role=role,
                    filename_prefix=prefix_item.text().strip(),
                    exposure_us=exposure.value(),
                    laser_state=str(laser.currentData()),
                    quality_mode=str(quality.currentData()),
                    frames=frames.value(),
                    settle_frames=settle.value(),
                    image_format=image_format,
                    instruction_template=self._INSTRUCTIONS.get(role, "准备好后采集姿态 {pose_id}。"),
                )
            )
        return tuple(result)


__all__ = ["CaptureRecipeTable"]
