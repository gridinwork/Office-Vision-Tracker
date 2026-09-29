"""Dark main window for Office Vision Tracker."""

from __future__ import annotations

import logging
from datetime import datetime

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QImage, QPalette, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.camera_worker import CameraWorker
from app.settings import (
    DEFAULT_PROCESSING_SCALE,
    DEFAULT_RESOLUTION,
    PROCESSING_SCALES,
    RESOLUTIONS,
    SCREENSHOTS_DIR,
)

logger = logging.getLogger("office_vision.ui")

STYLESHEET = """
QWidget {
    color: #e6e8eb;
    font-family: "Segoe UI";
    font-size: 13px;
}
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {
    background: #16181d;
}
QFrame#card, QFrame#videoFrame {
    background: #22262e;
    border: 1px solid #313743;
    border-radius: 12px;
}
QFrame#videoFrame {
    background: #0e1014;
}
QLabel#title {
    font-size: 22px;
    font-weight: 700;
    background: transparent;
}
QLabel#subtitle, QLabel#hint, QLabel#statusKey, QLabel#section {
    background: transparent;
}
QLabel#subtitle, QLabel#hint {
    color: #9aa3b2;
}
QLabel#section {
    font-size: 13px;
    font-weight: 700;
    color: #d5dbe3;
}
QLabel#statusKey {
    color: #9aa3b2;
}
QLabel#statusValue {
    font-weight: 600;
}
QPushButton {
    background: #2c323c;
    border: 1px solid #3c4452;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 650;
}
QPushButton:hover { background: #38404c; }
QPushButton:disabled {
    color: #6d7480;
    background: #1c2026;
    border-color: #2a3038;
}
QPushButton#start {
    background: #2f6fed;
    border: none;
    color: white;
}
QPushButton#start:hover { background: #4b82f0; }
QPushButton#start:disabled { background: #24324d; color: #8ea0bd; }
QPushButton#stop {
    background: #3a2428;
    border: 1px solid #6a3a42;
    color: #ffc1c1;
}
QPushButton#stop:hover { background: #4a2c32; }
QComboBox {
    background: #1a1e25;
    border: 1px solid #3c4452;
    border-radius: 8px;
    padding: 6px 10px;
    min-height: 20px;
}
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #22262e;
    color: #e6e8eb;
    selection-background-color: #2f6fed;
    border: 1px solid #3c4452;
}
QCheckBox { spacing: 8px; background: transparent; }
"""


def apply_dark_palette(app) -> None:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#16181d"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e6e8eb"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#1a1e25"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#22262e"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e6e8eb"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#2c323c"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e6e8eb"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2f6fed"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#8b939f"))
    app.setPalette(palette)


class VideoView(QWidget):
    """Camera image painted with the widget's aspect ratio, not the frame size."""

    def __init__(self) -> None:
        super().__init__()
        self._image: QImage | None = None
        self._scaled: QImage | None = None
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)

    def sizeHint(self) -> QSize:
        return QSize(960, 540)

    def set_image(self, image: QImage | None) -> None:
        self._image = image
        self._rescale()
        self.update()

    def resizeEvent(self, event) -> None:
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self) -> None:
        if self._image is None or self._image.isNull() or self.width() < 2 or self.height() < 2:
            self._scaled = None
            return
        self._scaled = self._image.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0e1014"))
        if self._scaled is None:
            painter.setPen(QColor("#8b939f"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Camera is stopped")
            return
        x = (self.width() - self._scaled.width()) // 2
        y = (self.height() - self._scaled.height()) // 2
        painter.drawImage(x, y, self._scaled)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Office Vision Tracker")
        self.resize(1280, 820)
        self.setMinimumSize(1100, 700)
        self._last_image: QImage | None = None
        self._last_dialog = ""
        self._camera_running = False
        self._has_camera = False
        self._values: dict[str, QLabel] = {}
        self._debug_widgets: list[QWidget] = []

        self.worker = CameraWorker(self)
        self.worker.cameras_ready.connect(self._on_cameras)
        self.worker.mediapipe_status.connect(self._on_mediapipe)
        self.worker.camera_status.connect(self._on_camera_status)
        self.worker.camera_running.connect(self._on_running)
        self.worker.error_message.connect(self._on_error)
        self.worker.frame_ready.connect(self._on_frame)

        self._build_ui()
        self.worker.start()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(12)
        header = QVBoxLayout()
        title = QLabel("Office Vision Tracker")
        title.setObjectName("title")
        subtitle = QLabel("Webcam presence, pose, face, and hand tracking")
        subtitle.setObjectName("subtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        left.addLayout(header)
        left.addWidget(self._build_toolbar())

        video_frame = QFrame()
        video_frame.setObjectName("videoFrame")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(8, 8, 8, 8)
        self.video = VideoView()
        video_layout.addWidget(self.video)
        left.addWidget(video_frame, stretch=1)
        outer.addLayout(left, stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(340)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(12)
        side_layout.addWidget(self._build_detection_card())
        side_layout.addWidget(self._build_status_card())
        side_layout.addStretch(1)
        scroll.setWidget(side)
        outer.addWidget(scroll)

        self.setStyleSheet(STYLESHEET)
        self._set_debug_visible(False)

    def _build_toolbar(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        controls.addLayout(self._labeled_combo("Camera", self._make_camera_combo()))
        controls.addLayout(self._labeled_combo("Resolution", self._make_resolution_combo()))
        controls.addLayout(self._labeled_combo("Processing Scale", self._make_scale_combo()))
        controls.addStretch(1)
        layout.addLayout(controls)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.start_btn = QPushButton("START CAMERA")
        self.start_btn.setObjectName("start")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_camera)
        self.stop_btn = QPushButton("STOP CAMERA")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_camera)
        self.shot_btn = QPushButton("Save Screenshot")
        self.shot_btn.clicked.connect(self._save_screenshot)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_cameras)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addWidget(self.shot_btn)
        buttons.addWidget(self.refresh_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.hint = QLabel("Detecting cameras...")
        self.hint.setObjectName("hint")
        layout.addWidget(self.hint)
        return card

    def _labeled_combo(self, caption: str, combo: QComboBox) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(4)
        label = QLabel(caption)
        label.setObjectName("statusKey")
        column.addWidget(label)
        column.addWidget(combo)
        return column

    def _make_camera_combo(self) -> QComboBox:
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Detecting cameras...", -1)
        self.camera_combo.currentIndexChanged.connect(self._push_options)
        return self.camera_combo

    def _make_resolution_combo(self) -> QComboBox:
        self.resolution_combo = QComboBox()
        for width, height in RESOLUTIONS:
            self.resolution_combo.addItem(f"{width}x{height}", (width, height))
        default_index = RESOLUTIONS.index(DEFAULT_RESOLUTION)
        self.resolution_combo.setCurrentIndex(default_index)
        self.resolution_combo.currentIndexChanged.connect(self._push_options)
        return self.resolution_combo

    def _make_scale_combo(self) -> QComboBox:
        self.scale_combo = QComboBox()
        self.scale_combo.setToolTip("MediaPipe runs on a smaller copy of the frame. The preview stays full size.")
        for scale in PROCESSING_SCALES:
            self.scale_combo.addItem(f"{scale}%", scale)
        default_index = PROCESSING_SCALES.index(DEFAULT_PROCESSING_SCALE)
        self.scale_combo.setCurrentIndex(default_index)
        self.scale_combo.currentIndexChanged.connect(self._push_options)
        return self.scale_combo

    def _build_detection_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        title = QLabel("Detection")
        title.setObjectName("section")
        layout.addWidget(title)
        self.person_box = self._check("Person Box", True)
        self.skeleton_box = self._check("Body Skeleton", True)
        self.face_box = self._check("Face", True)
        self.hands_box = self._check("Hands", True)
        self.hands_box.setToolTip("Left and right follow the person, the way MediaPipe reports them.")
        self.face_mesh_box = self._check("Face Mesh", False)
        self.trails_box = self._check("Motion Trails", False)
        self.labels_box = self._check("Labels", True)
        self.debug_box = self._check("Debug", False)
        for checkbox in (
            self.person_box,
            self.skeleton_box,
            self.face_box,
            self.hands_box,
            self.face_mesh_box,
            self.trails_box,
            self.labels_box,
            self.debug_box,
        ):
            checkbox.toggled.connect(self._push_options)
            layout.addWidget(checkbox)
        self.debug_box.toggled.connect(self._set_debug_visible)
        return card

    def _check(self, text: str, checked: bool) -> QCheckBox:
        checkbox = QCheckBox(text)
        checkbox.setChecked(checked)
        return checkbox

    def _build_status_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(7)
        title = QLabel("Status")
        title.setObjectName("section")
        layout.addWidget(title)
        rows = [
            ("Camera", "STOPPED"),
            ("MediaPipe", "LOADING"),
            ("FPS", "—"),
            ("Person", "NOT DETECTED"),
            ("People", "0"),
            ("Face", "NOT DETECTED"),
            ("Left hand", "NOT DETECTED"),
            ("Right hand", "NOT DETECTED"),
            ("Movement", "—"),
            ("Gesture", "—"),
        ]
        for key, value in rows:
            layout.addLayout(self._status_row(key, value))
        debug_rows = [
            ("Pose ms", "—"),
            ("Hands ms", "—"),
            ("Face ms", "—"),
            ("Process", "—"),
            ("Frame", "—"),
            ("Resolution", "—"),
        ]
        for key, value in debug_rows:
            layout.addLayout(self._status_row(key, value, track_debug=True))
        return card

    def _status_row(self, key: str, initial: str, track_debug: bool = False) -> QHBoxLayout:
        row = QHBoxLayout()
        name = QLabel(key)
        name.setObjectName("statusKey")
        value = QLabel(initial)
        value.setObjectName("statusValue")
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(name)
        row.addStretch(1)
        row.addWidget(value)
        self._values[key] = value
        self._paint_value(key, initial)
        if track_debug:
            self._debug_widgets.extend((name, value))
        return row

    def _set_debug_visible(self, visible: bool) -> None:
        for widget in self._debug_widgets:
            widget.setVisible(visible)

    def _paint_value(self, key: str, text: str) -> None:
        label = self._values[key]
        label.setText(text)
        upper = text.upper()
        color = "#e6e8eb"
        if text in {"ACTIVE", "DETECTED"}:
            color = "#5cdb95"
        elif upper.startswith("MOVING") or "HAND UP" in upper:
            color = "#f0c36a"
        elif text == "STANDING":
            color = "#8eb6ff"
        elif text in {"STOPPED", "NOT DETECTED", "—", "ARMS DOWN"}:
            color = "#aeb6c2"
        elif "ERROR" in upper:
            color = "#ff8d8d"
        elif text in {"LOADING", "STARTING"}:
            color = "#f0c36a"
        label.setStyleSheet(f"color: {color}; font-weight: 600; background: transparent;")

    def _set_value(self, key: str, text: str) -> None:
        if key in self._values:
            self._paint_value(key, text)

    def _selected_camera(self) -> int:
        data = self.camera_combo.currentData()
        if data is None:
            return -1
        return int(data)

    def _push_options(self) -> None:
        resolution = self.resolution_combo.currentData()
        scale = self.scale_combo.currentData()
        if resolution is None or scale is None:
            return
        width, height = resolution
        self.worker.update_options(
            camera_index=self._selected_camera(),
            width=int(width),
            height=int(height),
            scale=float(scale) / 100.0,
            scale_percent=int(scale),
            person_box=self.person_box.isChecked(),
            skeleton=self.skeleton_box.isChecked(),
            face=self.face_box.isChecked(),
            hands=self.hands_box.isChecked(),
            face_mesh=self.face_mesh_box.isChecked(),
            motion_trails=self.trails_box.isChecked(),
            labels=self.labels_box.isChecked(),
            debug=self.debug_box.isChecked(),
        )

    def _start_camera(self) -> None:
        if self._selected_camera() < 0:
            self._on_error("Camera cannot be opened.")
            return
        self._last_dialog = ""
        self._push_options()
        self._set_value("Camera", "STARTING")
        self.start_btn.setEnabled(False)
        self.hint.setText("Starting camera...")
        self.worker.start_camera()

    def _stop_camera(self) -> None:
        self.worker.stop_camera()
        self.hint.setText("Stopping camera...")

    def _refresh_cameras(self) -> None:
        if self._camera_running:
            return
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItem("Detecting cameras...", -1)
        self.camera_combo.blockSignals(False)
        self.start_btn.setEnabled(False)
        self._has_camera = False
        self.hint.setText("Detecting cameras...")
        self.worker.request_scan()

    def _on_cameras(self, cameras: list) -> None:
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        if not cameras:
            self.camera_combo.addItem("No camera found", -1)
            self._has_camera = False
            self.hint.setText("No camera found. Connect a webcam and press Refresh.")
        else:
            for index in cameras:
                self.camera_combo.addItem(f"Camera {index}", index)
            preferred = 0 if 0 in cameras else cameras[0]
            self.camera_combo.setCurrentIndex(list(cameras).index(preferred))
            self._has_camera = True
            names = ", ".join(f"Camera {index}" for index in cameras)
            self.hint.setText(f"Found {names}.")
        self.camera_combo.blockSignals(False)
        self.start_btn.setEnabled(self._has_camera and not self._camera_running)
        self._push_options()
        logger.info("Camera list updated: %s", cameras)

    def _on_mediapipe(self, status: str) -> None:
        self._set_value("MediaPipe", status)

    def _on_camera_status(self, status: str) -> None:
        self._set_value("Camera", status)
        if status == "ACTIVE":
            self.hint.setText("Camera is active.")
        elif status == "STOPPED":
            self.hint.setText("Camera is stopped.")
            self._set_value("FPS", "—")
        elif status == "ERROR":
            self._set_value("FPS", "—")

    def _on_running(self, running: bool) -> None:
        self._camera_running = running
        self.start_btn.setEnabled((not running) and self._has_camera)
        self.stop_btn.setEnabled(running)
        self.refresh_btn.setEnabled(not running)

    def _on_error(self, message: str) -> None:
        logger.error(message)
        self.hint.setText(message)
        if message == self._last_dialog:
            return
        self._last_dialog = message
        QMessageBox.warning(self, "Office Vision Tracker", message)

    def _on_frame(self, image: QImage, status: dict) -> None:
        try:
            self._last_image = image
            self.video.set_image(image)
            self._set_value("FPS", f"{status.get('fps', 0):.0f}")
            self._set_value("Person", status.get("person", "NOT DETECTED"))
            self._set_value("People", str(status.get("people", 0)))
            self._set_value("Face", status.get("face", "NOT DETECTED"))
            self._set_value("Left hand", status.get("left_hand", "NOT DETECTED"))
            self._set_value("Right hand", status.get("right_hand", "NOT DETECTED"))
            self._set_value("Movement", status.get("movement", "—"))
            self._set_value("Gesture", status.get("gesture", "—"))
            timings = status.get("timings_ms", {})
            self._set_value("Pose ms", f"{timings.get('pose', 0.0):.1f} ms")
            self._set_value("Hands ms", f"{timings.get('hands', 0.0):.1f} ms")
            self._set_value("Face ms", f"{timings.get('face', 0.0):.1f} ms")
            self._set_value("Process", f"{timings.get('process', 0.0):.1f} ms")
            self._set_value("Frame", str(status.get("frame_number", 0)))
            self._set_value("Resolution",f"{status.get('frame_width', 0)}x{status.get('frame_height', 0)}")
        finally:
            self.worker.release_ui()

    def _save_screenshot(self) -> None:
        if self._last_image is None or self._last_image.isNull():
            QMessageBox.information(self, "Office Vision Tracker", "No camera frame to save.")
            return
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SCREENSHOTS_DIR / f"office_tracker_{stamp}.jpg"
        saved = self._last_image.save(str(path), "JPG", 95)
        if not saved:
            logger.error("Could not save screenshot %s", path)
            QMessageBox.warning(self, "Office Vision Tracker", "Could not save the screenshot.")
            return
        logger.info("Screenshot saved: %s", path)
        self.hint.setText(f"Saved {path.name}")

    def closeEvent(self, event) -> None:
        self.worker.shutdown()
        self.worker.wait(8000)
        event.accept()
