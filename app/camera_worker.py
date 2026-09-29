"""Camera capture and vision processing off the GUI thread."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from app.settings import (
    CAMERA_PROBE_COUNT,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_PROCESSING_SCALE,
    DEFAULT_RESOLUTION,
    TARGET_CAMERA_FPS,
)
from vision.drawing import draw_overlays
from vision.mediapipe_engine import MediaPipeEngine
from vision.types import empty_vision

logger = logging.getLogger("office_vision.camera")

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


def scan_cameras(limit: int = CAMERA_PROBE_COUNT) -> list[int]:
    found: list[int] = []
    for index in range(limit):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        opened = capture.isOpened()
        capture.release()
        if opened:
            found.append(index)
            logger.info("Camera detected: index %s", index)
    if not found:
        capture = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if capture.isOpened():
            found.append(0)
            logger.info("Camera detected via Media Foundation: index 0")
        capture.release()
    if not found:
        logger.info("No camera detected")
    return found


def _configure(capture: cv2.VideoCapture, width: int, height: int) -> None:
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, TARGET_CAMERA_FPS)
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        logger.debug("Camera buffer size is not supported", exc_info=True)


def _read_frame(capture: cv2.VideoCapture, attempts: int = 10):
    for _ in range(attempts):
        ok, frame = capture.read()
        if ok and frame is not None and frame.size > 0:
            return frame
        time.sleep(0.03)
    return None


def open_capture(index: int, width: int, height: int) -> cv2.VideoCapture | None:
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture.release()
            continue
        _configure(capture, width, height)
        frame = _read_frame(capture)
        if frame is not None:
            logger.info(
                "Camera opened index=%s backend=%s frame=%sx%s",
                index,
                backend,
                frame.shape[1],
                frame.shape[0],
            )
            return capture
        capture.release()
    logger.error("Camera cannot be opened (index %s, %sx%s)", index, width, height)
    return None


class CameraWorker(QThread):
    """Owns the webcam and MediaPipe so the Qt event loop stays responsive.

    ``frame_ready`` is the extension point for a future video recorder.
    Subscribe to annotated frames here instead of reading the camera again.
    """

    cameras_ready = Signal(list)
    mediapipe_status = Signal(str)
    camera_status = Signal(str)
    camera_running = Signal(bool)
    error_message = Signal(str)
    frame_ready = Signal(QImage, dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._quit = threading.Event()
        self._run_camera = threading.Event()
        self._scan_request = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._ui_lock = threading.Lock()
        self._ui_busy = False
        self._options = {
            "camera_index": DEFAULT_CAMERA_INDEX,
            "width": DEFAULT_RESOLUTION[0],
            "height": DEFAULT_RESOLUTION[1],
            "scale": DEFAULT_PROCESSING_SCALE / 100.0,
            "scale_percent": DEFAULT_PROCESSING_SCALE,
            "person_box": True,
            "skeleton": True,
            "face": True,
            "hands": True,
            "face_mesh": False,
            "motion_trails": False,
            "labels": True,
            "debug": False,
        }
        self.engine: MediaPipeEngine | None = None

    def update_options(self, **kwargs) -> None:
        with self._lock:
            self._options.update(kwargs)

    def _snapshot(self) -> dict:
        with self._lock:
            return dict(self._options)

    def start_camera(self) -> None:
        logger.info(
            "Start camera requested index=%s %sx%s",
            self._snapshot()["camera_index"],
            self._snapshot()["width"],
            self._snapshot()["height"],
        )
        self._run_camera.set()
        self._wake.set()

    def stop_camera(self) -> None:
        logger.info("Stop camera requested")
        self._run_camera.clear()

    def request_scan(self) -> None:
        self._scan_request.set()
        self._wake.set()

    def shutdown(self) -> None:
        self._quit.set()
        self._wake.set()
        self._run_camera.set()

    def release_ui(self) -> None:
        with self._ui_lock:
            self._ui_busy = False

    def run(self) -> None:
        self._emit_cameras()
        self._load_engine()
        while not self._quit.is_set():
            if self._scan_request.is_set() and not self._run_camera.is_set():
                self._scan_request.clear()
                self._emit_cameras()
                continue
            if not self._run_camera.is_set():
                self._wake.wait(0.1)
                self._wake.clear()
                continue
            if self._quit.is_set():
                break
            self._camera_loop()
        if self.engine is not None:
            self.engine.close()
            self.engine = None

    def _emit_cameras(self) -> None:
        try:
            self.cameras_ready.emit(scan_cameras())
        except Exception:
            logger.exception("Camera scan failed")
            self.cameras_ready.emit([])

    def _load_engine(self) -> None:
        self.mediapipe_status.emit("LOADING")
        try:
            self.engine = MediaPipeEngine()
        except FileNotFoundError:
            logger.error("Model file not found")
            self.engine = None
            self.mediapipe_status.emit("ERROR")
            self.error_message.emit("Model file not found.")
            return
        except Exception:
            logger.exception("Unable to initialize MediaPipe model")
            self.engine = None
            self.mediapipe_status.emit("ERROR")
            self.error_message.emit("Unable to initialize MediaPipe model.")
            return
        self.mediapipe_status.emit("ACTIVE")

    def _camera_loop(self) -> None:
        options = self._snapshot()
        index = int(options["camera_index"])
        width = int(options["width"])
        height = int(options["height"])
        if index < 0:
            self._run_camera.clear()
            self.camera_running.emit(False)
            self.camera_status.emit("ERROR")
            self.error_message.emit("Camera cannot be opened.")
            return

        capture = open_capture(index, width, height)
        if capture is None:
            self._run_camera.clear()
            self.camera_running.emit(False)
            self.camera_status.emit("ERROR")
            self.error_message.emit("Camera cannot be opened.")
            return

        if self.engine is not None:
            self.engine.reset()
        self.camera_running.emit(True)
        self.camera_status.emit("ACTIVE")
        logger.info("Camera started")
        frame_number = 0
        marks: deque[float] = deque(maxlen=20)
        reopen = False
        try:
            while self._run_camera.is_set() and not self._quit.is_set():
                options = self._snapshot()
                if (
                    int(options["camera_index"]) != index
                    or int(options["width"]) != width
                    or int(options["height"]) != height
                ):
                    logger.info("Camera settings changed, reopening")
                    reopen = True
                    return
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    logger.error("Camera read failed")
                    self._run_camera.clear()
                    self.camera_status.emit("ERROR")
                    self.error_message.emit("Camera cannot be opened.")
                    return

                frame_number += 1
                started = time.perf_counter()
                try:
                    if self.engine is None:
                        vision = empty_vision(frame.shape[1], frame.shape[0])
                    else:
                        vision = self.engine.process(frame, float(options["scale"]))
                except Exception:
                    logger.exception("Vision processing error")
                    vision = empty_vision(frame.shape[1], frame.shape[0])
                finished = time.perf_counter()
                vision.frame_number = frame_number
                vision.timings_ms["process"] = (finished - started) * 1000.0
                marks.append(finished)
                fps = 0.0
                if len(marks) >= 2:
                    span = marks[-1] - marks[0]
                    if span > 0:
                        fps = (len(marks) - 1) / span
                annotated = draw_overlays(frame, vision, options, fps=fps)
                self._publish(annotated, self._status(vision, fps, options))
        finally:
            capture.release()
            if reopen:
                logger.info("Camera reopening")
            else:
                self.camera_running.emit(False)
                self.camera_status.emit("STOPPED")
                logger.info("Camera stopped")

    def _status(self, vision, fps: float, options: dict) -> dict:
        return {
            "fps": fps,
            "person": "DETECTED" if vision.person_detected else "NOT DETECTED",
            "people": int(vision.person_count),
            "face": "DETECTED" if vision.face_detected else "NOT DETECTED",
            "left_hand": "DETECTED" if vision.left_hand_detected else "NOT DETECTED",
            "right_hand": "DETECTED" if vision.right_hand_detected else "NOT DETECTED",
            "movement": vision.movement,
            "gesture": vision.gesture,
            "frame_width": vision.frame_width,
            "frame_height": vision.frame_height,
            "frame_number": vision.frame_number,
            "timings_ms": dict(vision.timings_ms),
            "scale_percent": int(options.get("scale_percent", DEFAULT_PROCESSING_SCALE)),
        }

    def _publish(self, frame_bgr: np.ndarray, status: dict) -> None:
        if self._quit.is_set():
            return
        with self._ui_lock:
            if self._ui_busy:
                return
            self._ui_busy = True
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self.frame_ready.emit(image, status)
