"""Draw tracking overlays on a camera frame."""

from __future__ import annotations

import cv2
import numpy as np

from vision.connections import (
    BODY_CONNECTIONS,
    BODY_KEYPOINTS,
    FACE_KEY_INDICES,
    HAND_CONNECTIONS,
)
from vision.types import Box, FrameVision, Point

PERSON_COLOR = (149, 219, 92)
SKELETON_COLOR = (255, 196, 120)
JOINT_COLOR = (245, 248, 252)
FACE_COLOR = (90, 186, 230)
LEFT_HAND_COLOR = (250, 196, 90)
RIGHT_HAND_COLOR = (106, 138, 255)
HUD_BG = (22, 24, 28)
TEXT_COLOR = (236, 238, 241)

_FACE_MESH: list[tuple[int, int]] | None = None
_FACE_MESH_TRIED = False


def _face_mesh_connections() -> list[tuple[int, int]]:
    global _FACE_MESH, _FACE_MESH_TRIED
    if _FACE_MESH_TRIED:
        return _FACE_MESH or []
    _FACE_MESH_TRIED = True
    pairs: list[tuple[int, int]] = []
    try:
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
        for item in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
            start = getattr(item, "start", None)
            end = getattr(item, "end", None)
            if start is None or end is None:
                continue
            pairs.append((int(start), int(end)))
    except Exception:
        pairs = []
    _FACE_MESH = pairs
    return pairs


def _pt(point: Point, width: int, height: int) -> tuple[int, int]:
    return (
        int(max(0, min(width - 1, round(point.x)))),
        int(max(0, min(height - 1, round(point.y)))),
    )


def _thickness(width: int) -> int:
    return max(2, int(round(width / 640)))


def _font_scale(width: int) -> float:
    return max(0.45, width / 1700.0)


def _clamp_box(box: Box, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(width - 1, box.x))
    y1 = max(0, min(height - 1, box.y))
    x2 = max(0, min(width - 1, box.x + box.w))
    y2 = max(0, min(height - 1, box.y + box.h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def _badge(image: np.ndarray, text: str, origin_x: int, origin_y: int,
           fg: tuple[int, int, int], bg: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = _font_scale(image.shape[1]) * 0.95
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    height, width = image.shape[:2]
    x = max(0, min(origin_x, width - text_w - 12))
    y = max(text_h + 8, min(origin_y, height - baseline - 4))
    cv2.rectangle(image, (x, y - text_h - 6), (x + text_w + 10, y + baseline + 3), bg, -1)
    cv2.putText(image, text, (x + 5, y), font, scale, fg, thickness, cv2.LINE_AA)


def _draw_box(image: np.ndarray, box: Box, color: tuple[int, int, int], thickness: int) -> tuple[int, int, int, int] | None:
    rect = _clamp_box(box, image.shape[1], image.shape[0])
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return rect


def _draw_pose(image: np.ndarray, vision: FrameVision, options: dict, thickness: int) -> None:
    show_box = options.get("person_box", True)
    show_skeleton = options.get("skeleton", True)
    show_labels = options.get("labels", True)
    width, height = image.shape[1], image.shape[0]
    radius = max(3, thickness + 1)
    for index, pose in enumerate(vision.poses):
        if show_box and pose.box is not None:
            rect = _draw_box(image, pose.box, PERSON_COLOR, thickness)
            if rect and show_labels:
                x1, y1, _, y2 = rect
                percent = int(round(max(0.0, min(1.0, pose.confidence)) * 100))
                _badge(image, f"PERSON {percent}%", x1, y1 - 4, (18, 28, 16), PERSON_COLOR)
                if index == 0 and vision.movement not in {"", "—"}:
                    _badge(image, vision.movement, x1, y2 + 22, TEXT_COLOR, (28, 36, 32))
        if not show_skeleton:
            continue
        points = pose.points
        lower_body = {(23, 25), (25, 27), (27, 31), (24, 26), (26, 28), (28, 32)}
        for start, end in BODY_CONNECTIONS:
            if start >= len(points) or end >= len(points):
                continue
            a, b = points[start], points[end]
            minimum = 0.7 if (start, end) in lower_body else 0.55
            if a.visibility < minimum or b.visibility < minimum:
                continue
            cv2.line(image, _pt(a, width, height), _pt(b, width, height), SKELETON_COLOR, thickness, cv2.LINE_AA)
        if len(points) > 12 and points[0].visibility >= 0.55 and points[11].visibility >= 0.55 and points[12].visibility >= 0.55:
            neck = (int(round((points[11].x + points[12].x) / 2)), int(round((points[11].y + points[12].y) / 2)))
            cv2.line(image, _pt(points[0], width, height), neck, SKELETON_COLOR, thickness, cv2.LINE_AA)
        for key in BODY_KEYPOINTS:
            minimum = 0.7 if key in {25, 26, 27, 28, 31, 32} else 0.55
            if key >= len(points) or points[key].visibility < minimum:
                continue
            center = _pt(points[key], width, height)
            cv2.circle(image, center, radius, JOINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(image, center, radius, SKELETON_COLOR, 1, cv2.LINE_AA)
        if show_labels and index == 0 and vision.gesture in {"LEFT HAND UP", "RIGHT HAND UP", "BOTH HANDS UP"}:
            wrist_index = 15 if vision.gesture == "LEFT HAND UP" else 16
            if vision.gesture == "BOTH HANDS UP" and pose.box is not None:
                _badge(image, vision.gesture, pose.box.x, max(18, pose.box.y - 28), TEXT_COLOR, (48, 42, 24))
            elif wrist_index < len(points) and points[wrist_index].visibility >= 0.5:
                wrist = _pt(points[wrist_index], width, height)
                _badge(image, vision.gesture, wrist[0] + 8, wrist[1] - 8, TEXT_COLOR, (48, 42, 24))


def _draw_faces(image: np.ndarray, vision: FrameVision, options: dict, thickness: int) -> None:
    if not options.get("face", True):
        return
    width, height = image.shape[1], image.shape[0]
    mesh = _face_mesh_connections() if options.get("face_mesh", False) else []
    for face in vision.faces:
        if options.get("face_mesh", False) and face.points:
            if mesh:
                color = (150, 145, 120)
                for start, end in mesh:
                    if start >= len(face.points) or end >= len(face.points):
                        continue
                    cv2.line(image, _pt(face.points[start], width, height), _pt(face.points[end], width, height), color, 1, cv2.LINE_AA)
            else:
                for point in face.points[::2]:
                    cv2.circle(image, _pt(point, width, height), 1, (160, 170, 190), -1, cv2.LINE_AA)
        if face.box is not None:
            rect = _draw_box(image, face.box, FACE_COLOR, max(1, thickness - 1))
            if rect and options.get("labels", True):
                _badge(image, "FACE", rect[0], rect[1] - 4, (20, 28, 32), FACE_COLOR)
        if face.center is not None:
            cv2.circle(image, face.center, max(3, thickness), FACE_COLOR, -1, cv2.LINE_AA)
        if face.center is not None and face.arrow_end is not None:
            cv2.arrowedLine(image, face.center, face.arrow_end, FACE_COLOR, max(2, thickness - 1), cv2.LINE_AA, tipLength=0.28)
        for index in FACE_KEY_INDICES:
            if index < len(face.points):
                cv2.circle(image, _pt(face.points[index], width, height), max(2, thickness - 1), JOINT_COLOR, -1, cv2.LINE_AA)


def _draw_hands(image: np.ndarray, vision: FrameVision, options: dict, thickness: int) -> None:
    if not options.get("hands", True):
        return
    width, height = image.shape[1], image.shape[0]
    show_labels = options.get("labels", True)
    show_trails = options.get("motion_trails", False)
    for hand in vision.hands:
        color = LEFT_HAND_COLOR if hand.side == "Left" else RIGHT_HAND_COLOR
        if show_trails and len(hand.trail) >= 2:
            for index in range(1, len(hand.trail)):
                fade = index / len(hand.trail)
                faded = tuple(int(channel * (0.35 + 0.65 * fade)) for channel in color)
                cv2.line(image, hand.trail[index - 1], hand.trail[index], faded, max(1, thickness - 1), cv2.LINE_AA)
        if len(hand.points) >= 21:
            for start, end in HAND_CONNECTIONS:
                cv2.line(image, _pt(hand.points[start], width, height), _pt(hand.points[end], width, height), color, max(1, thickness - 1), cv2.LINE_AA)
            for point in hand.points:
                center = _pt(point, width, height)
                cv2.circle(image, center, max(2, thickness), JOINT_COLOR, -1, cv2.LINE_AA)
                cv2.circle(image, center, max(2, thickness), color, 1, cv2.LINE_AA)
        if hand.box is not None:
            rect = _draw_box(image, hand.box, color, max(1, thickness - 1))
            if rect and show_labels:
                title = "LEFT HAND" if hand.side == "Left" else "RIGHT HAND"
                motion = "MOVING" if hand.moving else "STATIC"
                _badge(image, f"{title} · {motion}", rect[0], rect[1] - 4, (20, 20, 24), color)


def _draw_hud(image: np.ndarray, lines: list[str]) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.48, image.shape[1] / 1800.0)
    thickness = 1
    line_h = max(20, int(22 * scale / 0.5))
    pad_x, pad_y = 12, 10
    widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    box_w = max(widths) + pad_x * 2 + 6
    box_h = line_h * len(lines) + pad_y
    height, width = image.shape[:2]
    x, y = 12, 12
    if x + box_w >= width or y + box_h >= height:
        return
    roi = image[y:y + box_h, x:x + box_w]
    overlay = np.full_like(roi, HUD_BG)
    image[y:y + box_h, x:x + box_w] = cv2.addWeighted(overlay, 0.78, roi, 0.22, 0)
    cv2.rectangle(image, (x, y), (x + 3, y + box_h - 1), PERSON_COLOR, -1)
    for index, line in enumerate(lines):
        color = TEXT_COLOR
        if line.startswith("FPS"):
            color = PERSON_COLOR
        elif line.startswith("STATE: MOVING"):
            color = (120, 196, 240)
        elif "NOT DETECTED" in line:
            color = (160, 166, 176)
        text_y = y + pad_y + 14 + index * line_h
        cv2.putText(image, line, (x + pad_x, text_y), font, scale, color, thickness, cv2.LINE_AA)


def _detected(flag: bool) -> str:
    return "DETECTED" if flag else "NOT DETECTED"


def draw_overlays(frame_bgr: np.ndarray, vision: FrameVision, options: dict, fps: float) -> np.ndarray:
    canvas = frame_bgr.copy()
    thickness = _thickness(canvas.shape[1])
    _draw_pose(canvas, vision, options, thickness)
    _draw_faces(canvas, vision, options, thickness)
    _draw_hands(canvas, vision, options, thickness)
    lines = [
        f"FPS: {fps:.0f}",
        f"PERSON: {_detected(vision.person_detected)}",
        f"FACE: {_detected(vision.face_detected)}",
        f"LEFT HAND: {_detected(vision.left_hand_detected)}",
        f"RIGHT HAND: {_detected(vision.right_hand_detected)}",
        f"STATE: {vision.movement}",
        f"GESTURE: {vision.gesture}",
    ]
    if options.get("debug", False):
        timings = vision.timings_ms
        lines.extend([
            f"POSE: {timings.get('pose', 0.0):.1f} ms",
            f"HANDS: {timings.get('hands', 0.0):.1f} ms",
            f"FACE: {timings.get('face', 0.0):.1f} ms",
            f"PROCESS: {timings.get('process', 0.0):.1f} ms",
            f"FRAME: {vision.frame_number}",
            f"CAMERA: {vision.frame_width}x{vision.frame_height}",
            f"SCALE: {int(options.get('scale_percent', 0))}%",
        ])
    _draw_hud(canvas, lines)
    return canvas
