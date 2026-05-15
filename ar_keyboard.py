from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

try:
    from pynput.keyboard import Controller, Key
except ImportError:
    Controller = None
    Key = None

try:
    from .hand_features import HAND_CONNECTIONS, create_hand_detector, detect_hands
except ImportError:
    from hand_features import HAND_CONNECTIONS, create_hand_detector, detect_hands


WINDOW_NAME = "AR Keyboard"
_HAS_DISPLAY: bool | None = None


@dataclass
class HandObservation:
    label: str
    score: float
    points: np.ndarray
    palm_center: np.ndarray
    index_tip: np.ndarray
    thumb_tip: np.ndarray
    pinch_center: np.ndarray
    pinch_distance: float
    palm_size: float
    is_pinching: bool
    is_fist: bool


@dataclass
class KeyButton:
    label: str
    output: str
    rect: tuple[int, int, int, int]


@dataclass
class KeyboardState:
    center: np.ndarray
    scale: float
    last_resize_distance: float | None = None
    last_resize_scale: float = 1.0
    last_move_center: np.ndarray | None = None
    typed_text: str = ""
    last_typed_at: float = 0.0
    pressed_hands: set[str] | None = None


class KeyboardEmitter:
    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.controller = None if dry_run or Controller is None else Controller()

    def press(self, output: str) -> None:
        if self.dry_run:
            return
        if self.controller is None:
            raise RuntimeError("pynput is not available. Install it with: pip install pynput")

        if output == "SPACE":
            self.controller.press(Key.space)
            self.controller.release(Key.space)
        elif output == "BACK":
            self.controller.press(Key.backspace)
            self.controller.release(Key.backspace)
        elif output == "ENTER":
            self.controller.press(Key.enter)
            self.controller.release(Key.enter)
        elif output == "TAB":
            self.controller.press(Key.tab)
            self.controller.release(Key.tab)
        else:
            self.controller.type(output)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Type with a virtual AR keyboard controlled by hand gestures.")
    parser.add_argument("--source", default="0", help="Webcam index, video path, or RTSP URL.")
    parser.add_argument("--hand-model", type=Path, default=here / "models" / "hand_landmarker.task")
    parser.add_argument("--hand-confidence", type=float, default=0.5)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--min-scale", type=float, default=0.65)
    parser.add_argument("--max-scale", type=float, default=1.75)
    parser.add_argument("--pinch-ratio", type=float, default=0.55)
    parser.add_argument("--fist-strictness", type=float, default=0.68)
    parser.add_argument("--debounce", type=float, default=0.28)
    parser.add_argument("--dry-run", action="store_true", help="Show pressed keys without sending physical keyboard input.")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror webcam/video frames.")
    parser.add_argument("--hide-hands", action="store_true", help="Hide hand landmark overlay.")
    return parser.parse_args()


def source_value(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def has_display() -> bool:
    global _HAS_DISPLAY
    if _HAS_DISPLAY is not None:
        return _HAS_DISPLAY
    try:
        cv2.namedWindow("__display_check__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__display_check__")
    except cv2.error:
        _HAS_DISPLAY = False
    else:
        _HAS_DISPLAY = True
    return _HAS_DISPLAY


def close_windows() -> None:
    if not has_display():
        return
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def hand_observations(
    hand_result,
    frame_shape: tuple[int, ...],
    pinch_ratio: float,
    fist_strictness: float,
) -> list[HandObservation]:
    if hand_result is None or not hand_result.hand_landmarks:
        return []

    height, width = frame_shape[:2]
    observations: list[HandObservation] = []
    handedness = hand_result.handedness or []
    for hand_index, landmarks in enumerate(hand_result.hand_landmarks):
        points = np.array(
            [[float(landmark.x) * width, float(landmark.y) * height] for landmark in landmarks],
            dtype=np.float32,
        )
        label = ""
        score = 1.0
        if hand_index < len(handedness) and handedness[hand_index]:
            label = str(handedness[hand_index][0].category_name)
            score = float(handedness[hand_index][0].score)

        palm_center = points[[0, 5, 9, 13, 17]].mean(axis=0)
        thumb_tip = points[4]
        index_tip = points[8]
        pinch_center = (thumb_tip + index_tip) * 0.5
        palm_size = max(28.0, float(np.linalg.norm(points[5] - points[17])), float(np.linalg.norm(points[0] - points[9])))
        pinch_distance = float(np.linalg.norm(thumb_tip - index_tip))
        observations.append(
            HandObservation(
                label=label,
                score=score,
                points=points,
                palm_center=palm_center,
                index_tip=index_tip,
                thumb_tip=thumb_tip,
                pinch_center=pinch_center,
                pinch_distance=pinch_distance,
                palm_size=palm_size,
                is_pinching=pinch_distance <= max(18.0, palm_size * pinch_ratio),
                is_fist=is_fist(points, fist_strictness),
            )
        )
    return observations


def is_fist(points: np.ndarray, strictness: float) -> bool:
    palm_center = points[[0, 5, 9, 13, 17]].mean(axis=0)
    palm_size = max(
        28.0,
        float(np.linalg.norm(points[5] - points[17])),
        float(np.linalg.norm(points[0] - points[9])),
    )
    curled = 0
    for tip_index in (8, 12, 16, 20):
        tip_distance = float(np.linalg.norm(points[tip_index] - palm_center))
        pip_distance = float(np.linalg.norm(points[tip_index - 2] - palm_center))
        mcp_distance = float(np.linalg.norm(points[tip_index - 3] - palm_center))
        if tip_distance < palm_size * strictness and tip_distance <= pip_distance * 0.98 and tip_distance <= mcp_distance * 1.22:
            curled += 1
    thumb_tucked = float(np.linalg.norm(points[4] - palm_center)) < palm_size * (strictness + 0.12)
    return curled >= 4 and thumb_tucked


def keyboard_buttons(state: KeyboardState) -> list[KeyButton]:
    unit = 54.0 * state.scale
    gap = 8.0 * state.scale
    height = 48.0 * state.scale
    rows = [
        [("Q", "q", 1.0), ("W", "w", 1.0), ("E", "e", 1.0), ("R", "r", 1.0), ("T", "t", 1.0), ("Y", "y", 1.0), ("U", "u", 1.0), ("I", "i", 1.0), ("O", "o", 1.0), ("P", "p", 1.0)],
        [("A", "a", 1.0), ("S", "s", 1.0), ("D", "d", 1.0), ("F", "f", 1.0), ("G", "g", 1.0), ("H", "h", 1.0), ("J", "j", 1.0), ("K", "k", 1.0), ("L", "l", 1.0)],
        [("Z", "z", 1.0), ("X", "x", 1.0), ("C", "c", 1.0), ("V", "v", 1.0), ("B", "b", 1.0), ("N", "n", 1.0), ("M", "m", 1.0), (".", ".", 1.0), (",", ",", 1.0)],
        [("SPACE", "SPACE", 4.0), ("BACK", "BACK", 1.6), ("ENTER", "ENTER", 1.8)],
    ]

    total_height = len(rows) * height + (len(rows) - 1) * gap
    top = state.center[1] - total_height * 0.5
    buttons: list[KeyButton] = []
    for row_index, row in enumerate(rows):
        row_width = sum(width_units * unit for _, _, width_units in row) + (len(row) - 1) * gap
        left = state.center[0] - row_width * 0.5
        y1 = top + row_index * (height + gap)
        for label, output, width_units in row:
            key_width = width_units * unit
            x1 = left
            x2 = left + key_width
            y2 = y1 + height
            buttons.append(KeyButton(label=label, output=output, rect=(int(x1), int(y1), int(x2), int(y2))))
            left = x2 + gap
    return buttons


def update_keyboard_transform(state: KeyboardState, hands: list[HandObservation], args: argparse.Namespace) -> str:
    fists = [hand for hand in hands if hand.is_fist]
    pinches = [hand for hand in hands if hand.is_pinching]

    if len(fists) >= 2:
        move_center = (fists[0].palm_center + fists[1].palm_center) * 0.5
        if state.last_move_center is not None:
            state.center += move_center - state.last_move_center
        state.last_move_center = move_center
        state.last_resize_distance = None
        return "move"

    state.last_move_center = None
    if len(pinches) >= 2:
        distance = float(np.linalg.norm(pinches[0].pinch_center - pinches[1].pinch_center))
        if state.last_resize_distance is None:
            state.last_resize_distance = max(1.0, distance)
            state.last_resize_scale = state.scale
        else:
            ratio = distance / max(1.0, state.last_resize_distance)
            state.scale = float(np.clip(state.last_resize_scale * ratio, args.min_scale, args.max_scale))
        return "resize"

    state.last_resize_distance = None
    return "type"


def key_at_point(buttons: list[KeyButton], point: np.ndarray) -> KeyButton | None:
    x, y = float(point[0]), float(point[1])
    for button in buttons:
        x1, y1, x2, y2 = button.rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            return button
    return None


def update_typing(
    state: KeyboardState,
    hands: list[HandObservation],
    buttons: list[KeyButton],
    emitter: KeyboardEmitter,
    now: float,
    debounce: float,
) -> KeyButton | None:
    active_hands = state.pressed_hands or set()
    current_pressed: set[str] = set()
    typed_button = None
    for index, hand in enumerate(hands):
        hand_id = hand.label or str(index)
        if hand.is_pinching:
            current_pressed.add(hand_id)
            if hand_id not in active_hands and now - state.last_typed_at >= debounce:
                button = key_at_point(buttons, hand.index_tip)
                if button is not None:
                    emitter.press(button.output)
                    append_typed_text(state, button.output)
                    state.last_typed_at = now
                    typed_button = button
        active_hands.discard(hand_id)
    state.pressed_hands = current_pressed
    return typed_button


def append_typed_text(state: KeyboardState, output: str) -> None:
    if output == "SPACE":
        state.typed_text += " "
    elif output == "BACK":
        state.typed_text = state.typed_text[:-1]
    elif output == "ENTER":
        state.typed_text += "\n"
    elif output == "TAB":
        state.typed_text += "\t"
    else:
        state.typed_text += output
    state.typed_text = state.typed_text[-80:]


def draw_keyboard(display: np.ndarray, buttons: list[KeyButton], active: KeyButton | None, mode: str) -> None:
    overlay = display.copy()
    for button in buttons:
        x1, y1, x2, y2 = button.rect
        fill = (55, 65, 80)
        if button is active:
            fill = (40, 180, 95)
        elif mode == "move":
            fill = (90, 90, 45)
        elif mode == "resize":
            fill = (80, 65, 110)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (235, 240, 245), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.62, display, 0.38, 0, dst=display)

    for button in buttons:
        x1, y1, x2, y2 = button.rect
        font_scale = 0.55 if len(button.label) > 1 else 0.8
        thickness = 2
        text_size, baseline = cv2.getTextSize(button.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = x1 + max(4, (x2 - x1 - text_size[0]) // 2)
        text_y = y1 + max(text_size[1] + 4, (y2 - y1 + text_size[1]) // 2)
        cv2.putText(display, button.label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_hands(display: np.ndarray, hands: list[HandObservation]) -> None:
    for hand in hands:
        color = (0, 255, 80) if hand.is_pinching else (255, 80, 180)
        if hand.is_fist:
            color = (0, 80, 255)
        for start, end in HAND_CONNECTIONS:
            cv2.line(display, tuple(hand.points[start].astype(int)), tuple(hand.points[end].astype(int)), color, 2, cv2.LINE_AA)
        cv2.circle(display, tuple(hand.index_tip.astype(int)), 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(display, tuple(hand.thumb_tip.astype(int)), 6, (80, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(display, tuple(hand.pinch_center.astype(int)), 10, color, 2, cv2.LINE_AA)


def draw_status(display: np.ndarray, state: KeyboardState, mode: str, dry_run: bool) -> None:
    status = f"mode {mode}   scale {state.scale:.2f}"
    if dry_run:
        status += "   dry-run"
    cv2.putText(display, status, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(display, status, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 230, 255), 2, cv2.LINE_AA)

    preview = state.typed_text.replace("\n", "\\n")[-48:]
    if preview:
        cv2.putText(display, preview, (18, display.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(display, preview, (18, display.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 120), 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    if not has_display():
        raise SystemExit("OpenCV display support is not available in this Python environment.")

    emitter = KeyboardEmitter(dry_run=args.dry_run)
    cap = cv2.VideoCapture(source_value(args.source))
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    detector = create_hand_detector(args.hand_model, args.max_hands, args.hand_confidence)
    keyboard: KeyboardState | None = None
    previous_time = time.monotonic()
    start_time = previous_time
    last_timestamp_ms = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()
            previous_time = now
            timestamp_ms = int((now - start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            if keyboard is None:
                height, width = frame.shape[:2]
                keyboard = KeyboardState(
                    center=np.array([width * 0.5, height * 0.62], dtype=np.float32),
                    scale=float(np.clip(args.scale, args.min_scale, args.max_scale)),
                    pressed_hands=set(),
                )

            hand_result = detect_hands(detector, frame, timestamp_ms)
            hands = hand_observations(
                hand_result=hand_result,
                frame_shape=frame.shape,
                pinch_ratio=args.pinch_ratio,
                fist_strictness=args.fist_strictness,
            )
            mode = update_keyboard_transform(keyboard, hands, args)
            buttons = keyboard_buttons(keyboard)
            typed_button = None
            if mode == "type":
                typed_button = update_typing(keyboard, hands, buttons, emitter, now, args.debounce)
            else:
                keyboard.pressed_hands = set()

            display = frame.copy()
            draw_keyboard(display, buttons, typed_button, mode)
            if not args.hide_hands:
                draw_hands(display, hands)
            draw_status(display, keyboard, mode, args.dry_run)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break
            if key == ord("r"):
                keyboard = None
    finally:
        detector.close()
        cap.release()
        close_windows()


if __name__ == "__main__":
    main()
