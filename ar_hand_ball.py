from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

try:
    from .hand_features import HAND_CONNECTIONS, create_hand_detector, detect_hands
except ImportError:
    from hand_features import HAND_CONNECTIONS, create_hand_detector, detect_hands


WINDOW_NAME = "Hand AR Ball"
_HAS_DISPLAY: bool | None = None


@dataclass
class HandObservation:
    label: str
    score: float
    points: np.ndarray
    palm_center: np.ndarray
    pinch_center: np.ndarray
    pinch_distance: float
    palm_size: float
    is_pinching: bool
    hand_angle: float


@dataclass
class BallState:
    center: np.ndarray
    velocity: np.ndarray
    radius: float
    target_radius: float
    angle: float = 0.0
    angular_velocity: float = 0.0
    held: bool = False
    held_by: str = ""
    catch_cooldown: float = 0.0


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Play with a digital AR ball using MediaPipe hand tracking.")
    parser.add_argument("--source", default="0", help="Webcam index, video path, or RTSP URL.")
    parser.add_argument("--hand-model", type=Path, default=here / "models" / "hand_landmarker.task")
    parser.add_argument("--hand-confidence", type=float, default=0.5)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--min-radius", type=float, default=28.0)
    parser.add_argument("--max-radius", type=float, default=150.0)
    parser.add_argument("--ball-scale", type=float, default=0.28)
    parser.add_argument("--gravity", type=float, default=1500.0)
    parser.add_argument("--toss-velocity", type=float, default=850.0)
    parser.add_argument("--pinch-ratio", type=float, default=0.55)
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


def hand_observations(hand_result, frame_shape: tuple[int, ...]) -> list[HandObservation]:
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

        palm_indices = [0, 5, 9, 13, 17]
        palm_center = points[palm_indices].mean(axis=0)
        pinch_center = (points[4] + points[8]) * 0.5
        pinch_distance = float(np.linalg.norm(points[4] - points[8]))
        palm_size = float(np.linalg.norm(points[5] - points[17]))
        is_pinching = palm_size > 1.0 and pinch_distance <= palm_size * 0.55
        wrist_to_middle = points[9] - points[0]
        hand_angle = math.atan2(float(wrist_to_middle[1]), float(wrist_to_middle[0]))
        observations.append(
            HandObservation(
                label=label,
                score=score,
                points=points,
                palm_center=palm_center,
                pinch_center=pinch_center,
                pinch_distance=pinch_distance,
                palm_size=palm_size,
                is_pinching=is_pinching,
                hand_angle=hand_angle,
            )
        )
    return observations


def choose_control_hands(hands: list[HandObservation]) -> list[HandObservation]:
    if len(hands) <= 2:
        return hands

    by_label: dict[str, HandObservation] = {}
    for hand in sorted(hands, key=lambda item: item.score, reverse=True):
        if hand.label in {"Left", "Right"} and hand.label not in by_label:
            by_label[hand.label] = hand
    chosen = [by_label[label] for label in ("Left", "Right") if label in by_label]
    if len(chosen) == 2:
        return chosen
    return sorted(hands, key=lambda item: item.score, reverse=True)[:2]


def angle_difference(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def update_ball(
    ball: BallState,
    hands: list[HandObservation],
    frame_shape: tuple[int, ...],
    dt: float,
    args: argparse.Namespace,
) -> None:
    height, width = frame_shape[:2]
    ball.catch_cooldown = max(0.0, ball.catch_cooldown - dt)
    controls = choose_control_hands(hands)
    can_catch = ball.catch_cooldown <= 0.0

    one_hand_control = one_hand_grip_control(ball, controls, args) if controls else None
    if one_hand_control is not None and can_catch:
        hand = one_hand_control
        previous_center = ball.center.copy()
        previous_angle = ball.angle
        ball.held = True
        ball.held_by = hand_key(hand)
        ball.center = 0.62 * ball.center + 0.38 * hand.pinch_center
        ball.velocity = 0.55 * ball.velocity + 0.45 * ((ball.center - previous_center) / max(dt, 1e-3))
        angle_delta = angle_difference(hand.hand_angle, previous_angle)
        ball.angular_velocity = 0.6 * ball.angular_velocity + 0.4 * (angle_delta / max(dt, 1e-3))
        ball.angle = previous_angle + angle_delta * min(1.0, 14.0 * dt)

        if not hand.is_pinching or ball.velocity[1] < -args.toss_velocity:
            ball.held = False
            ball.held_by = ""
            ball.catch_cooldown = 0.35
    elif controls and can_catch:
        control_center, target_radius, target_angle = control_from_hands(controls, ball, args)
        catch_distance = np.linalg.norm(control_center - ball.center)
        catch_radius = max(ball.radius * 2.4, 120.0)
        if ball.held or catch_distance <= catch_radius:
            previous_center = ball.center.copy()
            previous_angle = ball.angle
            ball.held = True
            ball.center = 0.72 * ball.center + 0.28 * control_center
            ball.velocity = 0.65 * ball.velocity + 0.35 * ((ball.center - previous_center) / max(dt, 1e-3))
            ball.target_radius = target_radius
            ball.radius += (ball.target_radius - ball.radius) * min(1.0, 10.0 * dt)
            angle_delta = angle_difference(target_angle, previous_angle)
            ball.angular_velocity = 0.65 * ball.angular_velocity + 0.35 * (angle_delta / max(dt, 1e-3))
            ball.angle = previous_angle + angle_delta * min(1.0, 12.0 * dt)
            ball.held_by = "hands"

            if ball.velocity[1] < -args.toss_velocity:
                ball.held = False
                ball.held_by = ""
                ball.catch_cooldown = 0.45
        else:
            ball.held = False
            ball.held_by = ""
    else:
        ball.held = False
        ball.held_by = ""

    if not ball.held:
        ball.velocity[1] += args.gravity * dt
        ball.velocity *= max(0.0, 1.0 - 0.08 * dt)
        ball.center += ball.velocity * dt
        ball.angle += ball.angular_velocity * dt
        ball.angular_velocity *= max(0.0, 1.0 - 0.18 * dt)
        bounce_ball(ball, width, height)


def one_hand_grip_control(ball: BallState, controls: list[HandObservation], args: argparse.Namespace) -> HandObservation | None:
    if len(controls) != 1:
        return None

    hand = controls[0]
    pinch_limit = max(18.0, hand.palm_size * args.pinch_ratio)
    is_pinching = hand.pinch_distance <= pinch_limit
    hand.is_pinching = is_pinching

    if ball.held and ball.held_by == hand_key(hand):
        return hand
    if not is_pinching:
        return None

    pinch_to_ball = float(np.linalg.norm(hand.pinch_center - ball.center))
    palm_to_ball = float(np.linalg.norm(hand.palm_center - ball.center))
    pickup_radius = max(ball.radius * 1.25, 55.0)
    if pinch_to_ball <= pickup_radius or palm_to_ball <= pickup_radius:
        return hand
    return None


def hand_key(hand: HandObservation) -> str:
    return hand.label or "hand"


def control_from_hands(
    controls: list[HandObservation],
    ball: BallState,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, float]:
    if len(controls) >= 2:
        first, second = controls[0], controls[1]
        if first.label == "Right" and second.label == "Left":
            first, second = second, first
        center = (first.palm_center + second.palm_center) * 0.5
        hand_vector = second.palm_center - first.palm_center
        hand_distance = float(np.linalg.norm(hand_vector))
        target_radius = float(np.clip(hand_distance * args.ball_scale, args.min_radius, args.max_radius))
        target_angle = math.atan2(float(hand_vector[1]), float(hand_vector[0]))
        return center, target_radius, target_angle

    hand = controls[0]
    return hand.palm_center, ball.target_radius, hand.hand_angle


def bounce_ball(ball: BallState, width: int, height: int) -> None:
    left = ball.radius
    right = width - ball.radius
    top = ball.radius
    bottom = height - ball.radius

    if ball.center[0] < left:
        ball.center[0] = left
        ball.velocity[0] = abs(ball.velocity[0]) * 0.78
        ball.angular_velocity += ball.velocity[1] / max(ball.radius, 1.0) * 0.12
    elif ball.center[0] > right:
        ball.center[0] = right
        ball.velocity[0] = -abs(ball.velocity[0]) * 0.78
        ball.angular_velocity += ball.velocity[1] / max(ball.radius, 1.0) * 0.12

    if ball.center[1] < top:
        ball.center[1] = top
        ball.velocity[1] = abs(ball.velocity[1]) * 0.65
    elif ball.center[1] > bottom:
        ball.center[1] = bottom
        ball.velocity[1] = -abs(ball.velocity[1]) * 0.72
        ball.velocity[0] *= 0.92
        if abs(ball.velocity[1]) < 70.0:
            ball.velocity[1] = 0.0


def draw_hands(display: np.ndarray, hands: list[HandObservation]) -> None:
    for hand in hands:
        for start, end in HAND_CONNECTIONS:
            start_point = tuple(hand.points[start].astype(int))
            end_point = tuple(hand.points[end].astype(int))
            cv2.line(display, start_point, end_point, (255, 80, 180), 2, cv2.LINE_AA)
        for point_index, point in enumerate(hand.points):
            radius = 5 if point_index in {4, 8, 12, 16, 20} else 3
            center = tuple(point.astype(int))
            cv2.circle(display, center, radius, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(display, center, radius + 1, (35, 35, 35), 1, cv2.LINE_AA)
        pinch_color = (0, 255, 80) if hand.is_pinching else (70, 70, 255)
        cv2.circle(display, tuple(hand.pinch_center.astype(int)), 9, pinch_color, 2, cv2.LINE_AA)


def draw_ball(display: np.ndarray, ball: BallState) -> None:
    center = ball.center.astype(int)
    radius = max(1, int(ball.radius))
    shadow_center = (int(center[0] + radius * 0.14), int(center[1] + radius * 0.72))
    cv2.ellipse(display, shadow_center, (max(4, radius // 2), max(2, radius // 7)), 0, 0, 360, (20, 20, 20), 2, cv2.LINE_AA)

    overlay = display.copy()

    for layer_radius in range(radius, 0, -4):
        ratio = layer_radius / max(radius, 1)
        base = np.array([42, 118, 245], dtype=np.float32)
        light = np.array([122, 213, 255], dtype=np.float32)
        color = tuple((base * ratio + light * (1.0 - ratio)).astype(int).tolist())
        cv2.circle(overlay, tuple(center), layer_radius, color, -1, cv2.LINE_AA)

    highlight = center + np.array([-radius * 0.32, -radius * 0.35], dtype=np.float32).astype(int)
    cv2.circle(overlay, tuple(highlight), max(3, radius // 5), (210, 245, 255), -1, cv2.LINE_AA)

    draw_rotating_bands(overlay, center.astype(np.float32), radius, ball.angle)
    alpha = 0.88
    cv2.addWeighted(overlay, alpha, display, 1.0 - alpha, 0, dst=display)

    cv2.circle(display, tuple(center), radius, (20, 35, 65), 2, cv2.LINE_AA)
    if ball.held:
        cv2.circle(display, tuple(center), radius + 6, (0, 255, 120), 2, cv2.LINE_AA)


def draw_rotating_bands(display: np.ndarray, center: np.ndarray, radius: int, angle: float) -> None:
    for offset in (-0.42, 0.0, 0.42):
        points = []
        for step in range(-60, 61):
            t = step / 60.0
            x = t * radius * 0.88
            y = math.sin(t * math.pi) * radius * 0.16 + offset * radius
            rotated = rotate_point(np.array([x, y], dtype=np.float32), angle) + center
            if np.linalg.norm(rotated - center) <= radius * 0.96:
                points.append(rotated.astype(np.int32))
        if len(points) >= 2:
            cv2.polylines(display, [np.array(points, dtype=np.int32)], False, (245, 250, 255), 2, cv2.LINE_AA)

    marker = rotate_point(np.array([radius * 0.52, 0.0], dtype=np.float32), angle) + center
    cv2.circle(display, tuple(marker.astype(int)), max(3, radius // 12), (255, 255, 255), -1, cv2.LINE_AA)


def rotate_point(point: np.ndarray, angle: float) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.array(
        [point[0] * cos_a - point[1] * sin_a, point[0] * sin_a + point[1] * cos_a],
        dtype=np.float32,
    )


def reset_ball(frame_shape: tuple[int, ...], args: argparse.Namespace) -> BallState:
    height, width = frame_shape[:2]
    radius = float(np.clip(min(width, height) * 0.09, args.min_radius, args.max_radius))
    return BallState(
        center=np.array([width * 0.5, height * 0.42], dtype=np.float32),
        velocity=np.array([0.0, 0.0], dtype=np.float32),
        radius=radius,
        target_radius=radius,
    )


def main() -> None:
    args = parse_args()
    if not has_display():
        raise SystemExit("OpenCV display support is not available in this Python environment.")

    cap = cv2.VideoCapture(source_value(args.source))
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    detector = create_hand_detector(args.hand_model, args.max_hands, args.hand_confidence)
    ball: BallState | None = None
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
            dt = min(0.05, max(1e-3, now - previous_time))
            previous_time = now
            timestamp_ms = int((now - start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            if ball is None:
                ball = reset_ball(frame.shape, args)

            hand_result = detect_hands(detector, frame, timestamp_ms)
            hands = hand_observations(hand_result, frame.shape)
            update_ball(ball, hands, frame.shape, dt, args)

            display = frame.copy()
            if not args.hide_hands:
                draw_hands(display, hands)
            draw_ball(display, ball)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break
            if key == ord("r"):
                ball = reset_ball(frame.shape, args)
    finally:
        detector.close()
        cap.release()
        close_windows()


if __name__ == "__main__":
    main()
