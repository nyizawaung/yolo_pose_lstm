from __future__ import annotations

import argparse
import math
import os
import random
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


WINDOW_NAME = "Hand AR Helicopter"
_HAS_DISPLAY: bool | None = None


@dataclass
class HandObservation:
    label: str
    score: float
    points: np.ndarray
    palm_center: np.ndarray
    index_tip: np.ndarray
    index_direction: np.ndarray
    is_fist: bool


@dataclass
class HelicopterState:
    center: np.ndarray
    velocity: np.ndarray
    color: tuple[int, int, int] = (48, 170, 245)
    angle: float = 0.0
    rotor_phase: float = 0.0
    facing: float = 1.0
    rocket_cooldown: float = 0.0
    hits: int = 0
    hit_flash: float = 0.0


@dataclass
class Rocket:
    center: np.ndarray
    velocity: np.ndarray
    owner: str = "P1"
    age: float = 0.0
    angle: float = 0.0
    smoke_timer: float = 0.0


@dataclass
class SmokeParticle:
    center: np.ndarray
    velocity: np.ndarray
    radius: float
    age: float = 0.0
    lifetime: float = 1.2


@dataclass
class Target:
    center: np.ndarray
    radius: float
    hit_flash: float = 0.0
    score: int = 0
    misses: int = 0


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fly a digital helicopter with your right index finger.")
    parser.add_argument("--source", default="0", help="Webcam index, video path, or RTSP URL.")
    parser.add_argument("--hand-model", type=Path, default=here / "models" / "hand_landmarker.task")
    parser.add_argument("--hand-confidence", type=float, default=0.5)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--helicopter-size", type=float, default=1.0)
    parser.add_argument("--follow-strength", type=float, default=0.22, help="Legacy tuning value; joystick mode uses --joystick-speed.")
    parser.add_argument("--joystick-speed", type=float, default=430.0)
    parser.add_argument("--joystick-deadzone", type=float, default=0.28)
    parser.add_argument("--fist-strictness", type=float, default=0.68)
    parser.add_argument("--rocket-speed", type=float, default=560.0)
    parser.add_argument("--rocket-cooldown", type=float, default=0.35)
    parser.add_argument("--max-rockets", type=int, default=8)
    parser.add_argument("--target-radius", type=float, default=34.0)
    parser.add_argument("--two-player", action="store_true")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror webcam/video frames.")
    parser.add_argument("--swap-hands", action="store_true", help="Swap detected left/right hand labels.")
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
    swap_hands: bool,
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
        if swap_hands:
            label = swapped_label(label)

        palm_center = points[[0, 5, 9, 13, 17]].mean(axis=0)
        index_direction = index_finger_direction(points)
        observations.append(
            HandObservation(
                label=label,
                score=score,
                points=points,
                palm_center=palm_center,
                index_tip=points[8],
                index_direction=index_direction,
                is_fist=is_fist(points, fist_strictness),
            )
        )
    return observations


def swapped_label(label: str) -> str:
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def index_finger_direction(points: np.ndarray) -> np.ndarray:
    direction = points[8] - points[5]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-3:
        return np.zeros(2, dtype=np.float32)
    return (direction / norm).astype(np.float32)


def is_fist(points: np.ndarray, strictness: float = 0.68) -> bool:
    palm_center = points[[0, 5, 9, 13, 17]].mean(axis=0)
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    palm_length = float(np.linalg.norm(points[0] - points[9]))
    palm_size = max(28.0, palm_width, palm_length)

    curled = 0
    for tip_index in (8, 12, 16, 20):
        tip_distance = float(np.linalg.norm(points[tip_index] - palm_center))
        pip_distance = float(np.linalg.norm(points[tip_index - 2] - palm_center))
        mcp_distance = float(np.linalg.norm(points[tip_index - 3] - palm_center))
        if tip_distance < palm_size * strictness and tip_distance <= pip_distance * 0.98 and tip_distance <= mcp_distance * 1.22:
            curled += 1
    thumb_tucked = float(np.linalg.norm(points[4] - palm_center)) < palm_size * (strictness + 0.12)
    return curled >= 4 and thumb_tucked


def select_hand(hands: list[HandObservation], label: str) -> HandObservation | None:
    matches = [hand for hand in hands if hand.label == label]
    if matches:
        return max(matches, key=lambda hand: hand.score)
    return None


def update_helicopter(
    helicopter: HelicopterState,
    right_hand: HandObservation | None,
    left_hand: HandObservation | None,
    rockets: list[Rocket],
    smoke: list[SmokeParticle],
    target: Target,
    frame_shape: tuple[int, ...],
    dt: float,
    args: argparse.Namespace,
) -> None:
    height, width = frame_shape[:2]
    helicopter.rocket_cooldown = max(0.0, helicopter.rocket_cooldown - dt)

    if right_hand is not None:
        joystick = joystick_vector(right_hand, args.joystick_deadzone)
        target_velocity = joystick * args.joystick_speed
        helicopter.velocity = 0.82 * helicopter.velocity + 0.18 * target_velocity
        helicopter.center += helicopter.velocity * dt

        if abs(joystick[0]) > 0.12:
            helicopter.facing = 1.0 if helicopter.velocity[0] >= 0.0 else -1.0
        target_angle = float(np.clip(helicopter.velocity[0] / 900.0, -0.45, 0.45))
        helicopter.angle += (target_angle - helicopter.angle) * min(1.0, 8.0 * dt)
    else:
        helicopter.velocity *= max(0.0, 1.0 - 2.5 * dt)
        helicopter.angle *= max(0.0, 1.0 - 3.0 * dt)

    margin = 42.0 * args.helicopter_size
    helicopter.center[0] = float(np.clip(helicopter.center[0], margin, width - margin))
    helicopter.center[1] = float(np.clip(helicopter.center[1], margin, height - margin))
    helicopter.rotor_phase += dt * (24.0 + min(18.0, np.linalg.norm(helicopter.velocity) / 90.0))

    if left_hand is not None and left_hand.is_fist and helicopter.rocket_cooldown <= 0.0:
        fire_rocket(helicopter, rockets, args)
        helicopter.rocket_cooldown = args.rocket_cooldown

    update_rockets(rockets, smoke, target, frame_shape, dt, args)
    update_smoke(smoke, dt)


def update_player_helicopter(
    helicopter: HelicopterState,
    hand: HandObservation | None,
    rockets: list[Rocket],
    frame_shape: tuple[int, ...],
    dt: float,
    args: argparse.Namespace,
    owner: str,
) -> None:
    height, width = frame_shape[:2]
    helicopter.rocket_cooldown = max(0.0, helicopter.rocket_cooldown - dt)
    helicopter.hit_flash = max(0.0, helicopter.hit_flash - dt)

    if hand is not None:
        joystick = joystick_vector(hand, args.joystick_deadzone)
        target_velocity = joystick * args.joystick_speed
        helicopter.velocity = 0.82 * helicopter.velocity + 0.18 * target_velocity
        helicopter.center += helicopter.velocity * dt
        if abs(joystick[0]) > 0.12:
            helicopter.facing = 1.0 if helicopter.velocity[0] >= 0.0 else -1.0
        if hand.is_fist and helicopter.rocket_cooldown <= 0.0:
            fire_rocket(helicopter, rockets, args, owner=owner)
            helicopter.rocket_cooldown = args.rocket_cooldown
    else:
        helicopter.velocity *= max(0.0, 1.0 - 2.5 * dt)

    target_angle = float(np.clip(helicopter.velocity[0] / 900.0, -0.45, 0.45))
    helicopter.angle += (target_angle - helicopter.angle) * min(1.0, 8.0 * dt)
    margin = 42.0 * args.helicopter_size
    helicopter.center[0] = float(np.clip(helicopter.center[0], margin, width - margin))
    helicopter.center[1] = float(np.clip(helicopter.center[1], margin, height - margin))
    helicopter.rotor_phase += dt * (24.0 + min(18.0, np.linalg.norm(helicopter.velocity) / 90.0))


def update_two_player_rockets(
    rockets: list[Rocket],
    smoke: list[SmokeParticle],
    players: dict[str, HelicopterState],
    frame_shape: tuple[int, ...],
    dt: float,
) -> None:
    height, width = frame_shape[:2]
    alive: list[Rocket] = []
    for rocket in rockets:
        rocket.age += dt
        rocket.smoke_timer += dt
        rocket.center += rocket.velocity * dt

        while rocket.smoke_timer >= 0.035:
            rocket.smoke_timer -= 0.035
            spawn_smoke(smoke, rocket)

        hit_player = None
        for player_name, helicopter in players.items():
            if player_name == rocket.owner:
                continue
            hit_radius = 42.0
            if np.linalg.norm(rocket.center - helicopter.center) <= hit_radius:
                hit_player = player_name
                break

        if hit_player is not None:
            players[rocket.owner].hits += 1
            players[hit_player].hit_flash = 0.45
            spawn_explosion(smoke, rocket.center)
            continue

        if -80 <= rocket.center[0] <= width + 80 and -80 <= rocket.center[1] <= height + 80 and rocket.age < 4.0:
            alive.append(rocket)
    rockets[:] = alive


def joystick_vector(hand: HandObservation, deadzone: float) -> np.ndarray:
    direction = hand.index_direction.copy()
    magnitude = float(np.linalg.norm(direction))
    if magnitude < deadzone:
        return np.zeros(2, dtype=np.float32)
    scaled = (magnitude - deadzone) / max(1e-3, 1.0 - deadzone)
    return (direction / max(magnitude, 1e-3) * min(1.0, scaled)).astype(np.float32)


def fire_rocket(helicopter: HelicopterState, rockets: list[Rocket], args: argparse.Namespace, owner: str = "P1") -> None:
    direction = np.array([helicopter.facing, 0.08 * math.sin(helicopter.angle)], dtype=np.float32)
    norm = max(1e-3, float(np.linalg.norm(direction)))
    direction /= norm
    size = 44.0 * args.helicopter_size
    center = helicopter.center + direction * (size * 1.2) + np.array([0.0, size * 0.16], dtype=np.float32)
    velocity = direction * args.rocket_speed + helicopter.velocity * 0.35
    rockets.append(
        Rocket(
            center=center.astype(np.float32),
            velocity=velocity.astype(np.float32),
            owner=owner,
            angle=math.atan2(float(direction[1]), float(direction[0])),
        )
    )
    del rockets[:-args.max_rockets]


def update_rockets(
    rockets: list[Rocket],
    smoke: list[SmokeParticle],
    target: Target,
    frame_shape: tuple[int, ...],
    dt: float,
    args: argparse.Namespace,
) -> None:
    height, width = frame_shape[:2]
    alive: list[Rocket] = []
    for rocket in rockets:
        rocket.age += dt
        rocket.smoke_timer += dt
        rocket.center += rocket.velocity * dt

        while rocket.smoke_timer >= 0.035:
            rocket.smoke_timer -= 0.035
            spawn_smoke(smoke, rocket)

        hit_target = np.linalg.norm(rocket.center - target.center) <= target.radius + 16.0
        if hit_target:
            target.score += 1
            target.hit_flash = 0.45
            spawn_explosion(smoke, rocket.center)
            respawn_target(target, frame_shape, args)
            continue

        if -80 <= rocket.center[0] <= width + 80 and -80 <= rocket.center[1] <= height + 80 and rocket.age < 4.0:
            alive.append(rocket)
        else:
            target.misses += 1
    rockets[:] = alive


def spawn_smoke(smoke: list[SmokeParticle], rocket: Rocket) -> None:
    direction = np.array([math.cos(rocket.angle), math.sin(rocket.angle)], dtype=np.float32)
    tail = rocket.center - direction * 22.0
    jitter = np.array([random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)], dtype=np.float32)
    velocity = -direction * random.uniform(35.0, 70.0) + jitter * 4.0
    smoke.append(
        SmokeParticle(
            center=(tail + jitter).astype(np.float32),
            velocity=velocity.astype(np.float32),
            radius=random.uniform(5.0, 12.0),
            lifetime=random.uniform(0.8, 1.45),
        )
    )
    del smoke[:-180]


def spawn_explosion(smoke: list[SmokeParticle], center: np.ndarray) -> None:
    for _ in range(30):
        angle = random.uniform(0.0, math.tau)
        speed = random.uniform(60.0, 260.0)
        velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
        smoke.append(
            SmokeParticle(
                center=center.astype(np.float32).copy(),
                velocity=velocity,
                radius=random.uniform(8.0, 22.0),
                lifetime=random.uniform(0.35, 0.9),
            )
        )
    del smoke[:-180]


def update_smoke(smoke: list[SmokeParticle], dt: float) -> None:
    alive: list[SmokeParticle] = []
    for particle in smoke:
        particle.age += dt
        particle.center += particle.velocity * dt
        particle.velocity *= max(0.0, 1.0 - 1.2 * dt)
        particle.velocity[1] -= 18.0 * dt
        particle.radius += 13.0 * dt
        if particle.age < particle.lifetime:
            alive.append(particle)
    smoke[:] = alive


def reset_target(frame_shape: tuple[int, ...], args: argparse.Namespace) -> Target:
    height, width = frame_shape[:2]
    target = Target(center=np.zeros(2, dtype=np.float32), radius=args.target_radius)
    respawn_target(target, frame_shape, args)
    return target


def respawn_target(target: Target, frame_shape: tuple[int, ...], args: argparse.Namespace) -> None:
    height, width = frame_shape[:2]
    margin = max(70.0, args.target_radius * 2.2)
    target.center = np.array(
        [
            random.uniform(margin, max(margin, width - margin)),
            random.uniform(margin, max(margin, height - margin)),
        ],
        dtype=np.float32,
    )
    target.radius = args.target_radius


def draw_hands(display: np.ndarray, hands: list[HandObservation], right_hand: HandObservation | None, left_hand: HandObservation | None) -> None:
    for hand in hands:
        if hand is right_hand:
            color = (80, 255, 80)
        elif hand is left_hand and hand.is_fist:
            color = (0, 80, 255)
        else:
            color = (255, 80, 180)

        for start, end in HAND_CONNECTIONS:
            cv2.line(display, tuple(hand.points[start].astype(int)), tuple(hand.points[end].astype(int)), color, 2, cv2.LINE_AA)
        for point_index, point in enumerate(hand.points):
            radius = 6 if point_index == 8 else 4
            cv2.circle(display, tuple(point.astype(int)), radius, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(display, tuple(point.astype(int)), radius + 1, (35, 35, 35), 1, cv2.LINE_AA)

        if hand is left_hand and hand.is_fist:
            cv2.circle(display, tuple(hand.palm_center.astype(int)), 22, (0, 80, 255), 3, cv2.LINE_AA)
        if hand is right_hand:
            arrow_end = hand.palm_center + hand.index_direction * 62.0
            cv2.arrowedLine(
                display,
                tuple(hand.palm_center.astype(int)),
                tuple(arrow_end.astype(int)),
                (80, 255, 80),
                3,
                cv2.LINE_AA,
                tipLength=0.25,
            )


def draw_target(display: np.ndarray, target: Target) -> None:
    center = tuple(target.center.astype(int))
    radius = int(target.radius)
    flash = max(0.0, min(1.0, target.hit_flash / 0.45))
    outer_color = (
        int(30 + 220 * flash),
        int(220 - 120 * flash),
        int(255 - 220 * flash),
    )
    cv2.circle(display, center, radius, outer_color, 3, cv2.LINE_AA)
    cv2.circle(display, center, max(5, radius // 2), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(display, (center[0] - radius - 8, center[1]), (center[0] + radius + 8, center[1]), outer_color, 2, cv2.LINE_AA)
    cv2.line(display, (center[0], center[1] - radius - 8), (center[0], center[1] + radius + 8), outer_color, 2, cv2.LINE_AA)


def draw_hud(display: np.ndarray, target: Target, elapsed_seconds: float) -> None:
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    text = f"time {minutes:02d}:{seconds:02d}   hits {target.score}   misses {target.misses}"
    cv2.putText(display, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(display, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (40, 230, 255), 2, cv2.LINE_AA)


def draw_two_player_hud(display: np.ndarray, players: dict[str, HelicopterState]) -> None:
    text = f"P1 hits {players['P1'].hits}   P2 hits {players['P2'].hits}"
    cv2.putText(display, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(display, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40, 230, 255), 2, cv2.LINE_AA)


def draw_helicopter(display: np.ndarray, helicopter: HelicopterState, size_scale: float) -> None:
    size = 46.0 * size_scale
    center = helicopter.center
    angle = helicopter.angle
    facing = helicopter.facing

    body_points = np.array(
        [
            [-0.85 * size, -0.30 * size],
            [0.62 * size, -0.30 * size],
            [0.95 * size, 0.00 * size],
            [0.58 * size, 0.33 * size],
            [-0.72 * size, 0.35 * size],
            [-1.00 * size, 0.05 * size],
        ],
        dtype=np.float32,
    )
    body_points[:, 0] *= facing
    body_points = transform_points(body_points, center, angle)
    body_color = helicopter.color
    if helicopter.hit_flash > 0.0:
        flash = min(1.0, helicopter.hit_flash / 0.45)
        body_color = tuple(int(channel * (1.0 - flash) + 255 * flash) for channel in body_color)
    cv2.fillPoly(display, [body_points.astype(np.int32)], body_color, cv2.LINE_AA)
    cv2.polylines(display, [body_points.astype(np.int32)], True, (15, 40, 75), 2, cv2.LINE_AA)

    cockpit = transform_points(np.array([[0.35 * size * facing, -0.22 * size]], dtype=np.float32), center, angle)[0]
    cv2.circle(display, tuple(cockpit.astype(int)), int(0.22 * size), (190, 240, 255), -1, cv2.LINE_AA)
    cv2.circle(display, tuple(cockpit.astype(int)), int(0.22 * size), (15, 70, 115), 2, cv2.LINE_AA)

    tail_start = transform_points(np.array([[-0.86 * size * facing, 0.0]], dtype=np.float32), center, angle)[0]
    tail_end = transform_points(np.array([[-2.05 * size * facing, -0.03 * size]], dtype=np.float32), center, angle)[0]
    cv2.line(display, tuple(tail_start.astype(int)), tuple(tail_end.astype(int)), (40, 130, 205), 8, cv2.LINE_AA)
    cv2.line(display, tuple(tail_start.astype(int)), tuple(tail_end.astype(int)), (15, 40, 75), 2, cv2.LINE_AA)

    tail_rotor_center = tail_end
    rotor_spin = helicopter.rotor_phase * 1.7
    for rotor_angle in (rotor_spin, rotor_spin + math.pi / 2):
        blade = np.array([math.cos(rotor_angle), math.sin(rotor_angle)], dtype=np.float32) * size * 0.28
        cv2.line(display, tuple((tail_rotor_center - blade).astype(int)), tuple((tail_rotor_center + blade).astype(int)), (235, 245, 255), 2, cv2.LINE_AA)

    mast_top = transform_points(np.array([[0.0, -0.48 * size]], dtype=np.float32), center, angle)[0]
    mast_base = transform_points(np.array([[0.0, -0.18 * size]], dtype=np.float32), center, angle)[0]
    cv2.line(display, tuple(mast_base.astype(int)), tuple(mast_top.astype(int)), (20, 35, 55), 4, cv2.LINE_AA)
    draw_main_rotor(display, mast_top, size, helicopter.rotor_phase)

    skid_left = transform_points(np.array([[-0.75 * size * facing, 0.58 * size], [0.65 * size * facing, 0.58 * size]], dtype=np.float32), center, angle)
    skid_right = transform_points(np.array([[-0.58 * size * facing, 0.42 * size], [0.48 * size * facing, 0.42 * size]], dtype=np.float32), center, angle)
    cv2.line(display, tuple(skid_left[0].astype(int)), tuple(skid_left[1].astype(int)), (20, 35, 55), 3, cv2.LINE_AA)
    cv2.line(display, tuple(skid_right[0].astype(int)), tuple(skid_right[1].astype(int)), (20, 35, 55), 3, cv2.LINE_AA)


def draw_main_rotor(display: np.ndarray, center: np.ndarray, size: float, phase: float) -> None:
    rotor_length = size * 1.35
    for blade_angle in (phase, phase + math.pi / 2):
        direction = np.array([math.cos(blade_angle), math.sin(blade_angle) * 0.18], dtype=np.float32)
        direction /= max(1e-3, float(np.linalg.norm(direction)))
        end_a = center - direction * rotor_length
        end_b = center + direction * rotor_length
        cv2.line(display, tuple(end_a.astype(int)), tuple(end_b.astype(int)), (245, 250, 255), 5, cv2.LINE_AA)
        cv2.line(display, tuple(end_a.astype(int)), tuple(end_b.astype(int)), (45, 80, 110), 1, cv2.LINE_AA)
    cv2.circle(display, tuple(center.astype(int)), max(4, int(size * 0.08)), (25, 45, 70), -1, cv2.LINE_AA)


def draw_rockets(display: np.ndarray, rockets: list[Rocket]) -> None:
    for rocket in rockets:
        direction = np.array([math.cos(rocket.angle), math.sin(rocket.angle)], dtype=np.float32)
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        nose = rocket.center + direction * 18.0
        tail = rocket.center - direction * 16.0
        body = np.array(
            [
                nose,
                tail + normal * 6.0,
                tail - normal * 6.0,
            ],
            dtype=np.float32,
        )
        cv2.fillPoly(display, [body.astype(np.int32)], (30, 30, 235), cv2.LINE_AA)
        cv2.polylines(display, [body.astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
        flame_tip = tail - direction * (14.0 + 8.0 * math.sin(rocket.age * 45.0))
        cv2.line(display, tuple(tail.astype(int)), tuple(flame_tip.astype(int)), (0, 180, 255), 4, cv2.LINE_AA)


def draw_smoke(display: np.ndarray, smoke: list[SmokeParticle]) -> None:
    if not smoke:
        return
    overlay = display.copy()
    for particle in smoke:
        life = max(0.0, min(1.0, particle.age / particle.lifetime))
        alpha_color = int(170 - 110 * life)
        color = (alpha_color, alpha_color, alpha_color)
        cv2.circle(overlay, tuple(particle.center.astype(int)), max(1, int(particle.radius)), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.42, display, 0.58, 0, dst=display)


def transform_points(points: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return points @ rotation.T + center


def reset_helicopter(frame_shape: tuple[int, ...]) -> HelicopterState:
    height, width = frame_shape[:2]
    return HelicopterState(
        center=np.array([width * 0.5, height * 0.45], dtype=np.float32),
        velocity=np.array([0.0, 0.0], dtype=np.float32),
    )


def reset_two_player_helicopters(frame_shape: tuple[int, ...]) -> dict[str, HelicopterState]:
    height, width = frame_shape[:2]
    return {
        "P1": HelicopterState(
            center=np.array([width * 0.28, height * 0.45], dtype=np.float32),
            velocity=np.array([0.0, 0.0], dtype=np.float32),
            color=(50, 220, 90),
            facing=1.0,
        ),
        "P2": HelicopterState(
            center=np.array([width * 0.72, height * 0.45], dtype=np.float32),
            velocity=np.array([0.0, 0.0], dtype=np.float32),
            color=(65, 110, 255),
            facing=-1.0,
        ),
    }


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
    helicopter: HelicopterState | None = None
    players: dict[str, HelicopterState] | None = None
    rockets: list[Rocket] = []
    smoke: list[SmokeParticle] = []
    target: Target | None = None
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

            if args.two_player and players is None:
                players = reset_two_player_helicopters(frame.shape)
            if not args.two_player and helicopter is None:
                helicopter = reset_helicopter(frame.shape)
            if not args.two_player and target is None:
                target = reset_target(frame.shape, args)
            if target is not None:
                target.hit_flash = max(0.0, target.hit_flash - dt)

            hand_result = detect_hands(detector, frame, timestamp_ms)
            hands = hand_observations(
                hand_result=hand_result,
                frame_shape=frame.shape,
                swap_hands=args.swap_hands,
                fist_strictness=args.fist_strictness,
            )
            right_hand = select_hand(hands, "Right")
            left_hand = select_hand(hands, "Left")
            if args.two_player:
                assert players is not None
                update_player_helicopter(players["P1"], right_hand, rockets, frame.shape, dt, args, owner="P1")
                update_player_helicopter(players["P2"], left_hand, rockets, frame.shape, dt, args, owner="P2")
                update_two_player_rockets(rockets, smoke, players, frame.shape, dt)
                update_smoke(smoke, dt)
            else:
                assert helicopter is not None and target is not None
                update_helicopter(
                    helicopter=helicopter,
                    right_hand=right_hand,
                    left_hand=left_hand,
                    rockets=rockets,
                    smoke=smoke,
                    target=target,
                    frame_shape=frame.shape,
                    dt=dt,
                    args=args,
                )

            display = frame.copy()
            draw_smoke(display, smoke)
            if args.two_player:
                assert players is not None
                draw_two_player_hud(display, players)
            else:
                assert target is not None
                draw_target(display, target)
                draw_hud(display, target, now - start_time)
            if not args.hide_hands:
                draw_hands(display, hands, right_hand, left_hand)
            draw_rockets(display, rockets)
            if args.two_player:
                assert players is not None
                draw_helicopter(display, players["P1"], args.helicopter_size)
                draw_helicopter(display, players["P2"], args.helicopter_size)
            else:
                assert helicopter is not None
                draw_helicopter(display, helicopter, args.helicopter_size)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break
            if key == ord("r"):
                rockets.clear()
                smoke.clear()
                if args.two_player:
                    players = reset_two_player_helicopters(frame.shape)
                else:
                    helicopter = reset_helicopter(frame.shape)
                    target = reset_target(frame.shape, args)
    finally:
        detector.close()
        cap.release()
        close_windows()


if __name__ == "__main__":
    main()
