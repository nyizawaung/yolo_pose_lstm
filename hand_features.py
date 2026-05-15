from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


HAND_LANDMARK_COUNT = 21
HAND_VALUES_PER_LANDMARK = 4
HAND_FEATURE_SIZE = HAND_LANDMARK_COUNT * HAND_VALUES_PER_LANDMARK
HANDS_FEATURE_SIZE = HAND_FEATURE_SIZE * 2
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)
HAND_SIDES = ("Left", "Right")
BODY_WRIST_INDEX = {
    "Left": 9,
    "Right": 10,
}


def ensure_hand_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmark model: {model_path}")
    urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, model_path)
    return model_path


def create_hand_detector(model_path: Path, max_hands: int, confidence: float):
    if mp is None:
        raise SystemExit(
            "MediaPipe is required for hand landmarks. Install it in the active venv with: "
            "pip install mediapipe"
        )

    model_path = ensure_hand_model(model_path)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_hands(hand_detector, frame: np.ndarray, timestamp_ms: int):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return hand_detector.detect_for_video(image, timestamp_ms)


def hand_result_to_feature(
    hand_result,
    frame_shape: tuple[int, ...],
    person_box: np.ndarray | None = None,
    pose_feature: np.ndarray | None = None,
) -> np.ndarray:
    feature = np.zeros((2, HAND_LANDMARK_COUNT, HAND_VALUES_PER_LANDMARK), dtype=np.float32)
    if hand_result is None or not hand_result.hand_landmarks:
        return feature.reshape(-1)

    height, width = frame_shape[:2]
    candidates_by_side: dict[str, tuple[float, list]] = {}
    for hand_index, hand_landmarks in enumerate(hand_result.hand_landmarks):
        label, confidence = hand_label_and_score(hand_result, hand_index)
        if label not in HAND_SIDES:
            continue
        if person_box is not None and not hand_matches_person(hand_landmarks, frame_shape, person_box, pose_feature, label):
            continue

        previous = candidates_by_side.get(label)
        if previous is None or confidence > previous[0]:
            candidates_by_side[label] = (confidence, hand_landmarks)

    for side_index, side in enumerate(HAND_SIDES):
        candidate = candidates_by_side.get(side)
        if candidate is None:
            continue
        confidence, hand_landmarks = candidate
        for landmark_index, landmark in enumerate(hand_landmarks[:HAND_LANDMARK_COUNT]):
            feature[side_index, landmark_index] = (
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
                float(confidence),
            )
    return feature.reshape(-1)


def hand_label_and_score(hand_result, hand_index: int) -> tuple[str, float]:
    if hand_index >= len(hand_result.handedness or []):
        return "", 1.0
    categories = hand_result.handedness[hand_index]
    if not categories:
        return "", 1.0
    return str(categories[0].category_name), float(categories[0].score)


def hand_matches_person(
    hand_landmarks,
    frame_shape: tuple[int, ...],
    person_box: np.ndarray,
    pose_feature: np.ndarray | None,
    side: str,
) -> bool:
    height, width = frame_shape[:2]
    wrist = hand_landmarks[0]
    wrist_xy = np.array([float(wrist.x), float(wrist.y)], dtype=np.float32)
    wrist_px = np.array([wrist_xy[0] * width, wrist_xy[1] * height], dtype=np.float32)

    if pose_feature is not None:
        pose = pose_feature.reshape(-1, 3)
        body_wrist_index = BODY_WRIST_INDEX[side]
        if body_wrist_index < len(pose) and pose[body_wrist_index, 2] > 0.1:
            body_wrist_xy = pose[body_wrist_index, :2]
            if np.linalg.norm(wrist_xy - body_wrist_xy) <= 0.18:
                return True

    x1, y1, x2, y2 = person_box.astype(np.float32)
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    padding_x = box_width * 0.25
    padding_y = box_height * 0.25
    return bool(
        x1 - padding_x <= wrist_px[0] <= x2 + padding_x
        and y1 - padding_y <= wrist_px[1] <= y2 + padding_y
    )
