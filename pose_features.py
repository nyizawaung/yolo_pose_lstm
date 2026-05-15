from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from .hand_features import HANDS_FEATURE_SIZE, detect_hands, hand_result_to_feature
except ImportError:
    from hand_features import HANDS_FEATURE_SIZE, detect_hands, hand_result_to_feature


KEYPOINT_COUNT = 17
BODY_FEATURE_SIZE = KEYPOINT_COUNT * 3
FEATURE_SIZE = BODY_FEATURE_SIZE
FEATURE_SIZE_WITH_HANDS = BODY_FEATURE_SIZE + HANDS_FEATURE_SIZE


def feature_size(include_hands: bool) -> int:
    return FEATURE_SIZE_WITH_HANDS if include_hands else BODY_FEATURE_SIZE


def list_action_videos(dataset_dir: Path) -> list[tuple[Path, str]]:
    videos: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        for video_path in sorted(class_dir.glob("*")):
            if video_path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}:
                videos.append((video_path, class_dir.name))
    return videos


def dataset_classes(dataset_dir: Path) -> list[str]:
    return sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())


def video_cache_path(video_path: Path, cache_dir: Path) -> Path:
    digest = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{video_path.stem}_{digest}.npz"


def load_feature_cache(cache_path: Path) -> tuple[np.ndarray, dict]:
    with np.load(cache_path, allow_pickle=False) as data:
        features = data["features"].astype(np.float32)
        meta = json.loads(str(data["meta"].item()))
    return features, meta


def save_feature_cache(cache_path: Path, features: np.ndarray, meta: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features.astype(np.float32),
        meta=np.array(json.dumps(meta)),
    )


def select_primary_person(result) -> int | None:
    if result.keypoints is None or result.keypoints.xyn is None:
        return None
    keypoints = result.keypoints.xyn
    if len(keypoints) == 0:
        return None

    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) == len(keypoints):
        xyxy = boxes.xyxy.detach().cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        return int(np.argmax(areas))

    conf = result.keypoints.conf
    if conf is not None:
        scores = conf.detach().cpu().numpy().mean(axis=1)
        return int(np.argmax(scores))
    return 0


def append_hand_feature(
    body_feature: np.ndarray,
    hand_result,
    frame_shape: tuple[int, ...] | None,
    include_hands: bool,
    person_box: np.ndarray | None = None,
) -> np.ndarray:
    if not include_hands:
        return body_feature

    if frame_shape is None:
        hand_feature = np.zeros(HANDS_FEATURE_SIZE, dtype=np.float32)
    else:
        hand_feature = hand_result_to_feature(
            hand_result=hand_result,
            frame_shape=frame_shape,
            person_box=person_box,
            pose_feature=body_feature,
        )
    return np.concatenate([body_feature, hand_feature]).astype(np.float32)


def result_to_feature(
    result,
    hand_result=None,
    frame_shape: tuple[int, ...] | None = None,
    include_hands: bool = False,
) -> np.ndarray:
    feature = np.zeros((KEYPOINT_COUNT, 3), dtype=np.float32)
    person_idx = select_primary_person(result)
    if person_idx is None:
        return append_hand_feature(feature.reshape(-1), hand_result, frame_shape, include_hands)

    xyn = result.keypoints.xyn[person_idx].detach().cpu().numpy().astype(np.float32)
    feature[:, :2] = xyn[:KEYPOINT_COUNT]
    if result.keypoints.conf is not None:
        conf = result.keypoints.conf[person_idx].detach().cpu().numpy().astype(np.float32)
        feature[:, 2] = conf[:KEYPOINT_COUNT]
    body_feature = feature.reshape(-1)

    person_box = None
    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > person_idx:
        person_box = boxes.xyxy[person_idx].detach().cpu().numpy().astype(np.float32)
    return append_hand_feature(body_feature, hand_result, frame_shape, include_hands, person_box)


def keypoint_to_feature(result, person_idx: int) -> np.ndarray:
    feature = np.zeros((KEYPOINT_COUNT, 3), dtype=np.float32)
    xyn = result.keypoints.xyn[person_idx].detach().cpu().numpy().astype(np.float32)
    feature[:, :2] = xyn[:KEYPOINT_COUNT]
    if result.keypoints.conf is not None:
        conf = result.keypoints.conf[person_idx].detach().cpu().numpy().astype(np.float32)
        feature[:, 2] = conf[:KEYPOINT_COUNT]
    return feature.reshape(-1)


def result_to_person_features(
    result,
    hand_result=None,
    frame_shape: tuple[int, ...] | None = None,
    include_hands: bool = False,
) -> list[dict[str, np.ndarray | float]]:
    if result.keypoints is None or result.keypoints.xyn is None or len(result.keypoints.xyn) == 0:
        return []

    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) == len(result.keypoints.xyn):
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        box_conf = boxes.conf.detach().cpu().numpy().astype(np.float32) if boxes.conf is not None else None
    else:
        xyxy = np.zeros((len(result.keypoints.xyn), 4), dtype=np.float32)
        box_conf = None

    if result.keypoints.conf is not None:
        pose_conf = result.keypoints.conf.detach().cpu().numpy().astype(np.float32).mean(axis=1)
    else:
        pose_conf = np.ones(len(result.keypoints.xyn), dtype=np.float32)

    people = []
    for person_idx in range(len(result.keypoints.xyn)):
        confidence = float(box_conf[person_idx]) if box_conf is not None else float(pose_conf[person_idx])
        body_feature = keypoint_to_feature(result, person_idx)
        people.append(
            {
                "feature": append_hand_feature(
                    body_feature,
                    hand_result,
                    frame_shape,
                    include_hands,
                    xyxy[person_idx],
                ),
                "box": xyxy[person_idx],
                "confidence": confidence,
            }
        )
    return people


def extract_frame_feature(
    pose_model,
    frame: np.ndarray,
    imgsz: int,
    device: str | None,
    hand_detector=None,
    timestamp_ms: int = 0,
    include_hands: bool = False,
):
    predict_kwargs = {"imgsz": imgsz, "verbose": False}
    if device:
        predict_kwargs["device"] = device
    result = pose_model.predict(frame, **predict_kwargs)[0]
    hand_result = detect_hands(hand_detector, frame, timestamp_ms) if include_hands and hand_detector is not None else None
    return result_to_feature(result, hand_result, frame.shape, include_hands), result


def extract_video_features(
    video_path: Path,
    pose_model,
    imgsz: int,
    frame_stride: int,
    device: str | None,
    hand_detector=None,
    include_hands: bool = False,
    timestamp_offset_ms: int = 0,
) -> tuple[np.ndarray, dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    features: list[np.ndarray] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_index = 0
    sampled_frames = 0
    last_timestamp_ms = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_stride == 0:
            if fps > 0:
                timestamp_ms = timestamp_offset_ms + int(frame_index * 1000 / fps)
            else:
                timestamp_ms = timestamp_offset_ms + sampled_frames
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms
            feature, _ = extract_frame_feature(
                pose_model=pose_model,
                frame=frame,
                imgsz=imgsz,
                device=device,
                hand_detector=hand_detector,
                timestamp_ms=timestamp_ms,
                include_hands=include_hands,
            )
            features.append(feature)
            sampled_frames += 1
        frame_index += 1

    cap.release()
    if not features:
        features.append(np.zeros(feature_size(include_hands), dtype=np.float32))

    meta = {
        "video": str(video_path),
        "total_frames": total_frames,
        "fps": fps,
        "frame_stride": frame_stride,
        "sampled_frames": len(features),
        "imgsz": imgsz,
        "include_hands": include_hands,
        "feature_size": feature_size(include_hands),
    }
    return np.stack(features).astype(np.float32), meta


def pad_or_trim_sequence(sequence: np.ndarray, sequence_length: int) -> np.ndarray:
    if len(sequence) >= sequence_length:
        return sequence[:sequence_length].astype(np.float32)
    padded = np.zeros((sequence_length, sequence.shape[1]), dtype=np.float32)
    padded[: len(sequence)] = sequence
    return padded


def make_windows(sequence: np.ndarray, sequence_length: int, stride: int) -> list[np.ndarray]:
    if len(sequence) <= sequence_length:
        return [pad_or_trim_sequence(sequence, sequence_length)]

    windows = []
    for start in range(0, len(sequence) - sequence_length + 1, stride):
        windows.append(sequence[start : start + sequence_length].astype(np.float32))

    last = sequence[-sequence_length:].astype(np.float32)
    if not windows or not np.array_equal(windows[-1], last):
        windows.append(last)
    return windows


def iter_batches(
    samples: list[tuple[np.ndarray, int]],
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(samples))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        x = np.stack([samples[idx][0] for idx in batch_indices]).astype(np.float32)
        y = np.array([samples[idx][1] for idx in batch_indices], dtype=np.int64)
        yield x, y
