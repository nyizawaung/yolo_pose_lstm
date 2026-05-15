from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

#python Action_LSTM/webcam_action.py --source 0 --person-confidence 0.35 --track-iou-threshold 0.3
#python Action_LSTM/webcam_action.py --source rtsp://USER:PASSWORD@CAMERA_IP:554/axis-media/media.amp --track-iou-threshold 0.3
#rtsp://admin:AIruca88@192.168.1.18:554/axis-media/media.amp # do not remove or override this line
try:
    from .model import ActionLSTM
    from .pose_features import FEATURE_SIZE, result_to_person_features, pad_or_trim_sequence
    from .hand_features import HAND_CONNECTIONS, create_hand_detector, detect_hands
except ImportError:
    from model import ActionLSTM
    from pose_features import FEATURE_SIZE, result_to_person_features, pad_or_trim_sequence
    from hand_features import HAND_CONNECTIONS, create_hand_detector, detect_hands


WINDOW_NAME = "YOLOv8 Pose + LSTM Action Recognition"
_HAS_DISPLAY: bool | None = None
COCO_SKELETON = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)


@dataclass
class PersonTrack:
    track_id: int
    box: np.ndarray
    frames: deque[np.ndarray]
    missed: int = 0
    label: str = "warming up"
    score: float = 0.0


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run webcam action recognition with YOLOv8 pose + LSTM.")
    parser.add_argument("--checkpoint", type=Path, default=here / "checkpoints" / "webcam_action_lstm.pt")
    parser.add_argument("--pose-model", default=None, help="Defaults to the pose model recorded in the checkpoint.")
    parser.add_argument("--source", default="0", help="Webcam index or video/RTSP URL.")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=0, help="For example 0, cuda:0, or cpu.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--person-confidence", type=float, default=0.25)
    parser.add_argument("--track-iou-threshold", type=float, default=0.25)
    parser.add_argument("--max-missed-frames", type=int, default=15)
    parser.add_argument("--draw-pose", action="store_true", help="Draw pose skeleton keypoints on the preview.")
    parser.add_argument("--keypoint-confidence", type=float, default=0.25)
    parser.add_argument("--draw-hands", action="store_true", help="Draw MediaPipe hand landmarks on the preview.")
    parser.add_argument("--hand-model", type=Path, default=here / "models" / "hand_landmarker.task")
    parser.add_argument("--hand-confidence", type=float, default=0.5)
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--no-display", action="store_true", help="Run without opening an OpenCV preview window.")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    return parser.parse_args()


def source_value(source: str):
    return int(source) if source.isdigit() else source


def normalize_device_arg(device_arg) -> str | None:
    return None if device_arg is None else str(device_arg)


def torch_device_from_arg(device_arg: str | None) -> torch.device:
    if device_arg and device_arg.startswith(("cuda", "cpu")):
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_action_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    classes = checkpoint["classes"]
    input_size = int(checkpoint.get("feature_size", config.get("feature_size", FEATURE_SIZE)))
    model = ActionLSTM(
        input_size=input_size,
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        num_classes=len(classes),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes, config, input_size


def predict_action(model: ActionLSTM, frames: deque[np.ndarray], device: torch.device, sequence_length: int) -> tuple[int, float]:
    sequence = np.stack(list(frames)).astype(np.float32)
    sequence = pad_or_trim_sequence(sequence, sequence_length)
    x = torch.from_numpy(sequence).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)[0]
    score, index = torch.max(probabilities, dim=0)
    return int(index.item()), float(score.item())


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def update_tracks(
    tracks: dict[int, PersonTrack],
    detections: list[dict[str, np.ndarray | float]],
    sequence_length: int,
    next_track_id: int,
    iou_threshold: float,
    max_missed_frames: int,
) -> int:
    unmatched_tracks = set(tracks)
    matched_tracks: set[int] = set()

    for detection in detections:
        box = detection["box"]
        feature = detection["feature"]
        best_track_id = None
        best_iou = 0.0
        for track_id in unmatched_tracks:
            overlap = iou(tracks[track_id].box, box)
            if overlap > best_iou:
                best_iou = overlap
                best_track_id = track_id

        if best_track_id is not None and best_iou >= iou_threshold:
            track = tracks[best_track_id]
            track.box = box
            track.frames.append(feature)
            track.missed = 0
            unmatched_tracks.remove(best_track_id)
            matched_tracks.add(best_track_id)
        else:
            tracks[next_track_id] = PersonTrack(
                track_id=next_track_id,
                box=box,
                frames=deque([feature], maxlen=sequence_length),
            )
            matched_tracks.add(next_track_id)
            next_track_id += 1

    for track_id in list(unmatched_tracks):
        track = tracks[track_id]
        track.missed += 1
        if track.missed > max_missed_frames:
            del tracks[track_id]

    return next_track_id


def draw_track(display: np.ndarray, track: PersonTrack) -> None:
    x1, y1, x2, y2 = track.box.astype(int)
    color = (0, 255, 0) if track.label != "uncertain" else (0, 220, 255)
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    text = f"ID {track.track_id} {track.label} {track.score:.2f}"
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    text_y = max(24, y1 - 8)
    cv2.rectangle(
        display,
        (x1, text_y - text_size[1] - baseline - 4),
        (x1 + text_size[0] + 8, text_y + baseline),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(display, text, (x1 + 4, text_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_pose(display: np.ndarray, result, confidence_threshold: float) -> None:
    if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return

    keypoints = result.keypoints.xy.detach().cpu().numpy()
    confidences = (
        result.keypoints.conf.detach().cpu().numpy()
        if result.keypoints.conf is not None
        else np.ones(keypoints.shape[:2], dtype=np.float32)
    )

    for person_points, person_confidences in zip(keypoints, confidences):
        visible = person_confidences >= confidence_threshold

        for start, end in COCO_SKELETON:
            if start >= len(person_points) or end >= len(person_points):
                continue
            if not (visible[start] and visible[end]):
                continue
            start_point = tuple(person_points[start].astype(int))
            end_point = tuple(person_points[end].astype(int))
            cv2.line(display, start_point, end_point, (255, 180, 0), 2, cv2.LINE_AA)

        for point, is_visible in zip(person_points, visible):
            if not is_visible:
                continue
            center = tuple(point.astype(int))
            cv2.circle(display, center, 4, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(display, center, 5, (40, 40, 40), thickness=1, lineType=cv2.LINE_AA)


def draw_hands(display: np.ndarray, hand_result) -> None:
    if hand_result is None or not hand_result.hand_landmarks:
        return

    height, width = display.shape[:2]
    handedness = hand_result.handedness or []

    for hand_index, hand_landmarks in enumerate(hand_result.hand_landmarks):
        points = [
            (int(landmark.x * width), int(landmark.y * height))
            for landmark in hand_landmarks
        ]

        for start, end in HAND_CONNECTIONS:
            cv2.line(display, points[start], points[end], (255, 80, 180), 2, cv2.LINE_AA)

        for point_index, point in enumerate(points):
            radius = 5 if point_index in {4, 8, 12, 16, 20} else 3
            cv2.circle(display, point, radius, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(display, point, radius + 1, (35, 35, 35), thickness=1, lineType=cv2.LINE_AA)

        if hand_index < len(handedness) and handedness[hand_index]:
            label = handedness[hand_index][0].category_name
            score = handedness[hand_index][0].score
            wrist_x, wrist_y = points[0]
            cv2.putText(
                display,
                f"{label} hand {score:.2f}",
                (wrist_x + 8, max(24, wrist_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 80, 180),
                2,
                cv2.LINE_AA,
            )


def main() -> None:
    args = parse_args()
    if not args.no_display and not has_display():
        raise SystemExit(
            "OpenCV in this Python environment was installed without GUI/window support, "
            "so cv2.imshow cannot run. Reinstall the GUI OpenCV package in the active venv, "
            "or run with --no-display."
        )

    yolo_device = normalize_device_arg(args.device)
    device = torch_device_from_arg(yolo_device)
    model, classes, config, checkpoint_feature_size = load_action_model(args.checkpoint, device)
    include_hands = bool(config.get("include_hands", checkpoint_feature_size > FEATURE_SIZE))
    pose_model_name = args.pose_model or config["pose_model"]
    imgsz = args.imgsz or int(config["imgsz"])
    sequence_length = int(config["sequence_length"])

    print(f"Loaded action model: {args.checkpoint}")
    print(f"Classes: {', '.join(classes)}")
    print(f"Feature input: {'body + hands' if include_hands else 'body only'} ({checkpoint_feature_size})")
    print(f"Loading pose model: {pose_model_name}")
    pose_model = YOLO(pose_model_name)
    hand_detector = (
        create_hand_detector(args.hand_model, args.max_hands, args.hand_confidence)
        if args.draw_hands or include_hands
        else None
    )

    cap = cv2.VideoCapture(source_value(args.source))
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    tracks: dict[int, PersonTrack] = {}
    next_track_id = 1
    last_hand_timestamp_ms = -1
    start_time = time.monotonic()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        predict_kwargs = {"imgsz": imgsz, "verbose": False}
        if yolo_device:
            predict_kwargs["device"] = yolo_device
        result = pose_model.predict(frame, **predict_kwargs)[0]
        hand_result = None
        if hand_detector is not None:
            hand_timestamp_ms = int((time.monotonic() - start_time) * 1000)
            if hand_timestamp_ms <= last_hand_timestamp_ms:
                hand_timestamp_ms = last_hand_timestamp_ms + 1
            last_hand_timestamp_ms = hand_timestamp_ms
            hand_result = detect_hands(hand_detector, frame, hand_timestamp_ms)
        detections = [
            person
            for person in result_to_person_features(
                result,
                hand_result=hand_result,
                frame_shape=frame.shape,
                include_hands=include_hands,
            )
            if float(person["confidence"]) >= args.person_confidence
        ]
        next_track_id = update_tracks(
            tracks=tracks,
            detections=detections,
            sequence_length=sequence_length,
            next_track_id=next_track_id,
            iou_threshold=args.track_iou_threshold,
            max_missed_frames=args.max_missed_frames,
        )

        for track in tracks.values():
            if len(track.frames) == sequence_length:
                class_index, score = predict_action(model, track.frames, device, sequence_length)
                track.score = score
                track.label = classes[class_index] if score >= args.confidence_threshold else "uncertain"

        display = frame.copy()
        if args.draw_pose:
            draw_pose(display, result, args.keypoint_confidence)
        if args.draw_hands:
            draw_hands(display, hand_result)
        for track in tracks.values():
            if track.missed == 0:
                draw_track(display, track)
        if not args.no_display:
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break

    cap.release()
    if hand_detector is not None:
        hand_detector.close()
    close_windows()


if __name__ == "__main__":
    main()
