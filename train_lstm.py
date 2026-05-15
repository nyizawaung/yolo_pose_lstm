from __future__ import annotations

import argparse
import json
import re
import random
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch
from torch import nn
from ultralytics import YOLO

try:
    from .model import ActionLSTM
    from .pose_features import (
        FEATURE_SIZE,
        dataset_classes,
        extract_video_features,
        feature_size,
        iter_batches,
        list_action_videos,
        load_feature_cache,
        make_windows,
        save_feature_cache,
        video_cache_path,
    )
    from .hand_features import create_hand_detector
except ImportError:
    from model import ActionLSTM
    from pose_features import (
        FEATURE_SIZE,
        dataset_classes,
        extract_video_features,
        feature_size,
        iter_batches,
        list_action_videos,
        load_feature_cache,
        make_windows,
        save_feature_cache,
        video_cache_path,
    )
    from hand_features import create_hand_detector


@dataclass
class TrainConfig:
    dataset_dir: str
    cache_dir: str
    checkpoint: str
    split_file: str
    pose_model: str
    imgsz: int
    frame_stride: int
    include_hands: bool
    hand_model: str
    hand_confidence: float
    max_hands: int
    feature_size: int
    sequence_length: int
    window_stride: int
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    epochs: int
    lr: float
    val_split: float
    seed: int


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train an LSTM action classifier from YOLOv8 pose keypoints.")
    parser.add_argument("--dataset-dir", type=Path, default=here / "action_dataset")
    parser.add_argument("--cache-dir", type=Path, default=here / "cache" / "pose_features")
    parser.add_argument("--checkpoint", type=Path, default=here / "checkpoints" / "action_lstm.pt")
    parser.add_argument("--split-file", type=Path, default=here / "splits" / "train_val_split.json")
    parser.add_argument("--pose-model", default=str(here / "yolov8m-pose.pt"))
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every Nth video frame for pose extraction.")
    parser.add_argument("--include-hands", action="store_true", help="Append MediaPipe hand landmarks to each LSTM frame.")
    parser.add_argument("--hand-model", type=Path, default=here / "models" / "hand_landmarker.task")
    parser.add_argument("--hand-confidence", type=float, default=0.5)
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--window-stride", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--device", default=0, help="Ultralytics/PyTorch device, for example 0, cuda:0, or cpu.")
    return parser.parse_args()


def subject_id(video_path: Path) -> str:
    match = re.search(r"(person\d+)", video_path.stem, flags=re.IGNORECASE)
    return match.group(1).lower() if match else video_path.stem


def split_videos(videos: list[tuple[Path, str]], val_split: float, seed: int) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    rng = random.Random(seed)
    train: list[tuple[Path, str]] = []
    val: list[tuple[Path, str]] = []
    by_class: dict[str, list[Path]] = {}
    for path, label in videos:
        by_class.setdefault(label, []).append(path)

    for label, paths in sorted(by_class.items()):
        paths = paths[:]
        by_subject: dict[str, list[Path]] = {}
        for path in paths:
            by_subject.setdefault(subject_id(path), []).append(path)

        if len(by_subject) > 1:
            subjects = sorted(by_subject)
            rng.shuffle(subjects)
            val_count = max(1, int(round(len(subjects) * val_split)))
            val_subjects = set(subjects[:val_count])
            val_paths = {path for subject in val_subjects for path in by_subject[subject]}
        else:
            rng.shuffle(paths)
            val_count = max(1, int(round(len(paths) * val_split)))
            val_paths = set(paths[:val_count])

        for path in paths:
            target = val if path in val_paths else train
            target.append((path, label))
    return train, val


def load_or_extract_features(
    videos: list[tuple[Path, str]],
    pose_model,
    args: argparse.Namespace,
) -> dict[Path, np.ndarray]:
    features_by_video: dict[Path, np.ndarray] = {}
    expected_feature_size = feature_size(args.include_hands)
    hand_detector = (
        create_hand_detector(args.hand_model, args.max_hands, args.hand_confidence)
        if args.include_hands
        else None
    )
    try:
        timestamp_offset_ms = 0
        for index, (video_path, label) in enumerate(videos, start=1):
            cache_path = video_cache_path(video_path, args.cache_dir)
            if cache_path.exists() and not args.rebuild_cache:
                features, meta = load_feature_cache(cache_path)
                cache_issue = cache_mismatch_reason(features, meta, args, expected_feature_size)
                if cache_issue:
                    print(
                        f"[{index:03d}/{len(videos):03d}] rebuilding {video_path.name}: "
                        f"{cache_issue}"
                    )
                    features, meta = extract_one_video(video_path, pose_model, args, hand_detector, timestamp_offset_ms)
                    timestamp_offset_ms += video_timestamp_span_ms(meta)
                    meta["label"] = label
                    save_feature_cache(cache_path, features, meta)
            else:
                print(f"[{index:03d}/{len(videos):03d}] extracting {label}: {video_path.name}")
                features, meta = extract_one_video(video_path, pose_model, args, hand_detector, timestamp_offset_ms)
                timestamp_offset_ms += video_timestamp_span_ms(meta)
                meta["label"] = label
                save_feature_cache(cache_path, features, meta)
            features_by_video[video_path] = features
    finally:
        if hand_detector is not None:
            hand_detector.close()
    return features_by_video


def video_timestamp_span_ms(meta: dict) -> int:
    fps = float(meta.get("fps", 0.0) or 0.0)
    total_frames = int(meta.get("total_frames", 0) or 0)
    if fps > 0 and total_frames > 0:
        return int(total_frames * 1000 / fps) + 1000
    return int(meta.get("sampled_frames", 0) or 0) + 1000


def cache_mismatch_reason(
    features: np.ndarray,
    meta: dict,
    args: argparse.Namespace,
    expected_feature_size: int,
) -> str | None:
    if features.shape[1] != expected_feature_size:
        return f"cached feature size {features.shape[1]} != {expected_feature_size}"
    if bool(meta.get("include_hands", False)) != bool(args.include_hands):
        return "cached hand setting does not match"
    if int(meta.get("frame_stride", 0)) != max(1, args.frame_stride):
        return "cached frame stride does not match"
    if int(meta.get("imgsz", 0)) != int(args.imgsz):
        return "cached image size does not match"
    if args.include_hands:
        if float(meta.get("hand_confidence", -1.0)) != float(args.hand_confidence):
            return "cached hand confidence does not match"
        if int(meta.get("max_hands", 0)) != int(args.max_hands):
            return "cached max hands does not match"
    return None


def extract_one_video(
    video_path: Path,
    pose_model,
    args: argparse.Namespace,
    hand_detector,
    timestamp_offset_ms: int = 0,
) -> tuple[np.ndarray, dict]:
    features, meta = extract_video_features(
        video_path=video_path,
        pose_model=pose_model,
        imgsz=args.imgsz,
        frame_stride=max(1, args.frame_stride),
        device=args.device,
        hand_detector=hand_detector,
        include_hands=args.include_hands,
        timestamp_offset_ms=timestamp_offset_ms,
    )
    meta["hand_confidence"] = args.hand_confidence
    meta["max_hands"] = args.max_hands
    meta["hand_model"] = str(args.hand_model)
    return features, meta


def build_samples(
    videos: list[tuple[Path, str]],
    features_by_video: dict[Path, np.ndarray],
    class_to_index: dict[str, int],
    sequence_length: int,
    window_stride: int,
) -> list[tuple[np.ndarray, int]]:
    samples: list[tuple[np.ndarray, int]] = []
    for video_path, label in videos:
        for window in make_windows(features_by_video[video_path], sequence_length, window_stride):
            samples.append((window, class_to_index[label]))
    return samples


def count_by_class(videos: list[tuple[Path, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, label in videos:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def save_split_file(
    split_file: Path,
    dataset_dir: Path,
    train_videos: list[tuple[Path, str]],
    val_videos: list[tuple[Path, str]],
    seed: int,
    val_split: float,
) -> None:
    split_file.parent.mkdir(parents=True, exist_ok=True)

    def item(video: tuple[Path, str]) -> dict[str, str]:
        path, label = video
        return {
            "path": str(path.relative_to(dataset_dir.parent)),
            "label": label,
            "subject": subject_id(path),
        }

    split_file.write_text(
        json.dumps(
            {
                "seed": seed,
                "val_split": val_split,
                "dataset_dir": str(dataset_dir),
                "train_count": len(train_videos),
                "val_count": len(val_videos),
                "train_by_class": count_by_class(train_videos),
                "val_by_class": count_by_class(val_videos),
                "train": [item(video) for video in sorted(train_videos)],
                "val": [item(video) for video in sorted(val_videos)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def run_epoch(
    model: nn.Module,
    samples: list[tuple[np.ndarray, int]],
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0.0
    total_items = 0

    with torch.set_grad_enabled(training):
        for x_np, y_np in iter_batches(samples, batch_size, shuffle=training, rng=rng):
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            batch_size_actual = y.shape[0]
            total_loss += float(loss.item()) * batch_size_actual
            total_correct += accuracy(logits, y) * batch_size_actual
            total_items += batch_size_actual

    return total_loss / max(1, total_items), total_correct / max(1, total_items)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    classes = dataset_classes(args.dataset_dir)
    videos = list_action_videos(args.dataset_dir)
    if not videos:
        raise SystemExit(f"No videos found in {args.dataset_dir}")

    print(f"Found {len(videos)} videos across {len(classes)} classes: {', '.join(classes)}")
    print(f"Loading pose model: {args.pose_model}")
    pose_model = YOLO(args.pose_model)
    expected_feature_size = feature_size(args.include_hands)
    print(f"Feature size: {expected_feature_size} ({'body + hands' if args.include_hands else 'body only'})")
    features_by_video = load_or_extract_features(videos, pose_model, args)

    train_videos, val_videos = split_videos(videos, args.val_split, args.seed)
    save_split_file(args.split_file, args.dataset_dir, train_videos, val_videos, args.seed, args.val_split)
    print(f"Train videos: {len(train_videos)}  Validation videos: {len(val_videos)}")
    print(f"Train split by class: {count_by_class(train_videos)}")
    print(f"Validation split by class: {count_by_class(val_videos)}")
    print(f"Saved split: {args.split_file}")

    class_to_index = {name: index for index, name in enumerate(classes)}
    train_samples = build_samples(train_videos, features_by_video, class_to_index, args.sequence_length, args.window_stride)
    val_samples = build_samples(val_videos, features_by_video, class_to_index, args.sequence_length, args.window_stride)
    print(f"Train windows: {len(train_samples)}  Validation windows: {len(val_samples)}")

    train_device = torch.device(args.device if args.device and str(args.device).startswith(("cuda", "cpu")) else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ActionLSTM(
        input_size=expected_feature_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=len(classes),
        dropout=args.dropout,
    ).to(train_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = -1.0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        dataset_dir=str(args.dataset_dir),
        cache_dir=str(args.cache_dir),
        checkpoint=str(args.checkpoint),
        split_file=str(args.split_file),
        pose_model=args.pose_model,
        imgsz=args.imgsz,
        frame_stride=args.frame_stride,
        include_hands=args.include_hands,
        hand_model=str(args.hand_model),
        hand_confidence=args.hand_confidence,
        max_hands=args.max_hands,
        feature_size=expected_feature_size,
        sequence_length=args.sequence_length,
        window_stride=args.window_stride,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_samples, optimizer, criterion, args.batch_size, train_device, np_rng)
        val_loss, val_acc = run_epoch(model, val_samples, None, criterion, args.batch_size, train_device, np_rng)
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "config": asdict(config),
                    "feature_size": expected_feature_size,
                    "best_val_acc": best_val_acc,
                },
                args.checkpoint,
            )

    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Saved checkpoint: {args.checkpoint}")
    args.checkpoint.with_suffix(".json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
