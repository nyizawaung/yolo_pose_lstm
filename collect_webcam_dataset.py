from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2

try:
    from prompt_toolkit.shortcuts import input_dialog, message_dialog, radiolist_dialog
except ImportError:
    input_dialog = None
    message_dialog = None
    radiolist_dialog = None


HERE = Path(__file__).resolve().parent
ACTIONS_FILE = HERE / "actions.txt"
DEFAULT_DATASET_DIR = HERE / "webcam_action_dataset"
CLIP_SECONDS = 4.0
DEFAULT_FPS = 20.0
WINDOW_NAME = "Webcam Dataset Collector"
_HAS_DISPLAY: bool | None = None


@dataclass(frozen=True)
class CaptureSettings:
    actions: list[str]
    output_dir: Path
    videos_per_action: int
    wait_seconds: float
    camera_source: str
    width: int
    height: int


def load_actions(actions_path: Path = ACTIONS_FILE) -> list[str]:
    if actions_path.exists():
        actions = [
            line.strip()
            for line in actions_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if actions:
            return unique_in_order(actions)

    if DEFAULT_DATASET_DIR.exists():
        fallback_actions = [
            path.name
            for path in DEFAULT_DATASET_DIR.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
        return sorted(fallback_actions)

    return []


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("._") or "action"


def source_value(source: str) -> int | str:
    source = source.strip()
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


def show_message(title: str, text: str) -> None:
    if message_dialog is not None:
        message_dialog(title=title, text=text).run()
    else:
        print(f"\n{title}\n{text}\n")


def prompt_text(title: str, text: str, default: str) -> str | None:
    if input_dialog is not None:
        return input_dialog(title=title, text=text, default=default).run()
    value = input(f"{text} [{default}]: ").strip()
    return value or default


def prompt_number(title: str, text: str, default: str, cast, minimum: float):
    while True:
        value = prompt_text(title, text, default)
        if value is None:
            return None
        try:
            parsed = cast(value)
        except ValueError:
            show_message("Invalid value", f"Enter a number for: {text}")
            continue
        if parsed < minimum:
            show_message("Invalid value", f"Value must be at least {minimum}.")
            continue
        return parsed


def choose_actions(actions: list[str]) -> list[str] | None:
    all_actions_value = "__all_actions__"
    if radiolist_dialog is not None:
        selected = radiolist_dialog(
            title="Action",
            text="Choose the action to collect, or collect every action.",
            values=[(all_actions_value, "All actions")] + [(action, action) for action in actions],
            default=actions[0],
        ).run()
        if selected is None:
            return None
        if selected == all_actions_value:
            return actions
        return [str(selected)]

    print("\nChoose action:")
    print("0. All actions")
    for index, action in enumerate(actions, start=1):
        print(f"{index}. {action}")
    while True:
        raw = input("Selection [1]: ").strip() or "1"
        try:
            selected_index = int(raw)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if selected_index == 0:
            return actions
        if 1 <= selected_index <= len(actions):
            return [actions[selected_index - 1]]
        print("Selection out of range.")


def prompt_settings() -> CaptureSettings | None:
    actions = load_actions()
    if not actions:
        show_message("No actions found", "Add one action name per line to actions.txt, then run this collector again.")
        return None

    selected_actions = choose_actions(actions)
    if not selected_actions:
        return None

    default_output = str(HERE / "webcam_action_dataset")
    output_text = prompt_text("Output folder", "Where should the videos be saved?", default_output)
    if output_text is None or not output_text.strip():
        return None

    videos_per_action = prompt_number("Videos per action", "How many videos per action?", "1", int, 1)
    if videos_per_action is None:
        return None

    wait_seconds = prompt_number("Wait time", "Seconds to wait before each clip?", "2.0", float, 0)
    if wait_seconds is None:
        return None

    camera_source = prompt_text("Camera source", "Webcam index or RTSP URL?", "0")
    if camera_source is None or not camera_source.strip():
        return None

    width = prompt_number("Width", "Capture width?", "640", int, 1)
    if width is None:
        return None

    height = prompt_number("Height", "Capture height?", "480", int, 1)
    if height is None:
        return None

    return CaptureSettings(
        actions=selected_actions,
        output_dir=Path(output_text.strip()).expanduser(),
        videos_per_action=videos_per_action,
        wait_seconds=wait_seconds,
        camera_source=camera_source.strip(),
        width=width,
        height=height,
    )


class CaptureCancelled(Exception):
    pass


def open_camera(settings: CaptureSettings) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source_value(settings.camera_source))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {settings.camera_source}")
    return cap


def camera_fps(cap: cv2.VideoCapture) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1.0 or fps > 120.0:
        return DEFAULT_FPS
    return fps


def next_output_path(output_dir: Path, action: str) -> Path:
    action_dir = output_dir / safe_name(action)
    action_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = safe_name(action)
    index = 1
    while True:
        candidate = action_dir / f"{base}_{timestamp}_{index:03d}.avi"
        if not candidate.exists():
            return candidate
        index += 1


def make_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if writer.isOpened():
        return writer

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if writer.isOpened():
        return writer

    raise RuntimeError(f"Could not create video writer for {path}.")


def draw_status(frame, lines: list[str], color: tuple[int, int, int]) -> None:
    y = 30
    for line in lines:
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        y += 32


def preview_wait(
    cap: cv2.VideoCapture,
    action: str,
    clip_index: int,
    total_clips: int,
    wait_seconds: float,
) -> None:
    start = time.monotonic()
    display = has_display()
    last_printed_second = None
    print(f"Action: {action} | clip {clip_index}/{total_clips} | waiting {wait_seconds:0.1f}s")

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read from camera during countdown.")

        remaining = max(0.0, wait_seconds - (time.monotonic() - start))
        if display:
            draw_status(
                frame,
                [
                    f"Action: {action}",
                    f"Clip {clip_index}/{total_clips}",
                    f"Recording starts in {remaining:0.1f}s",
                    "Press q or Esc to cancel",
                ],
                (0, 255, 255),
            )
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                raise CaptureCancelled
        else:
            printed_second = int(remaining)
            if printed_second != last_printed_second:
                print(f"Recording starts in {remaining:0.1f}s")
                last_printed_second = printed_second

        if remaining <= 0:
            break


def record_clip(
    cap: cv2.VideoCapture,
    output_path: Path,
    fps: float,
    action: str,
    clip_index: int,
    total_clips: int,
) -> int:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read from camera before recording.")

    height, width = frame.shape[:2]
    writer = make_writer(output_path, fps, (width, height))
    frames_written = 0
    start = time.monotonic()
    display = has_display()
    last_printed_second = None
    print(f"Recording {output_path}")

    try:
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= CLIP_SECONDS:
                break

            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read from camera during recording.")

            writer.write(frame)
            frames_written += 1

            remaining = max(0.0, CLIP_SECONDS - elapsed)
            if display:
                preview = frame.copy()
                draw_status(
                    preview,
                    [
                        f"REC Action: {action}",
                        f"Clip {clip_index}/{total_clips}",
                        f"{remaining:0.1f}s remaining",
                        str(output_path.name),
                    ],
                    (0, 0, 255),
                )
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    raise CaptureCancelled
            else:
                printed_second = int(remaining)
                if printed_second != last_printed_second:
                    print(f"{remaining:0.1f}s remaining")
                    last_printed_second = printed_second
    finally:
        writer.release()

    return frames_written


def capture_dataset(settings: CaptureSettings) -> None:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    cap = open_camera(settings)
    fps = camera_fps(cap)
    total_clips = len(settings.actions) * settings.videos_per_action
    clip_number = 1

    try:
        for action in settings.actions:
            for _ in range(settings.videos_per_action):
                output_path = next_output_path(settings.output_dir, action)
                preview_wait(cap, action, clip_number, total_clips, settings.wait_seconds)
                frames_written = record_clip(cap, output_path, fps, action, clip_number, total_clips)
                if frames_written == 0:
                    output_path.unlink(missing_ok=True)
                    raise RuntimeError(f"No frames were written for {output_path}.")
                clip_number += 1
    finally:
        cap.release()


def main() -> None:
    settings = prompt_settings()
    if settings is None:
        print("Capture cancelled.")
        return

    try:
        capture_dataset(settings)
    except CaptureCancelled:
        print("Capture cancelled before all clips were recorded.")
    else:
        print(f"Saved clips to: {settings.output_dir}")
    finally:
        close_windows()


if __name__ == "__main__":
    main()
