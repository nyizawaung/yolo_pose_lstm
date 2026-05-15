# YOLOv8 Pose + LSTM Action Recognition

This folder contains a small action-recognition project for the dataset in `action_dataset`.
It extracts human pose keypoints with YOLOv8 pose, caches those keypoints, trains a PyTorch LSTM, then runs live predictions from a webcam.

## Dataset

Detected classes: (can create easily by collect_webcam_dataset.py or other video annotator)

- `boxing`
- `handclapping`
- `handwaving`
- `jogging`
- `running`
- `walking`

The scripts expect this layout:

```text
Action_LSTM/
  action_dataset/
    boxing/*.avi
    handclapping/*.avi
    handwaving/*.avi
    jogging/*.avi
    running/*.avi
    walking/*.avi
```

## Train

From the repo root:

```bash
python Action_LSTM/train_lstm.py
```

The first run will download `Action_LSTM/yolov8n-pose.pt` if it is not already available, extract pose keypoints from all videos, and cache them in:

```text
Action_LSTM/cache/pose_features/
```

The best LSTM checkpoint is saved to:

```text
Action_LSTM/checkpoints/action_lstm.pt
```

Useful options:

```bash
python Action_LSTM/train_lstm.py --epochs 50 --sequence-length 30 --frame-stride 2
python Action_LSTM/train_lstm.py --epochs 50 --include-hands --rebuild-cache --checkpoint Action_LSTM/checkpoints/webcam_action_lstm.pt
python Action_LSTM/train_lstm.py --device cuda:0
python Action_LSTM/train_lstm.py --rebuild-cache
```

## Webcam Inference

After training:

```bash
python Action_LSTM/webcam_action.py --source 0 --draw-pose --draw-hands
```

Press `q` or `Esc` to quit.

The webcam runner tracks each detected person separately. Each person gets an independent LSTM pose window and an action label drawn above their own bounding box.

Useful options:

```bash
python Action_LSTM/webcam_action.py --source 0 --confidence-threshold 0.55
python Action_LSTM/webcam_action.py --source rtsp://user:pass@host:554/stream --device cuda:0
python Action_LSTM/webcam_action.py --source 0 --person-confidence 0.35 --track-iou-threshold 0.3
python Action_LSTM/webcam_action.py --source 0 --draw-pose --draw-hands --hand-confidence 0.35
```

## AR Hand Ball

Play with a digital ball using MediaPipe hand landmarks:

```bash
python Action_LSTM/ar_hand_ball.py --source 0
```

Pinch thumb and index finger near the ball to pick it up with one hand, then open your hand or flick upward to toss it. Use two hands to resize and rotate the ball. Press `r` to reset.

## AR Hand Helicopter

Fly a digital helicopter with your right index finger as a joystick and fire smoky rockets at a moving target by making a tight left-hand fist:

```bash
python Action_LSTM/ar_hand_helicopter.py --source 0
```

Point the right index finger up/down/left/right to fly in that direction. The target relocates after each hit. The overlay shows elapsed time, hits, and misses. If the hands feel reversed, add `--swap-hands`. Press `r` to reset.

Useful options:

```bash
python Action_LSTM/ar_hand_helicopter.py --source 0 --rocket-speed 420 --rocket-cooldown 0.25
python Action_LSTM/ar_hand_helicopter.py --source 0 --target-radius 44
python Action_LSTM/ar_hand_helicopter.py --source 0 --joystick-speed 520 --fist-strictness 0.62
python Action_LSTM/ar_hand_helicopter.py --source 0 --two-player
```

In two-player mode, there is no target. The right detected hand controls/fires player 1, the left detected hand controls/fires player 2, and the HUD only shows each player's hit count.

## AR Keyboard

Type into the focused desktop application with a virtual keyboard controlled by hand gestures:

```bash
python Action_LSTM/ar_keyboard.py --source 0
```

Pinch thumb and index finger over a key to type it. Pinch with both hands and move them apart/together to resize the keyboard. Make two fists to move the keyboard. Use `--dry-run` to test without sending physical keyboard input.

## Notes

- The train/validation split keeps `personXX` subjects separate when the filename includes a subject id, then falls back to video-level splitting for tiny ad hoc datasets.
- The body-only feature vector is `17 keypoints x (x, y, confidence) = 51` values per sampled frame.
- Training with `--include-hands` appends MediaPipe hand landmarks: `2 hands x 21 landmarks x (x, y, z, confidence)`, for `219` values per sampled frame.
- YOLO pose uses normalized keypoint coordinates, so the LSTM sees pose movement rather than raw pixels.
