# handMouse

Control your computer with your hand. A webcam, MediaPipe's gesture recognizer and a few lines of geometry turn finger positions into pointer movement and clicks.

> Part of a series of experiments on making computers understand hands and sound. See also [drag-kit](https://github.com/CarloFanelli/drag-kit) and the soundBoard.

## How it works

1. OpenCV grabs frames from the webcam.
2. MediaPipe's **Gesture Recognizer** (`gesture_recognizer.task`) returns hand landmarks and a named gesture (`Pointing_Up`, `Open_Palm`, …) with a confidence score.
3. The script tracks a few key landmarks (index tip, thumb tip, middle tip) and derives actions from their distances:
   - **Pointing_Up** gesture → the hand is "armed" and the index fingertip drives the pointer.
   - **Thumb + middle finger pinch** → click.
   - **Thumb + index pinch** → detected and drawn on screen (reserved for drag / secondary actions).
4. A control square is drawn on the video feed: moving your hand inside it maps to the whole screen, so you don't need big arm movements.

Landmarks, the pinch line and the current gesture with its score are rendered on the mirrored video feed for feedback.

## Requirements

- Python 3.10+
- A webcam

```bash
pip install opencv-python mediapipe pyautogui numpy
```

## Run

```bash
git clone https://github.com/CarloFanelli/handMouse.git
cd handMouse
python hand_recognition.py
```

Press `Ctrl+C` in the terminal to quit.

## Tuning

Two thresholds in `hand_recognition.py` control how sensitive the pinch detection is:

| Constant | Default | Meaning |
|---|---|---|
| `threshold` in `checkPinch()` | `0.06` | thumb–index distance (normalized) below which a pinch is detected |
| `threshold` in `click()` | `0.045` | thumb–middle distance below which a click fires |

Lower values require a tighter pinch; raise them if clicks are hard to trigger.

## Status

Prototype. Gesture and pinch detection run in real time; the `pyautogui.moveTo()` / `pyautogui.click()` calls are currently commented out in the main loop so you can watch the detection without your mouse jumping around. Uncomment them to take control of the pointer.

## Stack

Python · OpenCV · MediaPipe Tasks (Gesture Recognizer) · PyAutoGUI · NumPy
