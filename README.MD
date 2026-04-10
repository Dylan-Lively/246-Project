# Trace

Gesture-controlled 3-DOF robotic arm. A Raspberry Pi tracks your hand with MediaPipe and streams position packets over serial to an Arduino, which solves inverse kinematics and drives three stepper motors in real time.

---

## How it works

The Pi captures frames from the camera module, runs MediaPipe's `GestureRecognizer`, estimates hand depth from palm geometry, and sends a compact serial packet at up to 20 Hz. The gesture determines the control mode:

| Gesture | Mode | Behaviour |
|---|---|---|
| Victory | Relative | Accumulates hand deltas into tip position |
| Open palm | Absolute | Snaps tip directly to hand position |
| ILY | Mirror | Absolute with axes negated |
| Anything else | Hold | Arm stays put |

Serial format: `N:+0.023,+0.441,+0.318`

---

## Hardware

- Raspberry Pi (tested on Bookworm) + Pi camera module
- Arduino + 3× stepper drivers (A4988 or similar)
- 3-DOF stepper arm — L1 240 mm, L2 230 mm
- USB-serial connection between Pi and Arduino

---

## Setup

**On the Pi:**

```bash
sudo apt install -y python3-picamera2
pip install mediapipe opencv-python numpy pyserial --break-system-packages
```

Download the MediaPipe gesture model:

```bash
wget -O gesture_recognizer.task \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

**On the Arduino:**

Install [AccelStepper](https://www.airspayce.com/mikem/arduino/AccelStepper/) via Library Manager, then flash `arm_controller.ino`. The arm must be in the L position (shoulder up, forearm right) at power-on.

---

## Usage

```bash
# With arm connected
python hand_tracker_pi.py --port /dev/ttyUSB0

# Camera test, no serial
python hand_tracker_pi.py --no-serial
```

---

## Tuning

| Constant | Default | Effect |
|---|---|---|
| `SMOOTH_ALPHA` | `0.25` | EMA smoothing — lower = smoother, laggier |
| `DEADBAND` | `0.005` | Minimum movement before a packet is sent |
| `SEND_HZ` | `20` | Max packet rate |
| `CAM_FOV_H_DEG` | `79` | Horizontal FOV of your camera module |

Adjust link lengths, gear ratios, and joint limits in `arm_controller.ino` to match your build.