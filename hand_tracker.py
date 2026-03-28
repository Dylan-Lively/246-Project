"""
Hand Tracker → Arduino Serial
==============================
Stripped from the window-control version.
Sends filtered XYZ to Arduino at ~20Hz.

Serial format:  X:0.623,Y:0.441,Z:0.318\n

Install:  pip install mediapipe opencv-python numpy pyserial
Run:      python hand_tracker.py --port COM4
          python hand_tracker.py --port /dev/ttyUSB0
          python hand_tracker.py --no-serial     (debug, no Arduino needed)
"""

import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import argparse
from collections import deque

# ── MediaPipe setup ────────────────────────────────────────────────────────────
BaseOptions              = mp.tasks.BaseOptions
GestureRecognizer        = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode        = mp.tasks.vision.RunningMode

MODEL_PATH = "C:/Users/livel/VScode/246 Project/HandTracking/gesture_recognizer.task"

# ── Landmark indices (kept from your original) ─────────────────────────────────
WRIST      = 0
INDEX_MCP  = 5
MIDDLE_MCP = 9
RING_MCP   = 13
PINKY_MCP  = 17
PALM_PTS   = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

PALM_PAIRS = [
    (WRIST,     MIDDLE_MCP, 0.090),
    (INDEX_MCP, PINKY_MCP,  0.075),
    (WRIST,     INDEX_MCP,  0.065),
    (WRIST,     PINKY_MCP,  0.065),
]

# ── Config ─────────────────────────────────────────────────────────────────────
SERIAL_BAUD  = 115200
SEND_HZ      = 20

# Deadband: minimum change before sending an update.
# Stops arm tremor when your hand is still. Tune 0.01–0.05.
DEADBAND     = 0.02

# EMA alpha. Lower = smoother, laggier. 0.1–0.4.
SMOOTH_ALPHA = 0.25

# Z working range in metres.
# Hand at Z_NEAR → Z output 0.0 (closest = fully retracted)
# Hand at Z_FAR  → Z output 1.0 (furthest = fully extended)
# Tune to your actual working distance from the camera.
Z_NEAR = 0.25
Z_FAR  = 0.80

font = cv2.FONT_HERSHEY_SIMPLEX


# ── Depth estimation (unchanged from your original) ───────────────────────────
def estimate_depth(lm, img_w, img_h):
    focal = img_w / (2 * np.tan(np.radians(35)))
    depths = []
    for a, b, real_m in PALM_PAIRS:
        px = np.hypot((lm[a].x - lm[b].x) * img_w,
                      (lm[a].y - lm[b].y) * img_h)
        if px > 2:
            depths.append((focal * real_m) / px)
    if not depths:
        return None
    med  = np.median(depths)
    good = [d for d in depths if abs(d - med) / med < 0.30]
    return float(np.mean(good)) if good else med


# ── Palm centre normalised 0.0–1.0 ────────────────────────────────────────────
def palm_center_norm(lm, img_w, img_h):
    pts = np.array([[lm[i].x * img_w, lm[i].y * img_h] for i in PALM_PTS])
    c   = pts.mean(axis=0)
    return float(c[0] / img_w), float(c[1] / img_h)


# ── Simple EMA ────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None

    def update(self, v):
        if v is None:
            return self.value
        if self.value is None:
            self.value = v
        else:
            self.value = self.alpha * v + (1.0 - self.alpha) * self.value
        return self.value


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",      default=None)
    parser.add_argument("--no-serial", action="store_true")
    args = parser.parse_args()

    ser = None
    if not args.no_serial:
        if args.port is None:
            print("Provide --port or use --no-serial"); return
        ser = serial.Serial(args.port, SERIAL_BAUD, timeout=1)
        time.sleep(2)
        print(f"Serial open on {args.port}")
    else:
        print("No-serial debug mode — values printed to console")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,                          # only need one hand
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    ema_x     = EMA(SMOOTH_ALPHA)
    ema_y     = EMA(SMOOTH_ALPHA)
    ema_z     = EMA(SMOOTH_ALPHA)
    depth_buf = deque(maxlen=8)

    last_sent      = None
    last_send_time = 0.0
    send_interval  = 1.0 / SEND_HZ

    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            now   = time.time()

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize_for_video(mp_img, int(now * 1000))

            hand_visible = bool(result.hand_landmarks)

            if hand_visible:
                lm = result.hand_landmarks[0]

                # X and Y: normalised 0–1 across the camera frame
                raw_x, raw_y = palm_center_norm(lm, w, h)

                # Z: smooth depth then normalise to 0–1 within working range
                raw_depth = estimate_depth(lm, w, h)
                if raw_depth is not None:
                    depth_buf.append(raw_depth)

                smoothed_depth = None
                if depth_buf:
                    vals = list(depth_buf)
                    med  = np.median(vals)
                    good = [v for v in vals if abs(v - med) < 0.15]
                    smoothed_depth = float(np.mean(good)) if good else med

                raw_z = None
                if smoothed_depth is not None:
                    raw_z = (smoothed_depth - Z_NEAR) / (Z_FAR - Z_NEAR)
                    raw_z = max(0.0, min(1.0, raw_z))

                # Smooth all three axes
                sx = ema_x.update(raw_x)
                sy = ema_y.update(raw_y)
                sz = ema_z.update(raw_z)

                if None not in (sx, sy, sz):
                    if (now - last_send_time) >= send_interval:

                        # Deadband: only send if something moved enough
                        send = True
                        if last_sent is not None:
                            if (abs(sx - last_sent[0]) < DEADBAND and
                                abs(sy - last_sent[1]) < DEADBAND and
                                abs(sz - last_sent[2]) < DEADBAND):
                                send = False

                        if send:
                            packet = f"X:{sx:.3f},Y:{sy:.3f},Z:{sz:.3f}\n"
                            if ser:
                                ser.write(packet.encode())
                            else:
                                print(packet, end="")
                            last_sent      = (sx, sy, sz)
                            last_send_time = now

                # Overlay
                depth_str = f"{smoothed_depth:.2f}m" if smoothed_depth else "--"
                cv2.putText(frame,
                    f"X:{sx:.3f}  Y:{sy:.3f}  Z:{sz:.3f}  depth:{depth_str}",
                    (16, 40), font, 0.7, (0, 255, 180), 2)

                # Skeleton
                for a, b in mp.solutions.hands.HAND_CONNECTIONS:
                    ax, ay = int(lm[a].x * w), int(lm[a].y * h)
                    bx, by = int(lm[b].x * w), int(lm[b].y * h)
                    cv2.line(frame, (ax, ay), (bx, by), (0, 210, 210), 2)

            else:
                cv2.putText(frame, "No hand", (16, 40), font, 0.7, (80, 80, 80), 2)
                depth_buf.clear()
                # Don't clear EMA — if hand briefly disappears and returns,
                # smoothers resume from last known position rather than snapping.

            cv2.imshow("Hand Tracker", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cap.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()