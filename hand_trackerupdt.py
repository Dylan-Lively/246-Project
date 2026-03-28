"""
Hand Tracker → Arduino Serial
==============================
Detects gesture and sends prefix + arm-space XYZ to Arduino.

Gesture mapping:
    Closed_Fist  → N  (relative delta tracking)
    Victory      → P  (absolute IK snap)
    anything else → F  (hold)

Coordinate convention (converted to arm space before sending):
    Arm X = hand left/right       (MediaPipe X, same direction)
    Arm Y = hand forward/back     (MediaPipe Z flipped: closer = more negative MP Z)
    Arm Z = hand up/down          (MediaPipe Y flipped: MP Y=0 is top of frame)

Serial format:  "N:0.623,0.441,0.318\n"
                 ^ prefix

Install:  pip install mediapipe opencv-python numpy pyserial
Run:      python hand_trackerupdt.py --port COM4
          python hand_trackerupdt.py --no-serial
"""

import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import argparse
from collections import deque, Counter

# ── MediaPipe setup ─────────────────────────────────────────────────────────────
BaseOptions              = mp.tasks.BaseOptions
GestureRecognizer        = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode        = mp.tasks.vision.RunningMode

MODEL_PATH = "C:/Users/livel/VScode/246 Project/HandTracking/gesture_recognizer.task"

# ── Landmark indices ─────────────────────────────────────────────────────────────
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

# ── Gesture → serial prefix ──────────────────────────────────────────────────────
# Anything not listed here sends F (hold)
GESTURE_PREFIX = {
    "Closed_Fist": "N",   # relative delta tracking
    "Victory":     "P",   # absolute IK snap
}

# ── Config ───────────────────────────────────────────────────────────────────────
SERIAL_BAUD  = 115200
SEND_HZ      = 20

# EMA smoothing alpha per axis. Lower = smoother, laggier.
SMOOTH_ALPHA = 0.25

# Deadband — minimum change in any axis before sending.
# Applied after coordinate conversion, in normalised 0-1 units.
DEADBAND = 0.008

# Depth (Z) working range in metres.
# Hand at Z_NEAR → normalised 0.0
# Hand at Z_FAR  → normalised 1.0
Z_NEAR = 0.25
Z_FAR  = 0.80

# Gesture smoothing — majority vote over this many frames
# Prevents single-frame misdetections from switching modes
GESTURE_BUF_LEN = 7

font = cv2.FONT_HERSHEY_SIMPLEX


# ── Depth estimation ─────────────────────────────────────────────────────────────
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


# ── Palm centre, normalised 0-1 in MediaPipe frame ───────────────────────────────
def palm_center_norm(lm, img_w, img_h):
    pts = np.array([[lm[i].x * img_w, lm[i].y * img_h] for i in PALM_PTS])
    c   = pts.mean(axis=0)
    return float(c[0] / img_w), float(c[1] / img_h)


# ── Convert MediaPipe coords to arm space ────────────────────────────────────────
# MediaPipe:  X left/right (0=left, 1=right after flip)
#             Y up/down    (0=top,  1=bottom)
#             Z depth      (0=near, 1=far after normalisation)
#
# Arm space:  X left/right — same as MediaPipe X
#             Y forward/back — MediaPipe Z, same direction
#             Z up/down — MediaPipe Y flipped (0=top → 1=up in arm space)
def to_arm_space(mp_x, mp_y, mp_z):
    arm_x = mp_x           # left/right — unchanged
    arm_y = mp_z           # forward/back — depth maps directly
    arm_z = 1.0 - mp_y    # up/down — flip Y so up = larger value
    return arm_x, arm_y, arm_z


# ── EMA smoother ────────────────────────────────────────────────────────────────
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


# ── Main ─────────────────────────────────────────────────────────────────────────
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
        print("No-serial debug mode")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    # Smoothers in arm space
    ema_x = EMA(SMOOTH_ALPHA)
    ema_y = EMA(SMOOTH_ALPHA)
    ema_z = EMA(SMOOTH_ALPHA)

    depth_buf   = deque(maxlen=8)
    gesture_buf = deque(maxlen=GESTURE_BUF_LEN)

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

                # Raw MediaPipe coords
                mp_x, mp_y = palm_center_norm(lm, w, h)

                # Depth
                raw_depth = estimate_depth(lm, w, h)
                if raw_depth is not None:
                    depth_buf.append(raw_depth)

                smoothed_depth = None
                if depth_buf:
                    vals = list(depth_buf)
                    med  = np.median(vals)
                    good = [v for v in vals if abs(v - med) < 0.15]
                    smoothed_depth = float(np.mean(good)) if good else med

                mp_z = None
                if smoothed_depth is not None:
                    mp_z = (smoothed_depth - Z_NEAR) / (Z_FAR - Z_NEAR)
                    mp_z = max(0.0, min(1.0, mp_z))

                # Convert to arm space then smooth
                if mp_z is not None:
                    ax, ay, az = to_arm_space(mp_x, mp_y, mp_z)
                    sx = ema_x.update(ax)
                    sy = ema_y.update(ay)
                    sz = ema_z.update(az)
                else:
                    sx = ema_x.update(None)
                    sy = ema_y.update(None)
                    sz = ema_z.update(None)

                # Gesture detection with majority-vote smoothing
                raw_gesture = "None"
                if result.gestures and result.gestures[0]:
                    raw_gesture = result.gestures[0][0].category_name
                gesture_buf.append(raw_gesture)
                gesture = Counter(gesture_buf).most_common(1)[0][0]

                # Map gesture to prefix — default to F (hold)
                prefix = GESTURE_PREFIX.get(gesture, "F")

                # Send packet
                if None not in (sx, sy, sz):
                    if (now - last_send_time) >= send_interval:

                        # Deadband — skip if nothing moved and not a mode change
                        send = True
                        if last_sent is not None:
                            same_prefix = (prefix == last_sent[3])
                            dx = abs(sx - last_sent[0])
                            dy = abs(sy - last_sent[1])
                            dz = abs(sz - last_sent[2])
                            if same_prefix and dx < DEADBAND and dy < DEADBAND and dz < DEADBAND:
                                send = False

                        # Always send on prefix change so Arduino gets the mode switch immediately
                        if last_sent is None or prefix != last_sent[3]:
                            send = True

                        if send:
                            packet = f"{prefix}:{sx:.3f},{sy:.3f},{sz:.3f}\n"
                            if ser:
                                ser.write(packet.encode())
                            else:
                                print(packet, end="")
                            last_sent      = (sx, sy, sz, prefix)
                            last_send_time = now

                # HUD overlay
                depth_str = f"{smoothed_depth:.2f}m" if smoothed_depth else "--"
                cv2.putText(frame,
                    f"[{prefix}] X:{sx:.3f}  Y:{sy:.3f}  Z:{sz:.3f}  depth:{depth_str}",
                    (16, 40), font, 0.7, (0, 255, 180), 2)
                cv2.putText(frame,
                    f"gesture: {gesture}",
                    (16, 75), font, 0.6, (180, 180, 0), 2)

                # Skeleton
                for a, b in mp.solutions.hands.HAND_CONNECTIONS:
                    ax2, ay2 = int(lm[a].x * w), int(lm[a].y * h)
                    bx2, by2 = int(lm[b].x * w), int(lm[b].y * h)
                    cv2.line(frame, (ax2, ay2), (bx2, by2), (0, 210, 210), 2)

            else:
                # No hand — send hold
                if last_sent is None or last_sent[3] != "F":
                    packet = "F:0.500,0.500,0.500\n"
                    if ser:
                        ser.write(packet.encode())
                    else:
                        print(packet, end="")
                    last_sent = (0.5, 0.5, 0.5, "F")

                cv2.putText(frame, "No hand — holding", (16, 40),
                            font, 0.7, (80, 80, 80), 2)
                depth_buf.clear()
                gesture_buf.clear()

            cv2.imshow("Hand Tracker", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cap.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()