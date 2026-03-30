"""
Hand Tracker → Arduino Serial  (Raspberry Pi / Bookworm edition)
=================================================================
Detects gesture and sends prefix + real-world XYZ metres to Arduino.

Coordinate system (origin = centre of screen at 0.5m depth):
    X: left/right  (negative = left,  positive = right)
    Y: depth       (positive = closer to camera)
    Z: up/down     (negative = down,  positive = up)

Gesture mapping:
    Victory      → N  (relative delta tracking)
    Open_Palm    → P  (absolute IK snap)
    ILoveYou     → M  (mirror)
    anything else → F  (hold)

Serial format:  "N:0.1230,-0.0420,0.0318\n"

Install:
    sudo apt install -y python3-picamera2
    pip install mediapipe opencv-python numpy pyserial --break-system-packages
    wget -O gesture_recognizer.task \
      https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

Run:
    python hand_tracker_pi.py --port /dev/ttyUSB0
    python hand_tracker_pi.py --no-serial
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from collections import deque, Counter
from picamera2 import Picamera2

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("pyserial not found — serial disabled")

# ── MediaPipe setup ──────────────────────────────────────────────────────────
BaseOptions              = mp.tasks.BaseOptions
GestureRecognizer        = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode        = mp.tasks.vision.RunningMode

MODEL_PATH = "gesture_recognizer.task"

# ── Landmark indices ─────────────────────────────────────────────────────────
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

# ── Gesture → serial prefix ──────────────────────────────────────────────────
GESTURE_PREFIX = {
    "Victory":   "N",
    "Open_Palm": "P",
    "ILoveYou":  "M",
}

# ── Camera intrinsics ────────────────────────────────────────────────────────
CAM_FOV_H_DEG = 79
CAM_FOV_V_DEG = 51.3

# ── Depth clamp (metres) ─────────────────────────────────────────────────────
DEPTH_MIN = 0.10
DEPTH_MAX = 1.00

# ── Config ───────────────────────────────────────────────────────────────────
SERIAL_BAUD     = 115200
SEND_HZ         = 20
CAM_W           = 1280
CAM_H           = 720
SMOOTH_ALPHA    = 0.25
DEADBAND        = 0.005
GESTURE_BUF_LEN = 7

font = cv2.FONT_HERSHEY_SIMPLEX


# ── Depth estimation ─────────────────────────────────────────────────────────
def estimate_depth_3d(lm_screen, lm_world, img_w, img_h):
    focal = img_w / (2 * np.tan(np.radians(42)))
    depths = []
    for a, b, _ in PALM_PAIRS:
        wa = lm_world[a]
        wb = lm_world[b]
        real_m = np.sqrt(
            (wa.x - wb.x)**2 +
            (wa.y - wb.y)**2 +
            (wa.z - wb.z)**2
        )
        if real_m < 0.005:
            continue
        px = np.hypot(
            (lm_screen[a].x - lm_screen[b].x) * img_w,
            (lm_screen[a].y - lm_screen[b].y) * img_h
        )
        if px > 2:
            depths.append((focal * real_m) / px)
    if not depths:
        return None
    med  = np.median(depths)
    good = [d for d in depths if abs(d - med) / med < 0.30]
    return float(np.mean(good)) if good else med


# ── Palm centre, normalised 0-1 ──────────────────────────────────────────────
def palm_center_norm(lm, img_w, img_h):
    pts = np.array([[lm[i].x * img_w, lm[i].y * img_h] for i in PALM_PTS])
    c   = pts.mean(axis=0)
    return float(c[0] / img_w), float(c[1] / img_h)


# ── Unproject screen coords + depth to real-world metres ─────────────────────
def unproject(norm_x, norm_y, depth_m):
    cx = norm_x - 0.5
    cy = norm_y - 0.5

    half_w = depth_m * np.tan(np.radians(CAM_FOV_H_DEG / 2))
    half_h = depth_m * np.tan(np.radians(CAM_FOV_V_DEG / 2))

    x_m =  cx * 2 * half_w
    y_m =  0.5 - depth_m
    z_m = -cy * 2 * half_h

    return x_m, y_m, z_m


# ── EMA smoother ─────────────────────────────────────────────────────────────
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


# ── picamera2 capture ────────────────────────────────────────────────────────
def make_camera():
    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": (CAM_W, CAM_H), "format": "BGR888"},
        controls={"FrameRate": 30},
    )
    cam.configure(config)
    cam.start()
    time.sleep(1.0)
    return cam


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",      default=None)
    parser.add_argument("--no-serial", action="store_true")
    args = parser.parse_args()

    ser = None
    if not args.no_serial:
        if not SERIAL_AVAILABLE:
            print("pyserial not installed. Use --no-serial or install it.")
            return
        if args.port is None:
            print("Provide --port (e.g. /dev/ttyUSB0) or use --no-serial")
            return
        ser = serial.Serial(args.port, SERIAL_BAUD, timeout=1)
        time.sleep(2)
        print(f"Serial open on {args.port}")
    else:
        print("No-serial debug mode")

    print("Starting camera...")
    cam = make_camera()
    print(f"Camera running at {CAM_W}x{CAM_H}")

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    ema_x = EMA(SMOOTH_ALPHA)
    ema_y = EMA(SMOOTH_ALPHA)
    ema_z = EMA(SMOOTH_ALPHA)

    depth_buf   = deque(maxlen=8)
    gesture_buf = deque(maxlen=GESTURE_BUF_LEN)

    last_sent      = None
    last_send_time = 0.0
    send_interval  = 1.0 / SEND_HZ

    fps_buf  = deque(maxlen=30)
    fps_last = time.time()

    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            frame = cam.capture_array()
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            now   = time.time()

            fps_buf.append(now - fps_last)
            fps_last = now
            fps = 1.0 / (sum(fps_buf) / len(fps_buf)) if fps_buf else 0.0

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize_for_video(mp_img, int(now * 1000))

            hand_visible = bool(result.hand_landmarks)

            if hand_visible:
                lm_screen = result.hand_landmarks[0]
                lm_world  = result.hand_world_landmarks[0]

                mp_x, mp_y = palm_center_norm(lm_screen, w, h)
                raw_depth  = estimate_depth_3d(lm_screen, lm_world, w, h)
                if raw_depth is not None:
                    depth_buf.append(raw_depth)

                smoothed_depth = None
                if depth_buf:
                    vals = list(depth_buf)
                    med  = np.median(vals)
                    good = [v for v in vals if abs(v - med) < 0.15]
                    smoothed_depth = float(np.mean(good)) if good else med

                if smoothed_depth is not None:
                    smoothed_depth = max(DEPTH_MIN, min(DEPTH_MAX, smoothed_depth))

                if smoothed_depth is not None:
                    x_m, y_m, z_m = unproject(mp_x, mp_y, smoothed_depth)
                    sx = ema_x.update(x_m)
                    sy = ema_y.update(y_m)
                    sz = ema_z.update(z_m)
                else:
                    sx = ema_x.update(None)
                    sy = ema_y.update(None)
                    sz = ema_z.update(None)

                raw_gesture = "None"
                if result.gestures and result.gestures[0]:
                    raw_gesture = result.gestures[0][0].category_name
                gesture_buf.append(raw_gesture)
                gesture = Counter(gesture_buf).most_common(1)[0][0]

                prefix = GESTURE_PREFIX.get(gesture, "F")

                if None not in (sx, sy, sz):
                    if (now - last_send_time) >= send_interval:
                        send = True
                        if last_sent is not None:
                            same_prefix = (prefix == last_sent[3])
                            dx = abs(sx - last_sent[0])
                            dy = abs(sy - last_sent[1])
                            dz = abs(sz - last_sent[2])
                            if same_prefix and dx < DEADBAND and dy < DEADBAND and dz < DEADBAND:
                                send = False

                        if last_sent is None or prefix != last_sent[3]:
                            send = True

                        if send:
                            packet = f"{prefix}:{sx:.4f},{sy:.4f},{sz:.4f}\n"
                            print(packet, end="")
                            if ser:
                                ser.write(packet.encode())
                            last_sent      = (sx, sy, sz, prefix)
                            last_send_time = now

                cv2.putText(frame,
                    f"[{prefix}] X:{sx:.3f}m  Y:{sy:.3f}m  Z:{sz:.3f}m",
                    (16, 40), font, 0.7, (0, 255, 180), 2)
                cv2.putText(frame,
                    f"gesture: {gesture}",
                    (16, 75), font, 0.6, (180, 180, 0), 2)

                for a, b in mp.solutions.hands.HAND_CONNECTIONS:
                    ax2, ay2 = int(lm_screen[a].x * w), int(lm_screen[a].y * h)
                    bx2, by2 = int(lm_screen[b].x * w), int(lm_screen[b].y * h)
                    cv2.line(frame, (ax2, ay2), (bx2, by2), (0, 210, 210), 2)

                pts = np.array([[lm_screen[i].x * w, lm_screen[i].y * h] for i in PALM_PTS])
                cx2, cy2 = pts.mean(axis=0).astype(int)
                dot_radius = int(5 / smoothed_depth) if smoothed_depth else 8
                cv2.circle(frame, (cx2, cy2), dot_radius, (0, 255, 0), -1)

            else:
                cv2.putText(frame, "No hand — holding", (16, 40),
                            font, 0.7, (80, 80, 80), 2)
                depth_buf.clear()
                gesture_buf.clear()
                last_sent = None

            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                        font, 0.6, (200, 200, 200), 2)

            cv2.imshow("Hand Tracker", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cam.stop()
    if ser:
        ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()