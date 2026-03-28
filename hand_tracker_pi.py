"""
Hand Tracker → Arduino Serial  (Raspberry Pi / Bookworm edition)
=================================================================
Uses picamera2 instead of cv2.VideoCapture — required on Raspberry Pi OS
Bookworm because the old V4L2 camera stack was replaced by libcamera.

Detects gesture and sends prefix + arm-space XYZ to Arduino.

Gesture mapping:
    Closed_Fist  → N  (relative delta tracking)
    Victory      → P  (absolute IK snap)
    anything else → F  (hold)

Coordinate convention (converted to arm space before sending):
    Arm X = hand left/right       (MediaPipe X, same direction)
    Arm Y = hand forward/back     (MediaPipe Z depth estimate)
    Arm Z = hand up/down          (MediaPipe Y flipped)

Serial format:  "N:0.623,0.441,0.318\n"

Install (run these on the Pi):
    sudo apt install -y python3-picamera2
    pip install mediapipe opencv-python numpy pyserial --break-system-packages

    # Download the gesture recognizer model:
    wget -O gesture_recognizer.task \
      https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

Run:
    python hand_tracker_pi.py --port /dev/ttyUSB0
    python hand_tracker_pi.py --port /dev/ttyACM0
    python hand_tracker_pi.py --no-serial
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from collections import deque, Counter

# picamera2 — installed via apt on Bookworm, not pip
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

# Put the .task file in the same directory as this script, or give a full path
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
    "Closed_Fist": "N",
    "Victory":     "P",
}

# ── Config ───────────────────────────────────────────────────────────────────
SERIAL_BAUD  = 115200
SEND_HZ      = 20

# Camera resolution — Camera Module 3 native is 4608x2592 but that's too slow
# for MediaPipe. 1280x720 is a good balance on Pi 4/5.
# Drop to 640x480 on Pi 3 or if fps is low.
CAM_W = 1280
CAM_H = 720

SMOOTH_ALPHA    = 0.25
DEADBAND        = 0.008
Z_NEAR          = 0.25
Z_FAR           = 0.80
GESTURE_BUF_LEN = 7

font = cv2.FONT_HERSHEY_SIMPLEX


# ── Depth estimation ─────────────────────────────────────────────────────────
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


# ── Palm centre, normalised 0-1 ──────────────────────────────────────────────
def palm_center_norm(lm, img_w, img_h):
    pts = np.array([[lm[i].x * img_w, lm[i].y * img_h] for i in PALM_PTS])
    c   = pts.mean(axis=0)
    return float(c[0] / img_w), float(c[1] / img_h)


# ── MediaPipe → arm space ────────────────────────────────────────────────────
def to_arm_space(mp_x, mp_y, mp_z):
    arm_x = mp_x
    arm_y = mp_z
    arm_z = 1.0 - mp_y
    return arm_x, arm_y, arm_z


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
    """
    Configure picamera2 for video capture.
    Returns a started Picamera2 instance that produces BGR frames
    compatible with OpenCV.
    """
    cam = Picamera2()

    # Use the main stream at our chosen resolution.
    # format="BGR888" gives us OpenCV-compatible frames directly —
    # no colour conversion needed.
    config = cam.create_video_configuration(
        main={"size": (CAM_W, CAM_H), "format": "BGR888"},
        controls={"FrameRate": 30},
    )
    cam.configure(config)
    cam.start()

    # Let AGC/AWB settle before we start processing
    time.sleep(1.0)
    return cam


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",      default=None,
                        help="Serial port, e.g. /dev/ttyUSB0 or /dev/ttyACM0")
    parser.add_argument("--no-serial", action="store_true")
    args = parser.parse_args()

    # Serial setup
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

    # Camera setup — picamera2 instead of cv2.VideoCapture
    print("Starting camera...")
    cam = make_camera()
    print(f"Camera running at {CAM_W}x{CAM_H}")

    # MediaPipe gesture recognizer
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

    # FPS counter
    fps_buf  = deque(maxlen=30)
    fps_last = time.time()

    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            # ── Grab frame from picamera2 ──────────────────────────────────
            # capture_array() returns a numpy array in the format we configured
            # (BGR888 → shape HxWx3, dtype uint8) — identical to what
            # cv2.VideoCapture.read() gives you.
            frame = cam.capture_array()

            # Pi camera is not mirrored by default — flip so left/right
            # matches the user's perspective (same as the original code).
            frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            now  = time.time()

            # FPS
            fps_buf.append(now - fps_last)
            fps_last = now
            fps = 1.0 / (sum(fps_buf) / len(fps_buf)) if fps_buf else 0.0

            # ── MediaPipe ─────────────────────────────────────────────────
            # MediaPipe expects RGB — convert from BGR
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize_for_video(mp_img, int(now * 1000))

            hand_visible = bool(result.hand_landmarks)

            if hand_visible:
                lm = result.hand_landmarks[0]

                mp_x, mp_y = palm_center_norm(lm, w, h)

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

                if mp_z is not None:
                    ax, ay, az = to_arm_space(mp_x, mp_y, mp_z)
                    sx = ema_x.update(ax)
                    sy = ema_y.update(ay)
                    sz = ema_z.update(az)
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

            # FPS display
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                        font, 0.6, (200, 200, 200), 2)

            cv2.imshow("Hand Tracker", frame)

            # 'q' or ESC to quit
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    cam.stop()
    if ser:
        ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()