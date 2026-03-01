"""
MediaPipe Hand Tracker – Window Control Edition
================================================
LAUNCH: No window shown until you swipe a pointing hand fast upward from
        the bottom of the camera frame (Point_Down → Point gesture while
        moving quickly from low to high in frame).

Control gesture (ILoveYou/None → "Control"):
  - X/Y  → move window so your hand is under window center
  - Z     → resize: CLOSER = SMALLER, FARTHER = BIGGER

Secret quit: Control → Fist → Control within 1 second exits the program.

Keys: Q / ESC = quit   S = screenshot

Install:  pip install mediapipe opencv-python numpy
Run:      python hand_tracker.py
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
from collections import deque, Counter

# ── MediaPipe Tasks ────────────────────────────────────────────────────────────
BaseOptions              = mp.tasks.BaseOptions
GestureRecognizer        = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode        = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task")
MODEL_PATH = "C:/Users/livel/VScode/246 Project/HandTracking/gesture_recognizer.task"

GESTURE_LABELS = {
    "Closed_Fist": "Fist",
    "Open_Palm":   "Open Hand",
    "Pointing_Up": "Point",
    "Thumb_Up":    "Thumbs Up",
    "Thumb_Down":  "Thumbs Down",
    "Victory":     "Peace",
    "ILoveYou":    "Control",
    "None":        "Neutral",
}

# ── Landmark indices ───────────────────────────────────────────────────────────
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

HAND_COLORS = [(0, 210, 210), (210, 90, 255)]
font = cv2.FONT_HERSHEY_SIMPLEX

# ── Window control constants ───────────────────────────────────────────────────
WIN_BASE_SIZE  = (960, 540)
WIN_REF_DEPTH  = 0.50        # metres at which window = base size
WIN_MIN_SIZE   = (320, 180)
WIN_MAX_SIZE   = (1920, 1080)
WIN_NAME       = "Hand Tracker"

# ── Secret quit timing ─────────────────────────────────────────────────────────
QUIT_SEQUENCE_TIMEOUT = 1.0   # seconds

def get_screen_size():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.destroy()
        return sw, sh
    except Exception:
        return 1920, 1080


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


def palm_center(lm, img_w, img_h):
    pts = np.array([[lm[i].x * img_w, lm[i].y * img_h] for i in PALM_PTS])
    c   = pts.mean(axis=0)
    nx  = (c[0] / img_w - 0.5) * 2
    ny  = (c[1] / img_h - 0.5) * 2
    return int(c[0]), int(c[1]), nx, ny


class HandState:
    def __init__(self):
        self.depth_buf   = deque(maxlen=8)
        self.gesture_buf = deque(maxlen=7)
        self.nx_buf      = deque(maxlen=6)
        self.ny_buf      = deque(maxlen=6)

    def push_depth(self, z):
        if z is not None:
            self.depth_buf.append(z)

    def smooth_depth(self):
        if not self.depth_buf:
            return None
        vals = list(self.depth_buf)
        med  = np.median(vals)
        good = [v for v in vals if abs(v - med) < 0.15]
        return float(np.mean(good)) if good else med

    def push_gesture(self, g):
        self.gesture_buf.append(g)

    def smooth_gesture(self):
        return Counter(self.gesture_buf).most_common(1)[0][0] if self.gesture_buf else "—"

    def push_xy(self, nx, ny):
        self.nx_buf.append(nx)
        self.ny_buf.append(ny)

    def smooth_xy(self):
        if not self.nx_buf:
            return 0.0, 0.0
        return float(np.mean(self.nx_buf)), float(np.mean(self.ny_buf))


class WindowController:
    def __init__(self, screen_w, screen_h):
        self.screen_w  = screen_w
        self.screen_h  = screen_h
        self.win_w, self.win_h = WIN_BASE_SIZE
        self.win_x = (screen_w - self.win_w) // 2
        self.win_y = (screen_h - self.win_h) // 2
        self._smoothed_depth = WIN_REF_DEPTH

    def update(self, nx, ny, depth):
        if depth is not None:
            self._smoothed_depth = self._smoothed_depth * 0.85 + depth * 0.15

        d = max(0.15, self._smoothed_depth)

        # CLOSER (smaller d) → SMALLER window: size proportional to depth
        scale    = d / WIN_REF_DEPTH
        new_w    = int(WIN_BASE_SIZE[0] * scale)
        new_h    = int(WIN_BASE_SIZE[1] * scale)
        self.win_w = max(WIN_MIN_SIZE[0], min(WIN_MAX_SIZE[0], new_w))
        self.win_h = max(WIN_MIN_SIZE[1], min(WIN_MAX_SIZE[1], new_h))

        # Centre window under hand position on screen
        hand_sx = int((nx * 0.5 + 0.5) * self.screen_w)
        hand_sy = int((ny * 0.5 + 0.5) * self.screen_h)

        self.win_x = hand_sx - self.win_w // 2
        self.win_y = hand_sy - self.win_h // 2

        # Clamp to screen bounds
        self.win_x = max(0, min(self.screen_w - self.win_w,  self.win_x))
        self.win_y = max(0, min(self.screen_h - self.win_h,  self.win_y))

    def apply(self):
        cv2.resizeWindow(WIN_NAME, self.win_w, self.win_h)
        cv2.moveWindow(WIN_NAME, self.win_x, self.win_y)


class QuitDetector:
    """
    Detects: Control (held any duration) → Fist → Control within 1 second.
    The 1-second timer only starts when the fist is first seen.
    """
    def __init__(self):
        self._state        = "idle"   # idle | holding_control | saw_fist
        self._fist_time    = 0.0
        self._prev_gesture = None

    def update(self, gesture, now):
        # Only act on transitions
        if gesture == self._prev_gesture:
            return False
        self._prev_gesture = gesture

        if self._state == "idle":
            if gesture == "Control":
                self._state = "holding_control"

        elif self._state == "holding_control":
            if gesture == "Fist":
                # Start the 1-second timer NOW
                self._state     = "saw_fist"
                self._fist_time = now
            else:
                # Any other gesture breaks the sequence
                self._state = "idle"

        elif self._state == "saw_fist":
            if now - self._fist_time > QUIT_SEQUENCE_TIMEOUT:
                # Took too long to return to Control
                self._state = "idle"
            elif gesture == "Control":
                return True   # ← QUIT
            else:
                self._state = "idle"

        return False




def draw_skeleton(frame, lm, color, w, h):
    for a, b in HAND_CONNECTIONS:
        ax, ay = int(lm[a].x * w), int(lm[a].y * h)
        bx, by = int(lm[b].x * w), int(lm[b].y * h)
        cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)
    for lmk in lm:
        px, py = int(lmk.x * w), int(lmk.y * h)
        cv2.circle(frame, (px, py), 4, (255, 255, 255), -1)
        cv2.circle(frame, (px, py), 4, color, 1)


def draw_hud(frame, hand_results, fps, controlling):
    h, w = frame.shape[:2]
    rows    = max(1, len(hand_results))
    strip_h = 42 + rows * 68
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - strip_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)

    if controlling:
        cv2.rectangle(frame, (0, 0), (w, 6), (0, 255, 120), -1)
        cv2.putText(frame, "CONTROL MODE",
                    (12, 32), font, 0.60, (0, 255, 120), 2, cv2.LINE_AA)

    if not hand_results:
        cv2.putText(frame, "No hand detected",
                    (28, h - strip_h + 46), font, 0.68, (90, 90, 90), 1, cv2.LINE_AA)
    else:
        for i, (label, nx, ny, depth, gesture, color) in enumerate(hand_results):
            yb = h - strip_h + 28 + i * 68
            cv2.putText(frame, f"{label} Hand",
                        (18, yb), font, 0.48, color, 1, cv2.LINE_AA)
            fields = [
                ("X",       f"{nx:+.3f}",                       130),
                ("Y",       f"{ny:+.3f}",                       268),
                ("Z",       f"{depth:.2f} m" if depth else "—", 406),
                ("Gesture", gesture,                             560),
            ]
            for lb, val, x in fields:
                cv2.putText(frame, lb,  (x, yb),      font, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
                cv2.putText(frame, val, (x, yb + 27), font, 0.76, (240, 240, 240), 2, cv2.LINE_AA)

def main():

    screen_w, screen_h = get_screen_size()
    controller    = WindowController(screen_w, screen_h)
    quit_detector = QuitDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera."); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    window_open = False   # window hidden until Pointing_Up detected
    fullscreen  = True

    prev   = time.time()
    shot   = 0
    states = {0: HandState(), 1: HandState()}

    with GestureRecognizer.create_from_options(options) as recognizer:

        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize_for_video(mp_img, int(now * 1000))

            hand_results = []
            active = len(result.hand_landmarks) if result.hand_landmarks else 0
            controlling  = False
            should_quit  = False

            for idx in range(active):
                lm         = result.hand_landmarks[idx]
                handedness = result.handedness[idx][0].display_name
                handedness = "Left" if handedness == "Right" else "Right"
                color      = HAND_COLORS[idx % 2]
                state      = states[idx]

                cx, cy, nx, ny = palm_center(lm, w, h)
                state.push_depth(estimate_depth(lm, w, h))
                depth = state.smooth_depth()
                state.push_xy(nx, ny)

                raw_label = "—"
                if result.gestures and idx < len(result.gestures) and result.gestures[idx]:
                    cat = result.gestures[idx][0].category_name
                    raw_label = GESTURE_LABELS.get(cat, cat)
                state.push_gesture(raw_label)
                gesture = state.smooth_gesture()

                if not window_open and idx == 0:
                    if result.gestures and result.gestures[idx] and \
                       result.gestures[idx][0].category_name == "Pointing_Up":
                        window_open = True
                        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_FULLSCREEN)

                if not window_open:
                    continue

                draw_skeleton(frame, lm, color, w, h)
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 9, color, 2)

                if idx == 0:
                    if quit_detector.update(gesture, now):
                        should_quit = True

                if gesture == "Control" and not controlling:
                    controlling = True
                    if fullscreen:
                        fullscreen = False
                        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                                              cv2.WINDOW_NORMAL)
                    snx, sny = state.smooth_xy()
                    controller.update(snx, sny, depth)
                    controller.apply()

                hand_results.append((handedness, nx, ny, depth, gesture, color))

            for k in range(active, 2):
                states[k] = HandState()

            if window_open:
                draw_hud(frame, hand_results, fps, controlling)
                cv2.imshow(WIN_NAME, frame)

            if should_quit:
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s') and window_open:
                fname = f"shot_{shot:03d}.png"
                cv2.imwrite(fname, frame)
                print(f"Saved {fname}")
                shot += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()