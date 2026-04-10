import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from collections import deque, Counter

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

BaseOptions              = mp.tasks.BaseOptions
GestureRecognizer        = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode        = mp.tasks.vision.RunningMode

MODEL_PATH = "C:/Users/livel/VScode/246 Project/HandTracking/gesture_recognizer.task"

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

GESTURE_PREFIX = {
    "Victory":   "N",
    "Open_Palm": "P",
    "ILoveYou":  "M",
}

MODE_LABELS = {
    "N": "RELATIVE",
    "P": "ABSOLUTE",
    "M": "MIRROR",
    "F": "HOLD",
}

MODE_COLORS = {
    "N": (0, 255, 180),
    "P": (0, 180, 255),
    "M": (180, 0, 255),
    "F": (100, 100, 100),
}

CAM_FOV_H_DEG = 79
CAM_FOV_V_DEG = 51.3
DEPTH_MIN     = 0.10
DEPTH_MAX     = 1.00
SERIAL_BAUD   = 115200
SEND_HZ       = 20
SMOOTH_ALPHA  = 0.25
DEADBAND      = 0.005
GESTURE_BUF_LEN = 7

font       = cv2.FONT_HERSHEY_SIMPLEX
font_mono  = cv2.FONT_HERSHEY_PLAIN


def estimate_depth_3d(lm_screen, lm_world, img_w, img_h):
    focal = img_w / (2 * np.tan(np.radians(42)))
    depths = []
    for a, b, _ in PALM_PAIRS:
        wa, wb = lm_world[a], lm_world[b]
        real_m = np.sqrt((wa.x-wb.x)**2 + (wa.y-wb.y)**2 + (wa.z-wb.z)**2)
        if real_m < 0.005:
            continue
        px = np.hypot((lm_screen[a].x - lm_screen[b].x) * img_w,
                      (lm_screen[a].y - lm_screen[b].y) * img_h)
        if px > 2:
            depths.append((focal * real_m) / px)
    if not depths:
        return None
    med  = np.median(depths)
    good = [d for d in depths if abs(d - med) / med < 0.30]
    return float(np.mean(good)) if good else med


def palm_center_norm(lm, img_w, img_h):
    pts = np.array([[lm[i].x * img_w, lm[i].y * img_h] for i in PALM_PTS])
    c   = pts.mean(axis=0)
    return float(c[0] / img_w), float(c[1] / img_h)


def unproject(norm_x, norm_y, depth_m):
    cx = norm_x - 0.5
    cy = norm_y - 0.5
    half_w = depth_m * np.tan(np.radians(CAM_FOV_H_DEG / 2))
    half_h = depth_m * np.tan(np.radians(CAM_FOV_V_DEG / 2))
    x_m =  cx * 2 * half_w
    y_m =  0.5 - depth_m
    z_m = -cy * 2 * half_h
    return x_m, y_m, z_m


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


# ── UI helpers ───────────────────────────────────────────────────────────────

def draw_panel(frame, x, y, w, h, color=(255,255,255), alpha=0.15, border=True):
    """Semi-transparent filled rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha + 0.25, frame, 1 - (alpha + 0.25), 0, frame)
    if border:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)


def draw_bar(frame, x, y, w, h, value, lo, hi, color, label):
    """Horizontal bar — value in [lo, hi]."""
    t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    fill = int(t * w)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (60,60,60), -1)
    cv2.rectangle(frame, (x, y), (x+fill, y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
    cv2.putText(frame, f"{label}: {value:+.3f}", (x + w + 8, y + h - 2),
                font, 0.42, color, 1, cv2.LINE_AA)


def draw_crosshair(frame, cx, cy, size, color, thickness=1):
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), size // 2, color, thickness, cv2.LINE_AA)


def draw_corner_brackets(frame, x, y, w, h, color, size=12, thickness=2):
    pts = [(x,y),(x+w,y),(x,y+h),(x+w,y+h)]
    dirs = [(1,1),(-1,1),(1,-1),(-1,-1)]
    for (px,py),(dx,dy) in zip(pts, dirs):
        cv2.line(frame, (px, py), (px + dx*size, py), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy*size), color, thickness, cv2.LINE_AA)


def draw_depth_arc(frame, cx, cy, depth, d_min, d_max, color):
    """Arc that shrinks as hand gets closer."""
    t = 1.0 - max(0.0, min(1.0, (depth - d_min) / (d_max - d_min)))
    radius = int(18 + t * 28)
    sweep  = int(270 * t)
    cv2.ellipse(frame, (cx, cy), (radius, radius), -135, 0, sweep,
                color, 2, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────────────

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

    ema_x = EMA(SMOOTH_ALPHA)
    ema_y = EMA(SMOOTH_ALPHA)
    ema_z = EMA(SMOOTH_ALPHA)

    depth_buf   = deque(maxlen=8)
    gesture_buf = deque(maxlen=GESTURE_BUF_LEN)
    fps_buf     = deque(maxlen=30)
    fps_last    = time.time()

    last_sent      = None
    last_send_time = 0.0
    send_interval  = 1.0 / SEND_HZ

    sx = sy = sz = 0.0
    smoothed_depth = None
    prefix  = "F"
    gesture = "None"

    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

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
            color = MODE_COLORS.get(prefix, (255,255,255))

            if hand_visible:
                lm_screen = result.hand_landmarks[0]
                lm_world  = result.hand_world_landmarks[0]

                mp_x, mp_y = palm_center_norm(lm_screen, w, h)
                raw_depth  = estimate_depth_3d(lm_screen, lm_world, w, h)
                if raw_depth is not None:
                    depth_buf.append(raw_depth)

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
                    sx = ema_x.update(None) or 0.0
                    sy = ema_y.update(None) or 0.0
                    sz = ema_z.update(None) or 0.0

                raw_gesture = "None"
                if result.gestures and result.gestures[0]:
                    raw_gesture = result.gestures[0][0].category_name
                gesture_buf.append(raw_gesture)
                gesture = Counter(gesture_buf).most_common(1)[0][0]
                prefix  = GESTURE_PREFIX.get(gesture, "F")
                color   = MODE_COLORS.get(prefix, (255,255,255))

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

                # ── Skeleton ─────────────────────────────────────────────
                for a, b in mp.solutions.hands.HAND_CONNECTIONS:
                    ax2, ay2 = int(lm_screen[a].x * w), int(lm_screen[a].y * h)
                    bx2, by2 = int(lm_screen[b].x * w), int(lm_screen[b].y * h)
                    cv2.line(frame, (ax2, ay2), (bx2, by2), color, 1, cv2.LINE_AA)

                for lm_pt in lm_screen:
                    px2 = int(lm_pt.x * w)
                    py2 = int(lm_pt.y * h)
                    cv2.circle(frame, (px2, py2), 3, (255,255,255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (px2, py2), 3, color, 1, cv2.LINE_AA)

                # ── Palm tracking dot ─────────────────────────────────────
                pts  = np.array([[lm_screen[i].x * w, lm_screen[i].y * h] for i in PALM_PTS])
                pcx, pcy = pts.mean(axis=0).astype(int)
                dot_radius = int(5 / smoothed_depth) if smoothed_depth else 8

                draw_crosshair(frame, pcx, pcy, dot_radius + 10, color)
                if smoothed_depth is not None:
                    draw_depth_arc(frame, pcx, pcy, smoothed_depth, DEPTH_MIN, DEPTH_MAX, color)
                cv2.circle(frame, (pcx, pcy), dot_radius, color, -1, cv2.LINE_AA)

                # ── Corner brackets around hand bounding box ──────────────
                xs = [int(lm_screen[i].x * w) for i in range(21)]
                ys = [int(lm_screen[i].y * h) for i in range(21)]
                pad = 18
                bx1, by1 = max(0, min(xs)-pad), max(0, min(ys)-pad)
                bx2b, by2b = min(w, max(xs)+pad), min(h, max(ys)+pad)
                draw_corner_brackets(frame, bx1, by1, bx2b-bx1, by2b-by1, color)

            else:
                prefix  = "F"
                gesture = "None"
                color   = MODE_COLORS["F"]
                depth_buf.clear()
                gesture_buf.clear()
                last_sent = None

            # ════════════════════════════════════════════════════════════
            # HUD — top-left panel
            # ════════════════════════════════════════════════════════════
            panel_x, panel_y, panel_w, panel_h = 12, 12, 320, 180
            draw_panel(frame, panel_x, panel_y, panel_w, panel_h, color, alpha=0.12)

            # Project name
            cv2.putText(frame, "TRACE", (panel_x+10, panel_y+22),
                        font, 0.55, color, 1, cv2.LINE_AA)
            cv2.line(frame, (panel_x+10, panel_y+28),
                     (panel_x+panel_w-10, panel_y+28), color, 1)

            # Mode badge
            mode_label = MODE_LABELS.get(prefix, "HOLD")
            cv2.putText(frame, f"MODE  {mode_label}", (panel_x+10, panel_y+50),
                        font, 0.55, color, 1, cv2.LINE_AA)

            # Gesture
            cv2.putText(frame, f"GESTURE  {gesture.upper()}", (panel_x+10, panel_y+72),
                        font, 0.45, (180,180,180), 1, cv2.LINE_AA)

            # Axis bars
            bar_x = panel_x + 10
            bar_w = 120
            bh    = 8
            draw_bar(frame, bar_x, panel_y+92,  bar_w, bh, sx or 0, -0.5, 0.5, (0,200,255),  "X")
            draw_bar(frame, bar_x, panel_y+112, bar_w, bh, sy or 0, -0.5, 0.5, (0,255,120),  "Y")
            draw_bar(frame, bar_x, panel_y+132, bar_w, bh, sz or 0, -0.5, 0.5, (200,120,255),"Z")

            # Depth
            depth_str = f"{smoothed_depth:.3f} m" if smoothed_depth else "-- m"
            cv2.putText(frame, f"DEPTH  {depth_str}", (panel_x+10, panel_y+160),
                        font, 0.45, (180,180,180), 1, cv2.LINE_AA)

            # ── Top-right: FPS + status ───────────────────────────────────
            status     = "TRACKING" if hand_visible else "NO HAND"
            status_col = color if hand_visible else (80,80,80)
            fps_str    = f"FPS {fps:5.1f}"

            draw_panel(frame, w-160, 12, 148, 52, status_col, alpha=0.12)
            cv2.putText(frame, fps_str, (w-148, 34),
                        font, 0.55, (200,200,200), 1, cv2.LINE_AA)
            cv2.putText(frame, status, (w-148, 54),
                        font, 0.45, status_col, 1, cv2.LINE_AA)

            # ── Bottom: raw packet ────────────────────────────────────────
            if last_sent:
                pkt_str = f"{last_sent[3]}  {last_sent[0]:+.4f}  {last_sent[1]:+.4f}  {last_sent[2]:+.4f}"
                draw_panel(frame, 12, h-36, 380, 26, color, alpha=0.10)
                cv2.putText(frame, pkt_str, (22, h-17),
                            font_mono, 1.1, color, 1, cv2.LINE_AA)

            cv2.imshow("Trace", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cap.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()