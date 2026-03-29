"""
Hand Tracker → Arduino Serial
==============================
Detects gesture and sends prefix + real-world XYZ metres to Arduino.

Coordinate system (origin = centre of screen at 0.5m depth):
    X: left/right  (negative = left,  positive = right)
    Y: depth       (0 = camera,  centred so 0.5m depth = 0.0)
    Z: up/down     (negative = down,  positive = up)

Gesture mapping:
    Closed_Fist  → N  (relative delta tracking)
    Victory      → P  (absolute IK snap)
    anything else → F  (hold)

Serial format:  "N:0.1230,-0.0420,0.0318\n"
                 ^ prefix  ^ x      ^ y      ^ z  (all metres)

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
    (WRIST,     MIDDLE_MCP),
    (WRIST,     INDEX_MCP),
    (WRIST,     RING_MCP),
    (WRIST,     PINKY_MCP),
    (INDEX_MCP, PINKY_MCP),
    (INDEX_MCP, RING_MCP),
]

# ── Gesture → serial prefix ──────────────────────────────────────────────────────
# Anything not listed here sends F (hold)
GESTURE_PREFIX = {
    "Closed_Fist": "N",   # relative delta tracking
    "Victory":     "P",   # absolute IK snap
}
# ILY triggers solid box display mode (visual only, no serial change)
ILY_GESTURE = "ILoveYou"

# ── Camera intrinsics ─────────────────────────────────────────────────────────
# Measured: at 0.5m depth frame was 0.80m wide, 0.46m tall
# FOV_H = 2 * atan(0.40 / 0.50) = 77.3°
# FOV_V = 2 * atan(0.23 / 0.50) = 49.8°
CAM_FOV_H_DEG = 77.3
CAM_FOV_V_DEG = 49.8

# ── Depth clamp (metres) ──────────────────────────────────────────────────────
# Hand tracking disappears below ~0.20m so floor at 0.10m
DEPTH_MIN = 0.10
DEPTH_MAX = 1.00

# ── Config ───────────────────────────────────────────────────────────────────────
SERIAL_BAUD  = 115200
SEND_HZ      = 20

# EMA smoothing alpha per axis. Lower = smoother, laggier.
SMOOTH_ALPHA = 0.10

# Deadband — minimum change in any axis before sending (metres).
DEADBAND = 0.005

# Gesture smoothing — majority vote over this many frames
GESTURE_BUF_LEN = 7

font = cv2.FONT_HERSHEY_SIMPLEX


# ── Depth estimation (3D world landmarks) ───────────────────────────────────────
# Uses MediaPipe world landmarks for pose-invariant real-world scale.
# real_m is computed from actual hand geometry each frame — no hardcoded sizes.
def estimate_depth_debug(lm_screen, lm_world, img_w, img_h):
    focal = img_w / (2 * np.tan(np.radians(31.5)))
    depths = []
    labels = []
    for a, b in PALM_PAIRS:
        wa, wb = lm_world[a], lm_world[b]
        real_m = np.sqrt((wa.x - wb.x)**2 + (wa.y - wb.y)**2 + (wa.z - wb.z)**2)
        if real_m < 0.005:
            continue
        px = np.hypot(
            (lm_screen[a].x - lm_screen[b].x) * img_w,
            (lm_screen[a].y - lm_screen[b].y) * img_h
        )
        if px > 2:
            d = (focal * real_m) / px
            depths.append(d)
            labels.append(f"{a}-{b}:{d:.2f}m")

    if not depths:
        return None, []

    med  = np.median(depths)
    good = [d for d in depths if abs(d - med) / med < 0.30]
    final = float(np.mean(good)) if good else med

    debug_labels = []
    for d, lbl in zip(depths, labels):
        if abs(d - med) / med >= 0.30:
            debug_labels.append(("[X]" + lbl, False))   # rejected
        else:
            debug_labels.append(("[+]" + lbl, True))    # used

    return final, debug_labels


# ── Palm centre, normalised 0-1 in MediaPipe frame ───────────────────────────────
def palm_center_norm(lm, img_w, img_h):
    pts = np.array([[lm[i].x * img_w, lm[i].y * img_h] for i in PALM_PTS])
    c   = pts.mean(axis=0)
    return float(c[0] / img_w), float(c[1] / img_h)


# ── Unproject screen coords + depth to real-world metres ─────────────────────────
# Origin: centre of screen at 0.5m depth = (0, 0, 0)
#   X: left/right  — negative left, positive right
#   Y: depth       — 0 at camera, centred so 0.5m = 0.0
#   Z: up/down     — negative down, positive up
def unproject(norm_x, norm_y, depth_m):
    cx = norm_x - 0.5   # centre: 0.5 screen → 0.0
    cy = norm_y - 0.5   # centre: positive = down on screen

    # Live depth — physically correct at any distance
    # Same physical position maps to same X/Z regardless of depth
    half_w = depth_m * np.tan(np.radians(CAM_FOV_H_DEG / 2))
    half_h = depth_m * np.tan(np.radians(CAM_FOV_V_DEG / 2))

    x_m =  cx * 2 * half_w    # left/right
    y_m =  depth_m - 0.5      # forward/back centred at 0.5m
    z_m = -cy * 2 * half_h    # up/down — flip so screen-down = negative

    return x_m, y_m, z_m


# ── Workspace box (metres, same coordinate system as unproject) ──────────────────
BOX_X = (-0.4, 0.4)   # left/right
BOX_Y = (-0.4, 0.5)   # depth (centred: 0 = 0.5m from camera)
BOX_Z = (-0.3, 0.3)   # up/down

# How close the hand needs to be to an edge to occlude it (metres)
HAND_OCCLUDE_RADIUS = 0.10

# Corners: bit0=X, bit1=Y, bit2=Z
#   0=(x0,y0,z0)  1=(x1,y0,z0)  2=(x0,y1,z0)  3=(x1,y1,z0)
#   4=(x0,y0,z1)  5=(x1,y0,z1)  6=(x0,y1,z1)  7=(x1,y1,z1)
BOX_EDGES = [
    (0,1),(2,3),(4,5),(6,7),   # X edges
    (0,2),(1,3),(4,6),(5,7),   # Y edges
    (0,4),(1,5),(2,6),(3,7),   # Z edges
]

# Faces as quads for solid fill mode (ILY gesture)
BOX_FACES = [
    [0,1,3,2],   # front  (y0)
    [4,5,7,6],   # back   (y1)
    [0,1,5,4],   # bottom (z0)
    [2,3,7,6],   # top    (z1)
    [0,2,6,4],   # left   (x0)
    [1,3,7,5],   # right  (x1)
]


def project_point(world_x, world_y, world_z, img_w, img_h):
    """Project world-space point (metres) to pixel coordinates."""
    depth_m = world_y + 0.5   # world_y centred at 0.5m
    if depth_m <= 0.01:
        return None
    half_w = depth_m * np.tan(np.radians(CAM_FOV_H_DEG / 2))
    half_h = depth_m * np.tan(np.radians(CAM_FOV_V_DEG / 2))
    norm_x =  (world_x / (2 * half_w)) + 0.5
    norm_y = -(world_z / (2 * half_h)) + 0.5
    return (int(norm_x * img_w), int(norm_y * img_h))


def build_corners():
    corners = []
    for idx in range(8):
        xi = (idx >> 0) & 1
        yi = (idx >> 1) & 1
        zi = (idx >> 2) & 1
        corners.append((BOX_X[xi], BOX_Y[yi], BOX_Z[zi]))
    return corners


def edge_occluded_by_hand(c_a, c_b, hx, hy, hz):
    """True if hand is within HAND_OCCLUDE_RADIUS of any sampled point along edge."""
    for t in np.linspace(0, 1, 10):
        ex = c_a[0] + t * (c_b[0] - c_a[0])
        ey = c_a[1] + t * (c_b[1] - c_a[1])
        ez = c_a[2] + t * (c_b[2] - c_a[2])
        if np.sqrt((ex-hx)**2 + (ey-hy)**2 + (ez-hz)**2) < HAND_OCCLUDE_RADIUS:
            return True
    return False


def draw_workspace_box(frame, hand_xyz, img_w, img_h, solid=False):
    """
    Draw the 3D workspace box projected onto the frame.
    hand_xyz: (x, y, z) metres or None for no hand.
    solid:    True = filled translucent faces (ILY mode).
    Occlusion: edges behind hand depth OR within HAND_OCCLUDE_RADIUS are hidden.
    """
    corners  = build_corners()
    corn_px  = [project_point(wx, wy, wz, img_w, img_h) for wx, wy, wz in corners]

    no_hand  = hand_xyz is None
    hx       = hand_xyz[0] if not no_hand else 0.0
    hy       = hand_xyz[1] if not no_hand else BOX_Y[0] - 1.0   # far in front = nothing hidden
    hz       = hand_xyz[2] if not no_hand else 0.0

    if solid:
        overlay = frame.copy()
        for face in BOX_FACES:
            pts = [corn_px[i] for i in face]
            if any(p is None for p in pts):
                continue
            face_y = np.mean([corners[i][1] for i in face])
            if face_y <= hy + 0.05:   # only faces in front of / at hand
                cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], (0, 180, 255))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        # Solid edge outline on top
        for i, j in BOX_EDGES:
            p1, p2 = corn_px[i], corn_px[j]
            if p1 and p2:
                cv2.line(frame, p1, p2, (0, 220, 255), 2)
        return

    # Edge wireframe mode
    for i, j in BOX_EDGES:
        p1, p2 = corn_px[i], corn_px[j]
        if p1 is None or p2 is None:
            continue

        c_a, c_b = corners[i], corners[j]
        avg_y    = (c_a[1] + c_b[1]) / 2.0

        depth_hidden     = not no_hand and avg_y > hy
        proximity_hidden = not no_hand and edge_occluded_by_hand(c_a, c_b, hx, hy, hz)

        if depth_hidden or proximity_hidden:
            # Faint stub at each endpoint, gap in middle
            def interp(a, b, t):
                return (int(a[0] + t*(b[0]-a[0])), int(a[1] + t*(b[1]-a[1])))
            cv2.line(frame, p1, interp(p1, p2, 0.2), (45, 45, 45), 1)
            cv2.line(frame, p2, interp(p2, p1, 0.2), (45, 45, 45), 1)
        else:
            cv2.line(frame, p1, p2, (0, 180, 255), 2)


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

    # Smoothers in real-world metres
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
                lm_screen = result.hand_landmarks[0]
                lm_world  = result.hand_world_landmarks[0]

                # Raw MediaPipe screen coords (normalised 0-1)
                mp_x, mp_y = palm_center_norm(lm_screen, w, h)

                # Depth estimation using 3D world landmarks
                raw_depth, depth_debug = estimate_depth_debug(lm_screen, lm_world, w, h)
                if raw_depth is not None:
                    depth_buf.append(raw_depth)

                smoothed_depth = None
                if depth_buf:
                    vals = list(depth_buf)
                    med  = np.median(vals)
                    good = [v for v in vals if abs(v - med) < 0.15]
                    smoothed_depth = float(np.mean(good)) if good else med

                # Clamp depth to valid range
                if smoothed_depth is not None:
                    smoothed_depth = max(DEPTH_MIN, min(DEPTH_MAX, smoothed_depth))

                # Unproject to real-world metres then smooth
                if smoothed_depth is not None:
                    x_m, y_m, z_m = unproject(mp_x, mp_y, smoothed_depth)
                    sx = ema_x.update(x_m)
                    sy = ema_y.update(y_m)
                    sz = ema_z.update(z_m)
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
                            packet = f"{prefix}:{sx:.4f},{sy:.4f},{sz:.4f}\n"
                            if ser:
                                ser.write(packet.encode())
                            else:
                                print(packet, end="")
                            last_sent      = (sx, sy, sz, prefix)
                            last_send_time = now

                # HUD overlay
                cv2.putText(frame,
                    f"[{prefix}] X:{sx:.3f}m  Y:{sy:.3f}m  Z:{sz:.3f}m",
                    (16, 40), font, 0.7, (0, 255, 180), 2)
                cv2.putText(frame,
                    f"gesture: {gesture}",
                    (16, 75), font, 0.6, (180, 180, 0), 2)

                # Depth buffer variance — shows stability
                if len(depth_buf) > 2:
                    variance = float(np.std(list(depth_buf)))
                    cv2.putText(frame,
                        f"depth std: {variance:.4f}m  buf: {len(depth_buf)}",
                        (16, 100), font, 0.5, (200, 200, 0), 1)

                # Per-pair depth debug — green=used, red=rejected
                for i, (lbl, used) in enumerate(depth_debug):
                    color = (0, 220, 0) if used else (0, 0, 220)
                    cv2.putText(frame, lbl, (16, 125 + i * 22), font, 0.45, color, 1)

                # Workspace box — drawn before skeleton so hand appears on top
                hand_xyz_for_box = (sx, sy, sz) if None not in (sx, sy, sz) else None
                solid_box = (gesture == ILY_GESTURE)
                draw_workspace_box(frame, hand_xyz_for_box, w, h, solid=solid_box)

                # Skeleton
                for a, b in mp.solutions.hands.HAND_CONNECTIONS:
                    ax2, ay2 = int(lm_screen[a].x * w), int(lm_screen[a].y * h)
                    bx2, by2 = int(lm_screen[b].x * w), int(lm_screen[b].y * h)
                    cv2.line(frame, (ax2, ay2), (bx2, by2), (0, 210, 210), 2)

                pts = np.array([[lm_screen[i].x * w, lm_screen[i].y * h] for i in PALM_PTS])
                cx, cy = pts.mean(axis=0).astype(int)

                # Scale dot with depth
                if smoothed_depth is not None:
                    dot_radius = int(5 / smoothed_depth)
                else:
                    dot_radius = 8

                cv2.circle(frame, (cx, cy), dot_radius, (0, 255, 0), -1)

            else:
                # No hand — send hold at origin (centre screen, 0.5m depth)
                if last_sent is None or last_sent[3] != "F":
                    packet = "F:0.0000,0.0000,0.0000\n"
                    if ser:
                        ser.write(packet.encode())
                    else:
                        print(packet, end="")
                    last_sent = (0.0, 0.0, 0.0, "F")

                cv2.putText(frame, "No hand — holding", (16, 40),
                            font, 0.7, (80, 80, 80), 2)
                draw_workspace_box(frame, None, w, h)  # no hand — show full box
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