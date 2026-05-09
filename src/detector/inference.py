"""
Photonic Mosquito Defense System v4
- Auto-screenshot on species lock
- Keyboard controls: P=pause, S=screenshot, Q=quit
- Resizable display window (output video stays full resolution)
"""

import cv2
import csv
import os
import numpy as np
from collections import Counter
from datetime import datetime
from ultralytics import YOLO

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH     = "models/mosquito_species_v1/weights/best.pt"
VIDEO_PATH     = "data/raw/mosquito_2.mp4"
OUTPUT_DIR     = "outputs"
SCREENSHOT_DIR = "outputs/screenshots"
os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

CONF_THRESHOLD = 0.40
IOU_THRESHOLD  = 0.50
TRAIL_LENGTH   = 25
PREDICT_STEPS  = 12
LOCK_CONF      = 0.75
DISPLAY_SCALE  = 0.5    # 0.5 = half size display; change to 0.6 or 0.7 as needed

# ── species colors (BGR) ──────────────────────────────────────────────────────
CLASS_COLORS = {
    "Aedes":              (0,   165, 255),
    "Anopheles funestus": (0,   0,   255),
    "Anopheles gambiae":  (0,   0,   200),
    "Anopheles other":    (100, 0,   200),
    "Culex":              (255, 150, 0  ),
    "Mansonia":           (0,   200, 200),
    "Non-mosquito":       (150, 150, 150),
}
CLASS_NAMES = list(CLASS_COLORS.keys())

# ── load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded.\n")
print("Controls:")
print("  P = pause/resume")
print("  S = save screenshot (works while paused or playing)")
print("  Q = quit\n")

# ── video setup ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

out = cv2.VideoWriter(
    f"{OUTPUT_DIR}/detection_output_v4.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ── event log ─────────────────────────────────────────────────────────────────
log_file = open(f"{OUTPUT_DIR}/detection_log_v4.csv", "w", newline="")
writer   = csv.writer(log_file)
writer.writerow([
    "timestamp", "frame", "track_id", "species", "locked",
    "confidence", "x_center", "y_center", "bbox_w", "bbox_h",
    "predicted_x", "predicted_y"
])

# ── tracking state ────────────────────────────────────────────────────────────
track_history        = {}
track_species_log    = {}
track_locked_species = {}
screenshot_taken     = set()   # track IDs already auto-screenshotted

frame_num        = 0
total_detections = 0
paused           = False

# ── helper functions ──────────────────────────────────────────────────────────
def get_stable_species(tid, current_species, current_conf):
    if tid not in track_locked_species and current_conf >= LOCK_CONF:
        track_locked_species[tid] = current_species
    if tid in track_locked_species:
        return track_locked_species[tid], True
    history  = track_species_log.get(tid, [current_species])
    weighted = history + history[-5:]
    stable   = Counter(weighted).most_common(1)[0][0]
    return stable, False


def predict_position(history, steps=PREDICT_STEPS):
    if len(history) < 3:
        return None
    recent = history[-6:]
    vx = np.mean([recent[i][0] - recent[i-1][0] for i in range(1, len(recent))])
    vy = np.mean([recent[i][1] - recent[i-1][1] for i in range(1, len(recent))])
    lx, ly = history[-1]
    px = max(0, min(w - 1, int(lx + vx * steps)))
    py = max(0, min(h - 1, int(ly + vy * steps)))
    return (px, py)


def draw_trail(frame, history, color):
    for i in range(1, len(history)):
        alpha     = i / len(history)
        thickness = max(1, int(alpha * 3))
        faded     = tuple(int(c * alpha) for c in color)
        cv2.line(frame, history[i-1], history[i], faded, thickness)


def draw_reticle(frame, current_pos, predicted_pos, color=(0, 0, 255)):
    x, y = current_pos
    size = 28
    x = max(size+15, min(w-size-15, x))
    y = max(size+20, min(h-size-5,  y))

    cv2.circle(frame, (x, y), size,     color, 2)
    cv2.circle(frame, (x, y), 4,        color, -1)
    cv2.circle(frame, (x, y), size + 6, color, 1)

    gap = 6
    cv2.line(frame, (x-size-gap-8, y), (x-size-gap,     y), color, 2)
    cv2.line(frame, (x+size+gap,   y), (x+size+gap+8,   y), color, 2)
    cv2.line(frame, (x, y-size-gap-8), (x, y-size-gap    ), color, 2)
    cv2.line(frame, (x, y+size+gap),   (x, y+size+gap+8  ), color, 2)

    cv2.putText(frame, "TARGET LOCKED",
                (x-52, y-size-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    if predicted_pos is not None:
        px, py    = predicted_pos
        px        = max(10, min(w-10, px))
        py        = max(10, min(h-10, py))
        d         = 8
        pts       = np.array([[px,py-d],[px+d,py],[px,py+d],[px-d,py]], np.int32)
        pred_color = (0, 100, 255)
        cv2.polylines(frame, [pts], True, pred_color, 1)
        cv2.line(frame, (x, y), (px, py), pred_color, 1)
        cv2.putText(frame, "INTERCEPT", (px+10, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, pred_color, 1)


def save_screenshot(frame, label="manual"):
    filename = f"{SCREENSHOT_DIR}/{label}_frame{frame_num}.jpg"
    cv2.imwrite(filename, frame)
    print(f"📸 Screenshot saved: {filename}")


def draw_hud(frame, detection_count, paused):
    cv2.rectangle(frame, (0, 0), (400, 105), (0, 0, 0), -1)
    cv2.putText(frame, "PHOTONIC MOSQUITO DEFENSE SYSTEM",
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 100), 1)
    cv2.putText(frame, f"Frame: {frame_num}  |  Active targets: {detection_count}",
                (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
    cv2.putText(frame, f"Total detections: {total_detections}  |  mAP@50: 0.971",
                (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
    cv2.putText(frame, f"Model: YOLOv8n  |  Species: 7  |  CPU only",
                (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 150), 1)
    cv2.putText(frame, f"[P] Pause  [S] Screenshot  [Q] Quit",
                (8, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 180, 100), 1)

    # paused indicator
    if paused:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 255), 3)
        cv2.putText(frame, "PAUSED — press any key to resume",
                    (w//2 - 180, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


# ── main loop ─────────────────────────────────────────────────────────────────
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num      += 1
        detection_count = 0

        results = model.track(
            frame,
            conf    = CONF_THRESHOLD,
            iou     = IOU_THRESHOLD,
            persist = True,
            verbose = False
        )[0]

        if results.boxes is not None and len(results.boxes) > 0:
            has_ids = results.boxes.id is not None

            for i, box in enumerate(results.boxes):
                cls_id  = int(box.cls[0])
                species = CLASS_NAMES[cls_id]
                conf    = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                bw = x2 - x1
                bh = y2 - y1

                tid = int(results.boxes.id[i]) if has_ids else i

                if tid not in track_history:
                    track_history[tid]     = []
                    track_species_log[tid] = []

                track_history[tid].append((cx, cy))
                track_species_log[tid].append(species)
                if len(track_history[tid]) > TRAIL_LENGTH:
                    track_history[tid].pop(0)

                stable_species, is_locked = get_stable_species(tid, species, conf)
                stable_color = CLASS_COLORS[stable_species]
                predicted    = predict_position(track_history[tid])

                # draw
                draw_trail(frame, track_history[tid], stable_color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), stable_color, 2)

                lock_icon = " [LOCKED]" if is_locked else ""
                label     = f"ID:{tid} {stable_species}{lock_icon} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame,
                              (x1, y1-th-8), (x1+tw+4, y1),
                              stable_color, -1)
                cv2.putText(frame, label, (x1+2, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (255, 255, 255), 1)

                cv2.circle(frame, (cx, cy), 4, stable_color, -1)

                if stable_species != "Non-mosquito":
                    draw_reticle(frame, (cx, cy), predicted, color=(0, 0, 255))
                    detection_count += 1

                    # auto-screenshot on first lock per track
                    if is_locked and tid not in screenshot_taken:
                        screenshot_taken.add(tid)
                        label_clean = stable_species.replace(" ", "_")
                        save_screenshot(
                            frame,
                            label=f"AUTO_target{tid}_{label_clean}"
                        )

                pred_x = predicted[0] if predicted else cx
                pred_y = predicted[1] if predicted else cy
                writer.writerow([
                    datetime.now().isoformat(), frame_num, tid,
                    stable_species, is_locked, f"{conf:.3f}",
                    cx, cy, bw, bh, pred_x, pred_y
                ])

            total_detections += detection_count

        # draw HUD onto frame
        draw_hud(frame, detection_count, paused)

        # write full-resolution frame to output video
        out.write(frame)

    # ── display (scaled down for screen) ─────────────────────────────────────
    display_frame = cv2.resize(
        frame, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
    cv2.imshow("Photonic Mosquito Defense System", display_frame)

    # ── keyboard handling ─────────────────────────────────────────────────────
    key = cv2.waitKey(1 if not paused else 50) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("p"):
        paused = not paused
        print("⏸  Paused." if paused else "▶  Resumed.")
    elif key == ord("s"):
        save_screenshot(frame, label="manual")

# ── cleanup ───────────────────────────────────────────────────────────────────
cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()

print(f"\n✅ Complete.")
print(f"   Frames processed : {frame_num}")
print(f"   Total detections : {total_detections}")
print(f"   Screenshots saved: {len(screenshot_taken)} auto + manual")
print(f"   Output video     : {OUTPUT_DIR}/detection_output_v4.mp4")
print(f"   Screenshots dir  : {SCREENSHOT_DIR}/")
print(f"   Detection log    : {OUTPUT_DIR}/detection_log_v4.csv")