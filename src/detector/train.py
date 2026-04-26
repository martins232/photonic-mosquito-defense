from ultralytics import YOLO
import torch
import os

# ── confirm GPU availability ──────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device.upper()}")
if device == "cpu":
    print("No GPU detected — training on CPU. This will take longer but works fine.")

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_YAML   = os.path.abspath("dataset/data.yaml")
PROJECT_DIR = os.path.abspath("models")
RUN_NAME    = "mosquito_species_v1"

# ── load base model ───────────────────────────────────────────────────────────
# yolov8n = nano (fastest, least accurate) — good starting point on CPU
# yolov8s = small (better accuracy, still manageable on CPU)
model = YOLO("yolov8n.pt")

# ── training configuration ────────────────────────────────────────────────────
results = model.train(
    data       = DATA_YAML,
    epochs     = 60,
    imgsz      = 640,
    batch      = 8,          # lower if you get memory errors; raise to 16 if fine
    device     = device,
    project    = PROJECT_DIR,
    name       = RUN_NAME,
    exist_ok   = True,

    # optimizer
    optimizer  = "AdamW",
    lr0        = 0.001,
    lrf        = 0.01,       # final LR = lr0 * lrf

    # augmentation — helps smaller classes like Non-mosquito
    hsv_h      = 0.015,      # hue shift
    hsv_s      = 0.7,        # saturation
    hsv_v      = 0.4,        # brightness
    flipud     = 0.3,        # vertical flip (insects can appear any orientation)
    fliplr     = 0.5,        # horizontal flip
    mosaic     = 1.0,        # mosaic augmentation (combines 4 images — great for small datasets)
    mixup      = 0.1,        # blends two images — helps generalization
    degrees    = 15,         # random rotation ±15°
    scale      = 0.5,        # random scale ±50%

    # logging & saving
    save       = True,
    save_period= 5,         # save checkpoint every 5 epochs
    plots      = True,       # generate training curve plots automatically
    verbose    = True,
)

print("\n✅ Training complete.")
print(f"Best weights: {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")