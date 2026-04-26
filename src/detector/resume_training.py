from ultralytics import YOLO
import torch
import os

# ── This script resumes training from last saved checkpoint ──────────────────

LAST_CHECKPOINT = "models/mosquito_species_v1/weights/last.pt"
DATA_YAML       = os.path.abspath("dataset/data.yaml")
device          = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(LAST_CHECKPOINT):
    print("❌ No checkpoint found. Check the path:")
    print(f"   {LAST_CHECKPOINT}")
    exit()

print(f"✅ Checkpoint found: {LAST_CHECKPOINT}")
print(f"   Resuming training on: {device.upper()}")

model = YOLO(LAST_CHECKPOINT)  # load from checkpoint, not base model

results = model.train(
    data       = DATA_YAML,
    epochs     = 60,          # keep same total — YOLO tracks how many already ran
    imgsz      = 640,
    batch      = 8,
    device     = device,
    project    = "models",
    name       = "mosquito_species_v1",
    exist_ok   = True,        # CRITICAL — prevents overwriting existing run
    resume     = True,        # CRITICAL — tells YOLO to continue, not restart

    # keep all same settings as original run
    optimizer  = "AdamW",
    lr0        = 0.001,
    lrf        = 0.01,
    hsv_h      = 0.015,
    hsv_s      = 0.7,
    hsv_v      = 0.4,
    flipud     = 0.3,
    fliplr     = 0.5,
    mosaic     = 1.0,
    mixup      = 0.1,
    degrees    = 15,
    scale      = 0.5,
    save       = True,
    save_period= 2,
    plots      = True,
    verbose    = True,
)

print("\n✅ Training resumed and completed.")