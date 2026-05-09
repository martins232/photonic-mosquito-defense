import os

for split in ["train", "valid", "test"]:
    img_dir = f"dataset/{split}/images"
    lbl_dir = f"dataset/{split}/labels"
    
    imgs = set(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg"))
    lbls = set(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".txt"))
    
    print(f"\n{split.upper()}")
    print(f"  Images : {len(imgs)}")
    print(f"  Labels : {len(lbls)}")
    print(f"  Unmatched images (no label): {len(imgs - lbls)}")
    print(f"  Unmatched labels (no image): {len(lbls - imgs)}")
    
    
# src/detector/evaluate.py
from ultralytics import YOLO
import os

MODEL_PATH = "models/mosquito_species_v1/weights/best.pt"
DATA_YAML  = os.path.abspath("dataset/data.yaml")

model   = YOLO(MODEL_PATH)
metrics = model.val(data=DATA_YAML, split="test")

class_names = [
    "Aedes", "Anopheles funestus", "Anopheles gambiae",
    "Anopheles other", "Culex", "Mansonia", "Non-mosquito"
]

print("\n" + "="*55)
print("  MOSQUITO SPECIES DETECTOR — TEST SET RESULTS")
print("="*55)
print(f"  Overall mAP@50      : {metrics.box.map50:.3f}")
print(f"  Overall mAP@50-95   : {metrics.box.map:.3f}")
print(f"  Precision           : {metrics.box.mp:.3f}")
print(f"  Recall              : {metrics.box.mr:.3f}")
print("\n  Per-class mAP@50:")
print("  " + "-"*40)

for i, name in enumerate(class_names):
    try:
        class_map = metrics.box.maps[i]
        bar = "█" * int(class_map * 20)
        print(f"  {name:<25} {class_map:.3f}  {bar}")
    except IndexError:
        print(f"  {name:<25} N/A")

print("="*55)