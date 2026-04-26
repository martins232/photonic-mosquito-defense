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