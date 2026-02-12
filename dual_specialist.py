"""
DAL-Shemagh v11 — Dual Specialist Models
# ============================================================
# 1. SETUP & PATHS (Portable Relative Paths)
# ============================================================
# Assumes data is in ./data (Repo Structure)
Idea: Train two separate models to avoid confusion and NMS issues.
  1. Model A: Detects ONLY Heads (class 0)
  2. Model B: Detects ONLY Shemaghs (class 1)
  3. Inference: Combine detections + Containment Logic
Start small with yolov8n to test the concept quickly.
"""
import os
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

# Configuration
# Robust Path Detection
POSSIBLE_PATHS = [
    "/kaggle/input/dal-shemagh-identification",  # Standard Kaggle
    "/kaggle/input/dal-shemagh-detection-challenge", # Alternate Kaggle
    "./data/dal-shemagh-detection-challenge",    # Local Nested
    "./data",                                    # Local Flat
    "../input/dal-shemagh-identification"        # Relative Parent
]

ROOT_DIR = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p) and os.path.exists(f"{p}/images/train"):
        ROOT_DIR = p
        break

if ROOT_DIR is None:
    print(f"ERROR: Could not find dataset in {POSSIBLE_PATHS}")
    print("Files in current dir:", os.listdir("."))
    if os.path.exists("./data"): print("Files in ./data:", os.listdir("./data"))
    exit(1)
    
WORK_DIR = "."

# Update Ultralytics for H100
os.system('pip install -U ultralytics')

# Data Prep
train_csv_path = f"{ROOT_DIR}/train.csv"
train_dir = f"{ROOT_DIR}/images/train"
test_dir = f"{ROOT_DIR}/images/test"

print(f"Using Data at: {ROOT_DIR}")

# H100 Settings (Adjusted for OOM)
BATCH_SIZE = 8
EPOCHS = 100
IMGSZ = 1280
WORK_DIR_HEAD = "./yolo_head"
WORK_DIR_SHEMAGH = "./yolo_shemagh"
SEED = int(os.getenv("SEED", "42"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.2"))

# ══════════════════════════════════════════════════════════════════════════════
# 1. Prepare Split Datasets (Head-only / Shemagh-only)
# ══════════════════════════════════════════════════════════════════════════════
def prepare_specialist_data(class_id, work_dir, *, val_ratio=VAL_RATIO, seed=SEED):
    print(f"\nPreparing dataset for Class {class_id} in {work_dir}...")

    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1 (got {val_ratio})")

    work_dir_path = Path(work_dir)
    for sub in ("images", "labels"):
        existing = work_dir_path / sub
        if existing.exists():
            shutil.rmtree(existing)

    # Create dirs
    for split in ("train", "val"):
        (work_dir_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (work_dir_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Stratified split: ensure both splits contain examples of the target class (if any exist).
    root_images = Path(ROOT_DIR) / "images" / "train"
    root_labels = Path(ROOT_DIR) / "labels" / "train"
    all_files = sorted([p.name for p in root_images.iterdir() if p.suffix.lower() == ".jpg"])

    positive_files = []
    negative_files = []
    for fname in all_files:
        src_lbl = root_labels / f"{Path(fname).stem}.txt"
        has_class = False
        if src_lbl.exists():
            for line in src_lbl.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    if int(parts[0]) == class_id:
                        has_class = True
                        break
                except ValueError:
                    continue
        (positive_files if has_class else negative_files).append(fname)

    rng = random.Random(seed)
    rng.shuffle(positive_files)
    rng.shuffle(negative_files)

    n_val_pos = int(round(len(positive_files) * val_ratio))
    if positive_files and n_val_pos == 0:
        n_val_pos = 1
    n_val_neg = int(round(len(negative_files) * val_ratio))

    val_files = positive_files[:n_val_pos] + negative_files[:n_val_neg]
    train_files = positive_files[n_val_pos:] + negative_files[n_val_neg:]
    rng.shuffle(train_files)
    rng.shuffle(val_files)
    
    # Process files
    stats = {
        "train": {"images": len(train_files), "images_with_obj": 0, "instances": 0},
        "val": {"images": len(val_files), "images_with_obj": 0, "instances": 0},
    }
    for split, files in [("train", train_files), ("val", val_files)]:
        for fname in files:
            # Copy Image
            shutil.copy2(root_images / fname, work_dir_path / "images" / split / fname)
            
            # Filter Label
            lbl_name = f"{Path(fname).stem}.txt"
            src_lbl = root_labels / lbl_name
            dst_lbl = work_dir_path / "labels" / split / lbl_name
            
            if os.path.exists(src_lbl):
                has_obj = False
                with open(src_lbl, "r") as f_in, open(dst_lbl, "w") as f_out:
                    for line in f_in:
                        parts = line.split()
                        if len(parts) != 5:
                            continue
                        try:
                            src_class_id = int(parts[0])
                        except ValueError:
                            continue
                        if src_class_id == class_id:
                            # Remap class to 0 (single-class model), remap back at inference.
                            f_out.write(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                            stats[split]["instances"] += 1
                            has_obj = True
                if has_obj:
                    stats[split]["images_with_obj"] += 1
            else:
                open(dst_lbl, "w").close()  # Empty

    print(
        f"Class {class_id} split stats | "
        f"train: {stats['train']['images']} imgs, {stats['train']['images_with_obj']} with obj, {stats['train']['instances']} instances | "
        f"val: {stats['val']['images']} imgs, {stats['val']['images_with_obj']} with obj, {stats['val']['instances']} instances"
    )
    
    # Config YAML
    yaml_content = f"""path: {os.path.abspath(work_dir)}
train: images/train
val: images/val
nc: 1
names: ['{'Head' if class_id==0 else 'Shemagh'}']
"""
    with open(f"{work_dir}/data.yaml", 'w') as f:
        f.write(yaml_content)

prepare_specialist_data(0, WORK_DIR_HEAD)
prepare_specialist_data(1, WORK_DIR_SHEMAGH)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Train Two Models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  TRAINING HEAD MODEL (Class 0)")
print("="*60)
model_head = YOLO('yolov8x.pt')
# Strong HSV Agumentation to handle blue/green tinted images
# H100 Training (1280) with Heavy Model (X)
model_head.train(data=f"{WORK_DIR_HEAD}/data.yaml", epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH_SIZE, 
                 project='./models', name='head_model', 
                 hsv_h=0.3, hsv_s=0.7, hsv_v=0.4, degrees=10.0, fliplr=0.5, patience=50)

print("\n" + "="*60)
print("  TRAINING SHEMAGH MODEL (Class 1) — Medium Model @ 640px")
print("="*60)
# Shemagh has far fewer instances (~100 train) so yolov8x overfits badly
# Use yolov8m (medium) at 640px with extra augmentation for better generalization
model_shemagh = YOLO('yolov8m.pt')
model_shemagh.train(data=f"{WORK_DIR_SHEMAGH}/data.yaml", epochs=EPOCHS, imgsz=640, batch=16,
                    project='./models', name='shemagh_model',
                    hsv_h=0.3, hsv_s=0.7, hsv_v=0.4, degrees=15.0, fliplr=0.5,
                    mosaic=1.0, mixup=0.3, copy_paste=0.1, patience=50)

# ══════════════════════════════════════════════════════════════════════════════
# 3. Inference & Logic
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  DUAL MODEL INFERENCE (Heavy + TTA)")
print("="*60)

def get_containment(head_box, shemagh_box):
    # Normalized xywh
    hx, hy, hw, hh = head_box
    sx, sy, sw, sh = shemagh_box
    
    # Convert to corners
    hx1, hy1, hx2, hy2 = hx-hw/2, hy-hh/2, hx+hw/2, hy+hh/2
    sx1, sy1, sx2, sy2 = sx-sw/2, sy-sh/2, sx+sw/2, sy+sh/2
    
    # Intersection
    ix1 = max(hx1, sx1); iy1 = max(hy1, sy1)
    ix2 = min(hx2, sx2); iy2 = min(hy2, sy2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter_area = iw * ih
    
    head_area = hw * hh
    if head_area <= 0: return 0
    return inter_area / head_area

test_dir = f"{ROOT_DIR}/images/test"
test_files = sorted(os.listdir(test_dir))
submission = []

count = 0
total = len(test_files)

for fname in test_files:
    img_path = f"{test_dir}/{fname}"
    if count % 50 == 0: print(f"Processing {count}/{total}...")
    count += 1
    
    # Predict Head (Balanced Confidence, TTA Enabled for mAP boost)
    res_h = model_head.predict(img_path, conf=0.15, imgsz=IMGSZ, augment=True, verbose=False)[0]
    heads = []
    for box in res_h.boxes:
        heads.append(box.xywhn[0].tolist() + [float(box.conf[0])]) # x,y,w,h,conf

    # Predict Shemagh (Medium Model @ 640px, TTA Enabled)
    res_s = model_shemagh.predict(img_path, conf=0.15, imgsz=640, augment=True, verbose=False)[0]
    shemaghs = []
    for box in res_s.boxes:
        shemaghs.append(box.xywhn[0].tolist() + [float(box.conf[0])])

    # Sort lists by confidence descending (High -> Low)
    heads.sort(key=lambda x: x[4], reverse=True)
    shemaghs.sort(key=lambda x: x[4], reverse=True)

    # 1. Logic for Right Place (F1 Score)
    # Use ONLY the Single Best Head and Single Best Shemagh
    rp = 0
    if len(heads) > 0 and len(shemaghs) > 0:
        best_h = heads[0]      # Highest confidence head
        best_s = shemaghs[0]   # Highest confidence shemagh
        
        # Only consider valid top candidates (> 0.25)
        if best_h[4] > 0.25 and best_s[4] > 0.25:
            # Check containment (Strict > 20%)
            h_box = best_h[:4]
            s_box = best_s[:4]
            if get_containment(h_box, s_box) > 0.20:
                rp = 1

    # 2. Prediction String for mAP Score
    # Report ALL boxes (down to 0.01) to maximize Recall
    parts = []
    for h in heads:
        parts.extend(["0", f"{h[4]:.4f}", f"{h[0]:.4f}", f"{h[1]:.4f}", f"{h[2]:.4f}", f"{h[3]:.4f}"])
    for s in shemaghs:
        parts.extend(["1", f"{s[4]:.4f}", f"{s[0]:.4f}", f"{s[1]:.4f}", f"{s[2]:.4f}", f"{s[3]:.4f}"])
    
    pred_str = " ".join(parts) if parts else "-"
    submission.append([fname, rp, pred_str])

# 4. Save Submission
print("Done! Saved submission_dual.csv")
right_place_count = sum([x[1] for x in submission])
print(f"right_place=1: {right_place_count}")

# Save the normal submission
with open('submission_dual.csv', 'w') as f:
    f.write("filename,right_place,prediction_string\n")
    for row in submission:
        f.write(f"{row[0]},{row[1]},{row[2]}\n")
