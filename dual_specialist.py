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
import shutil
import pandas as pd
from ultralytics import YOLO
import yaml

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

# H100 Settings
BATCH_SIZE = 32
EPOCHS = 100
IMGSZ = 1280
WORK_DIR_HEAD = "./yolo_head"
WORK_DIR_SHEMAGH = "./yolo_shemagh"

# ══════════════════════════════════════════════════════════════════════════════
# 1. Prepare Split Datasets (Head-only / Shemagh-only)
# ══════════════════════════════════════════════════════════════════════════════
def prepare_specialist_data(class_id, work_dir):
    print(f"\nPreparing dataset for Class {class_id} in {work_dir}...")
    
    # Create dirs
    for split in ['train', 'val']:
        os.makedirs(f"{work_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{work_dir}/labels/{split}", exist_ok=True)

    # We use the full training set for training (no split script here for brevity, 
    # but in real run we'd split. For now, let's just use all for max performance)
    # Actually, let's do a simple 80/20 split on file list
    all_files = sorted([f for f in os.listdir(f"{ROOT_DIR}/images/train") if f.endswith('.jpg')])
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    # Process files
    for split, files in [('train', train_files), ('val', val_files)]:
        for fname in files:
            # Copy Image
            shutil.copy(f"{ROOT_DIR}/images/train/{fname}", f"{work_dir}/images/{split}/{fname}")
            
            # Filter Label
            lbl_name = fname.replace('.jpg', '.txt')
            src_lbl = f"{ROOT_DIR}/labels/train/{lbl_name}"
            dst_lbl = f"{work_dir}/labels/{split}/{lbl_name}"
            
            if os.path.exists(src_lbl):
                with open(src_lbl, 'r') as f_in, open(dst_lbl, 'w') as f_out:
                    lines = f_in.readlines()
                    has_obj = False
                    for line in lines:
                        parts = line.split()
                        if int(parts[0]) == class_id:
                            # Remap class to 0 (since it's a single-class model)
                            # Actually, keep original class ID to avoid confusion later?
                            # YOLO expects 0-indexed for single class usually. 
                            # Let's map everything to class 0 for the model, then remap back at inference.
                            f_out.write(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                            has_obj = True
                    if not has_obj:
                        pass # Empty file created automatically by open/close? No, 'w' creates it.
            else:
                open(dst_lbl, 'w').close() # Empty
    
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
print("  TRAINING SHEMAGH MODEL (Class 1)")
print("="*60)
model_shemagh = YOLO('yolov8x.pt')
model_shemagh.train(data=f"{WORK_DIR_SHEMAGH}/data.yaml", epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH_SIZE,
                    project='./models', name='shemagh_model',
                    hsv_h=0.3, hsv_s=0.7, hsv_v=0.4, degrees=10.0, fliplr=0.5, patience=50)

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

    # Predict Shemagh (Balanced Confidence, TTA Enabled)
    res_s = model_shemagh.predict(img_path, conf=0.15, imgsz=IMGSZ, augment=True, verbose=False)[0]
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
