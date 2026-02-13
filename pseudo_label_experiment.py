"""
DAL-Shemagh v21 — Pseudo-Labeling (Self-Training)
# ============================================================
# STRATEGY:
# 1. Use best model (RT-DETR-L, 0.777 LB) to label the Test Set.
# 2. Filter for high confidence boxes (> 0.60).
# 3. Combine TRAIN + PSEUDO-TEST data.
# 4. Retrain RT-DETR-L on this larger, domain-adapted dataset.
# 5. Generate submission with the new model.
# ============================================================
"""
import os
import sys
import shutil
import yaml
from ultralytics import YOLO, RTDETR
from distutils.dir_util import copy_tree

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
CONF_THRESHOLD = 0.60  # Only trust high-conf predictions for pseudo-labels
EPOCHS = 50            # Fine-tune epochs
IMG_SIZE = 640

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL = os.path.join(SCRIPT_DIR, "models", "map_rtdetr_l_best.pt")

POSSIBLE_PATHS = [
    "/kaggle/input/dal-shemagh-identification",
    "/kaggle/input/dal-shemagh-detection-challenge",
    "./data/dal-shemagh-detection-challenge",
    "./data",
    "../input/dal-shemagh-identification"
]

ROOT_DIR = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p) and os.path.exists(f"{p}/images/test"):
        ROOT_DIR = p
        break

if ROOT_DIR is None:
    print("ERROR: Dataset not found")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Prepare Pseudo-Dataset
# ══════════════════════════════════════════════════════════════════════════════
WORK_DIR = "./pseudo_dataset"
IMG_DIR = f"{WORK_DIR}/images/train"
LBL_DIR = f"{WORK_DIR}/labels/train"

if os.path.exists(WORK_DIR): shutil.rmtree(WORK_DIR)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

print(f"\n1. PREPARING DATASET in {WORK_DIR}...")

# Symlink ORIGINAL TRAIN data
orig_train_img = f"{ROOT_DIR}/images/train"
orig_train_lbl = f"{ROOT_DIR}/labels/train"

print("   Linking original training data...")
for f in os.listdir(orig_train_img):
    if f.endswith('.jpg'):
        os.symlink(os.path.abspath(f"{orig_train_img}/{f}"), f"{IMG_DIR}/{f}")
for f in os.listdir(orig_train_lbl):
    if f.endswith('.txt'):
         # Copy labels (don't symlink, we might want to modify/view them)
        shutil.copy(f"{orig_train_lbl}/{f}", f"{LBL_DIR}/{f}")

# Symlink TEST data (images only)
print("   Linking test data...")
test_dir = f"{ROOT_DIR}/images/test"
test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
for f in test_files:
    os.symlink(os.path.abspath(f"{test_dir}/{f}"), f"{IMG_DIR}/{f}")

print(f"   Total Images: {len(os.listdir(IMG_DIR))}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Generate Pseudo-Labels
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n2. GENERATING PSEUDO-LABELS (Conf > {CONF_THRESHOLD})...")
print(f"   Model: {os.path.basename(BEST_MODEL)}")

model = RTDETR(BEST_MODEL)
pseudo_count = 0

for i, fname in enumerate(test_files):
    img_path = f"{test_dir}/{fname}"
    
    # TTA Inference for best quality labels
    results = model.predict(img_path, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, augment=True, verbose=False)[0]
    
    label_path = f"{LBL_DIR}/{fname.replace('.jpg', '.txt')}"
    
    lines = []
    if hasattr(results, 'boxes'):
        for box in results.boxes:
            cls = int(box.cls[0])
            # Save normalized xywh
            x, y, w, h = box.xywhn[0].tolist()
            lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    
    # Write label file (even if empty, to confirm it was processed, 
    # though for training usually we only care about files with targets or background)
    # RT-DETR handles empty label files as background images.
    with open(label_path, 'w') as f:
        f.write("\n".join(lines))
        
    if lines: pseudo_count += 1
    if i % 100 == 0: print(f"   Processed {i}/{len(test_files)}...")

print(f"   Pseudo-labels generated for {pseudo_count}/{len(test_files)} test images.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Create YAML
# ══════════════════════════════════════════════════════════════════════════════
yaml_content = f"""path: {os.path.abspath(WORK_DIR)}
train: images/train
val: images/train  # Validate on everything (or split if you prefer)
names:
  0: person
  1: shemagh
"""
with open(f"{WORK_DIR}/data.yaml", 'w') as f:
    f.write(yaml_content)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Train
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n3. TRAINING (Fine-tuning for {EPOCHS} epochs)...")

# Load the best model to fine-tune
train_model = RTDETR(BEST_MODEL)
train_model.train(
    data=f"{WORK_DIR}/data.yaml",
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=8,
    project='pseudo_runs',
    name='rtdetr_pseudo',
    exist_ok=True,
    augment=True,      # Keep augmentations
    optimizer='auto'
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Final Submission (Hybrid v20 Logic: New mAP Model + Old F1 Specialists)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n4. GENERATING SUBMISSION...")

NEW_MODEL_PATH = "pseudo_runs/rtdetr_pseudo/weights/best.pt"
F1_HEAD = os.path.join(SCRIPT_DIR, "models", "head_m640_best.pt")
F1_SHEM = os.path.join(SCRIPT_DIR, "models", "shemagh_m640_best.pt")

if not os.path.exists(NEW_MODEL_PATH):
    print("Optimization failed? best.pt not found. Using last known best.")
    NEW_MODEL_PATH = BEST_MODEL

# Load models
map_model = RTDETR(NEW_MODEL_PATH)
f1_head = YOLO(F1_HEAD)
f1_shem = YOLO(F1_SHEM)

submission = []
count = 0

def get_overlap(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax1, ay1, ax2, ay2 = ax-aw/2, ay-ah/2, ax+aw/2, ay+ah/2
    bx1, by1, bx2, by2 = bx-bw/2, by-bh/2, bx+bw/2, by+bh/2
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    min_area = min(aw * ah, bw * bh)
    if min_area <= 0: return 0
    return inter / min_area

for fname in test_files:
    img_path = f"{test_dir}/{fname}"
    if count % 100 == 0: print(f"   Predicting {count}...")
    count += 1
    
    # F1 Logic (Right Place)
    rp = 0
    try:
        res_h = f1_head.predict(img_path, conf=0.15, augment=True, verbose=False)[0]
        res_s = f1_shem.predict(img_path, conf=0.15, augment=True, verbose=False)[0]
        h_boxes = [b.xywhn[0].tolist() for b in res_h.boxes]
        s_boxes = [b.xywhn[0].tolist() for b in res_s.boxes]
        if h_boxes and s_boxes:
            for h in h_boxes:
                for s in s_boxes:
                    if get_overlap(h, s) > 0.10:
                        rp = 1; break
                if rp: break
    except: rp=0
    
    # mAP Logic (New Pseudo Model)
    # TTA Enabled
    try:
        res_map = map_model.predict(img_path, conf=0.15, imgsz=640, augment=True, verbose=False)[0]
        parts = []
        if hasattr(res_map, 'boxes'):
            for box in res_map.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x, y, w, h = box.xywhn[0].tolist()
                parts.extend([str(cls), f"{conf:.4f}", f"{x:.4f}", f"{y:.4f}", f"{w:.4f}", f"{h:.4f}"])
        pred_str = " ".join(parts) if parts else "-"
    except: pred_str = "-"
    
    submission.append([fname, rp, pred_str])

with open('submission_pseudo.csv', 'w') as f:
    f.write("filename,right_place,prediction_string\n")
    for row in submission:
        f.write(f"{row[0]},{row[1]},{row[2]}\n")

print("\nSaved submission_pseudo.csv")
