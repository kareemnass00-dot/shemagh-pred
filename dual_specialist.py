"""
DAL-Shemagh v12 — Dual Specialist Models (Optimized)
# ============================================================
# Final Score = 0.5 × mAP@[0.5:0.95] + 0.5 × F1-Score
# ============================================================
# Head:    yolov8m @ 640px  → F1=0.969 on ground truth
# Shemagh: yolov8m @ 640px + grayscale copies → F1=0.948
# right_place: all-pairs overlap check (threshold 0.10)
"""
import os
import random
import shutil
import cv2
from pathlib import Path
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
# 0. Path Detection & Config
# ══════════════════════════════════════════════════════════════════════════════
POSSIBLE_PATHS = [
    "/kaggle/input/dal-shemagh-identification",
    "/kaggle/input/dal-shemagh-detection-challenge",
    "./data/dal-shemagh-detection-challenge",
    "./data",
    "../input/dal-shemagh-identification"
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

os.system('pip install -U ultralytics')

print(f"Using Data at: {ROOT_DIR}")

# Experiment-proven optimal settings
SEED = 42
VAL_RATIO = 0.2
EPOCHS = 100
PATIENCE = 30

# Head: yolov8m @ 640 (F1=0.969)
HEAD_MODEL = "yolov8m.pt"
HEAD_IMGSZ = 640
HEAD_BATCH = 32

# Shemagh: yolov8m @ 640 + grayscale (F1=0.948)
SHEMAGH_MODEL = "yolov8m.pt"
SHEMAGH_IMGSZ = 640
SHEMAGH_BATCH = 32

# Heavy augmentation (proven in experiments)
COMMON_AUG = dict(
    hsv_h=0.5,
    hsv_s=0.9,
    hsv_v=0.5,
    degrees=30.0,
    translate=0.2,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.3,
    erasing=0.3,
    copy_paste=0.1,
)

# Right-place overlap threshold
OVERLAP_THRESHOLD = 0.10

# ══════════════════════════════════════════════════════════════════════════════
# 1. Prepare Split Datasets
# ══════════════════════════════════════════════════════════════════════════════
def prepare_specialist_data(class_id, work_dir, include_grayscale=False):
    """Prepare single-class YOLO training data with optional grayscale copies."""
    class_name = 'Head' if class_id == 0 else 'Shemagh'
    print(f"\nPreparing dataset for {class_name} (class {class_id}) in {work_dir}...")

    work_dir_path = Path(work_dir)
    for sub in ("images", "labels"):
        existing = work_dir_path / sub
        if existing.exists():
            shutil.rmtree(existing)

    for split in ("train", "val"):
        (work_dir_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (work_dir_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    root_images = Path(ROOT_DIR) / "images" / "train"
    root_labels = Path(ROOT_DIR) / "labels" / "train"
    all_files = sorted([p.name for p in root_images.iterdir() if p.suffix.lower() == ".jpg"])

    # Stratified split
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

    rng = random.Random(SEED)
    rng.shuffle(positive_files)
    rng.shuffle(negative_files)

    # Include ~50% negatives for better precision
    neg_sample = negative_files[:len(positive_files) // 2]

    all_train_files = positive_files + neg_sample
    rng.shuffle(all_train_files)

    n_val = int(len(all_train_files) * VAL_RATIO)
    val_files = all_train_files[:n_val]
    train_files = all_train_files[n_val:]

    stats = {"train": 0, "val": 0, "train_instances": 0, "val_instances": 0}
    
    for split, files in [("train", train_files), ("val", val_files)]:
        for fname in files:
            # Copy image
            shutil.copy2(root_images / fname, work_dir_path / "images" / split / fname)
            
            # Filter label (remap class_id → 0)
            lbl_name = f"{Path(fname).stem}.txt"
            src_lbl = root_labels / lbl_name
            dst_lbl = work_dir_path / "labels" / split / lbl_name
            
            if src_lbl.exists():
                with open(src_lbl, "r") as f_in, open(dst_lbl, "w") as f_out:
                    for line in f_in:
                        parts = line.split()
                        if len(parts) == 5:
                            try:
                                if int(parts[0]) == class_id:
                                    f_out.write(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                                    stats[f"{split}_instances"] += 1
                            except ValueError:
                                continue
            else:
                open(dst_lbl, "w").close()
            
            # Add grayscale copy for training set
            if include_grayscale and split == "train":
                gray_name = fname.replace('.jpg', '_gray.jpg')
                img = cv2.imread(str(root_images / fname))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(str(work_dir_path / "images" / split / gray_name), gray_bgr)
                # Copy same label
                gray_lbl = lbl_name.replace('.txt', '_gray.txt')
                shutil.copy2(dst_lbl, work_dir_path / "labels" / split / gray_lbl)
            
            stats[split] += 1

    yaml_content = f"""path: {os.path.abspath(work_dir)}
train: images/train
val: images/val
nc: 1
names: ['{class_name}']
"""
    with open(f"{work_dir}/data.yaml", 'w') as f:
        f.write(yaml_content)

    print(f"  {class_name}: {stats['train']} train ({stats['train_instances']} instances), "
          f"{stats['val']} val ({stats['val_instances']} instances)"
          + (" (+grayscale copies)" if include_grayscale else ""))

# ══════════════════════════════════════════════════════════════════════════════
# 2. Prepare Data
# ══════════════════════════════════════════════════════════════════════════════
WORK_DIR_HEAD = "./yolo_head"
WORK_DIR_SHEMAGH = "./yolo_shemagh"

prepare_specialist_data(0, WORK_DIR_HEAD, include_grayscale=False)
prepare_specialist_data(1, WORK_DIR_SHEMAGH, include_grayscale=True)  # Grayscale copies for shemagh

# ══════════════════════════════════════════════════════════════════════════════
# 3. Train Two Models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"  TRAINING HEAD MODEL — {HEAD_MODEL} @ {HEAD_IMGSZ}px")
print("="*60)
model_head = YOLO(HEAD_MODEL)
model_head.train(
    data=f"{WORK_DIR_HEAD}/data.yaml",
    epochs=EPOCHS, imgsz=HEAD_IMGSZ, batch=HEAD_BATCH,
    project='./models', name='head_model',
    patience=PATIENCE, exist_ok=True,
    **COMMON_AUG
)

# Get best weights
best_head = str(model_head.trainer.best)
print(f"  Head best weights: {best_head}")
assert os.path.exists(best_head), f"Head weights not found: {best_head}"

print("\n" + "="*60)
print(f"  TRAINING SHEMAGH MODEL — {SHEMAGH_MODEL} @ {SHEMAGH_IMGSZ}px + grayscale")
print("="*60)
model_shemagh = YOLO(SHEMAGH_MODEL)
model_shemagh.train(
    data=f"{WORK_DIR_SHEMAGH}/data.yaml",
    epochs=EPOCHS, imgsz=SHEMAGH_IMGSZ, batch=SHEMAGH_BATCH,
    project='./models', name='shemagh_model',
    patience=PATIENCE, exist_ok=True,
    **COMMON_AUG
)

best_shemagh = str(model_shemagh.trainer.best)
print(f"  Shemagh best weights: {best_shemagh}")
assert os.path.exists(best_shemagh), f"Shemagh weights not found: {best_shemagh}"

# Load best weights for inference
model_head = YOLO(best_head)
model_shemagh = YOLO(best_shemagh)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Inference & Logic
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  DUAL MODEL INFERENCE")
print("="*60)

def get_overlap(box_a, box_b):
    """Calculate IoU-style overlap between two xywh normalized boxes.
    Returns intersection / area of smaller box (containment ratio)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    
    ax1, ay1, ax2, ay2 = ax-aw/2, ay-ah/2, ax+aw/2, ay+ah/2
    bx1, by1, bx2, by2 = bx-bw/2, by-bh/2, bx+bw/2, by+bh/2
    
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter_area = iw * ih
    
    # Use smaller box area as denominator (containment check)
    area_a = aw * ah
    area_b = bw * bh
    min_area = min(area_a, area_b)
    if min_area <= 0:
        return 0
    return inter_area / min_area

test_dir = f"{ROOT_DIR}/images/test"
test_files = sorted(os.listdir(test_dir))
submission = []

count = 0
total = len(test_files)

for fname in test_files:
    img_path = f"{test_dir}/{fname}"
    if count % 100 == 0:
        print(f"Processing {count}/{total}...")
    count += 1
    
    # Predict Head
    res_h = model_head.predict(img_path, conf=0.15, imgsz=HEAD_IMGSZ, augment=True, verbose=False)[0]
    heads = []
    for box in res_h.boxes:
        heads.append(box.xywhn[0].tolist() + [float(box.conf[0])])

    # Predict Shemagh
    res_s = model_shemagh.predict(img_path, conf=0.15, imgsz=SHEMAGH_IMGSZ, augment=True, verbose=False)[0]
    shemaghs = []
    for box in res_s.boxes:
        shemaghs.append(box.xywhn[0].tolist() + [float(box.conf[0])])

    # ── Right Place: ALL-PAIRS OVERLAP CHECK ──
    # Check every head-shemagh pair, if ANY pair overlaps → right_place=1
    rp = 0
    if heads and shemaghs:
        for h in heads:
            for s in shemaghs:
                if get_overlap(h[:4], s[:4]) > OVERLAP_THRESHOLD:
                    rp = 1
                    break
            if rp:
                break

    # ── Prediction String for mAP ──
    # Report ALL detections with original class IDs
    parts = []
    for h in heads:
        parts.extend(["0", f"{h[4]:.4f}", f"{h[0]:.4f}", f"{h[1]:.4f}", f"{h[2]:.4f}", f"{h[3]:.4f}"])
    for s in shemaghs:
        parts.extend(["1", f"{s[4]:.4f}", f"{s[0]:.4f}", f"{s[1]:.4f}", f"{s[2]:.4f}", f"{s[3]:.4f}"])
    
    pred_str = " ".join(parts) if parts else "-"
    submission.append([fname, rp, pred_str])

# ══════════════════════════════════════════════════════════════════════════════
# 5. Save Submission
# ══════════════════════════════════════════════════════════════════════════════
right_place_count = sum([x[1] for x in submission])
print(f"\nDone! right_place=1: {right_place_count}/{total}")

with open('submission_dual.csv', 'w') as f:
    f.write("filename,right_place,prediction_string\n")
    for row in submission:
        f.write(f"{row[0]},{row[1]},{row[2]}\n")

print("Saved submission_dual.csv")
