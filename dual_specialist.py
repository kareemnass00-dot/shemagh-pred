"""
DAL-Shemagh v12 — Dual Specialist Models (Optimized)
# ============================================================
# Final Score = 0.5 × mAP@[0.5:0.95] + 0.5 × F1-Score
# ============================================================
# Head:    yolov8m @ 640px  → F1=0.969 on ground truth
# Shemagh: yolov8m @ 640px + grayscale copies → F1=0.948
# right_place: all-pairs overlap check (threshold 0.10)
#
# Mode: Uses pre-trained weights from models/ folder (no training needed)
"""
import os
import sys
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
    if os.path.exists(p) and os.path.exists(f"{p}/images/test"):
        ROOT_DIR = p
        break

if ROOT_DIR is None:
    print(f"ERROR: Could not find dataset in {POSSIBLE_PATHS}")
    print("Files in current dir:", os.listdir("."))
    if os.path.exists("./data"): print("Files in ./data:", os.listdir("./data"))
    sys.exit(1)

os.system('pip install -U ultralytics')

print(f"Using Data at: {ROOT_DIR}")

# Model paths (relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HEAD_WEIGHTS = os.path.join(SCRIPT_DIR, "models", "head_m640_best.pt")
SHEMAGH_WEIGHTS = os.path.join(SCRIPT_DIR, "models", "shemagh_m640_best.pt")

# Inference settings (must match training)
HEAD_IMGSZ = 640
SHEMAGH_IMGSZ = 640

# Right-place overlap threshold
OVERLAP_THRESHOLD = 0.10

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load Pre-trained Models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  LOADING PRE-TRAINED MODELS")
print("="*60)

assert os.path.exists(HEAD_WEIGHTS), f"Head weights not found: {HEAD_WEIGHTS}"
assert os.path.exists(SHEMAGH_WEIGHTS), f"Shemagh weights not found: {SHEMAGH_WEIGHTS}"

model_head = YOLO(HEAD_WEIGHTS)
print(f"  Head model loaded: {HEAD_WEIGHTS}")

model_shemagh = YOLO(SHEMAGH_WEIGHTS)
print(f"  Shemagh model loaded: {SHEMAGH_WEIGHTS}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Inference & Logic
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  DUAL MODEL INFERENCE")
print("="*60)

def get_overlap(box_a, box_b):
    """Calculate overlap between two xywh normalized boxes.
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
# 3. Save Submission
# ══════════════════════════════════════════════════════════════════════════════
right_place_count = sum([x[1] for x in submission])
print(f"\nDone! right_place=1: {right_place_count}/{total}")

with open('submission_dual.csv', 'w') as f:
    f.write("filename,right_place,prediction_string\n")
    for row in submission:
        f.write(f"{row[0]},{row[1]},{row[2]}\n")

print("Saved submission_dual.csv")
