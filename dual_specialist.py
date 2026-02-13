"""
DAL-Shemagh v20 — RT-DETR-L (TTA) + F1 Specialists
# ============================================================
# Reverted to v17 architecture (Separate F1 Specialists) because
# it scored 0.774 vs v19's 0.723.
#
# IMPROVEMENT:
# Enabled Test Time Augmentation (TTA) for the RT-DETR-L model.
# This averages predictions across flips/scales to improve
# robustness and mAP.
# ============================================================
"""
import os
import sys
from ultralytics import YOLO, RTDETR

# ══════════════════════════════════════════════════════════════════════════════
# 0. Path Detection
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
    sys.exit(1)

print(f"Using Data at: {ROOT_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# Model Paths
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
M = lambda name: os.path.join(SCRIPT_DIR, "models", name)

# 1. F1 Specialists (Champion Logic from v17)
F1_HEAD_WEIGHTS    = M("head_m640_best.pt")
F1_SHEMAGH_WEIGHTS = M("shemagh_m640_best.pt")

# 2. mAP Model (Champion RT-DETR-L)
MAP_WEIGHTS        = M("map_rtdetr_l_best.pt")

# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load Models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  LOADING MODELS (v17 Logic + TTA)")
print("="*60)

for label, path in [("F1 Head", F1_HEAD_WEIGHTS), ("F1 Shemagh", F1_SHEMAGH_WEIGHTS), ("RT-DETR", MAP_WEIGHTS)]:
    if not os.path.exists(path):
        print(f"ERROR: {label} not found at {path}")
        sys.exit(1)
    print(f"  ✓ {label}: {os.path.basename(path)}")

f1_head    = YOLO(F1_HEAD_WEIGHTS)
f1_shemagh = YOLO(F1_SHEMAGH_WEIGHTS)
map_model  = RTDETR(MAP_WEIGHTS)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Inference
# ══════════════════════════════════════════════════════════════════════════════
test_dir = f"{ROOT_DIR}/images/test"
test_files = sorted(os.listdir(test_dir))
submission = []

# Settings
F1_CONF = 0.15
F1_OVERLAP = 0.10
MAP_CONF = 0.20 # Slightly lower conf for TTA to capture more recall

count = 0
total = len(test_files)
total_head = 0
total_shem = 0

print("\n" + "="*60)
print(f"  Running Inference on {total} files (TTA ENABLED)...")
print("="*60)

for fname in test_files:
    img_path = f"{test_dir}/{fname}"
    if count % 50 == 0:
        print(f"Processing {count}/{total}...")
    count += 1
    
    # ── 1. Right Place Classification (F1 Models) ──
    rp = 0
    try:
        # F1 models always use TTA (augment=True)
        res_h = f1_head.predict(img_path, conf=F1_CONF, augment=True, verbose=False)[0]
        res_s = f1_shemagh.predict(img_path, conf=F1_CONF, augment=True, verbose=False)[0]
        
        heads = [b.xywhn[0].tolist() for b in res_h.boxes]
        shems = [b.xywhn[0].tolist() for b in res_s.boxes]
        
        if heads and shems:
            for h in heads:
                for s in shems:
                    if get_overlap(h, s) > F1_OVERLAP:
                        rp = 1
                        break
                if rp: break
    except Exception:
        rp = 0

    # ── 2. mAP Detection (RT-DETR) ──
    pred_str = "-"
    try:
        # ENABLE TTA HERE (augment=True)
        res_map = map_model.predict(img_path, conf=MAP_CONF, imgsz=640, augment=True, verbose=False)[0]
        
        parts = []
        if hasattr(res_map, 'boxes'):
            for box in res_map.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x, y, w, h = box.xywhn[0].tolist()
                
                parts.extend([str(cls), f"{conf:.4f}", f"{x:.4f}", f"{y:.4f}", f"{w:.4f}", f"{h:.4f}"])
                if cls == 0: total_head += 1
                else: total_shem += 1
        
        pred_str = " ".join(parts) if parts else "-"
    except Exception as e:
        print(f"Error {fname}: {e}")
        pred_str = "-"

    submission.append([fname, rp, pred_str])

# ══════════════════════════════════════════════════════════════════════════════
# 3. Save
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"  DONE.")
print(f"  right_place=1: {sum(x[1] for x in submission)}")
print(f"  Head Boxes: {total_head}")
print(f"  Shem Boxes: {total_shem}")
print(f"  Total Boxes: {total_head + total_shem} (Avg {(total_head+total_shem)/total:.2f}/img)")
print("="*60)

with open('submission_dual_tta.csv', 'w') as f:
    f.write("filename,right_place,prediction_string\n")
    for row in submission:
        f.write(f"{row[0]},{row[1]},{row[2]}\n")

print("Saved submission_dual_tta.csv")
