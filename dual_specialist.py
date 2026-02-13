"""
DAL-Shemagh v13 — Hybrid Submission
# ============================================================
# Final Score = 0.5 × mAP@[0.5:0.95] + 0.5 × F1-Score
#
# Strategy:
#   F1 → Dual specialist models (head + shemagh) + all-pairs overlap
#   mAP → Single multi-class model + postprocessing
#
# Postprocessing for mAP:
#   - Top-K detections per class per image (cap FPs)
#   - Class-specific confidence thresholds
#   - NMS already handled by YOLO, but we tune iou threshold
# ============================================================
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

# ══════════════════════════════════════════════════════════════════════════════
# Model Paths
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# F1 models (specialist, single-class)
HEAD_WEIGHTS = os.path.join(SCRIPT_DIR, "models", "head_m640_best.pt")
SHEMAGH_WEIGHTS = os.path.join(SCRIPT_DIR, "models", "shemagh_m640_best.pt")

# mAP model (multi-class) — UPDATE THIS after running map_experiment.py
MAP_WEIGHTS = os.path.join(SCRIPT_DIR, "models", "map_best.pt")

# ══════════════════════════════════════════════════════════════════════════════
# Inference Settings
# ══════════════════════════════════════════════════════════════════════════════
# F1 inference
F1_HEAD_IMGSZ = 640
F1_SHEMAGH_IMGSZ = 640
F1_CONF = 0.15
OVERLAP_THRESHOLD = 0.10

# mAP inference — postprocessing params
MAP_IMGSZ = 960           # Must match training resolution
MAP_CONF = 0.01           # Low conf to maximize recall (mAP cares about ranking)
MAP_IOU_THRESH = 0.5      # NMS IoU threshold
MAP_MAX_DET_PER_CLASS = 5  # Cap detections per class per image
MAP_HEAD_CONF_MIN = 0.05   # Drop head boxes below this
MAP_SHEM_CONF_MIN = 0.05   # Drop shemagh boxes below this

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load Models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  LOADING MODELS")
print("="*60)

# F1 specialist models
assert os.path.exists(HEAD_WEIGHTS), f"Head weights not found: {HEAD_WEIGHTS}"
assert os.path.exists(SHEMAGH_WEIGHTS), f"Shemagh weights not found: {SHEMAGH_WEIGHTS}"
model_head = YOLO(HEAD_WEIGHTS)
model_shemagh = YOLO(SHEMAGH_WEIGHTS)
print(f"  F1 head model:    {HEAD_WEIGHTS}")
print(f"  F1 shemagh model: {SHEMAGH_WEIGHTS}")

# mAP multi-class model
use_map_model = os.path.exists(MAP_WEIGHTS)
if use_map_model:
    model_map = YOLO(MAP_WEIGHTS)
    print(f"  mAP model:        {MAP_WEIGHTS}")
else:
    model_map = None
    print(f"  ⚠ mAP model not found at {MAP_WEIGHTS}")
    print(f"    Will use F1 specialist detections for prediction_string")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Inference
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  DUAL PIPELINE INFERENCE")
print("="*60)

def get_overlap(box_a, box_b):
    """Overlap = intersection / smaller box area."""
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

def postprocess_map_boxes(results, imgsz):
    """Extract and postprocess boxes from mAP model.
    - Separate by class
    - Apply class-specific confidence threshold
    - Cap at top-K per class (sorted by confidence)
    Returns list of (class_id, conf, x, y, w, h) tuples.
    """
    heads = []
    shemaghs = []
    
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x, y, w, h = box.xywhn[0].tolist()
        
        if cls == 0 and conf >= MAP_HEAD_CONF_MIN:
            heads.append((conf, x, y, w, h))
        elif cls == 1 and conf >= MAP_SHEM_CONF_MIN:
            shemaghs.append((conf, x, y, w, h))
    
    # Sort by confidence descending, cap at top-K
    heads.sort(reverse=True)
    shemaghs.sort(reverse=True)
    heads = heads[:MAP_MAX_DET_PER_CLASS]
    shemaghs = shemaghs[:MAP_MAX_DET_PER_CLASS]
    
    # Convert to output format
    boxes = []
    for conf, x, y, w, h in heads:
        boxes.append((0, conf, x, y, w, h))
    for conf, x, y, w, h in shemaghs:
        boxes.append((1, conf, x, y, w, h))
    
    return boxes

def specialist_boxes(res_h, res_s):
    """Extract boxes from F1 specialist models (fallback for mAP)."""
    boxes = []
    for box in res_h.boxes:
        conf = float(box.conf[0])
        x, y, w, h = box.xywhn[0].tolist()
        boxes.append((0, conf, x, y, w, h))
    for box in res_s.boxes:
        conf = float(box.conf[0])
        x, y, w, h = box.xywhn[0].tolist()
        boxes.append((1, conf, x, y, w, h))
    return boxes

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
    
    # ── F1 PIPELINE (specialist models → right_place) ──
    res_h = model_head.predict(img_path, conf=F1_CONF, imgsz=F1_HEAD_IMGSZ, augment=True, verbose=False)[0]
    heads_f1 = [box.xywhn[0].tolist() + [float(box.conf[0])] for box in res_h.boxes]
    
    res_s = model_shemagh.predict(img_path, conf=F1_CONF, imgsz=F1_SHEMAGH_IMGSZ, augment=True, verbose=False)[0]
    shemaghs_f1 = [box.xywhn[0].tolist() + [float(box.conf[0])] for box in res_s.boxes]
    
    # All-pairs overlap check
    rp = 0
    if heads_f1 and shemaghs_f1:
        for h in heads_f1:
            for s in shemaghs_f1:
                if get_overlap(h[:4], s[:4]) > OVERLAP_THRESHOLD:
                    rp = 1
                    break
            if rp: break
    
    # ── mAP PIPELINE (multi-class model → prediction_string) ──
    if use_map_model:
        res_map = model_map.predict(img_path, conf=MAP_CONF, imgsz=MAP_IMGSZ, 
                                     iou=MAP_IOU_THRESH, augment=True, verbose=False)[0]
        boxes = postprocess_map_boxes(res_map, MAP_IMGSZ)
    else:
        # Fallback: use specialist detections
        boxes = specialist_boxes(res_h, res_s)
    
    # Build prediction string
    parts = []
    for cls, conf, x, y, w, h in boxes:
        parts.extend([str(cls), f"{conf:.4f}", f"{x:.4f}", f"{y:.4f}", f"{w:.4f}", f"{h:.4f}"])
    
    pred_str = " ".join(parts) if parts else "-"
    submission.append([fname, rp, pred_str])

# ══════════════════════════════════════════════════════════════════════════════
# 3. Save Submission
# ══════════════════════════════════════════════════════════════════════════════
right_place_count = sum([x[1] for x in submission])
total_boxes = sum(len(x[2].split()) // 6 for x in submission if x[2] != '-')
print(f"\nDone!")
print(f"  right_place=1: {right_place_count}/{total}")
print(f"  Total boxes: {total_boxes}")
print(f"  Avg boxes/image: {total_boxes/total:.1f}")

with open('submission_dual.csv', 'w') as f:
    f.write("filename,right_place,prediction_string\n")
    for row in submission:
        f.write(f"{row[0]},{row[1]},{row[2]}\n")

print("Saved submission_dual.csv")
