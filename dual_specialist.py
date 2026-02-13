"""
DAL-Shemagh v14 — Quad Specialist Submission
# ============================================================
# Final Score = 0.5 × mAP@[0.5:0.95] + 0.5 × F1-Score
#
# 4 models, each a specialist:
#   F1 head:     head_m640_best.pt    (right_place)
#   F1 shemagh:  shemagh_m640_best.pt (right_place)
#   mAP head:    map_head_s640_best.pt    → head boxes (class 0)
#   mAP shemagh: map_shem_m640_best.pt    → shemagh boxes (class 1)
#
# Postprocessing for mAP:
#   - Top-K per class per image
#   - Class-specific confidence thresholds
#   - TTA (augment=True)
# ============================================================
"""
import os
import sys
from ultralytics import YOLO

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
    print("Files in current dir:", os.listdir("."))
    if os.path.exists("./data"): print("Files in ./data:", os.listdir("./data"))
    sys.exit(1)

os.system('pip install -U ultralytics')
print(f"Using Data at: {ROOT_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# Model Paths
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
M = lambda name: os.path.join(SCRIPT_DIR, "models", name)

# F1 specialists (single-class, for right_place)
F1_HEAD_WEIGHTS    = M("head_m640_best.pt")
F1_SHEMAGH_WEIGHTS = M("shemagh_m640_best.pt")

# mAP specialists (multi-class, but we pick best per-class)
MAP_HEAD_WEIGHTS   = M("map_head_s640_best.pt")    # s_640: head mAP=0.8652
MAP_SHEM_WEIGHTS   = M("map_shem_m640_best.pt")    # m_640: shem mAP=0.6562

# ══════════════════════════════════════════════════════════════════════════════
# Inference Settings
# ══════════════════════════════════════════════════════════════════════════════
# F1 pipeline
F1_IMGSZ = 640
F1_CONF = 0.15
OVERLAP_THRESHOLD = 0.10

# mAP pipeline — postprocessing
MAP_IMGSZ = 640
MAP_CONF = 0.01              # Low for max recall (mAP ranks by confidence)
MAP_MAX_DET_PER_CLASS = 5    # Cap FPs
MAP_HEAD_CONF_MIN = 0.05     # Floor for head boxes
MAP_SHEM_CONF_MIN = 0.05     # Floor for shemagh boxes

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load All 4 Models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  LOADING 4 SPECIALIST MODELS")
print("="*60)

for label, path in [("F1 head", F1_HEAD_WEIGHTS), ("F1 shemagh", F1_SHEMAGH_WEIGHTS),
                     ("mAP head", MAP_HEAD_WEIGHTS), ("mAP shemagh", MAP_SHEM_WEIGHTS)]:
    assert os.path.exists(path), f"{label} weights not found: {path}"
    print(f"  ✓ {label}: {os.path.basename(path)}")

f1_head    = YOLO(F1_HEAD_WEIGHTS)
f1_shemagh = YOLO(F1_SHEMAGH_WEIGHTS)
map_head   = YOLO(MAP_HEAD_WEIGHTS)
map_shem   = YOLO(MAP_SHEM_WEIGHTS)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Inference
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  QUAD SPECIALIST INFERENCE")
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
    
    # ────────────────────────────────────────────────────────
    # F1 PIPELINE → right_place
    # ────────────────────────────────────────────────────────
    res_fh = f1_head.predict(img_path, conf=F1_CONF, imgsz=F1_IMGSZ, augment=True, verbose=False)[0]
    heads_f1 = [box.xywhn[0].tolist() for box in res_fh.boxes]
    
    res_fs = f1_shemagh.predict(img_path, conf=F1_CONF, imgsz=F1_IMGSZ, augment=True, verbose=False)[0]
    shemaghs_f1 = [box.xywhn[0].tolist() for box in res_fs.boxes]
    
    rp = 0
    if heads_f1 and shemaghs_f1:
        for h in heads_f1:
            for s in shemaghs_f1:
                if get_overlap(h, s) > OVERLAP_THRESHOLD:
                    rp = 1
                    break
            if rp: break
    
    # ────────────────────────────────────────────────────────
    # mAP PIPELINE → prediction_string
    # ────────────────────────────────────────────────────────
    # Head boxes from s_640 model (class 0 only)
    res_mh = map_head.predict(img_path, conf=MAP_CONF, imgsz=MAP_IMGSZ, augment=True, verbose=False)[0]
    head_boxes = []
    for box in res_mh.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 0 and conf >= MAP_HEAD_CONF_MIN:
            x, y, w, h = box.xywhn[0].tolist()
            head_boxes.append((conf, x, y, w, h))
    head_boxes.sort(reverse=True)
    head_boxes = head_boxes[:MAP_MAX_DET_PER_CLASS]
    
    # Shemagh boxes from m_640 model (class 1 only)
    res_ms = map_shem.predict(img_path, conf=MAP_CONF, imgsz=MAP_IMGSZ, augment=True, verbose=False)[0]
    shem_boxes = []
    for box in res_ms.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls == 1 and conf >= MAP_SHEM_CONF_MIN:
            x, y, w, h = box.xywhn[0].tolist()
            shem_boxes.append((conf, x, y, w, h))
    shem_boxes.sort(reverse=True)
    shem_boxes = shem_boxes[:MAP_MAX_DET_PER_CLASS]
    
    # Build prediction string
    parts = []
    for conf, x, y, w, h in head_boxes:
        parts.extend(["0", f"{conf:.4f}", f"{x:.4f}", f"{y:.4f}", f"{w:.4f}", f"{h:.4f}"])
    for conf, x, y, w, h in shem_boxes:
        parts.extend(["1", f"{conf:.4f}", f"{x:.4f}", f"{y:.4f}", f"{w:.4f}", f"{h:.4f}"])
    
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
