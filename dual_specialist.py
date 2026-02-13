"""
DAL-Shemagh v17 — RT-DETR-L mAP + F1 Specialists
# ============================================================
# mAP: RT-DETR-L @ 640 (mAP50-95=0.948 val)
# F1:  YOLOv8m specialists (head + shemagh) → right_place
# Inference conf=0.25
# ============================================================
# Notes:
# - RT-DETR provides cleaner boxes, no NMS artifacts.
# - F1 specialists handle the classification task.
# ============================================================
"""
import os
import sys
from ultralytics import YOLO, RTDETR  # Using RTDETR class

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

os.system('pip install -U ultralytics')
print(f"Using Data at: {ROOT_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# Model Paths
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
M = lambda name: os.path.join(SCRIPT_DIR, "models", name)

# F1 specialists (for right_place)
F1_HEAD_WEIGHTS    = M("head_m640_best.pt")
F1_SHEMAGH_WEIGHTS = M("shemagh_m640_best.pt")

# mAP model — RT-DETR-L (mAP50-95=0.948 on val)
MAP_WEIGHTS = M("map_rtdetr_l_best.pt")

# ══════════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════════
F1_IMGSZ = 640
F1_CONF = 0.15
OVERLAP_THRESHOLD = 0.10

# mAP inference — RT-DETR works well with standard conf
MAP_IMGSZ = 640
MAP_CONF = 0.25

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load Models
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  LOADING MODELS (RT-DETR + YOLOv8)")
print("="*60)

for label, path in [("F1 head", F1_HEAD_WEIGHTS), ("F1 shemagh", F1_SHEMAGH_WEIGHTS),
                     ("mAP rtdetr", MAP_WEIGHTS)]:
    assert os.path.exists(path), f"{label} not found: {path}"
    print(f"  ✓ {label}: {os.path.basename(path)}")

f1_head    = YOLO(F1_HEAD_WEIGHTS)
f1_shemagh = YOLO(F1_SHEMAGH_WEIGHTS)
map_model  = RTDETR(MAP_WEIGHTS)  # Use RTDETR class

# ══════════════════════════════════════════════════════════════════════════════
# 2. Inference
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  INFERENCE")
print("="*60)

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

test_dir = f"{ROOT_DIR}/images/test"
test_files = sorted(os.listdir(test_dir))
submission = []

count = 0
total = len(test_files)
total_head_boxes = 0
total_shem_boxes = 0

for fname in test_files:
    img_path = f"{test_dir}/{fname}"
    if count % 100 == 0:
        print(f"Processing {count}/{total}...")
    count += 1
    
    # ── F1 PIPELINE → right_place ──
    try:
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
    except Exception as e:
        print(f"Error in F1 prediction for {fname}: {e}")
        rp = 0
    
    # ── mAP PIPELINE → prediction_string ──
    # RT-DETR prediction
    try:
        res_map = map_model.predict(img_path, conf=MAP_CONF, imgsz=MAP_IMGSZ, augment=False, verbose=False)[0] # RT-DETR usually no TTA
        
        parts = []
        if hasattr(res_map, 'boxes'):
            for box in res_map.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x, y, w, h = box.xywhn[0].tolist()
                parts.extend([str(cls), f"{conf:.4f}", f"{x:.4f}", f"{y:.4f}", f"{w:.4f}", f"{h:.4f}"])
                if cls == 0: total_head_boxes += 1
                else: total_shem_boxes += 1
        
        pred_str = " ".join(parts) if parts else "-"
    except Exception as e:
        print(f"Error in mAP prediction for {fname}: {e}")
        pred_str = "-"

    submission.append([fname, rp, pred_str])

# ══════════════════════════════════════════════════════════════════════════════
# 3. Save
# ══════════════════════════════════════════════════════════════════════════════
right_place_count = sum([x[1] for x in submission])
total_boxes = total_head_boxes + total_shem_boxes
print(f"\nDone!")
print(f"  right_place=1: {right_place_count}/{total}")
print(f"  Head boxes: {total_head_boxes}, Shemagh boxes: {total_shem_boxes}")
print(f"  Total boxes: {total_boxes}, Avg/img: {total_boxes/total:.1f}")

with open('submission_dual.csv', 'w') as f:
    f.write("filename,right_place,prediction_string\n")
    for row in submission:
        f.write(f"{row[0]},{row[1]},{row[2]}\n")

print("Saved submission_dual.csv")
