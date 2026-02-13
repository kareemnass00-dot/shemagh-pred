"""
DAL-Shemagh v19 — PURE RT-DETR-L (Single Model for Everything)
# ============================================================
# We dropped the Ensemble (0.761) because it was worse than
# single RT-DETR-L (0.774).
#
# Now we drop the separate F1 specialists (YOLOv8m).
# We trust RT-DETR-L for EVERYTHING:
# 1. mAP boxes: RT-DETR-L boxes
# 2. right_place: If RT-DETR-L finds intersecting Head+Shemagh -> 1
# ============================================================
"""
import os
import sys
from ultralytics import RTDETR

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
# Model Path
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The champion model
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "map_rtdetr_l_best.pt")

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# Helper Function
# ══════════════════════════════════════════════════════════════════════════════
def get_iou(box_a, box_b):
    """Calculates Intersection over Union (IoU)"""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax1, ay1, ax2, ay2 = ax-aw/2, ay-ah/2, ax+aw/2, ay+ah/2
    bx1, by1, bx2, by2 = bx-bw/2, by-bh/2, bx+bw/2, by+bh/2
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = (aw * ah) + (bw * bh) - inter
    if union <= 0: return 0
    return inter / union

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load Model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"  LOADING RT-DETR-L: {os.path.basename(MODEL_PATH)}")
print("="*60)
model = RTDETR(MODEL_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Inference
# ══════════════════════════════════════════════════════════════════════════════
test_dir = f"{ROOT_DIR}/images/test"
test_files = sorted(os.listdir(test_dir))
submission = []

CONF_THRESH = 0.25 # Baseline conf
IOU_THRESH = 0.10  # Overlap threshold for right_place logic

count = 0
for fname in test_files:
    img_path = f"{test_dir}/{fname}"
    if count % 100 == 0:
        print(f"Processing {count}/{len(test_files)}...")
    count += 1
    
    try:
        # Standard inference
        res = model.predict(img_path, conf=CONF_THRESH, imgsz=640, verbose=False)[0]
        
        parts = []
        heads = []
        shemaghs = []
        
        if hasattr(res, 'boxes'):
            for box in res.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x, y, w, h = box.xywhn[0].tolist()
                
                parts.extend([str(cls), f"{conf:.4f}", f"{x:.4f}", f"{y:.4f}", f"{w:.4f}", f"{h:.4f}"])
                
                if cls == 0:
                    heads.append([x, y, w, h])
                elif cls == 1:
                    shemaghs.append([x, y, w, h])
        
        # Logic: If we found intersecting Head + Shemagh -> right_place=1
        rp = 0
        if heads and shemaghs:
            for h in heads:
                for s in shemaghs:
                    if get_iou(h, s) > 0.0: # Even minor overlap counts
                        rp = 1
                        break
                if rp: break
                
        pred_str = " ".join(parts) if parts else "-"
        
    except Exception as e:
        print(f"Error {fname}: {e}")
        rp = 0
        pred_str = "-"
    
    submission.append([fname, rp, pred_str])

# ══════════════════════════════════════════════════════════════════════════════
# 3. Save
# ══════════════════════════════════════════════════════════════════════════════
full_boxes = sum([len(x[2].split())//6 for x in submission if x[2] != '-'])
rp_ones = sum([x[1] for x in submission])

print("\n" + "="*60)
print(f"  DONE: {len(submission)} files processed")
print(f"  right_place=1 count: {rp_ones}")
print(f"  Total boxes: {full_boxes}")
print("="*60)

with open('submission_pure_rtdetr.csv', 'w') as f:
    f.write("filename,right_place,prediction_string\n")
    for row in submission:
        f.write(f"{row[0]},{row[1]},{row[2]}\n")

print("Saved submission_pure_rtdetr.csv")
