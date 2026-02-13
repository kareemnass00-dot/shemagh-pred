"""
DAL-Shemagh v18 — 4-Model WBF Ensemble
# ============================================================
# Ensemble of 4 models using Weighted Box Fusion (WBF):
# 1. RT-DETR-L (mAP ~0.948)
# 2. YOLO11n   (mAP ~0.933)
# 3. YOLO11s   (mAP ~0.925)
# 4. RT-DETR-X (mAP ~0.919)
#
# WBF creates a "consensus" prediction that is cleaner and
# more accurate than any single model. This is how you win.
#
# F1 Specialists (Head/Shemagh) still used for right_place logic.
# ============================================================
"""
import os
import sys
import numpy as np
from ultralytics import YOLO, RTDETR

# Install dependencies if missing
try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    os.system('pip install -U ensemble-boxes')
    from ensemble_boxes import weighted_boxes_fusion

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

# F1 specialists (for right_place)
F1_HEAD_WEIGHTS    = M("head_m640_best.pt")
F1_SHEMAGH_WEIGHTS = M("shemagh_m640_best.pt")

# Ensemble Models (Detection)
ENSEMBLE_MODELS = [
    # (weight_path, model_type, weight_in_ensemble)
    (M("map_rtdetr_l_best.pt"), RTDETR, 2.0), # Strongest model (0.948) gets double weight
    (M("map_y11n_best.pt"),      YOLO,   1.0), # Fast & accurate (0.933)
    (M("map_y11s.pt"),           YOLO,   1.0), # Good backup (0.925)
    (M("map_rtdetr_x_best.pt"),  RTDETR, 1.0), # Diverse architecture (0.919)
]

# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════
def get_overlap(box_a, box_b):
    """Calculates intersection over minimum area for classification logic"""
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
print("  LOADING MODELS")
print("="*60)

# Load F1 models
f1_head    = YOLO(F1_HEAD_WEIGHTS)
f1_shemagh = YOLO(F1_SHEMAGH_WEIGHTS)

# Load Ensemble Detection Models
detection_models = []
weights_list = [] # Weights for WBF aggregation

for path, ModelClass, w in ENSEMBLE_MODELS:
    if os.path.exists(path):
        print(f"  ✓ Loaded Detection Model ({w}x): {os.path.basename(path)}")
        m = ModelClass(path)
        detection_models.append(m)
        weights_list.append(w)
    else:
        print(f"  ⚠ WARNING: Model not found: {path} (Skipping)")

if not detection_models:
    print("ERROR: No detection models loaded!")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Inference Loop
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"  STARTING INFERENCE ON {len(detection_models)} MODELS")
print("="*60)

test_dir = f"{ROOT_DIR}/images/test"
test_files = sorted(os.listdir(test_dir))
submission = []

params = {
    'conf': 0.15,       # Low conf for ensemble (WBF will filter/merge)
    'imgsz': 640
}
wbf_iou_thr = 0.55      # IoU threshold for merging boxes
wbf_skip_box_thr = 0.10 # Discard boxes with low score after fusion

# Statistics
total_processed = 0
total_dets = 0

for idx, fname in enumerate(test_files):
    img_path = f"{test_dir}/{fname}"
    if idx % 50 == 0:
        print(f"Processing {idx}/{len(test_files)}...")

    # 1. Run F1 Specialists (Classification Logic)
    rp = 0
    try:
        # TTA used for classification robustness
        res_h = f1_head.predict(img_path, conf=0.15, augment=True, verbose=False)[0]
        res_s = f1_shemagh.predict(img_path, conf=0.15, augment=True, verbose=False)[0]
        
        heads = [b.xywhn[0].tolist() for b in res_h.boxes]
        shemaghs = [b.xywhn[0].tolist() for b in res_s.boxes]
        
        if heads and shemaghs:
            for h in heads:
                for s in shemaghs:
                    if get_overlap(h, s) > 0.10: # Overlap threshold
                        rp = 1
                        break
                if rp: break
    except Exception as e:
        # Fallback to 0
        pass

    # 2. Run Detection Ensemble (mAP Logic)
    boxes_list = []
    scores_list = []
    labels_list = []
    
    try:
        for model in detection_models:
            # Standard inference (no TTA for speed/stability in ensemble unless needed)
            res = model.predict(img_path, conf=params['conf'], imgsz=params['imgsz'], verbose=False)[0]
            
            # Extract boxes
            if len(res.boxes) > 0:
                # wrapper for normalized xyxy
                b_xyxyn = res.boxes.xyxyn.cpu().numpy().tolist()
                b_conf  = res.boxes.conf.cpu().numpy().tolist()
                b_cls   = res.boxes.cls.cpu().numpy().tolist()
                
                boxes_list.append(b_xyxyn)
                scores_list.append(b_conf)
                labels_list.append(b_cls)
            else:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])

        # 3. Apply Weighted Box Fusion
        # Models with empty preds are handled by WBF gracefully
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights_list,
            iou_thr=wbf_iou_thr,
            skip_box_thr=wbf_skip_box_thr,
            conf_type='avg' # Average scores of merged boxes
        )
        
        # 4. Format Prediction String
        parts = []
        for i in range(len(boxes)):
            cls = int(labels[i])
            score = float(scores[i])
            x1, y1, x2, y2 = boxes[i]
            
            # Convert xyxy normalized back to xywh normalized for submission
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w/2
            cy = y1 + h/2
            
            parts.extend([str(cls), f"{score:.4f}", f"{cx:.4f}", f"{cy:.4f}", f"{w:.4f}", f"{h:.4f}"])
            total_dets += 1
            
        pred_str = " ".join(parts) if parts else "-"

    except Exception as e:
        print(f"Ensemble Error {fname}: {e}")
        pred_str = "-"

    submission.append([fname, rp, pred_str])
    total_processed += 1

# ══════════════════════════════════════════════════════════════════════════════
# 3. Save Submission
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nCompleted {total_processed} files.")
print(f"Total Detections: {total_dets} (Avg {total_dets/total_processed:.2f}/img)")

out_file = "submission_ensemble_wbf.csv"
with open(out_file, 'w') as f:
    f.write("filename,right_place,prediction_string\n")
    for row in submission:
        f.write(f"{row[0]},{row[1]},{row[2]}\n")

print(f"Saved {out_file}")
