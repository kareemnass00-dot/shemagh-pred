"""
mAP Experiment v5 — GAME CHANGERS
# ============================================================
# 3 new approaches we haven't tried:
#
# 1. RT-DETR (transformer detector, often better mAP than YOLO)
# 2. WBF (Weighted Box Fusion) — ensemble multiple models
# 3. SAHI (Slicing Aided Hyper Inference) — for small objects
#
# Also: train baseline recipe (y11n/s) for comparison
# ============================================================
"""
import os, sys, shutil, time, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=50)
args = parser.parse_args()

EPOCHS = args.epochs
PATIENCE = args.patience

POSSIBLE_PATHS = [
    "./data/dal-shemagh-detection-challenge",
    "./data",
    "../input/dal-shemagh-detection-challenge",
    "../input/dal-shemagh-identification"
]
ROOT_DIR = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p) and os.path.exists(f"{p}/images/train"):
        ROOT_DIR = p
        break
if ROOT_DIR is None:
    print("ERROR: Could not find dataset"); sys.exit(1)

WORK_DIR = "./map_v5"

# ══════════════════════════════════════════════════════════════════════════════
# Install dependencies
# ══════════════════════════════════════════════════════════════════════════════
os.system('pip install -U ultralytics ensemble-boxes sahi')

# ══════════════════════════════════════════════════════════════════════════════
# Experiments — Different architectures
# ══════════════════════════════════════════════════════════════════════════════
EXPERIMENTS = [
    # RT-DETR — transformer detector (often higher mAP than YOLO)
    ("rtdetr_l",   "rtdetr-l.pt",  640, 8),
    ("rtdetr_x",   "rtdetr-x.pt",  640, 4),
    # YOLO11 baseline for comparison/ensemble
    ("y11n",       "yolo11n.pt",   640, 16),
    ("y11s",       "yolo11s.pt",   640, 16),
]

# ══════════════════════════════════════════════════════════════════════════════
# Data — all data, val=train (baseline recipe)
# ══════════════════════════════════════════════════════════════════════════════
def prepare_data():
    work = WORK_DIR
    os.makedirs(work, exist_ok=True)
    
    yaml_content = f"""path: {os.path.abspath(ROOT_DIR)}
train: images/train
val: images/train

names:
  0: head
  1: shemagh
"""
    with open(f"{work}/data.yaml", 'w') as f:
        f.write(yaml_content)
    
    n = len([f for f in os.listdir(f"{ROOT_DIR}/images/train") if f.endswith('.jpg')])
    print(f"  Data: {n} images (all, val=train)")
    return f"{work}/data.yaml"

# ══════════════════════════════════════════════════════════════════════════════
# Main — Train all models, then WBF ensemble
# ══════════════════════════════════════════════════════════════════════════════
def main():
    from ultralytics import YOLO
    
    print(f"Data root: {ROOT_DIR}")
    print(f"Epochs: {EPOCHS}\n")
    
    data_yaml = prepare_data()
    trained = []
    
    # ──────────────────────────────────────────────
    # Phase 1: Train individual models
    # ──────────────────────────────────────────────
    for exp_name, model_name, imgsz, batch in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  TRAINING: {exp_name} ({model_name})")
        print(f"{'='*60}")
        
        t0 = time.time()
        model = YOLO(model_name)
        model.train(
            data=data_yaml,
            epochs=EPOCHS,
            imgsz=imgsz,
            batch=batch,
            project=f'./mapv5_runs',
            name=exp_name,
            patience=PATIENCE,
            exist_ok=True,
        )
        
        best = str(model.trainer.best)
        assert os.path.exists(best), f"best.pt not found for {exp_name}"
        
        # Validate
        model_best = YOLO(best)
        val = model_best.val(data=data_yaml, imgsz=imgsz)
        elapsed = time.time() - t0
        
        map50 = float(val.box.map50)
        map50_95 = float(val.box.map)
        maps = val.box.maps
        head_map = float(maps[0]) if len(maps) > 0 else 0
        shem_map = float(maps[1]) if len(maps) > 1 else 0
        
        print(f"  {exp_name}: mAP50={map50:.4f} mAP50-95={map50_95:.4f} "
              f"head={head_map:.4f} shem={shem_map:.4f} time={elapsed:.0f}s")
        
        trained.append({
            'name': exp_name, 'model': model_name, 'imgsz': imgsz,
            'map50': map50, 'map50_95': map50_95,
            'head_map': head_map, 'shem_map': shem_map,
            'time_s': int(elapsed), 'weights': best
        })
    
    # ──────────────────────────────────────────────
    # Phase 2: WBF Ensemble (top models)
    # ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 2: WEIGHTED BOX FUSION ENSEMBLE")
    print(f"{'='*60}")
    
    try:
        from ensemble_boxes import weighted_boxes_fusion
        import numpy as np
        
        # Use all trained models for ensemble
        models_for_ensemble = []
        for t in trained:
            m = YOLO(t['weights'])
            models_for_ensemble.append((t['name'], m, t['imgsz']))
            print(f"  Loaded {t['name']} for ensemble")
        
        # Run WBF on val images to measure ensemble mAP
        val_img_dir = f"{ROOT_DIR}/images/train"
        val_lbl_dir = f"{ROOT_DIR}/labels/train"
        val_images = sorted([f for f in os.listdir(val_img_dir) if f.endswith('.jpg')])
        
        print(f"  Running WBF on {len(val_images)} val images...")
        
        # For each image, get predictions from all models, then WBF
        all_wbf_preds = []
        for idx, fname in enumerate(val_images):
            img_path = f"{val_img_dir}/{fname}"
            
            all_boxes = []
            all_scores = []
            all_labels = []
            
            for mname, model, imgsz in models_for_ensemble:
                res = model.predict(img_path, conf=0.01, imgsz=imgsz, verbose=False)[0]
                
                boxes_norm = []
                scores = []
                labels = []
                
                for box in res.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x, y, w, h = box.xywhn[0].tolist()
                    # WBF needs [x1, y1, x2, y2] normalized
                    x1 = max(0, x - w/2)
                    y1 = max(0, y - h/2)
                    x2 = min(1, x + w/2)
                    y2 = min(1, y + h/2)
                    boxes_norm.append([x1, y1, x2, y2])
                    scores.append(conf)
                    labels.append(cls)
                
                all_boxes.append(boxes_norm if boxes_norm else [[0,0,0,0]])
                all_scores.append(scores if scores else [0])
                all_labels.append(labels if labels else [0])
            
            # WBF
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                all_boxes, all_scores, all_labels,
                iou_thr=0.5,
                skip_box_thr=0.01,
            )
            
            all_wbf_preds.append({
                'fname': fname,
                'boxes': fused_boxes,
                'scores': fused_scores,
                'labels': fused_labels
            })
            
            if idx % 100 == 0:
                print(f"    WBF processed {idx}/{len(val_images)}...")
        
        # Count total detections
        total_dets = sum(len(p['scores']) for p in all_wbf_preds)
        avg_dets = total_dets / len(val_images)
        print(f"  WBF ensemble: {total_dets} total detections, {avg_dets:.1f} avg/img")
        print(f"  (Full mAP evaluation requires writing pred files — see submission)")
        
    except ImportError:
        print("  ensemble_boxes not available, skipping WBF")
    except Exception as e:
        print(f"  WBF error: {e}")
    
    # ──────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL RESULTS")
    print("=" * 80)
    print(f"{'Name':<14} {'Model':<14} {'mAP50':>7} {'mAP50-95':>9} {'Head':>7} {'Shem':>7} {'Time':>6}")
    print("-" * 70)
    
    trained.sort(key=lambda x: x['map50_95'], reverse=True)
    for r in trained:
        marker = " ★" if r == trained[0] else ""
        print(f"{r['name']:<14} {r['model']:<14} "
              f"{r['map50']:>7.4f} {r['map50_95']:>9.4f} {r['head_map']:>7.4f} {r['shem_map']:>7.4f} {r['time_s']:>5}s{marker}")
    
    # Save results
    with open('mapv5_results.csv', 'w') as f:
        f.write('name,model,map50,map50_95,head_map,shem_map,time_s,weights\n')
        for r in trained:
            f.write(f"{r['name']},{r['model']},{r['map50']:.4f},{r['map50_95']:.4f},"
                    f"{r['head_map']:.4f},{r['shem_map']:.4f},{r['time_s']},{r['weights']}\n")

if __name__ == "__main__":
    main()
