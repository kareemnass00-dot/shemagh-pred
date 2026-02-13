"""
mAP Experiment v4 — Exact baseline recipe + variants
# ============================================================
# LESSON: Oversampling negatives = overfitting. DON'T DO IT.
# Just use all data as-is, val=train, like baseline.
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

WORK_DIR = "./yolo11_v4"

# ══════════════════════════════════════════════════════════════════════════════
# Experiments — NO oversampling, just vary model size
# ══════════════════════════════════════════════════════════════════════════════
EXPERIMENTS = [
    ("y11n_baseline", "yolo11n.pt", 640, 16),  # Exact baseline
    ("y11s_baseline", "yolo11s.pt", 640, 16),  # One step up
    ("y11m_baseline", "yolo11m.pt", 640, 16),  # Medium
]

# ══════════════════════════════════════════════════════════════════════════════
# Data — Just point to original data, val=train (like baseline)
# ══════════════════════════════════════════════════════════════════════════════
def prepare_data():
    """Exact baseline setup: all data, val=train, no oversampling."""
    
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
    
    img_dir = f"{ROOT_DIR}/images/train"
    n_imgs = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    print(f"  Using {n_imgs} images (all data, val=train, NO oversampling)")
    
    return f"{work}/data.yaml"

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
os.system('pip install -U ultralytics')

def main():
    from ultralytics import YOLO
    
    print(f"Data root: {ROOT_DIR}")
    print(f"Epochs: {EPOCHS}\n")
    
    data_yaml = prepare_data()
    results_list = []
    
    for exp_name, model_name, imgsz, batch in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  {exp_name}: {model_name} @ {imgsz}")
        print(f"{'='*60}")
        
        t0 = time.time()
        
        model = YOLO(model_name)
        model.train(
            data=data_yaml,
            epochs=EPOCHS,
            imgsz=imgsz,
            batch=batch,
            project='./y11v4_experiments',
            name=exp_name,
            patience=PATIENCE,
            exist_ok=True,
        )
        
        best_weights = str(model.trainer.best)
        assert os.path.exists(best_weights), f"best.pt not found"
        
        model_best = YOLO(best_weights)
        val_results = model_best.val(data=data_yaml, imgsz=imgsz)
        
        elapsed = time.time() - t0
        
        map50 = float(val_results.box.map50)
        map50_95 = float(val_results.box.map)
        per_class = val_results.box.maps
        head_map = float(per_class[0]) if len(per_class) > 0 else 0
        shem_map = float(per_class[1]) if len(per_class) > 1 else 0
        
        print(f"\n  {exp_name}: mAP50={map50:.4f} mAP50-95={map50_95:.4f} head={head_map:.4f} shem={shem_map:.4f} time={elapsed:.0f}s")
        
        results_list.append({
            'name': exp_name, 'model': model_name, 'imgsz': imgsz,
            'map50': map50, 'map50_95': map50_95,
            'head_map': head_map, 'shem_map': shem_map,
            'time_s': int(elapsed), 'weights': best_weights
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("  FINAL RESULTS — YOLO11 BASELINE RECIPE (NO OVERSAMPLING)")
    print("=" * 80)
    print(f"{'Name':<18} {'Model':<14} {'mAP50':>7} {'mAP50-95':>9} {'Head':>7} {'Shem':>7} {'Time':>6}")
    print("-" * 75)
    
    results_list.sort(key=lambda x: x['map50_95'], reverse=True)
    for r in results_list:
        marker = " ★" if r == results_list[0] else ""
        print(f"{r['name']:<18} {r['model']:<14} "
              f"{r['map50']:>7.4f} {r['map50_95']:>9.4f} {r['head_map']:>7.4f} {r['shem_map']:>7.4f} {r['time_s']:>5}s{marker}")

if __name__ == "__main__":
    main()
