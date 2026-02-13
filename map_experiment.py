"""
mAP Experiment v3 — YOLO11 (matching baseline architecture)
# ============================================================
# Key insight: baseline uses yolo11n with ALL data, light aug, conf=0.25
# We test: yolo11n, yolo11s, yolo11m at 640px
# Train on ALL data (val=train) for max data usage
# ============================================================
"""
import os, sys, shutil, random, time, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=50)
args = parser.parse_args()

EPOCHS = args.epochs
PATIENCE = args.patience

# Auto-detect data path
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
    print(f"ERROR: Could not find dataset"); sys.exit(1)

WORK_DIR = "./yolo11_map_exp"

# ══════════════════════════════════════════════════════════════════════════════
# Experiments — YOLO11 models
# ══════════════════════════════════════════════════════════════════════════════
EXPERIMENTS = [
    # (name, model, imgsz, batch)
    ("y11n_640", "yolo11n.pt", 640, 32),   # Baseline model
    ("y11s_640", "yolo11s.pt", 640, 32),   # One step up
    ("y11m_640", "yolo11m.pt", 640, 16),   # Medium
]

# Light augmentation (matching baseline style)
AUG = dict(
    hsv_h=0.015,
    hsv_s=0.1,
    hsv_v=0.1,
    degrees=10.0,
    translate=0.1,
    scale=0.1,
    fliplr=0.5,
    mosaic=0.0,       # No mosaic (baseline doesn't use it)
    mixup=0.0,
    erasing=0.0,
    copy_paste=0.0,
    close_mosaic=0,
)

# ══════════════════════════════════════════════════════════════════════════════
# Prepare Data — ALL data, val=train (like baseline)
# ══════════════════════════════════════════════════════════════════════════════
def prepare_data():
    """Use ALL training data. Val = train (like baseline)."""
    
    work = WORK_DIR
    if os.path.exists(work):
        shutil.rmtree(work)
    
    # Just symlink to original data
    os.makedirs(work, exist_ok=True)
    
    img_dir = f"{ROOT_DIR}/images/train"
    lbl_dir = f"{ROOT_DIR}/labels/train"
    
    all_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    # Count classes
    head_count = shem_count = both_count = neg_count = 0
    for f in all_files:
        lbl = f"{lbl_dir}/{f.replace('.jpg', '.txt')}"
        has_h = has_s = False
        if os.path.exists(lbl):
            with open(lbl) as lf:
                for line in lf:
                    if line.strip().startswith('0 '): has_h = True
                    elif line.strip().startswith('1 '): has_s = True
        if has_h and has_s: both_count += 1
        elif has_h: head_count += 1
        elif has_s: shem_count += 1
        else: neg_count += 1
    
    print(f"  All data: {len(all_files)} images")
    print(f"    Both: {both_count}, Head only: {head_count}, Shem only: {shem_count}, Neg: {neg_count}")
    
    # data.yaml — val=train (like baseline)
    yaml_content = f"""path: {os.path.abspath(ROOT_DIR)}
train: images/train
val: images/train

names:
  0: head
  1: shemagh
"""
    with open(f"{work}/data.yaml", 'w') as f:
        f.write(yaml_content)
    
    return f"{work}/data.yaml"

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
os.system('pip install -U ultralytics')

def main():
    from ultralytics import YOLO
    
    print(f"Data root: {ROOT_DIR}")
    print(f"Epochs: {EPOCHS}")
    print(f"Running {len(EXPERIMENTS)} YOLO11 experiments...\n")
    
    data_yaml = prepare_data()
    results_list = []
    
    for exp_name, model_name, imgsz, batch in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  YOLO11 EXPERIMENT: {exp_name}")
        print(f"  Model: {model_name} | ImgSz: {imgsz} | Batch: {batch}")
        print(f"{'='*60}")
        
        t0 = time.time()
        
        model = YOLO(model_name)
        model.train(
            data=data_yaml,
            epochs=EPOCHS,
            imgsz=imgsz,
            batch=batch,
            project='./y11_experiments',
            name=exp_name,
            patience=PATIENCE,
            exist_ok=True,
            **AUG
        )
        
        best_weights = str(model.trainer.best)
        assert os.path.exists(best_weights), f"best.pt not found for {exp_name}"
        
        # Validate
        model_best = YOLO(best_weights)
        val_results = model_best.val(data=data_yaml, imgsz=imgsz)
        
        elapsed = time.time() - t0
        
        map50 = float(val_results.box.map50)
        map50_95 = float(val_results.box.map)
        per_class = val_results.box.maps
        head_map = float(per_class[0]) if len(per_class) > 0 else 0
        shem_map = float(per_class[1]) if len(per_class) > 1 else 0
        
        print(f"\n  RESULTS for {exp_name}:")
        print(f"  mAP@50    = {map50:.4f}")
        print(f"  mAP@50-95 = {map50_95:.4f}")
        print(f"  Head mAP  = {head_map:.4f}")
        print(f"  Shem mAP  = {shem_map:.4f}")
        print(f"  Time: {elapsed:.0f}s")
        
        results_list.append({
            'name': exp_name, 'model': model_name, 'imgsz': imgsz,
            'map50': map50, 'map50_95': map50_95,
            'head_map': head_map, 'shem_map': shem_map,
            'time_s': int(elapsed), 'weights': best_weights
        })
        
        with open('y11_experiment_results.csv', 'w') as f:
            f.write('name,model,imgsz,map50,map50_95,head_map,shem_map,time_s\n')
            for r in results_list:
                f.write(f"{r['name']},{r['model']},{r['imgsz']},"
                        f"{r['map50']:.4f},{r['map50_95']:.4f},{r['head_map']:.4f},{r['shem_map']:.4f},{r['time_s']}\n")
    
    # Summary
    print("\n" + "=" * 80)
    print("  FINAL RESULTS — YOLO11 mAP EXPERIMENTS")
    print("=" * 80)
    print(f"{'Name':<14} {'Model':<14} {'ImgSz':<6} {'mAP50':>7} {'mAP50-95':>9} {'Head':>7} {'Shem':>7} {'Time':>6}")
    print("-" * 75)
    
    results_list.sort(key=lambda x: x['map50_95'], reverse=True)
    for r in results_list:
        marker = " ★" if r == results_list[0] else ""
        print(f"{r['name']:<14} {r['model']:<14} {r['imgsz']:<6} "
              f"{r['map50']:>7.4f} {r['map50_95']:>9.4f} {r['head_map']:>7.4f} {r['shem_map']:>7.4f} {r['time_s']:>5}s{marker}")
    
    print(f"\n🏆 BEST: {results_list[0]['name']} (mAP@50-95={results_list[0]['map50_95']:.4f})")

if __name__ == "__main__":
    main()
