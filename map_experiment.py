"""
mAP Experiment v3 — YOLO11 + Oversampled Negatives
# ============================================================
# Fixes the 73% background FP problem:
#   1. YOLO11 (newer architecture)
#   2. Oversample negatives 2x for ~1:1 ratio
#   3. Light augmentation (no mosaic)
#   4. conf=0.25 at inference (like baseline)
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
    print("ERROR: Could not find dataset"); sys.exit(1)

WORK_DIR = "./yolo11_map_exp"

# ══════════════════════════════════════════════════════════════════════════════
# Experiments
# ══════════════════════════════════════════════════════════════════════════════
EXPERIMENTS = [
    # (name, model, imgsz, batch)
    ("y11n_640", "yolo11n.pt", 640, 32),
    ("y11s_640", "yolo11s.pt", 640, 32),
    ("y11m_640", "yolo11m.pt", 640, 16),
]

# Light augmentation (matching baseline)
AUG = dict(
    hsv_h=0.015,
    hsv_s=0.1,
    hsv_v=0.1,
    degrees=10.0,
    translate=0.1,
    scale=0.1,
    fliplr=0.5,
    mosaic=0.0,
    mixup=0.0,
    erasing=0.0,
    copy_paste=0.0,
    close_mosaic=0,
)

# ══════════════════════════════════════════════════════════════════════════════
# Prepare Data — ALL images + OVERSAMPLED negatives
# ══════════════════════════════════════════════════════════════════════════════
def prepare_data():
    """
    - Use all training images
    - Oversample negatives 2x so model learns "nothing is here"
    - Current ratio: 448 positive : 203 negative (2.2:1)
    - After 2x oversample: 448 positive : 406 negative (~1.1:1)
    """
    work = WORK_DIR
    if os.path.exists(work):
        shutil.rmtree(work)
    
    for sub in ['images/train', 'labels/train']:
        os.makedirs(f"{work}/{sub}", exist_ok=True)
    
    img_dir = f"{ROOT_DIR}/images/train"
    lbl_dir = f"{ROOT_DIR}/labels/train"
    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    positives = []
    negatives = []
    
    for f in all_files:
        lbl_path = f"{lbl_dir}/{f.replace('.jpg', '.txt')}"
        has_annotation = False
        if os.path.exists(lbl_path):
            content = open(lbl_path).read().strip()
            if content:
                has_annotation = True
        
        if has_annotation:
            positives.append(f)
        else:
            negatives.append(f)
    
    print(f"  Original: {len(positives)} positive, {len(negatives)} negative")
    
    # Copy all positive images + labels
    for f in positives:
        shutil.copy(f"{img_dir}/{f}", f"{work}/images/train/{f}")
        lbl_src = f"{lbl_dir}/{f.replace('.jpg', '.txt')}"
        lbl_dst = f"{work}/labels/train/{f.replace('.jpg', '.txt')}"
        shutil.copy(lbl_src, lbl_dst)
    
    # Copy all negative images + empty labels
    for f in negatives:
        shutil.copy(f"{img_dir}/{f}", f"{work}/images/train/{f}")
        open(f"{work}/labels/train/{f.replace('.jpg', '.txt')}", 'w').close()
    
    # Oversample negatives 2x (copy with suffix)
    for copy_idx in range(1, 3):  # 2 extra copies
        for f in negatives:
            stem = f.replace('.jpg', '')
            new_name = f"{stem}_neg{copy_idx}.jpg"
            shutil.copy(f"{img_dir}/{f}", f"{work}/images/train/{new_name}")
            open(f"{work}/labels/train/{new_name.replace('.jpg', '.txt')}", 'w').close()
    
    total_neg = len(negatives) * 3  # original + 2 copies
    total = len(positives) + total_neg
    print(f"  After oversample: {len(positives)} positive, {total_neg} negative")
    print(f"  Ratio: {len(positives)/total_neg:.1f}:1 (pos:neg)")
    print(f"  Total training images: {total}")
    
    # data.yaml — val=train (like baseline)
    yaml_content = f"""path: {os.path.abspath(work)}
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
    print(f"Epochs: {EPOCHS}\n")
    
    data_yaml = prepare_data()
    results_list = []
    
    for exp_name, model_name, imgsz, batch in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {exp_name}")
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
    print("  FINAL RESULTS — YOLO11 + OVERSAMPLED NEGATIVES")
    print("=" * 80)
    print(f"{'Name':<14} {'Model':<14} {'ImgSz':<6} {'mAP50':>7} {'mAP50-95':>9} {'Head':>7} {'Shem':>7} {'Time':>6}")
    print("-" * 75)
    
    results_list.sort(key=lambda x: x['map50_95'], reverse=True)
    for r in results_list:
        marker = " ★" if r == results_list[0] else ""
        print(f"{r['name']:<14} {r['model']:<14} {r['imgsz']:<6} "
              f"{r['map50']:>7.4f} {r['map50_95']:>9.4f} {r['head_map']:>7.4f} {r['shem_map']:>7.4f} {r['time_s']:>5}s{marker}")

if __name__ == "__main__":
    main()
