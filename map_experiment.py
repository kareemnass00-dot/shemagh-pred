"""
mAP-Focused Training — Maximize bounding box quality (mAP@[0.5:0.95])
# ============================================================
# Strategy: Single model, both classes, high resolution, large backbone
# - Includes negatives in train+val (empty labels = pure FP detector)
# - Stratified split (ensures shemagh representation in val)
# - Close-mosaic for last 10 epochs (tighter boxes)
# - Low geometric distortion (preserve box alignment)
# - Per-class mAP reporting
# ============================================================
"""
import os, sys, shutil, random, time, argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='mAP-Focused Experiment Grid')
parser.add_argument('--epochs', type=int, default=150, help='Training epochs (default: 150)')
parser.add_argument('--patience', type=int, default=40, help='Early stopping patience (default: 40)')
args = parser.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
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
    print(f"ERROR: Could not find dataset in {POSSIBLE_PATHS}")
    print("Files in current dir:", os.listdir("."))
    if os.path.exists("./data"): print("Files in ./data:", os.listdir("./data"))
    sys.exit(1)

WORK_DIR = "./yolo_map_exp"

# ══════════════════════════════════════════════════════════════════════════════
# Experiment Configs
# ══════════════════════════════════════════════════════════════════════════════
EXPERIMENTS = [
    # Top 2 from v1 results
    ("l_960",        "yolov8l.pt",  960,  12, "moderate"),
    ("x_1280_light", "yolov8x.pt",  1280, 4,  "light"),
]

# Moderate aug: enough diversity, low geometric distortion for tight boxes
AUG_MODERATE = dict(
    hsv_h=0.3,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=5.0,       # Very mild rotation (tight boxes)
    translate=0.1,
    scale=0.2,         # Less scale variation
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,         # No mixup (destroys box edges)
    erasing=0.0,       # No erasing
    copy_paste=0.1,
    close_mosaic=10,   # Disable mosaic last 10 epochs for tighter boxes
)

# Light aug: minimal distortion, no mosaic at all
AUG_LIGHT = dict(
    hsv_h=0.015,
    hsv_s=0.2,
    hsv_v=0.2,
    degrees=0.0,       # No rotation
    translate=0.05,
    scale=0.1,
    fliplr=0.5,
    mosaic=0.0,        # NO mosaic
    mixup=0.0,
    erasing=0.0,
    copy_paste=0.0,
    close_mosaic=0,
)

# Heavy aug: for comparison (we expect this to hurt mAP50-95)
AUG_HEAVY = dict(
    hsv_h=0.5,
    hsv_s=0.9,
    hsv_v=0.5,
    degrees=30.0,
    translate=0.2,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.3,
    erasing=0.3,
    copy_paste=0.1,
    close_mosaic=10,
)

AUG_MAP = {"moderate": AUG_MODERATE, "light": AUG_LIGHT, "heavy": AUG_HEAVY}

# ══════════════════════════════════════════════════════════════════════════════
# 1. Prepare Data — BOTH classes, WITH negatives, STRATIFIED split
# ══════════════════════════════════════════════════════════════════════════════
def prepare_map_data():
    """Prepare YOLO data with both classes.
    - Includes ALL images (annotated + negatives with empty labels)
    - Stratified split: ensures both classes represented in val
    - Prints per-class counts
    """
    
    work = WORK_DIR
    if os.path.exists(work):
        shutil.rmtree(work)
    
    for split in ['train', 'val']:
        os.makedirs(f"{work}/images/{split}", exist_ok=True)
        os.makedirs(f"{work}/labels/{split}", exist_ok=True)
    
    img_dir = f"{ROOT_DIR}/images/train"
    lbl_dir = f"{ROOT_DIR}/labels/train"
    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    # Categorize images for stratification
    has_head_only = []      # only head annotations
    has_shemagh_only = []   # only shemagh annotations
    has_both = []           # both classes
    negatives = []          # no annotations
    
    for f in all_files:
        lbl_path = f"{lbl_dir}/{f.replace('.jpg', '.txt')}"
        has_head = False
        has_shem = False
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as lf:
                for line in lf:
                    line = line.strip()
                    if line.startswith('0 '): has_head = True
                    elif line.startswith('1 '): has_shem = True
        
        if has_head and has_shem: has_both.append(f)
        elif has_head: has_head_only.append(f)
        elif has_shem: has_shemagh_only.append(f)
        else: negatives.append(f)
    
    print(f"  Dataset breakdown:")
    print(f"    Both classes: {len(has_both)}")
    print(f"    Head only:    {len(has_head_only)}")
    print(f"    Shemagh only: {len(has_shemagh_only)}")
    print(f"    Negatives:    {len(negatives)}")
    
    # Stratified 80/20 split per category
    random.seed(42)
    train_files = []
    val_files = []
    
    for group_name, group in [("both", has_both), ("head_only", has_head_only), 
                               ("shem_only", has_shemagh_only), ("negatives", negatives)]:
        random.shuffle(group)
        n_val = max(1, int(len(group) * 0.2)) if group else 0
        val_files.extend(group[:n_val])
        train_files.extend(group[n_val:])
        print(f"    {group_name}: {len(group) - n_val} train, {n_val} val")
    
    random.shuffle(train_files)
    random.shuffle(val_files)
    
    # Copy files
    head_train = head_val = shem_train = shem_val = 0
    
    for split, files in [('train', train_files), ('val', val_files)]:
        for f in files:
            # Copy image
            shutil.copy(f"{img_dir}/{f}", f"{work}/images/{split}/{f}")
            
            # Copy label or create empty
            lbl_src = f"{lbl_dir}/{f.replace('.jpg', '.txt')}"
            lbl_dst = f"{work}/labels/{split}/{f.replace('.jpg', '.txt')}"
            if os.path.exists(lbl_src):
                content = open(lbl_src).read().strip()
                if content:
                    with open(lbl_dst, 'w') as out:
                        out.write(content + '\n')
                    # Count instances
                    for line in content.split('\n'):
                        if line.startswith('0 '):
                            if split == 'train': head_train += 1
                            else: head_val += 1
                        elif line.startswith('1 '):
                            if split == 'train': shem_train += 1
                            else: shem_val += 1
                else:
                    open(lbl_dst, 'w').close()
            else:
                open(lbl_dst, 'w').close()  # Empty = negative
    
    print(f"\n  Instance counts:")
    print(f"    Head:    {head_train} train, {head_val} val")
    print(f"    Shemagh: {shem_train} train, {shem_val} val")
    
    # data.yaml
    yaml_content = f"""path: {os.path.abspath(work)}
train: images/train
val: images/val

names:
  0: head
  1: shemagh
"""
    with open(f"{work}/data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"\n  Total: {len(train_files)} train, {len(val_files)} val (including negatives)")
    return f"{work}/data.yaml"

# ══════════════════════════════════════════════════════════════════════════════
# 2. Main — Run Experiments
# ══════════════════════════════════════════════════════════════════════════════
os.system('pip install -U ultralytics')

def main():
    from ultralytics import YOLO
    
    print(f"Data root: {ROOT_DIR}")
    print(f"Epochs: {EPOCHS}, Patience: {PATIENCE}")
    print(f"Running {len(EXPERIMENTS)} mAP experiments...\n")
    
    # Prepare data once
    data_yaml = prepare_map_data()
    
    results = []
    
    for exp_name, model_name, imgsz, batch, aug_name in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  mAP EXPERIMENT: {exp_name}")
        print(f"  Model: {model_name} | ImgSz: {imgsz} | Batch: {batch} | Aug: {aug_name}")
        print(f"{'='*60}")
        
        t0 = time.time()
        aug = AUG_MAP[aug_name]
        
        # Train
        model = YOLO(model_name)
        train_results = model.train(
            data=data_yaml,
            epochs=EPOCHS,
            imgsz=imgsz,
            batch=batch,
            project='./map_experiments',
            name=exp_name,
            patience=PATIENCE,
            exist_ok=True,
            **aug
        )
        
        # Get best weights
        best_weights = str(model.trainer.best)
        print(f"  Trainer reports best at: {best_weights}")
        
        if not os.path.exists(best_weights):
            import subprocess
            result = subprocess.run(['find', '.', '-path', f'*map_experiments*{exp_name}*best.pt'],
                                    capture_output=True, text=True)
            found = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
            if found:
                best_weights = found[0]
                print(f"  Found via search: {best_weights}")
        
        assert os.path.exists(best_weights), f"FATAL: best.pt not found for {exp_name}"
        
        # Validate to get mAP metrics
        model_best = YOLO(best_weights)
        val_results = model_best.val(data=data_yaml, imgsz=imgsz)
        
        elapsed = time.time() - t0
        
        # Extract mAP metrics
        map50 = float(val_results.box.map50)
        map50_95 = float(val_results.box.map)
        
        # Per-class mAP
        per_class = val_results.box.maps
        head_map = float(per_class[0]) if len(per_class) > 0 else 0
        shem_map = float(per_class[1]) if len(per_class) > 1 else 0
        
        print(f"\n  RESULTS for {exp_name}:")
        print(f"  mAP@50    = {map50:.4f}")
        print(f"  mAP@50-95 = {map50_95:.4f}")
        print(f"  Head mAP  = {head_map:.4f}")
        print(f"  Shem mAP  = {shem_map:.4f}")
        print(f"  Time: {elapsed:.0f}s")
        
        results.append({
            'name': exp_name,
            'model': model_name,
            'imgsz': imgsz,
            'aug': aug_name,
            'map50': map50,
            'map50_95': map50_95,
            'head_map': head_map,
            'shem_map': shem_map,
            'time_s': int(elapsed),
            'weights': best_weights
        })
        
        # Save incrementally
        with open('map_experiment_results.csv', 'w') as f:
            f.write('name,model,imgsz,aug,map50,map50_95,head_map,shem_map,time_s\n')
            for r in results:
                f.write(f"{r['name']},{r['model']},{r['imgsz']},{r['aug']},"
                        f"{r['map50']:.4f},{r['map50_95']:.4f},{r['head_map']:.4f},{r['shem_map']:.4f},{r['time_s']}\n")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("  FINAL RESULTS — mAP EXPERIMENTS")
    print("=" * 80)
    print(f"{'Name':<16} {'Model':<12} {'ImgSz':<6} {'Aug':<10} {'mAP50':>7} {'mAP50-95':>9} {'Head':>7} {'Shem':>7} {'Time':>6}")
    print("-" * 80)
    
    results.sort(key=lambda x: x['map50_95'], reverse=True)
    for r in results:
        marker = " ★" if r == results[0] else ""
        print(f"{r['name']:<16} {r['model']:<12} {r['imgsz']:<6} {r['aug']:<10} "
              f"{r['map50']:>7.4f} {r['map50_95']:>9.4f} {r['head_map']:>7.4f} {r['shem_map']:>7.4f} {r['time_s']:>5}s{marker}")
    
    print(f"\n🏆 BEST mAP: {results[0]['name']} (mAP@50-95={results[0]['map50_95']:.4f})")
    print(f"   Estimated Final Score (with F1=0.918): {0.5 * results[0]['map50_95'] + 0.5 * 0.918:.4f}")
    print(f"   Saved to map_experiment_results.csv")

if __name__ == "__main__":
    main()
