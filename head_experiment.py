"""
Head Detection Experiment Grid — Find the best config for >95% head detection.
Runs on H100 Kaggle. Evaluates against labels.txt ground truth.

Experiments:
  - Model sizes: yolov8s, yolov8m, yolov8l
  - Image sizes: 640, 960, 1280
  - Augmentation: heavy (handles B&W, color filters, side angles)
  - Plus: grayscale copies variant
"""
import os, sys, shutil, csv, time, argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Head Detection Experiment Grid')
parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: 100)')
parser.add_argument('--patience', type=int, default=30, help='Early stopping patience (default: 30)')
args = parser.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
EPOCHS = args.epochs
PATIENCE = args.patience

# Auto-detect data path (same logic as dual_specialist.py)
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

# Labels file (same directory as this script)
LABELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.txt")

WORK_DIR = "./yolo_head_exp"
RESULTS_FILE = "head_experiment_results.csv"

# ══════════════════════════════════════════════════════════════════════════════
# Experiment Configs
# ══════════════════════════════════════════════════════════════════════════════
EXPERIMENTS = [
    # (name, model, imgsz, batch, extra_aug_kwargs)
    ("s_640",   "yolov8s.pt", 640,  32, {}),
    ("s_960",   "yolov8s.pt", 960,  16, {}),
    ("m_640",   "yolov8m.pt", 640,  32, {}),
    ("m_960",   "yolov8m.pt", 960,  16, {}),
    ("m_1280",  "yolov8m.pt", 1280, 8,  {}),
    ("l_640",   "yolov8l.pt", 640,  16, {}),
    ("l_960",   "yolov8l.pt", 960,  8,  {}),
    ("l_1280",  "yolov8l.pt", 1280, 4,  {}),
    # Grayscale copies variant (uses yolov8m@640 but with grayscale training data)
    ("m_640_gray", "yolov8m.pt", 640, 32, {"grayscale": True}),
]

# Common heavy augmentation (handles B&W, color filters, side angles)
COMMON_AUG = dict(
    hsv_h=0.5,       # Strong hue shift
    hsv_s=0.9,       # Extreme saturation (can desaturate toward grayscale)
    hsv_v=0.5,       # Value/brightness variation
    degrees=30.0,    # Rotation for side angles
    translate=0.2,   # Translation
    scale=0.5,       # Scale variation
    fliplr=0.5,      # Horizontal flip
    mosaic=1.0,      # Mosaic augmentation
    mixup=0.3,       # Mixup blending
    erasing=0.3,     # Random erasing (partial occlusion)
    copy_paste=0.1,  # Copy-paste aug
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Prepare Data
# ══════════════════════════════════════════════════════════════════════════════
def prepare_head_data(include_grayscale=False):
    """Prepare head-only YOLO training data."""
    import cv2
    
    tag = "_gray" if include_grayscale else ""
    work = f"{WORK_DIR}{tag}"
    
    # Clean
    if os.path.exists(work):
        shutil.rmtree(work)
    
    for split in ['train', 'val']:
        os.makedirs(f"{work}/images/{split}", exist_ok=True)
        os.makedirs(f"{work}/labels/{split}", exist_ok=True)
    
    # Get all training images
    img_dir = f"{ROOT_DIR}/images/train"
    lbl_dir = f"{ROOT_DIR}/labels/train"
    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    # Filter: only images that have HEAD annotations (class 0)
    head_files = []
    for f in all_files:
        lbl_path = f"{lbl_dir}/{f.replace('.jpg', '.txt')}"
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as lf:
                for line in lf:
                    if line.strip().startswith('0 '):
                        head_files.append(f)
                        break
    
    # Also include some negative examples (no head) for better precision
    no_head_files = [f for f in all_files if f not in head_files]
    # Include 30% negatives
    import random
    random.seed(42)
    neg_sample = random.sample(no_head_files, min(len(no_head_files), len(head_files) // 2))
    
    all_train_files = head_files + neg_sample
    random.shuffle(all_train_files)
    
    # 80/20 split
    split_idx = int(len(all_train_files) * 0.8)
    train_files = all_train_files[:split_idx]
    val_files = all_train_files[split_idx:]
    
    for split, files in [('train', train_files), ('val', val_files)]:
        for f in files:
            # Copy image
            shutil.copy(f"{img_dir}/{f}", f"{work}/images/{split}/{f}")
            
            # Copy label (head only = class 0)
            lbl_path = f"{lbl_dir}/{f.replace('.jpg', '.txt')}"
            out_lbl = f"{work}/labels/{split}/{f.replace('.jpg', '.txt')}"
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as lf:
                    head_lines = [l for l in lf if l.strip().startswith('0 ')]
                with open(out_lbl, 'w') as of:
                    of.writelines(head_lines)
            else:
                # Empty label file (negative example)
                open(out_lbl, 'w').close()
            
            # Add grayscale copy if requested
            if include_grayscale and split == 'train':
                gray_name = f.replace('.jpg', '_gray.jpg')
                img = cv2.imread(f"{img_dir}/{f}")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(f"{work}/images/{split}/{gray_name}", gray_bgr)
                # Same label
                shutil.copy(out_lbl, f"{work}/labels/{split}/{gray_name.replace('.jpg', '.txt')}")
    
    # data.yaml
    yaml_content = f"""path: {os.path.abspath(work)}
train: images/train
val: images/val

names:
  0: head
"""
    with open(f"{work}/data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"  Data prepared: {len(train_files)} train, {len(val_files)} val" + 
          (" (+grayscale copies)" if include_grayscale else ""))
    return f"{work}/data.yaml"

# ══════════════════════════════════════════════════════════════════════════════
# 2. Load Ground Truth
# ══════════════════════════════════════════════════════════════════════════════
def load_ground_truth():
    """Load labels.txt ground truth for head detection."""
    gt = {}
    with open(LABELS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                gt[parts[0]] = int(parts[1])  # head column
    return gt

# ══════════════════════════════════════════════════════════════════════════════
# 3. Evaluate Head Detection
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_head_detection(model, test_dir, gt, imgsz, conf_thresholds=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]):
    """Run inference on test images and evaluate against ground truth."""
    
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')],
                         key=lambda x: int(x.replace('.jpg', '')))
    
    # Run inference on ALL test images
    detections = {}  # {filename: max_confidence}
    print(f"  Running inference on {len(test_images)} test images...")
    
    for img_file in test_images:
        img_path = os.path.join(test_dir, img_file)
        results = model.predict(img_path, conf=0.05, imgsz=imgsz, verbose=False)[0]
        
        max_conf = 0
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > max_conf:
                max_conf = conf
        
        detections[img_file] = max_conf
    
    # Evaluate at multiple confidence thresholds
    best_f1 = 0
    best_thresh = 0
    best_stats = None
    
    for thresh in conf_thresholds:
        tp = fp = fn = tn = 0
        for fname in gt:
            gt_val = gt[fname]
            pred_val = 1 if detections.get(fname, 0) > thresh else 0
            if gt_val == 1 and pred_val == 1: tp += 1
            elif gt_val == 0 and pred_val == 1: fp += 1
            elif gt_val == 1 and pred_val == 0: fn += 1
            else: tn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_stats = {
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': precision, 'recall': recall, 'f1': f1,
                'accuracy': accuracy, 'thresh': thresh
            }
    
    return best_stats, detections

# ══════════════════════════════════════════════════════════════════════════════
# 4. Main — Run All Experiments
# ══════════════════════════════════════════════════════════════════════════════

# Upgrade ultralytics first
os.system('pip install -U ultralytics')

def main():
    from ultralytics import YOLO
    import glob
    
    test_dir = f"{ROOT_DIR}/images/test"
    
    gt = load_ground_truth()
    print(f"Ground truth: {sum(gt.values())} heads out of {len(gt)} images")
    print(f"Data root: {ROOT_DIR}")
    print(f"Test dir: {test_dir} ({len(os.listdir(test_dir))} files)")
    print(f"Labels file: {LABELS_FILE}")
    print(f"Running {len(EXPERIMENTS)} experiments...\n")
    
    results = []
    
    for exp_name, model_name, imgsz, batch, extra in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {exp_name}")
        print(f"  Model: {model_name} | ImgSz: {imgsz} | Batch: {batch}")
        print(f"{'='*60}")
        
        t0 = time.time()
        
        # Prepare data
        use_gray = extra.get("grayscale", False)
        data_yaml = prepare_head_data(include_grayscale=use_gray)
        
        # Train
        model = YOLO(model_name)
        train_results = model.train(
            data=data_yaml,
            epochs=EPOCHS,
            imgsz=imgsz,
            batch=batch,
            project='./head_experiments',
            name=exp_name,
            patience=PATIENCE,
            exist_ok=True,
            **COMMON_AUG
        )
        
        # Get best weights — multiple strategies, guaranteed to find it
        best_weights = str(model.trainer.best)
        print(f"  Trainer reports best at: {best_weights}")
        
        if not os.path.exists(best_weights):
            # Strategy 2: search common locations
            search_paths = [
                f"./head_experiments/{exp_name}/weights/best.pt",
                f"./runs/detect/head_experiments/{exp_name}/weights/best.pt",
                f"./runs/head_experiments/{exp_name}/weights/best.pt",
            ]
            for sp in search_paths:
                if os.path.exists(sp):
                    best_weights = sp
                    print(f"  Found at alternate path: {best_weights}")
                    break
        
        if not os.path.exists(best_weights):
            # Strategy 3: brute force find
            import subprocess
            result = subprocess.run(['find', '.', '-path', f'*{exp_name}*best.pt'], 
                                    capture_output=True, text=True)
            found = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
            if found:
                best_weights = found[0]
                print(f"  Found via search: {best_weights}")
        
        assert os.path.exists(best_weights), \
            f"FATAL: best.pt not found for {exp_name}. Trainer said: {model.trainer.best}"
        
        model_best = YOLO(best_weights)
        
        # Evaluate
        stats, detections = evaluate_head_detection(model_best, test_dir, gt, imgsz)
        
        elapsed = time.time() - t0
        
        print(f"\n  RESULTS for {exp_name}:")
        print(f"  Best threshold: {stats['thresh']}")
        print(f"  TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} TN={stats['tn']}")
        print(f"  Precision={stats['precision']:.3f} Recall={stats['recall']:.3f} F1={stats['f1']:.3f}")
        print(f"  Accuracy={stats['accuracy']:.3f}")
        print(f"  Time: {elapsed:.0f}s")
        
        results.append({
            'name': exp_name,
            'model': model_name,
            'imgsz': imgsz,
            'batch': batch,
            'best_thresh': stats['thresh'],
            'tp': stats['tp'], 'fp': stats['fp'],
            'fn': stats['fn'], 'tn': stats['tn'],
            'precision': stats['precision'],
            'recall': stats['recall'],
            'f1': stats['f1'],
            'accuracy': stats['accuracy'],
            'time_s': int(elapsed),
            'weights_path': best_weights
        })
        
        # Save results incrementally (after every experiment, even failed ones)
        with open(RESULTS_FILE, 'w') as f:
            f.write('name,model,imgsz,batch,best_thresh,tp,fp,fn,tn,precision,recall,f1,accuracy,time_s\n')
            for r in results:
                f.write(f"{r['name']},{r['model']},{r['imgsz']},{r['batch']},{r['best_thresh']},"
                        f"{r['tp']},{r['fp']},{r['fn']},{r['tn']},"
                        f"{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f},{r['accuracy']:.4f},{r['time_s']}\n")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Name':<14} {'Model':<12} {'ImgSz':<6} {'Thresh':<7} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6} {'Time':>5}")
    print("-" * 80)
    
    results.sort(key=lambda x: x['f1'], reverse=True)
    for r in results:
        marker = " ★" if r == results[0] else ""
        print(f"{r['name']:<14} {r['model']:<12} {r['imgsz']:<6} {r['best_thresh']:<7} "
              f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {r['accuracy']:>6.3f} {r['time_s']:>4}s{marker}")
    
    print(f"\n🏆 BEST: {results[0]['name']} (F1={results[0]['f1']:.4f})")
    print(f"   Saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
