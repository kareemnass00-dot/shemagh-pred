"""
mAP-Focused Training v2 — fixes to map_experiment.py
# ============================================================
# Goals:
#   - Improve mAP@[0.5:0.95] via better box quality + fewer false positives.
#
# Key fixes vs v1:
#   1) Include negatives (empty labels) in train/val (at least val is required).
#   2) Stratified split by {has_head, has_shemagh} to stabilize rare-class mAP.
#   3) "Light" augmentation is truly light (no mosaic/mixup/copy-paste).
#   4) Use close_mosaic to turn off mosaic late for tighter boxes.
#
# Notes:
#   - This script creates a fresh dataset copy under --work-dir for reproducibility.
#   - Keep your original scripts untouched.
"""

import argparse
import csv
import os
import random
import shutil
import sys
import time
from pathlib import Path


def find_dataset_root() -> str:
    possible_paths = [
        "/kaggle/input/dal-shemagh-identification",
        "/kaggle/input/dal-shemagh-detection-challenge",
        "./data/dal-shemagh-detection-challenge",
        "./data",
        "../input/dal-shemagh-detection-challenge",
        "../input/dal-shemagh-identification",
    ]
    for p in possible_paths:
        if os.path.exists(p) and os.path.exists(f"{p}/images/train"):
            return p

    print(f"ERROR: Could not find dataset in {possible_paths}")
    print("Files in current dir:", os.listdir("."))
    if os.path.exists("./data"):
        print("Files in ./data:", os.listdir("./data"))
    sys.exit(1)


def parse_label_file(label_path: Path) -> tuple[bool, bool, int, int]:
    """Return (has_head, has_shemagh, head_instances, shemagh_instances)."""
    if not label_path.exists():
        return False, False, 0, 0

    has_head = False
    has_shemagh = False
    head_instances = 0
    shemagh_instances = 0

    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
        if class_id == 0:
            has_head = True
            head_instances += 1
        elif class_id == 1:
            has_shemagh = True
            shemagh_instances += 1

    return has_head, has_shemagh, head_instances, shemagh_instances


def stratified_split(
    files: list[str],
    *,
    file_flags: dict[str, tuple[bool, bool]],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Stratify by (has_head, has_shemagh) bucket."""
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1 (got {val_ratio})")

    buckets: dict[tuple[bool, bool], list[str]] = {
        (False, False): [],
        (True, False): [],
        (False, True): [],
        (True, True): [],
    }
    for fname in files:
        buckets[file_flags[fname]].append(fname)

    rng = random.Random(seed)
    for bucket_files in buckets.values():
        rng.shuffle(bucket_files)

    train: list[str] = []
    val: list[str] = []

    def bucket_n_val(n: int) -> int:
        if n <= 1:
            return 0
        desired = int(round(n * val_ratio))
        if desired == 0:
            desired = 1
        if desired >= n:
            desired = n - 1
        return desired

    print("Split buckets (total images):")
    for (has_head, has_shemagh), bucket_files in buckets.items():
        n = len(bucket_files)
        n_val = bucket_n_val(n)
        print(f"  has_head={has_head} has_shemagh={has_shemagh}: {n} (val={n_val}, train={n - n_val})")
        val.extend(bucket_files[:n_val])
        train.extend(bucket_files[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def prepare_map_data(
    *,
    root_dir: str,
    work_dir: str,
    val_ratio: float,
    seed: int,
) -> str:
    """Prepare YOLO data with BOTH classes (head=0, shemagh=1), including negatives."""
    work = Path(work_dir)
    if work.exists():
        shutil.rmtree(work)

    for split in ("train", "val"):
        (work / "images" / split).mkdir(parents=True, exist_ok=True)
        (work / "labels" / split).mkdir(parents=True, exist_ok=True)

    img_dir = Path(root_dir) / "images" / "train"
    lbl_dir = Path(root_dir) / "labels" / "train"
    all_files = sorted([p.name for p in img_dir.iterdir() if p.suffix.lower() == ".jpg"])

    # Precompute class presence per image (for stratification + stats).
    file_flags: dict[str, tuple[bool, bool]] = {}
    instance_counts: dict[str, tuple[int, int]] = {}
    for fname in all_files:
        label_path = lbl_dir / f"{Path(fname).stem}.txt"
        has_head, has_shemagh, head_n, shem_n = parse_label_file(label_path)
        file_flags[fname] = (has_head, has_shemagh)
        instance_counts[fname] = (head_n, shem_n)

    train_files, val_files = stratified_split(
        all_files,
        file_flags=file_flags,
        val_ratio=val_ratio,
        seed=seed,
    )

    def copy_split(split: str, files: list[str]) -> dict[str, int]:
        stats = {
            "images": 0,
            "images_with_head": 0,
            "images_with_shemagh": 0,
            "head_instances": 0,
            "shemagh_instances": 0,
            "negatives": 0,
        }
        for fname in files:
            stats["images"] += 1

            src_img = img_dir / fname
            dst_img = work / "images" / split / fname
            shutil.copy2(src_img, dst_img)

            src_lbl = lbl_dir / f"{Path(fname).stem}.txt"
            dst_lbl = work / "labels" / split / f"{Path(fname).stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                dst_lbl.write_text("")

            has_head, has_shemagh = file_flags[fname]
            head_n, shem_n = instance_counts[fname]

            if has_head:
                stats["images_with_head"] += 1
                stats["head_instances"] += head_n
            if has_shemagh:
                stats["images_with_shemagh"] += 1
                stats["shemagh_instances"] += shem_n
            if not has_head and not has_shemagh:
                stats["negatives"] += 1

        return stats

    train_stats = copy_split("train", train_files)
    val_stats = copy_split("val", val_files)

    def fmt_stats(name: str, s: dict[str, int]) -> str:
        return (
            f"{name}: {s['images']} imgs | "
            f"head: {s['images_with_head']} imgs / {s['head_instances']} inst | "
            f"shemagh: {s['images_with_shemagh']} imgs / {s['shemagh_instances']} inst | "
            f"neg: {s['negatives']}"
        )

    print("\nDataset stats:")
    print(" ", fmt_stats("train", train_stats))
    print(" ", fmt_stats("val  ", val_stats))

    yaml_content = f"""path: {work.resolve()}
train: images/train
val: images/val

nc: 2
names:
  0: head
  1: shemagh
"""
    (work / "data.yaml").write_text(yaml_content)
    return str(work / "data.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="mAP-Focused Experiment Grid (v2)")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs (default: 150)")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience (default: 40)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Default imgsz for single-run mode")
    parser.add_argument("--batch", type=int, default=8, help="Default batch for single-run mode")
    parser.add_argument("--seed", type=int, default=42, help="Split seed (default: 42)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (default: 0.2)")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic for last N epochs (default: 10)")
    parser.add_argument("--work-dir", type=str, default="./yolo_map_exp_v2", help="Prepared dataset output dir")
    parser.add_argument("--project", type=str, default="./map_experiments_v2", help="Ultralytics project dir")
    parser.add_argument("--results-csv", type=str, default="map_experiment_v2_results.csv", help="Results CSV path")
    parser.add_argument("--val-conf", type=float, default=0.001, help="Validation conf (default: 0.001)")
    parser.add_argument("--val-iou", type=float, default=0.65, help="Validation NMS IoU (default: 0.65)")
    parser.add_argument(
        "--mode",
        type=str,
        default="grid",
        choices=["grid", "single"],
        help="Run full grid or one config (default: grid)",
    )
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="Single-run model (default: yolov8x.pt)")
    parser.add_argument("--aug", type=str, default="moderate", choices=["light", "moderate", "heavy"], help="Aug profile")
    args = parser.parse_args()

    epochs = args.epochs
    patience = args.patience
    seed = args.seed
    val_ratio = args.val_ratio
    close_mosaic = args.close_mosaic

    root_dir = find_dataset_root()
    print(f"Data root: {root_dir}")
    print(f"Epochs: {epochs} | Patience: {patience} | Seed: {seed} | Val ratio: {val_ratio}")

    # Upgrade ultralytics (kept consistent with your other scripts)
    os.system("pip install -U ultralytics")

    # Prepare dataset once (shared across all experiments)
    data_yaml = prepare_map_data(
        root_dir=root_dir,
        work_dir=args.work_dir,
        val_ratio=val_ratio,
        seed=seed,
    )

    # Augmentation profiles: light is truly light (no mosaic/mixup/copy_paste).
    aug_light = dict(
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.05,
        scale=0.15,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        erasing=0.0,
        copy_paste=0.0,
    )
    aug_moderate = dict(
        hsv_h=0.2,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=10.0,
        translate=0.1,
        scale=0.25,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.05,
        erasing=0.0,
        copy_paste=0.05,
    )
    aug_heavy = dict(
        hsv_h=0.4,
        hsv_s=0.8,
        hsv_v=0.4,
        degrees=20.0,
        translate=0.2,
        scale=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        erasing=0.0,
        copy_paste=0.1,
    )
    aug_map = {"light": aug_light, "moderate": aug_moderate, "heavy": aug_heavy}

    # Experiment grid
    experiments = [
        # (name, model, imgsz, batch, aug_profile)
        ("x_1280_mod", "yolov8x.pt", 1280, 4, "moderate"),
        ("x_960_mod", "yolov8x.pt", 960, 8, "moderate"),
        ("l_1280_mod", "yolov8l.pt", 1280, 6, "moderate"),
        ("l_960_mod", "yolov8l.pt", 960, 10, "moderate"),
        ("x_1280_light", "yolov8x.pt", 1280, 4, "light"),
        ("x_1280_heavy", "yolov8x.pt", 1280, 4, "heavy"),
    ]

    if args.mode == "single":
        exp_name = f"single_{Path(args.model).stem}_{args.imgsz}_{args.aug}"
        experiments = [(exp_name, args.model, args.imgsz, args.batch, args.aug)]

    from ultralytics import YOLO

    results: list[dict[str, object]] = []

    for exp_name, model_name, imgsz, batch, aug_name in experiments:
        print(f"\n{'=' * 72}")
        print(f"mAP EXPERIMENT v2: {exp_name}")
        print(f"Model: {model_name} | ImgSz: {imgsz} | Batch: {batch} | Aug: {aug_name} | close_mosaic: {close_mosaic}")
        print(f"{'=' * 72}")

        t0 = time.time()
        aug = aug_map[aug_name]

        model = YOLO(model_name)
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=args.project,
            name=exp_name,
            patience=patience,
            exist_ok=True,
            close_mosaic=close_mosaic,
            **aug,
        )

        best_weights = str(model.trainer.best)
        if not os.path.exists(best_weights):
            best_guess = list(Path(args.project).rglob(f"{exp_name}/weights/best.pt"))
            if best_guess:
                best_weights = str(best_guess[0])

        assert os.path.exists(best_weights), f"FATAL: best.pt not found for {exp_name}"
        print(f"Best weights: {best_weights}")

        model_best = YOLO(best_weights)
        val_results = model_best.val(
            data=data_yaml,
            imgsz=imgsz,
            conf=args.val_conf,
            iou=args.val_iou,
        )

        elapsed = time.time() - t0

        map50 = float(val_results.box.map50)
        map50_95 = float(val_results.box.map)
        per_class = val_results.box.maps
        head_map = float(per_class[0]) if len(per_class) > 0 else 0.0
        shem_map = float(per_class[1]) if len(per_class) > 1 else 0.0

        print("\nRESULTS:")
        print(f"  mAP@50      = {map50:.4f}")
        print(f"  mAP@50-95   = {map50_95:.4f}")
        print(f"  head mAP    = {head_map:.4f}")
        print(f"  shemagh mAP = {shem_map:.4f}")
        print(f"  time        = {elapsed:.0f}s")

        results.append(
            {
                "name": exp_name,
                "model": model_name,
                "imgsz": imgsz,
                "batch": batch,
                "aug": aug_name,
                "close_mosaic": close_mosaic,
                "val_conf": args.val_conf,
                "val_iou": args.val_iou,
                "map50": map50,
                "map50_95": map50_95,
                "head_map": head_map,
                "shem_map": shem_map,
                "time_s": int(elapsed),
                "weights": best_weights,
            }
        )

        # Save incrementally (so you don't lose runs if interrupted).
        with open(args.results_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "model",
                    "imgsz",
                    "batch",
                    "aug",
                    "close_mosaic",
                    "val_conf",
                    "val_iou",
                    "map50",
                    "map50_95",
                    "head_map",
                    "shem_map",
                    "time_s",
                    "weights",
                ],
            )
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    results.sort(key=lambda x: float(x["map50_95"]), reverse=True)
    best = results[0]
    print("\n" + "=" * 80)
    print("FINAL RESULTS — mAP EXPERIMENTS v2")
    print("=" * 80)
    print(
        f"BEST: {best['name']} | mAP@50-95={float(best['map50_95']):.4f} "
        f"(head={float(best['head_map']):.4f}, shemagh={float(best['shem_map']):.4f})"
    )
    print(f"Saved to {args.results_csv}")


if __name__ == "__main__":
    main()

