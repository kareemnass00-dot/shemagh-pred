"""
Step 2: Generate OOF predictions for the current top model set.

What this script does:
1) Builds stratified K folds from train data (including negatives).
2) Trains each selected model on K-1 folds and predicts the held-out fold.
3) Exports box-level OOF predictions with IoU-based TP labels for reranking.

Outputs (inside --out-dir):
- folds_manifest.csv
- oof_images.csv
- oof_ground_truth.csv
- oof_predictions.csv
- oof_runs.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class ModelSpec:
    tag: str
    weights: str
    imgsz: int
    batch: int

    @property
    def is_rtdetr(self) -> bool:
        token = f"{self.tag} {self.weights}".lower()
        return "rtdetr" in token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OOF predictions for reranking.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model spec as 'tag,weights,imgsz,batch'. "
            "Can be repeated. Example: --model rtdetr_l,rtdetr-l.pt,640,8"
        ),
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs per fold (default: 80)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (default: 20)")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers (default: 8)")
    parser.add_argument("--device", type=str, default=None, help="Device passed to Ultralytics (e.g. 0, 0,1, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--pred-conf", type=float, default=0.001, help="Prediction conf for OOF export (default: 0.001)")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per image (default: 300)")
    parser.add_argument("--tp-iou", type=float, default=0.50, help="IoU threshold for TP label (default: 0.50)")
    parser.add_argument("--project", type=str, default="./oof_runs", help="Ultralytics project output dir")
    parser.add_argument("--out-dir", type=str, default="./oof_stage2", help="OOF export output dir")
    parser.add_argument("--root-dir", type=str, default=None, help="Dataset root override")
    parser.add_argument("--use-symlinks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-runs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rebuild-fold-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def default_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(tag="rtdetr_l", weights="rtdetr-l.pt", imgsz=640, batch=8),
        ModelSpec(tag="y11s", weights="yolo11s.pt", imgsz=640, batch=16),
        ModelSpec(tag="y11n", weights="yolo11n.pt", imgsz=640, batch=16),
    ]


def parse_model_specs(raw_specs: list[str]) -> list[ModelSpec]:
    if not raw_specs:
        return default_model_specs()

    specs: list[ModelSpec] = []
    for raw in raw_specs:
        parts = [token.strip() for token in raw.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Invalid --model '{raw}'. Expected tag,weights,imgsz,batch")
        tag, weights, imgsz_text, batch_text = parts
        specs.append(
            ModelSpec(
                tag=tag,
                weights=weights,
                imgsz=int(imgsz_text),
                batch=int(batch_text),
            )
        )
    return specs


def find_dataset_root(override: str | None) -> Path:
    if override:
        root = Path(override)
        if (root / "images" / "train").exists() and (root / "labels" / "train").exists():
            return root
        raise FileNotFoundError(f"Invalid --root-dir: {override}")

    possible_paths = [
        "/kaggle/input/dal-shemagh-identification",
        "/kaggle/input/dal-shemagh-detection-challenge",
        "./data/dal-shemagh-detection-challenge",
        "./data",
        "../input/dal-shemagh-detection-challenge",
        "../input/dal-shemagh-identification",
    ]
    for path in possible_paths:
        root = Path(path)
        if (root / "images" / "train").exists() and (root / "labels" / "train").exists():
            return root

    raise FileNotFoundError(f"Could not find dataset in: {possible_paths}")


def list_train_images(root_dir: Path) -> list[str]:
    image_dir = root_dir / "images" / "train"
    return sorted([path.name for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS])


def parse_label_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            continue
        if class_id not in (0, 1):
            continue
        if w <= 0 or h <= 0:
            continue
        boxes.append((class_id, x, y, w, h))
    return boxes


def bucket_key(boxes: list[tuple[int, float, float, float, float]]) -> tuple[bool, bool]:
    has_head = any(class_id == 0 for class_id, *_ in boxes)
    has_shemagh = any(class_id == 1 for class_id, *_ in boxes)
    return has_head, has_shemagh


def build_stratified_folds(
    files: list[str],
    labels_by_file: dict[str, list[tuple[int, float, float, float, float]]],
    n_folds: int,
    seed: int,
) -> list[list[str]]:
    if n_folds < 2:
        raise ValueError("--folds must be at least 2")
    if n_folds > len(files):
        raise ValueError("--folds cannot exceed number of images")

    buckets: dict[tuple[bool, bool], list[str]] = {
        (False, False): [],
        (True, False): [],
        (False, True): [],
        (True, True): [],
    }
    for name in files:
        buckets[bucket_key(labels_by_file[name])].append(name)

    rng = random.Random(seed)
    folds: list[list[str]] = [[] for _ in range(n_folds)]

    for key, bucket in buckets.items():
        rng.shuffle(bucket)
        start = rng.randrange(n_folds)
        for index, filename in enumerate(bucket):
            fold_index = (start + index) % n_folds
            folds[fold_index].append(filename)
        print(f"Bucket {key}: {len(bucket)} images")

    for index in range(n_folds):
        folds[index].sort(key=lambda name: (not name.split(".")[0].isdigit(), name))
        print(f"Fold {index}: {len(folds[index])} validation images")

    return folds


def link_or_copy(src: Path, dst: Path, use_symlinks: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if use_symlinks:
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def prepare_fold_dataset(
    root_dir: Path,
    fold_dir: Path,
    train_files: list[str],
    val_files: list[str],
    use_symlinks: bool,
    rebuild: bool,
) -> Path:
    if rebuild and fold_dir.exists():
        shutil.rmtree(fold_dir)

    for split in ("train", "val"):
        (fold_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    src_img_dir = root_dir / "images" / "train"
    src_lbl_dir = root_dir / "labels" / "train"

    def copy_split(split: str, files: list[str]) -> None:
        out_img = fold_dir / "images" / split
        out_lbl = fold_dir / "labels" / split
        for filename in files:
            src_img = src_img_dir / filename
            src_lbl = src_lbl_dir / f"{Path(filename).stem}.txt"
            dst_img = out_img / filename
            dst_lbl = out_lbl / f"{Path(filename).stem}.txt"

            link_or_copy(src_img, dst_img, use_symlinks)
            if src_lbl.exists():
                if not dst_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)
            elif not dst_lbl.exists():
                dst_lbl.write_text("")

    copy_split("train", train_files)
    copy_split("val", val_files)

    data_yaml = fold_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {fold_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "nc: 2",
                "names:",
                "  0: head",
                "  1: shemagh",
            ]
        )
        + "\n"
    )
    return data_yaml


def get_iou_xywh(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax1, ay1, ax2, ay2 = ax - aw / 2.0, ay - ah / 2.0, ax + aw / 2.0, ay + ah / 2.0
    bx1, by1, bx2, by2 = bx - bw / 2.0, by - bh / 2.0, bx + bw / 2.0, by + bh / 2.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(aw, 0.0) * max(ah, 0.0)
    area_b = max(bw, 0.0) * max(bh, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions_to_gt(
    preds: list[tuple[int, float, float, float, float, float]],
    gts: list[tuple[int, float, float, float, float]],
    iou_threshold: float,
) -> tuple[list[float], list[int]]:
    max_ious = [0.0] * len(preds)
    tp_flags = [0] * len(preds)

    gt_by_class: dict[int, list[int]] = {}
    for gt_index, (class_id, *_rest) in enumerate(gts):
        gt_by_class.setdefault(class_id, []).append(gt_index)

    pred_by_class: dict[int, list[int]] = {}
    for pred_index, (class_id, *_rest) in enumerate(preds):
        pred_by_class.setdefault(class_id, []).append(pred_index)

    for class_id, pred_indices in pred_by_class.items():
        gt_indices = gt_by_class.get(class_id, [])
        if not gt_indices:
            continue

        pred_indices.sort(key=lambda index: preds[index][1], reverse=True)
        used_gt: set[int] = set()

        for pred_index in pred_indices:
            _, _, px, py, pw, ph = preds[pred_index]
            pred_box = (px, py, pw, ph)

            best_iou_any = 0.0
            best_iou_free = 0.0
            best_gt_free = -1

            for gt_index in gt_indices:
                _, gx, gy, gw, gh = gts[gt_index]
                iou = get_iou_xywh(pred_box, (gx, gy, gw, gh))
                if iou > best_iou_any:
                    best_iou_any = iou
                if gt_index not in used_gt and iou > best_iou_free:
                    best_iou_free = iou
                    best_gt_free = gt_index

            max_ious[pred_index] = best_iou_any
            if best_gt_free >= 0 and best_iou_free >= iou_threshold:
                tp_flags[pred_index] = 1
                used_gt.add(best_gt_free)

    return max_ious, tp_flags


def confidence_rank_per_class(preds: list[tuple[int, float, float, float, float, float]]) -> list[int]:
    ranks = [0] * len(preds)
    by_class: dict[int, list[int]] = {}
    for index, (class_id, *_rest) in enumerate(preds):
        by_class.setdefault(class_id, []).append(index)
    for _, indices in by_class.items():
        indices.sort(key=lambda idx: preds[idx][1], reverse=True)
        for rank, pred_index in enumerate(indices, start=1):
            ranks[pred_index] = rank
    return ranks


def get_detector_class(spec: ModelSpec, yolo_class: type, rtdetr_class: type) -> type:
    return rtdetr_class if spec.is_rtdetr else yolo_class


def main() -> None:
    args = parse_args()
    try:
        model_specs = parse_model_specs(args.model)
        root_dir = find_dataset_root(args.root_dir)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Dataset root: {root_dir}")
    print("Models:")
    for spec in model_specs:
        print(f"  - {spec.tag}: {spec.weights} (imgsz={spec.imgsz}, batch={spec.batch})")

    from ultralytics import RTDETR, YOLO

    train_images = list_train_images(root_dir)
    label_dir = root_dir / "labels" / "train"
    image_dir = root_dir / "images" / "train"

    labels_by_file = {
        filename: parse_label_boxes(label_dir / f"{Path(filename).stem}.txt") for filename in train_images
    }
    folds = build_stratified_folds(train_images, labels_by_file, args.folds, args.seed)

    out_dir = Path(args.out_dir)
    project_dir = Path(args.project)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    folds_manifest_path = out_dir / "folds_manifest.csv"
    image_rows_path = out_dir / "oof_images.csv"
    gt_rows_path = out_dir / "oof_ground_truth.csv"
    pred_rows_path = out_dir / "oof_predictions.csv"
    run_rows_path = out_dir / "oof_runs.csv"

    with (
        open(folds_manifest_path, "w", newline="") as fold_handle,
        open(image_rows_path, "w", newline="") as image_handle,
        open(gt_rows_path, "w", newline="") as gt_handle,
        open(pred_rows_path, "w", newline="") as pred_handle,
        open(run_rows_path, "w", newline="") as run_handle,
    ):
        fold_writer = csv.DictWriter(fold_handle, fieldnames=["filename", "fold", "is_val"])
        image_writer = csv.DictWriter(
            image_handle,
            fieldnames=["filename", "fold", "has_head", "has_shemagh", "head_count", "shemagh_count", "total_gt_boxes"],
        )
        gt_writer = csv.DictWriter(
            gt_handle,
            fieldnames=["filename", "fold", "class_id", "x", "y", "w", "h"],
        )
        pred_writer = csv.DictWriter(
            pred_handle,
            fieldnames=[
                "filename",
                "fold",
                "model_tag",
                "model_weights",
                "class_id",
                "confidence",
                "x",
                "y",
                "w",
                "h",
                "area",
                "aspect",
                "rank_in_class",
                "pred_count_image",
                "pred_count_class_image",
                "gt_count_class_image",
                "max_iou",
                "is_tp_iou50",
            ],
        )
        run_writer = csv.DictWriter(
            run_handle,
            fieldnames=[
                "model_tag",
                "fold",
                "weights_in",
                "weights_best",
                "val_images",
                "pred_boxes",
                "tp_iou50",
                "precision_iou50",
            ],
        )
        fold_writer.writeheader()
        image_writer.writeheader()
        gt_writer.writeheader()
        pred_writer.writeheader()
        run_writer.writeheader()

        all_files_set = set(train_images)
        for fold_index, val_files in enumerate(folds):
            val_set = set(val_files)
            train_files = sorted(all_files_set - val_set, key=lambda name: (not name.split(".")[0].isdigit(), name))

            for filename in train_images:
                fold_writer.writerow({"filename": filename, "fold": fold_index, "is_val": int(filename in val_set)})

            fold_data_dir = out_dir / "datasets" / f"fold_{fold_index}"
            data_yaml = prepare_fold_dataset(
                root_dir=root_dir,
                fold_dir=fold_data_dir,
                train_files=train_files,
                val_files=val_files,
                use_symlinks=args.use_symlinks,
                rebuild=args.rebuild_fold_data,
            )

            gt_by_file = {filename: labels_by_file[filename] for filename in val_files}
            for filename in val_files:
                boxes = gt_by_file[filename]
                head_count = sum(1 for class_id, *_ in boxes if class_id == 0)
                shemagh_count = sum(1 for class_id, *_ in boxes if class_id == 1)
                image_writer.writerow(
                    {
                        "filename": filename,
                        "fold": fold_index,
                        "has_head": int(head_count > 0),
                        "has_shemagh": int(shemagh_count > 0),
                        "head_count": head_count,
                        "shemagh_count": shemagh_count,
                        "total_gt_boxes": len(boxes),
                    }
                )
                for class_id, x, y, w, h in boxes:
                    gt_writer.writerow(
                        {
                            "filename": filename,
                            "fold": fold_index,
                            "class_id": class_id,
                            "x": f"{x:.6f}",
                            "y": f"{y:.6f}",
                            "w": f"{w:.6f}",
                            "h": f"{h:.6f}",
                        }
                    )

            for model_spec in model_specs:
                run_name = f"{model_spec.tag}_fold{fold_index}"
                run_dir = project_dir / run_name
                best_weights = run_dir / "weights" / "best.pt"

                if not (args.reuse_runs and best_weights.exists()):
                    print(f"\nTraining {run_name}...")
                    detector_class = get_detector_class(model_spec, YOLO, RTDETR)
                    detector = detector_class(model_spec.weights)

                    train_kwargs = {
                        "data": str(data_yaml),
                        "epochs": args.epochs,
                        "patience": args.patience,
                        "imgsz": model_spec.imgsz,
                        "batch": model_spec.batch,
                        "project": str(project_dir),
                        "name": run_name,
                        "exist_ok": True,
                        "workers": args.workers,
                        "seed": args.seed + fold_index,
                        "deterministic": args.deterministic,
                    }
                    if args.device is not None:
                        train_kwargs["device"] = args.device

                    detector.train(**train_kwargs)

                    if hasattr(detector, "trainer") and getattr(detector.trainer, "best", None):
                        candidate = Path(str(detector.trainer.best))
                        if candidate.exists():
                            best_weights = candidate
                else:
                    print(f"\nReusing {best_weights}")

                if not best_weights.exists():
                    raise FileNotFoundError(f"best.pt not found for {run_name}: {best_weights}")

                print(f"Predicting OOF boxes for {run_name}...")
                detector_class = get_detector_class(model_spec, YOLO, RTDETR)
                predictor = detector_class(str(best_weights))

                fold_pred_count = 0
                fold_tp_count = 0

                for image_index, filename in enumerate(val_files):
                    image_path = image_dir / filename
                    result = predictor.predict(
                        str(image_path),
                        conf=args.pred_conf,
                        imgsz=model_spec.imgsz,
                        max_det=args.max_det,
                        verbose=False,
                        augment=False,
                    )[0]

                    preds: list[tuple[int, float, float, float, float, float]] = []
                    if hasattr(result, "boxes") and result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(float(box.cls[0]))
                            if class_id not in (0, 1):
                                continue
                            confidence = float(box.conf[0])
                            x, y, w, h = map(float, box.xywhn[0].tolist())
                            preds.append((class_id, confidence, x, y, w, h))

                    max_ious, tp_flags = match_predictions_to_gt(preds, gt_by_file[filename], args.tp_iou)
                    ranks = confidence_rank_per_class(preds)

                    class_pred_count: dict[int, int] = {0: 0, 1: 0}
                    class_gt_count: dict[int, int] = {0: 0, 1: 0}
                    for class_id, *_ in preds:
                        class_pred_count[class_id] = class_pred_count.get(class_id, 0) + 1
                    for class_id, *_ in gt_by_file[filename]:
                        class_gt_count[class_id] = class_gt_count.get(class_id, 0) + 1

                    for pred_index, (class_id, conf, x, y, w, h) in enumerate(preds):
                        area = w * h
                        aspect = w / h if h > 0 else 0.0
                        tp_flag = tp_flags[pred_index]
                        fold_pred_count += 1
                        fold_tp_count += tp_flag

                        pred_writer.writerow(
                            {
                                "filename": filename,
                                "fold": fold_index,
                                "model_tag": model_spec.tag,
                                "model_weights": str(best_weights),
                                "class_id": class_id,
                                "confidence": f"{conf:.8f}",
                                "x": f"{x:.8f}",
                                "y": f"{y:.8f}",
                                "w": f"{w:.8f}",
                                "h": f"{h:.8f}",
                                "area": f"{area:.8f}",
                                "aspect": f"{aspect:.8f}",
                                "rank_in_class": ranks[pred_index],
                                "pred_count_image": len(preds),
                                "pred_count_class_image": class_pred_count.get(class_id, 0),
                                "gt_count_class_image": class_gt_count.get(class_id, 0),
                                "max_iou": f"{max_ious[pred_index]:.8f}",
                                "is_tp_iou50": tp_flag,
                            }
                        )

                    if image_index % 50 == 0:
                        print(f"  Fold {fold_index} {model_spec.tag}: {image_index}/{len(val_files)}")

                precision = (fold_tp_count / fold_pred_count) if fold_pred_count else 0.0
                run_writer.writerow(
                    {
                        "model_tag": model_spec.tag,
                        "fold": fold_index,
                        "weights_in": model_spec.weights,
                        "weights_best": str(best_weights),
                        "val_images": len(val_files),
                        "pred_boxes": fold_pred_count,
                        "tp_iou50": fold_tp_count,
                        "precision_iou50": f"{precision:.6f}",
                    }
                )

    print("\nDone.")
    print(f"Saved: {folds_manifest_path}")
    print(f"Saved: {image_rows_path}")
    print(f"Saved: {gt_rows_path}")
    print(f"Saved: {pred_rows_path}")
    print(f"Saved: {run_rows_path}")


if __name__ == "__main__":
    main()
