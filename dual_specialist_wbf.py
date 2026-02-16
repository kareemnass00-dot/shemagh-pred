"""
DAL-Shemagh — Dual Specialist (F1) + mAP Ensemble (WBF + head-conditioned rescoring)
# =================================================================================
# Keeps the "F1 specialist" strategy for `right_place`, and upgrades the mAP pipeline
# for `prediction_string`:
#   - Supports ensembling multiple mAP models via Weighted Box Fusion (WBF)
#   - Optional head-conditioned *soft* rescoring for shemagh boxes (ranking boost)
#
# This script is NEW and does not modify your existing scripts.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable


def find_dataset_root() -> str:
    possible_paths = [
        "/kaggle/input/dal-shemagh-identification",
        "/kaggle/input/dal-shemagh-detection-challenge",
        "./data/dal-shemagh-detection-challenge",
        "./data",
        "../input/dal-shemagh-identification",
        "../input/dal-shemagh-detection-challenge",
    ]
    for p in possible_paths:
        if os.path.exists(p) and os.path.exists(f"{p}/images/test"):
            return p

    print(f"ERROR: Could not find dataset in {possible_paths}")
    sys.exit(1)


def area_xyxy(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    union = area_xyxy(a) + area_xyxy(b) - inter
    return inter / union if union > 0 else 0.0


def containment_ratio(head_xyxy: list[float], shemagh_xyxy: list[float]) -> float:
    """How much of the head area is covered by the shemagh box."""
    hx1, hy1, hx2, hy2 = head_xyxy
    sx1, sy1, sx2, sy2 = shemagh_xyxy

    ix1 = max(hx1, sx1)
    iy1 = max(hy1, sy1)
    ix2 = min(hx2, sx2)
    iy2 = min(hy2, sy2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    head_area = area_xyxy(head_xyxy)
    return inter / head_area if head_area > 0 else 0.0


def xyxy_to_xywh(box_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = box_xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    return [x, y, w, h]


def clip01(box_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = box_xyxy
    x1 = min(1.0, max(0.0, x1))
    y1 = min(1.0, max(0.0, y1))
    x2 = min(1.0, max(0.0, x2))
    y2 = min(1.0, max(0.0, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def weighted_box_fusion(
    *,
    boxes: list[list[float]],
    scores: list[float],
    model_ids: list[int],
    model_weights: list[float],
    iou_thr: float,
    skip_box_thr: float,
) -> tuple[list[list[float]], list[float]]:
    """Simple WBF with at most one box per model per cluster."""
    assert len(boxes) == len(scores) == len(model_ids)
    if not boxes:
        return [], []

    kept = [(i, scores[i]) for i in range(len(scores)) if scores[i] >= skip_box_thr]
    if not kept:
        return [], []
    kept.sort(key=lambda t: t[1], reverse=True)

    clusters: list[dict[str, object]] = []

    def recompute(cluster: dict[str, object]) -> None:
        by_model: dict[int, tuple[list[float], float]] = cluster["by_model"]  # type: ignore[assignment]
        weighted_sum = [0.0, 0.0, 0.0, 0.0]
        weighted_total = 0.0
        score_sum = 0.0
        weight_sum = 0.0

        for mid, (b, s) in by_model.items():
            w = float(model_weights[mid]) if mid < len(model_weights) else 1.0
            coeff = s * w
            weighted_total += coeff
            weighted_sum[0] += b[0] * coeff
            weighted_sum[1] += b[1] * coeff
            weighted_sum[2] += b[2] * coeff
            weighted_sum[3] += b[3] * coeff
            score_sum += s * w
            weight_sum += w

        fused = [v / weighted_total for v in weighted_sum] if weighted_total > 0 else list(by_model.values())[0][0]
        cluster["box"] = clip01(fused)
        cluster["score"] = (score_sum / weight_sum) if weight_sum > 0 else float(list(by_model.values())[0][1])

    for idx, _ in kept:
        box = boxes[idx]
        score = float(scores[idx])
        mid = int(model_ids[idx])

        best_j = -1
        best_iou = 0.0
        for j, cl in enumerate(clusters):
            cl_box = cl["box"]  # type: ignore[assignment]
            iou = iou_xyxy(box, cl_box)  # type: ignore[arg-type]
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j == -1:
            cluster: dict[str, object] = {
                "by_model": {mid: (box, score)},
                "box": box,
                "score": score,
            }
            recompute(cluster)
            clusters.append(cluster)
            continue

        cl = clusters[best_j]
        by_model = cl["by_model"]  # type: ignore[assignment]
        existing = by_model.get(mid)
        if existing is None or score > float(existing[1]):
            by_model[mid] = (box, score)
            recompute(cl)

    out_boxes = [cl["box"] for cl in clusters]  # type: ignore[list-item]
    out_scores = [float(cl["score"]) for cl in clusters]
    order = sorted(range(len(out_scores)), key=lambda i: out_scores[i], reverse=True)
    out_boxes = [out_boxes[i] for i in order]
    out_scores = [out_scores[i] for i in order]
    return out_boxes, out_scores


def numeric_sort_key(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    if stem.isdigit():
        return (0, f"{int(stem):010d}")
    return (1, name)


def get_overlap_xywh(box_a: list[float], box_b: list[float]) -> float:
    """Overlap ratio used by your F1 logic (intersection / min(area))."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    min_area = min(aw * ah, bw * bh)
    return (inter / min_area) if min_area > 0 else 0.0


def parse_weights_list(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    for it in items:
        if not it:
            continue
        out.append(it)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual specialist + mAP ensemble (WBF)")
    parser.add_argument("--f1-head", type=str, default="models/head_m640_best.pt", help="F1 head weights path")
    parser.add_argument("--f1-shemagh", type=str, default="models/shemagh_m640_best.pt", help="F1 shemagh weights path")
    parser.add_argument(
        "--map-weights",
        type=str,
        nargs="+",
        default=["models/map_y11n_best.pt"],
        help="One or more mAP model weights (ensemble).",
    )
    parser.add_argument("--map-model-weights", type=float, nargs="*", default=[], help="Optional per-model weights.")
    parser.add_argument("--f1-imgsz", type=int, default=640)
    parser.add_argument("--f1-conf", type=float, default=0.15)
    parser.add_argument("--f1-overlap", type=float, default=0.10)

    parser.add_argument("--map-imgsz", type=int, default=640)
    parser.add_argument("--map-conf", type=float, default=0.25)
    parser.add_argument("--map-iou", type=float, default=0.70, help="NMS IoU for Ultralytics predict()")
    parser.add_argument("--map-max-det", type=int, default=300)
    parser.add_argument("--map-augment", action="store_true", help="Enable TTA augment=True for mAP models")

    parser.add_argument("--wbf-iou-head", type=float, default=0.55)
    parser.add_argument("--wbf-iou-shemagh", type=float, default=0.55)
    parser.add_argument("--wbf-skip", type=float, default=0.0)

    parser.add_argument("--rescore-shemagh", action="store_true", help="Enable head-conditioned soft rescoring")
    parser.add_argument("--containment-thresh", type=float, default=0.20)
    parser.add_argument("--rescore-min", type=float, default=0.80)
    parser.add_argument("--rescore-max", type=float, default=1.15)
    parser.add_argument("--rescore-power", type=float, default=1.0)

    parser.add_argument("--limit", type=int, default=0, help="Process only first N test images (0=all)")
    parser.add_argument("--out", type=str, default="submission_dual_wbf.csv")
    args = parser.parse_args()

    root_dir = find_dataset_root()
    test_dir = Path(root_dir) / "images" / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

    script_dir = Path(__file__).resolve().parent

    f1_head_path = Path(args.f1_head)
    f1_shemagh_path = Path(args.f1_shemagh)
    if not f1_head_path.is_absolute():
        f1_head_path = script_dir / f1_head_path
    if not f1_shemagh_path.is_absolute():
        f1_shemagh_path = script_dir / f1_shemagh_path

    map_weight_paths = [Path(p) for p in parse_weights_list(args.map_weights)]
    resolved_map_paths: list[Path] = []
    for p in map_weight_paths:
        resolved_map_paths.append(p if p.is_absolute() else (script_dir / p))

    for label, p in [
        ("F1 head", f1_head_path),
        ("F1 shemagh", f1_shemagh_path),
        ("mAP weights", resolved_map_paths[0] if resolved_map_paths else Path("<none>")),
    ]:
        if label != "mAP weights":
            if not p.exists():
                raise FileNotFoundError(f"{label} weights not found: {p}")

    for p in resolved_map_paths:
        if not p.exists():
            raise FileNotFoundError(f"mAP weights not found: {p}")

    # Per-model weights default to 1.0
    model_weights = list(args.map_model_weights)
    if len(model_weights) < len(resolved_map_paths):
        model_weights.extend([1.0] * (len(resolved_map_paths) - len(model_weights)))
    if len(model_weights) > len(resolved_map_paths):
        model_weights = model_weights[: len(resolved_map_paths)]

    # Upgrade ultralytics (kept consistent with your other scripts)
    os.system("pip install -U ultralytics")
    from ultralytics import YOLO

    print(f"Using data at: {root_dir}")
    print("Loading models:")
    print(f"  F1 head:    {f1_head_path.name}")
    print(f"  F1 shemagh: {f1_shemagh_path.name}")
    for i, p in enumerate(resolved_map_paths):
        print(f"  mAP[{i}]:     {p.name} (w={model_weights[i]:g})")

    f1_head = YOLO(str(f1_head_path))
    f1_shemagh = YOLO(str(f1_shemagh_path))
    map_models = [YOLO(str(p)) for p in resolved_map_paths]

    test_files = sorted([p.name for p in test_dir.iterdir() if p.suffix.lower() == ".jpg"], key=numeric_sort_key)
    if args.limit and args.limit > 0:
        test_files = test_files[: args.limit]

    submission_rows: list[list[str | int]] = []
    total_head_boxes = 0
    total_shem_boxes = 0
    right_place_count = 0

    for idx, fname in enumerate(test_files):
        img_path = str(test_dir / fname)
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(test_files)}...")

        # --------------------
        # F1 pipeline (right_place)
        # --------------------
        res_fh = f1_head.predict(img_path, conf=args.f1_conf, imgsz=args.f1_imgsz, augment=True, verbose=False)[0]
        heads_f1 = [box.xywhn[0].tolist() for box in res_fh.boxes]

        res_fs = f1_shemagh.predict(img_path, conf=args.f1_conf, imgsz=args.f1_imgsz, augment=True, verbose=False)[0]
        shemaghs_f1 = [box.xywhn[0].tolist() for box in res_fs.boxes]

        rp = 0
        if heads_f1 and shemaghs_f1:
            for h in heads_f1:
                for s in shemaghs_f1:
                    if get_overlap_xywh(h, s) > args.f1_overlap:
                        rp = 1
                        break
                if rp:
                    break

        right_place_count += rp

        # --------------------
        # mAP pipeline (prediction_string): ensemble + WBF + rescoring
        # --------------------
        head_boxes: list[list[float]] = []
        head_scores: list[float] = []
        head_model_ids: list[int] = []

        shem_boxes: list[list[float]] = []
        shem_scores: list[float] = []
        shem_model_ids: list[int] = []

        for mid, model in enumerate(map_models):
            res = model.predict(
                img_path,
                conf=args.map_conf,
                imgsz=args.map_imgsz,
                iou=args.map_iou,
                max_det=args.map_max_det,
                augment=args.map_augment,
                verbose=False,
            )[0]
            for box in res.boxes:
                cls = int(box.cls[0])
                score = float(box.conf[0])
                xyxy = clip01(box.xyxyn[0].tolist())
                if cls == 0:
                    head_boxes.append(xyxy)
                    head_scores.append(score)
                    head_model_ids.append(mid)
                elif cls == 1:
                    shem_boxes.append(xyxy)
                    shem_scores.append(score)
                    shem_model_ids.append(mid)

        fused_head_boxes, fused_head_scores = weighted_box_fusion(
            boxes=head_boxes,
            scores=head_scores,
            model_ids=head_model_ids,
            model_weights=model_weights,
            iou_thr=args.wbf_iou_head,
            skip_box_thr=args.wbf_skip,
        )

        # Best head for rescoring shemagh
        best_head = fused_head_boxes[0] if fused_head_boxes else None

        if args.rescore_shemagh and best_head is not None and shem_boxes:
            t = max(1e-9, float(args.containment_thresh))
            min_f = float(args.rescore_min)
            max_f = float(args.rescore_max)
            power = max(0.01, float(args.rescore_power))

            rescored: list[float] = []
            for b, s in zip(shem_boxes, shem_scores, strict=True):
                r = containment_ratio(best_head, b)
                # Soft factor in [min_f, max_f], saturating at r>=t.
                x = min(1.0, max(0.0, r / t))
                x = x**power
                factor = min_f + (max_f - min_f) * x
                rescored.append(min(1.0, max(0.0, s * factor)))
            shem_scores = rescored

        fused_shem_boxes, fused_shem_scores = weighted_box_fusion(
            boxes=shem_boxes,
            scores=shem_scores,
            model_ids=shem_model_ids,
            model_weights=model_weights,
            iou_thr=args.wbf_iou_shemagh,
            skip_box_thr=args.wbf_skip,
        )

        # Build prediction_string (keep lots of boxes; WBF reduces duplicates)
        parts: list[str] = []
        for b, s in zip(fused_head_boxes, fused_head_scores, strict=True):
            x, y, w, h = xyxy_to_xywh(b)
            parts.extend(["0", f"{s:.4f}", f"{x:.4f}", f"{y:.4f}", f"{w:.4f}", f"{h:.4f}"])
        for b, s in zip(fused_shem_boxes, fused_shem_scores, strict=True):
            x, y, w, h = xyxy_to_xywh(b)
            parts.extend(["1", f"{s:.4f}", f"{x:.4f}", f"{y:.4f}", f"{w:.4f}", f"{h:.4f}"])

        total_head_boxes += len(fused_head_boxes)
        total_shem_boxes += len(fused_shem_boxes)

        pred_str = " ".join(parts) if parts else "-"
        submission_rows.append([fname, rp, pred_str])

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "right_place", "prediction_string"])
        for row in submission_rows:
            writer.writerow(row)

    total = len(test_files)
    total_boxes = total_head_boxes + total_shem_boxes
    print("\nDone!")
    print(f"  right_place=1: {right_place_count}/{total}")
    print(f"  Head boxes: {total_head_boxes}, Shemagh boxes: {total_shem_boxes}")
    print(f"  Total boxes: {total_boxes}, Avg/img: {total_boxes/total:.1f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

