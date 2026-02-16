"""
Local scoring and diagnostics for DAL-Shemagh submissions.

What this script can score exactly:
- right_place F1 against labels.txt
- head/shemagh presence metrics against labels.txt

What this script provides as mAP-oriented proxies (not official mAP):
- per-class precision/recall/F1 trends over confidence thresholds
- box count and confidence distributions
- overlap-based right_place consistency from predicted boxes
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass


@dataclass
class GroundTruthRow:
    filename: str
    head: int
    shemagh: int
    right_place: int


@dataclass
class ParsedPrediction:
    right_place: int
    boxes_head: list[tuple[float, list[float]]]
    boxes_shemagh: list[tuple[float, list[float]]]
    invalid_chunks: int


def normalize_field(name: str) -> str:
    return name.strip().lower().lstrip("#").strip()


def parse_int01(value: str) -> int:
    try:
        number = int(float(value))
    except Exception:
        return 0
    return 1 if number > 0 else 0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * q
    left = int(idx)
    right = min(left + 1, len(sorted_values) - 1)
    frac = idx - left
    return float(sorted_values[left] * (1.0 - frac) + sorted_values[right] * frac)


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, len(y_true))
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def get_overlap_xywh(box_a: list[float], box_b: list[float]) -> float:
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
    min_area = min(max(aw * ah, 0.0), max(bw * bh, 0.0))
    return inter / min_area if min_area > 0 else 0.0


def parse_prediction_string(prediction_string: str) -> tuple[list[tuple[float, list[float]]], list[tuple[float, list[float]]], int]:
    if not prediction_string or prediction_string.strip() == "-":
        return [], [], 0

    tokens = prediction_string.split()
    boxes_head: list[tuple[float, list[float]]] = []
    boxes_shemagh: list[tuple[float, list[float]]] = []
    invalid_chunks = 0

    for start in range(0, len(tokens), 6):
        chunk = tokens[start : start + 6]
        if len(chunk) < 6:
            invalid_chunks += 1
            continue
        try:
            class_id = int(float(chunk[0]))
            confidence = float(chunk[1])
            x = float(chunk[2])
            y = float(chunk[3])
            w = float(chunk[4])
            h = float(chunk[5])
        except Exception:
            invalid_chunks += 1
            continue

        box = [x, y, w, h]
        if class_id == 0:
            boxes_head.append((confidence, box))
        elif class_id == 1:
            boxes_shemagh.append((confidence, box))

    return boxes_head, boxes_shemagh, invalid_chunks


def load_labels(path: str) -> dict[str, GroundTruthRow]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No headers found in labels file: {path}")
        header_map = {normalize_field(field): field for field in reader.fieldnames}

        required = ["filename", "head", "shemagh", "right_place"]
        missing = [field for field in required if field not in header_map]
        if missing:
            raise ValueError(f"Missing columns in labels file {path}: {missing}")

        labels: dict[str, GroundTruthRow] = {}
        for row in reader:
            filename = row[header_map["filename"]].strip()
            if not filename:
                continue
            labels[filename] = GroundTruthRow(
                filename=filename,
                head=parse_int01(row[header_map["head"]]),
                shemagh=parse_int01(row[header_map["shemagh"]]),
                right_place=parse_int01(row[header_map["right_place"]]),
            )
    return labels


def load_submission(path: str) -> dict[str, ParsedPrediction]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No headers found in submission file: {path}")
        header_map = {normalize_field(field): field for field in reader.fieldnames}

        required = ["filename", "right_place", "prediction_string"]
        missing = [field for field in required if field not in header_map]
        if missing:
            raise ValueError(f"Missing columns in submission file {path}: {missing}")

        rows: dict[str, ParsedPrediction] = {}
        for row in reader:
            filename = row[header_map["filename"]].strip()
            if not filename:
                continue
            right_place = parse_int01(row[header_map["right_place"]])
            boxes_head, boxes_shemagh, invalid_chunks = parse_prediction_string(row[header_map["prediction_string"]].strip())
            rows[filename] = ParsedPrediction(
                right_place=right_place,
                boxes_head=boxes_head,
                boxes_shemagh=boxes_shemagh,
                invalid_chunks=invalid_chunks,
            )
    return rows


def threshold_metrics(y_true: list[int], max_conf: list[float], thresholds: list[float]) -> dict[str, object]:
    sweep: list[dict[str, float]] = []
    best: dict[str, float] | None = None

    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in max_conf]
        metrics = binary_metrics(y_true, y_pred)
        item = {
            "threshold": threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        }
        sweep.append(item)
        if best is None or item["f1"] > best["f1"]:
            best = item

    return {"best": best or {}, "sweep": sweep}


def evaluate_submission(
    labels: dict[str, GroundTruthRow],
    submission: dict[str, ParsedPrediction],
    overlap_threshold: float,
) -> dict[str, object]:
    filenames = sorted(labels.keys(), key=lambda name: (not name.split(".")[0].isdigit(), name))

    y_rp_true: list[int] = []
    y_rp_pred: list[int] = []
    y_rp_from_boxes: list[int] = []

    y_head_true: list[int] = []
    y_head_pred: list[int] = []
    y_shem_true: list[int] = []
    y_shem_pred: list[int] = []

    head_max_conf: list[float] = []
    shem_max_conf: list[float] = []

    head_box_counts: list[float] = []
    shem_box_counts: list[float] = []
    head_conf_all: list[float] = []
    shem_conf_all: list[float] = []

    missing_rows = 0
    invalid_chunks_total = 0

    for filename in filenames:
        gt = labels[filename]
        pred = submission.get(filename)
        if pred is None:
            missing_rows += 1
            pred = ParsedPrediction(right_place=0, boxes_head=[], boxes_shemagh=[], invalid_chunks=0)

        invalid_chunks_total += pred.invalid_chunks

        y_rp_true.append(gt.right_place)
        y_rp_pred.append(pred.right_place)

        has_head = 1 if pred.boxes_head else 0
        has_shemagh = 1 if pred.boxes_shemagh else 0

        y_head_true.append(gt.head)
        y_head_pred.append(has_head)
        y_shem_true.append(gt.shemagh)
        y_shem_pred.append(has_shemagh)

        head_box_counts.append(float(len(pred.boxes_head)))
        shem_box_counts.append(float(len(pred.boxes_shemagh)))

        if pred.boxes_head:
            class_head_conf = [score for score, _ in pred.boxes_head]
            head_max_conf.append(max(class_head_conf))
            head_conf_all.extend(class_head_conf)
        else:
            head_max_conf.append(0.0)

        if pred.boxes_shemagh:
            class_shem_conf = [score for score, _ in pred.boxes_shemagh]
            shem_max_conf.append(max(class_shem_conf))
            shem_conf_all.extend(class_shem_conf)
        else:
            shem_max_conf.append(0.0)

        rp_overlap = 0
        if pred.boxes_head and pred.boxes_shemagh:
            for _, box_head in pred.boxes_head:
                for _, box_shem in pred.boxes_shemagh:
                    if get_overlap_xywh(box_head, box_shem) > overlap_threshold:
                        rp_overlap = 1
                        break
                if rp_overlap:
                    break
        y_rp_from_boxes.append(rp_overlap)

    rp_metrics = binary_metrics(y_rp_true, y_rp_pred)
    rp_overlap_metrics = binary_metrics(y_rp_true, y_rp_from_boxes)
    head_presence_metrics = binary_metrics(y_head_true, y_head_pred)
    shem_presence_metrics = binary_metrics(y_shem_true, y_shem_pred)

    thresholds = [i / 100.0 for i in range(0, 101, 5)]
    head_threshold = threshold_metrics(y_head_true, head_max_conf, thresholds)
    shem_threshold = threshold_metrics(y_shem_true, shem_max_conf, thresholds)

    proxy_map = 0.5 * (head_presence_metrics["f1"] + shem_presence_metrics["f1"])
    proxy_final = 0.5 * rp_metrics["f1"] + 0.5 * proxy_map

    diagnostics = {
        "files_scored": len(filenames),
        "missing_submission_rows": missing_rows,
        "invalid_prediction_chunks": invalid_chunks_total,
        "box_stats": {
            "head_total": int(sum(head_box_counts)),
            "shemagh_total": int(sum(shem_box_counts)),
            "head_avg_per_image": sum(head_box_counts) / max(1, len(head_box_counts)),
            "shemagh_avg_per_image": sum(shem_box_counts) / max(1, len(shem_box_counts)),
            "images_with_no_boxes": int(sum(1 for h, s in zip(head_box_counts, shem_box_counts, strict=True) if h == 0 and s == 0)),
            "head_p90_per_image": percentile(head_box_counts, 0.90),
            "shemagh_p90_per_image": percentile(shem_box_counts, 0.90),
        },
        "conf_stats": {
            "head_conf_p10": percentile(head_conf_all, 0.10),
            "head_conf_p50": percentile(head_conf_all, 0.50),
            "head_conf_p90": percentile(head_conf_all, 0.90),
            "shemagh_conf_p10": percentile(shem_conf_all, 0.10),
            "shemagh_conf_p50": percentile(shem_conf_all, 0.50),
            "shemagh_conf_p90": percentile(shem_conf_all, 0.90),
        },
        "rp_submission_vs_overlap_disagreement": int(
            sum(1 for a, b in zip(y_rp_pred, y_rp_from_boxes, strict=True) if a != b)
        ),
    }

    return {
        "right_place": rp_metrics,
        "right_place_from_boxes": rp_overlap_metrics,
        "head_presence": head_presence_metrics,
        "shemagh_presence": shem_presence_metrics,
        "head_threshold_sweep": head_threshold,
        "shemagh_threshold_sweep": shem_threshold,
        "proxy_map_from_presence_f1": proxy_map,
        "proxy_final_score": proxy_final,
        "diagnostics": diagnostics,
    }


def format_float(value: float) -> str:
    return f"{value:.4f}"


def print_summary(path: str, result: dict[str, object], detailed: bool) -> None:
    rp = result["right_place"]
    head = result["head_presence"]
    shem = result["shemagh_presence"]
    rp_boxes = result["right_place_from_boxes"]
    diag = result["diagnostics"]

    print(f"\n=== {path} ===")
    print(
        "RightPlace  "
        f"F1={format_float(rp['f1'])} P={format_float(rp['precision'])} "
        f"R={format_float(rp['recall'])} TP={int(rp['tp'])} FP={int(rp['fp'])} FN={int(rp['fn'])}"
    )
    print(
        "HeadPresence "
        f"F1={format_float(head['f1'])} P={format_float(head['precision'])} "
        f"R={format_float(head['recall'])}"
    )
    print(
        "ShemPresence "
        f"F1={format_float(shem['f1'])} P={format_float(shem['precision'])} "
        f"R={format_float(shem['recall'])}"
    )
    print(
        "ProxyScores "
        f"proxy_map={format_float(result['proxy_map_from_presence_f1'])} "
        f"proxy_final={format_float(result['proxy_final_score'])}"
    )
    print(
        "BoxStats    "
        f"head_total={diag['box_stats']['head_total']} "
        f"shem_total={diag['box_stats']['shemagh_total']} "
        f"head_avg={format_float(diag['box_stats']['head_avg_per_image'])} "
        f"shem_avg={format_float(diag['box_stats']['shemagh_avg_per_image'])} "
        f"empty_images={diag['box_stats']['images_with_no_boxes']}"
    )
    print(
        "Diagnostics "
        f"missing_rows={diag['missing_submission_rows']} "
        f"invalid_chunks={diag['invalid_prediction_chunks']} "
        f"rp_overlap_f1={format_float(rp_boxes['f1'])} "
        f"rp_vs_overlap_disagree={diag['rp_submission_vs_overlap_disagreement']}"
    )

    if not detailed:
        return

    head_best = result["head_threshold_sweep"]["best"]
    shem_best = result["shemagh_threshold_sweep"]["best"]
    print(
        "BestHeadThr "
        f"t={format_float(head_best['threshold'])} f1={format_float(head_best['f1'])} "
        f"p={format_float(head_best['precision'])} r={format_float(head_best['recall'])}"
    )
    print(
        "BestShemThr "
        f"t={format_float(shem_best['threshold'])} f1={format_float(shem_best['f1'])} "
        f"p={format_float(shem_best['precision'])} r={format_float(shem_best['recall'])}"
    )
    print("ThresholdTrend(class,thr,p,r,f1,fp,fn)")
    for class_name, sweep in [
        ("head", result["head_threshold_sweep"]["sweep"]),
        ("shemagh", result["shemagh_threshold_sweep"]["sweep"]),
    ]:
        for row in sweep:
            threshold = row["threshold"]
            if threshold in (0.10, 0.20, 0.30, 0.40, 0.50):
                print(
                    f"  {class_name},{format_float(threshold)},"
                    f"{format_float(row['precision'])},{format_float(row['recall'])},"
                    f"{format_float(row['f1'])},{int(row['fp'])},{int(row['fn'])}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local scorer + diagnostics for DAL-Shemagh submissions")
    parser.add_argument("--labels", type=str, default="labels.txt", help="Path to labels file")
    parser.add_argument(
        "--submission",
        type=str,
        action="append",
        default=[],
        help="Submission CSV path (can be used multiple times). If omitted, uses --glob",
    )
    parser.add_argument("--glob", type=str, default="submission*.csv", help="Submission glob when --submission is omitted")
    parser.add_argument("--overlap-threshold", type=float, default=0.10, help="Overlap threshold for box-derived right_place")
    parser.add_argument("--detailed", action="store_true", help="Print threshold sweep details")
    parser.add_argument("--json-out", type=str, default="", help="Optional JSON output path")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    if not labels:
        raise ValueError(f"No label rows found in {args.labels}")

    submissions = args.submission or sorted(glob.glob(args.glob))
    if not submissions:
        raise ValueError(f"No submissions found. Use --submission or adjust --glob ({args.glob})")

    results_by_file: dict[str, dict[str, object]] = {}
    auto_detailed = args.detailed or len(submissions) == 1

    for sub_path in submissions:
        if not os.path.exists(sub_path):
            print(f"Skipping missing submission: {sub_path}")
            continue
        sub = load_submission(sub_path)
        result = evaluate_submission(labels, sub, args.overlap_threshold)
        results_by_file[sub_path] = result
        print_summary(sub_path, result, detailed=auto_detailed)

    if not results_by_file:
        raise ValueError("No valid submission files evaluated")

    ranked = sorted(
        ((path, result["proxy_final_score"]) for path, result in results_by_file.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    print("\n=== Proxy Ranking (higher is better) ===")
    for path, score in ranked:
        print(f"{path}: {format_float(score)}")

    if args.json_out:
        with open(args.json_out, "w", newline="") as handle:
            json.dump(results_by_file, handle, indent=2)
        print(f"\nSaved JSON results to {args.json_out}")


if __name__ == "__main__":
    main()

