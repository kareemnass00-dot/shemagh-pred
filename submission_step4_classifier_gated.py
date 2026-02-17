"""
Step 4: Build submission with classifier-driven constraints (no labels.txt leakage).

Pipeline:
1) Run map detectors to get candidate boxes.
2) Re-score candidates with Step-3 reranker.
3) Blend reranker with raw confidence.
4) Apply soft class-presence gating from head/shemagh specialist confidences.
5) Ensemble with class-wise WBF.
6) Predict right_place using specialist overlap logic.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "confidence",
    "conf_logit",
    "x",
    "y",
    "w",
    "h",
    "area",
    "log_area",
    "aspect",
    "log_aspect_abs",
    "center_dist",
    "rank_in_class",
    "pred_count_image",
    "pred_count_class_image",
    "rel_rank_cls",
    "inv_rank_cls",
]

CATEGORICAL_FEATURES = [
    "model_tag",
    "class_id",
]


@dataclass
class Candidate:
    class_id: int
    confidence: float
    x: float
    y: float
    w: float
    h: float
    model_id: int
    model_tag: str
    rerank_score_raw: float = 0.0
    rerank_score: float = 0.0
    score_final: float = 0.0

    def xyxy(self) -> list[float]:
        x1 = self.x - self.w / 2.0
        y1 = self.y - self.h / 2.0
        x2 = self.x + self.w / 2.0
        y2 = self.y + self.h / 2.0
        return clip01([x1, y1, x2, y2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step-4 classifier-gated rerank + WBF submission builder")
    parser.add_argument("--f1-head", type=str, default="models/head_m640_best.pt")
    parser.add_argument("--f1-shemagh", type=str, default="models/shemagh_m640_best.pt")
    parser.add_argument(
        "--map-weights",
        type=str,
        nargs="+",
        default=["models/map_rtdetr_l_best.pt", "models/map_y11s.pt", "models/map_y11n.pt"],
    )
    parser.add_argument("--map-tags", type=str, nargs="*", default=[])
    parser.add_argument("--map-model-weights", type=float, nargs="*", default=[])
    parser.add_argument("--reranker-model", type=str, default="reranker_stage3/reranker_model.joblib")
    parser.add_argument("--reranker-metrics", type=str, default="reranker_stage3/reranker_metrics.json")
    parser.add_argument("--blend-alpha", type=float, default=None, help="If omitted, read from reranker_metrics.json")

    parser.add_argument("--f1-imgsz", type=int, default=640)
    parser.add_argument("--f1-conf", type=float, default=0.15)
    parser.add_argument("--f1-overlap", type=float, default=0.10)

    parser.add_argument("--map-imgsz", type=int, default=640)
    parser.add_argument("--map-conf", type=float, default=0.001)
    parser.add_argument("--map-iou", type=float, default=0.70)
    parser.add_argument("--map-max-det", type=int, default=300)
    parser.add_argument("--map-augment", action="store_true")

    parser.add_argument("--gate-min-factor", type=float, default=0.40)
    parser.add_argument("--gate-max-factor", type=float, default=1.10)
    parser.add_argument("--gate-absent-thr", type=float, default=0.15)
    parser.add_argument("--gate-present-thr", type=float, default=0.60)

    parser.add_argument("--hard-absent-thr-head", type=float, default=-1.0, help="Disable with negative")
    parser.add_argument("--hard-absent-thr-shemagh", type=float, default=-1.0, help="Disable with negative")
    parser.add_argument("--hard-keep-topk-head", type=int, default=1)
    parser.add_argument("--hard-keep-topk-shemagh", type=int, default=1)

    parser.add_argument("--wbf-iou-head", type=float, default=0.55)
    parser.add_argument("--wbf-iou-shemagh", type=float, default=0.55)
    parser.add_argument("--wbf-skip", type=float, default=0.001)
    parser.add_argument("--post-conf-head", type=float, default=0.01)
    parser.add_argument("--post-conf-shemagh", type=float, default=0.01)
    parser.add_argument("--post-topk-head", type=int, default=3)
    parser.add_argument("--post-topk-shemagh", type=int, default=2)

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=str, default="submission_step4.csv")
    return parser.parse_args()


def parse_model_weights(raw_weights: list[float], n_models: int) -> list[float]:
    weights = list(raw_weights)
    if len(weights) < n_models:
        weights.extend([1.0] * (n_models - len(weights)))
    if len(weights) > n_models:
        weights = weights[:n_models]
    return [float(max(0.0, value)) for value in weights]


def parse_model_tags(raw_tags: list[str], model_paths: list[Path]) -> list[str]:
    tags = [tag.strip() for tag in raw_tags if tag.strip()]
    if len(tags) < len(model_paths):
        tags.extend([path.stem for path in model_paths[len(tags) :]])
    if len(tags) > len(model_paths):
        tags = tags[: len(model_paths)]
    return tags


def normalize_score(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def numeric_sort_key(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    if stem.isdigit():
        return (0, f"{int(stem):010d}")
    return (1, name)


def find_dataset_root() -> Path:
    possible_paths = [
        "/kaggle/input/dal-shemagh-identification",
        "/kaggle/input/dal-shemagh-detection-challenge",
        "./data/dal-shemagh-detection-challenge",
        "./data",
        "../input/dal-shemagh-identification",
        "../input/dal-shemagh-detection-challenge",
    ]
    for path in possible_paths:
        root = Path(path)
        if (root / "images" / "test").exists():
            return root
    raise FileNotFoundError(f"Could not find dataset in {possible_paths}")


def resolve_path(script_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (script_dir / path)


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
    min_area = min(aw * ah, bw * bh)
    return (inter / min_area) if min_area > 0 else 0.0


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


def xyxy_to_xywh(box_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = box_xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    return [x, y, w, h]


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
    if inter <= 0:
        return 0.0
    union = area_xyxy(a) + area_xyxy(b) - inter
    return inter / union if union > 0 else 0.0


def weighted_box_fusion(
    boxes: list[list[float]],
    scores: list[float],
    model_ids: list[int],
    model_weights: list[float],
    iou_thr: float,
    skip_box_thr: float,
) -> tuple[list[list[float]], list[float]]:
    if not boxes:
        return [], []

    keep = [(index, scores[index]) for index in range(len(boxes)) if scores[index] >= skip_box_thr]
    if not keep:
        return [], []
    keep.sort(key=lambda item: item[1], reverse=True)

    clusters: list[dict[str, object]] = []

    def recompute_cluster(cluster: dict[str, object]) -> None:
        by_model: dict[int, tuple[list[float], float]] = cluster["by_model"]  # type: ignore[assignment]
        weighted_sum = [0.0, 0.0, 0.0, 0.0]
        weighted_total = 0.0
        score_sum = 0.0
        weight_sum = 0.0
        for model_id, (box, score) in by_model.items():
            w = model_weights[model_id] if model_id < len(model_weights) else 1.0
            coeff = score * w
            weighted_total += coeff
            weighted_sum[0] += box[0] * coeff
            weighted_sum[1] += box[1] * coeff
            weighted_sum[2] += box[2] * coeff
            weighted_sum[3] += box[3] * coeff
            score_sum += score * w
            weight_sum += w
        if weighted_total > 0:
            fused_box = [value / weighted_total for value in weighted_sum]
        else:
            fused_box = list(next(iter(by_model.values()))[0])
        cluster["box"] = clip01(fused_box)
        cluster["score"] = score_sum / weight_sum if weight_sum > 0 else float(next(iter(by_model.values()))[1])

    for index, _ in keep:
        box = boxes[index]
        score = float(scores[index])
        model_id = int(model_ids[index])

        best_cluster = -1
        best_iou = 0.0
        for cluster_index, cluster in enumerate(clusters):
            cluster_box = cluster["box"]  # type: ignore[assignment]
            iou = iou_xyxy(box, cluster_box)  # type: ignore[arg-type]
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_cluster = cluster_index

        if best_cluster < 0:
            new_cluster: dict[str, object] = {
                "by_model": {model_id: (box, score)},
                "box": box,
                "score": score,
            }
            recompute_cluster(new_cluster)
            clusters.append(new_cluster)
            continue

        cluster = clusters[best_cluster]
        by_model = cluster["by_model"]  # type: ignore[assignment]
        existing = by_model.get(model_id)
        if existing is None or score > float(existing[1]):
            by_model[model_id] = (box, score)
            recompute_cluster(cluster)

    out_boxes = [cluster["box"] for cluster in clusters]  # type: ignore[list-item]
    out_scores = [float(cluster["score"]) for cluster in clusters]
    order = sorted(range(len(out_scores)), key=lambda idx: out_scores[idx], reverse=True)
    out_boxes = [out_boxes[idx] for idx in order]
    out_scores = [out_scores[idx] for idx in order]
    return out_boxes, out_scores


def to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for col in ("x", "y", "w", "h"):
        data[col] = to_numeric(data[col], default=0.0)
    data["confidence"] = to_numeric(data["confidence"], default=0.0).clip(0.0, 1.0)
    data["class_id"] = to_numeric(data["class_id"], default=0).astype(int)
    data["area"] = (data["w"] * data["h"]).clip(lower=0.0)
    data["aspect"] = data["w"] / data["h"].clip(lower=1e-6)
    data["pred_count_image"] = to_numeric(data["pred_count_image"], default=1.0).clip(lower=1.0)
    data["pred_count_class_image"] = to_numeric(data["pred_count_class_image"], default=1.0).clip(lower=1.0)
    data["rank_in_class"] = to_numeric(data["rank_in_class"], default=1.0).clip(lower=1.0)

    conf = data["confidence"].clip(1e-6, 1.0 - 1e-6)
    data["conf_logit"] = np.log(conf / (1.0 - conf))
    data["log_area"] = np.log(data["area"].clip(lower=1e-8))
    data["log_aspect_abs"] = np.abs(np.log(data["aspect"].clip(lower=1e-6)))
    data["center_dist"] = np.sqrt((data["x"] - 0.5) ** 2 + (data["y"] - 0.5) ** 2)
    data["rel_rank_cls"] = data["rank_in_class"] / data["pred_count_class_image"].clip(lower=1.0)
    data["inv_rank_cls"] = 1.0 / data["rank_in_class"].clip(lower=1.0)
    return data


def select_blend_alpha(explicit_alpha: float | None, metrics_path: Path) -> float:
    if explicit_alpha is not None:
        return float(np.clip(explicit_alpha, 0.0, 1.0))
    if metrics_path.exists():
        try:
            payload = json_load(metrics_path)
            settings = payload.get("settings", {})
            value = settings.get("blend_alpha_selected")
            if value is not None:
                return float(np.clip(float(value), 0.0, 1.0))
        except Exception:
            pass
    return 1.0


def json_load(path: Path) -> dict[str, object]:
    import json

    return json.loads(path.read_text())


def presence_factor(
    presence_prob: float,
    absent_thr: float,
    present_thr: float,
    min_factor: float,
    max_factor: float,
) -> float:
    p = normalize_score(presence_prob)
    if present_thr <= absent_thr:
        return max_factor if p >= present_thr else min_factor
    if p <= absent_thr:
        return min_factor
    if p >= present_thr:
        return max_factor
    ratio = (p - absent_thr) / (present_thr - absent_thr)
    return min_factor + (max_factor - min_factor) * ratio


def hard_prune(
    items: list[Candidate],
    absent_prob: float,
    absent_threshold: float,
    keep_topk: int,
) -> list[Candidate]:
    if absent_threshold < 0:
        return items
    if absent_prob >= absent_threshold:
        return items
    if keep_topk <= 0:
        return []
    items_sorted = sorted(items, key=lambda item: item.score_final, reverse=True)
    return items_sorted[:keep_topk]


def is_rtdetr_weight(path: Path, tag: str) -> bool:
    token = f"{path.name} {tag}".lower()
    return "rtdetr" in token


def predict_specialist_presence(result) -> tuple[list[list[float]], float]:
    boxes_xywh: list[list[float]] = []
    max_conf = 0.0
    if hasattr(result, "boxes") and result.boxes is not None:
        for box in result.boxes:
            boxes_xywh.append(box.xywhn[0].tolist())
            max_conf = max(max_conf, float(box.conf[0]))
    return boxes_xywh, normalize_score(max_conf)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    dataset_root = find_dataset_root()
    test_dir = dataset_root / "images" / "test"

    f1_head_path = resolve_path(script_dir, args.f1_head)
    f1_shemagh_path = resolve_path(script_dir, args.f1_shemagh)
    reranker_model_path = resolve_path(script_dir, args.reranker_model)
    reranker_metrics_path = resolve_path(script_dir, args.reranker_metrics)
    map_paths = [resolve_path(script_dir, raw) for raw in args.map_weights]

    required_paths = [
        ("f1_head", f1_head_path),
        ("f1_shemagh", f1_shemagh_path),
        ("reranker_model", reranker_model_path),
    ]
    for label, path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    for path in map_paths:
        if not path.exists():
            raise FileNotFoundError(f"map model not found: {path}")

    blend_alpha = select_blend_alpha(args.blend_alpha, reranker_metrics_path)
    model_tags = parse_model_tags(args.map_tags, map_paths)
    model_weights = parse_model_weights(args.map_model_weights, len(map_paths))

    print(f"Using data at: {dataset_root}")
    print(f"Blend alpha: {blend_alpha:.3f}")
    print("Map models:")
    for idx, path in enumerate(map_paths):
        print(f"  [{idx}] {model_tags[idx]} -> {path.name} (weight={model_weights[idx]:.3f})")

    from ultralytics import RTDETR, YOLO

    f1_head = YOLO(str(f1_head_path))
    f1_shemagh = YOLO(str(f1_shemagh_path))

    map_models = []
    for tag, path in zip(model_tags, map_paths, strict=True):
        cls = RTDETR if is_rtdetr_weight(path, tag) else YOLO
        map_models.append((tag, cls(str(path))))

    reranker = joblib.load(reranker_model_path)

    test_files = sorted([path.name for path in test_dir.iterdir() if path.suffix.lower() == ".jpg"], key=numeric_sort_key)
    if args.limit > 0:
        test_files = test_files[: args.limit]

    rows: list[list[str | int]] = []
    total_head = 0
    total_shemagh = 0
    rp_positive = 0

    for index, filename in enumerate(test_files):
        if index % 100 == 0:
            print(f"Processing {index}/{len(test_files)}...")

        image_path = str(test_dir / filename)

        res_head = f1_head.predict(
            image_path,
            conf=args.f1_conf,
            imgsz=args.f1_imgsz,
            augment=True,
            verbose=False,
        )[0]
        res_shemagh = f1_shemagh.predict(
            image_path,
            conf=args.f1_conf,
            imgsz=args.f1_imgsz,
            augment=True,
            verbose=False,
        )[0]

        head_boxes_f1, p_head = predict_specialist_presence(res_head)
        shemagh_boxes_f1, p_shemagh = predict_specialist_presence(res_shemagh)

        rp = 0
        if head_boxes_f1 and shemagh_boxes_f1:
            for head_box in head_boxes_f1:
                for shemagh_box in shemagh_boxes_f1:
                    if get_overlap_xywh(head_box, shemagh_box) > args.f1_overlap:
                        rp = 1
                        break
                if rp:
                    break
        rp_positive += rp

        candidates: list[Candidate] = []
        for model_id, (model_tag, model) in enumerate(map_models):
            result = model.predict(
                image_path,
                conf=args.map_conf,
                imgsz=args.map_imgsz,
                iou=args.map_iou,
                max_det=args.map_max_det,
                augment=args.map_augment,
                verbose=False,
            )[0]

            model_candidates: list[Candidate] = []
            if hasattr(result, "boxes") and result.boxes is not None:
                for box in result.boxes:
                    class_id = int(float(box.cls[0]))
                    if class_id not in (0, 1):
                        continue
                    conf = normalize_score(float(box.conf[0]))
                    x, y, w, h = map(float, box.xywhn[0].tolist())
                    if w <= 0.0 or h <= 0.0:
                        continue
                    model_candidates.append(
                        Candidate(
                            class_id=class_id,
                            confidence=conf,
                            x=x,
                            y=y,
                            w=w,
                            h=h,
                            model_id=model_id,
                            model_tag=model_tag,
                        )
                    )

            if not model_candidates:
                continue

            by_class: dict[int, list[int]] = {0: [], 1: []}
            for local_index, item in enumerate(model_candidates):
                by_class[item.class_id].append(local_index)
            for class_id in (0, 1):
                indices = by_class[class_id]
                indices.sort(key=lambda idx_local: model_candidates[idx_local].confidence, reverse=True)
                class_count = len(indices)
                for rank, idx_local in enumerate(indices, start=1):
                    item = model_candidates[idx_local]
                    rows_map = {
                        "filename": filename,
                        "model_tag": item.model_tag,
                        "class_id": item.class_id,
                        "confidence": item.confidence,
                        "x": item.x,
                        "y": item.y,
                        "w": item.w,
                        "h": item.h,
                        "rank_in_class": rank,
                        "pred_count_image": len(model_candidates),
                        "pred_count_class_image": class_count,
                    }
                    row_df = pd.DataFrame([rows_map])
                    row_df = ensure_features(row_df)
                    score_raw = float(
                        reranker.predict_proba(row_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1][0]
                    )
                    item.rerank_score_raw = normalize_score(score_raw)
                    item.rerank_score = normalize_score(
                        blend_alpha * item.confidence + (1.0 - blend_alpha) * item.rerank_score_raw
                    )

            candidates.extend(model_candidates)

        head_factor = presence_factor(
            p_head,
            absent_thr=args.gate_absent_thr,
            present_thr=args.gate_present_thr,
            min_factor=args.gate_min_factor,
            max_factor=args.gate_max_factor,
        )
        shemagh_factor = presence_factor(
            p_shemagh,
            absent_thr=args.gate_absent_thr,
            present_thr=args.gate_present_thr,
            min_factor=args.gate_min_factor,
            max_factor=args.gate_max_factor,
        )

        for item in candidates:
            factor = head_factor if item.class_id == 0 else shemagh_factor
            item.score_final = normalize_score(item.rerank_score * factor)

        head_candidates = [item for item in candidates if item.class_id == 0]
        shemagh_candidates = [item for item in candidates if item.class_id == 1]

        head_candidates = hard_prune(
            head_candidates,
            absent_prob=p_head,
            absent_threshold=args.hard_absent_thr_head,
            keep_topk=args.hard_keep_topk_head,
        )
        shemagh_candidates = hard_prune(
            shemagh_candidates,
            absent_prob=p_shemagh,
            absent_threshold=args.hard_absent_thr_shemagh,
            keep_topk=args.hard_keep_topk_shemagh,
        )

        def fuse_class(items: list[Candidate], iou_thr: float, post_conf: float, topk: int) -> tuple[list[list[float]], list[float]]:
            if not items:
                return [], []
            boxes = [item.xyxy() for item in items]
            scores = [item.score_final for item in items]
            model_ids = [item.model_id for item in items]
            fused_boxes, fused_scores = weighted_box_fusion(
                boxes=boxes,
                scores=scores,
                model_ids=model_ids,
                model_weights=model_weights,
                iou_thr=iou_thr,
                skip_box_thr=args.wbf_skip,
            )
            filtered = [
                (box, score)
                for box, score in zip(fused_boxes, fused_scores, strict=True)
                if score >= post_conf
            ]
            if topk > 0:
                filtered = filtered[:topk]
            if not filtered:
                return [], []
            return [pair[0] for pair in filtered], [pair[1] for pair in filtered]

        fused_head_boxes, fused_head_scores = fuse_class(
            head_candidates,
            iou_thr=args.wbf_iou_head,
            post_conf=args.post_conf_head,
            topk=args.post_topk_head,
        )
        fused_shemagh_boxes, fused_shemagh_scores = fuse_class(
            shemagh_candidates,
            iou_thr=args.wbf_iou_shemagh,
            post_conf=args.post_conf_shemagh,
            topk=args.post_topk_shemagh,
        )

        total_head += len(fused_head_boxes)
        total_shemagh += len(fused_shemagh_boxes)

        parts: list[str] = []
        for box, score in zip(fused_head_boxes, fused_head_scores, strict=True):
            x, y, w, h = xyxy_to_xywh(box)
            parts.extend(["0", f"{score:.4f}", f"{x:.4f}", f"{y:.4f}", f"{w:.4f}", f"{h:.4f}"])
        for box, score in zip(fused_shemagh_boxes, fused_shemagh_scores, strict=True):
            x, y, w, h = xyxy_to_xywh(box)
            parts.extend(["1", f"{score:.4f}", f"{x:.4f}", f"{y:.4f}", f"{w:.4f}", f"{h:.4f}"])
        prediction_string = " ".join(parts) if parts else "-"

        rows.append([filename, rp, prediction_string])

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "right_place", "prediction_string"])
        writer.writerows(rows)

    total_images = max(1, len(test_files))
    total_boxes = total_head + total_shemagh
    print("\nDone.")
    print(f"Saved: {out_path}")
    print(f"right_place=1: {rp_positive}/{len(test_files)}")
    print(f"Head boxes: {total_head} | Shemagh boxes: {total_shemagh} | Total boxes: {total_boxes}")
    print(f"Average boxes/image: {total_boxes/total_images:.2f}")


if __name__ == "__main__":
    main()
