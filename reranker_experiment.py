"""
Step 3: Train a box-level reranker from OOF predictions.

Expected input from Step 2:
- oof_stage2/oof_predictions.csv

Main outputs:
- reranker_stage3/reranker_model.joblib
- reranker_stage3/reranker_oof_scored.csv
- reranker_stage3/reranker_metrics.json
- reranker_stage3/reranker_fold_metrics.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OOF reranker for Step 3.")
    parser.add_argument("--oof-dir", type=str, default="./oof_stage2")
    parser.add_argument("--predictions-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="./reranker_stage3")
    parser.add_argument("--target-col", type=str, default="is_tp_iou50")
    parser.add_argument("--fold-col", type=str, default="fold")
    parser.add_argument("--score-col", type=str, default="confidence")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blend-alpha", type=float, default=None, help="Final score = alpha*confidence + (1-alpha)*rerank_raw")
    parser.add_argument("--auto-blend", action=argparse.BooleanOptionalAction, default=True, help="Auto-select alpha on OOF to avoid degrading AP")
    parser.add_argument("--score-csv", type=str, default=None)
    parser.add_argument("--score-out-csv", type=str, default=None)
    return parser.parse_args()


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"f1": float("nan"), "threshold": float("nan"), "precision": float("nan"), "recall": float("nan")}
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    denom = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    best_idx = int(np.argmax(f1))
    threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 1.0
    return {
        "f1": float(f1[best_idx]),
        "threshold": threshold,
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
    }


def to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "filename" not in data.columns:
        raise ValueError("Missing required column: filename")
    if "confidence" not in data.columns:
        raise ValueError("Missing required column: confidence")
    if "class_id" not in data.columns:
        raise ValueError("Missing required column: class_id")
    if "model_tag" not in data.columns:
        data["model_tag"] = "model"

    for col in ("x", "y", "w", "h"):
        if col not in data.columns:
            data[col] = 0.0
        data[col] = to_numeric(data[col], default=0.0)

    data["confidence"] = to_numeric(data["confidence"], default=0.0).clip(0.0, 1.0)
    data["class_id"] = to_numeric(data["class_id"], default=0).astype(int)
    data["model_tag"] = data["model_tag"].astype(str)

    if "area" not in data.columns:
        data["area"] = data["w"] * data["h"]
    data["area"] = to_numeric(data["area"], default=0.0).clip(lower=0.0)

    if "aspect" not in data.columns:
        data["aspect"] = data["w"] / data["h"].clip(lower=1e-6)
    data["aspect"] = to_numeric(data["aspect"], default=0.0)

    if "pred_count_image" not in data.columns:
        data["pred_count_image"] = data.groupby("filename")["filename"].transform("count")
    data["pred_count_image"] = to_numeric(data["pred_count_image"], default=1.0).clip(lower=1.0)

    if "pred_count_class_image" not in data.columns:
        data["pred_count_class_image"] = data.groupby(["filename", "class_id"])["filename"].transform("count")
    data["pred_count_class_image"] = to_numeric(data["pred_count_class_image"], default=1.0).clip(lower=1.0)

    if "rank_in_class" not in data.columns:
        data["rank_in_class"] = (
            data.sort_values(["filename", "class_id", "confidence"], ascending=[True, True, False])
            .groupby(["filename", "class_id"])
            .cumcount()
            + 1
        )
    data["rank_in_class"] = to_numeric(data["rank_in_class"], default=1.0).clip(lower=1.0)

    conf = data["confidence"].clip(1e-6, 1.0 - 1e-6)
    data["conf_logit"] = np.log(conf / (1.0 - conf))
    data["log_area"] = np.log(data["area"].clip(lower=1e-8))
    data["log_aspect_abs"] = np.abs(np.log(data["aspect"].clip(lower=1e-6)))
    data["center_dist"] = np.sqrt((data["x"] - 0.5) ** 2 + (data["y"] - 0.5) ** 2)
    data["rel_rank_cls"] = data["rank_in_class"] / data["pred_count_class_image"].clip(lower=1.0)
    data["inv_rank_cls"] = 1.0 / data["rank_in_class"].clip(lower=1.0)

    return data


def make_pipeline(c_value: float, max_iter: int, seed: int) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    clf = LogisticRegression(
        C=c_value,
        max_iter=max_iter,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate_scores(df: pd.DataFrame, target_col: str, base_col: str, rerank_col: str) -> dict[str, object]:
    y = to_numeric(df[target_col], default=0).astype(int).to_numpy()
    base = to_numeric(df[base_col], default=0.0).to_numpy()
    rerank = to_numeric(df[rerank_col], default=0.0).to_numpy()

    result: dict[str, object] = {}
    result["overall"] = {
        "n": int(len(df)),
        "positives": int(y.sum()),
        "ap_base": safe_ap(y, base),
        "ap_rerank": safe_ap(y, rerank),
        "best_f1_base": best_f1(y, base),
        "best_f1_rerank": best_f1(y, rerank),
    }

    per_class: dict[str, dict[str, float | int | dict[str, float]]] = {}
    for class_id in sorted(df["class_id"].unique()):
        class_mask = df["class_id"] == class_id
        y_c = to_numeric(df.loc[class_mask, target_col], default=0).astype(int).to_numpy()
        b_c = to_numeric(df.loc[class_mask, base_col], default=0.0).to_numpy()
        r_c = to_numeric(df.loc[class_mask, rerank_col], default=0.0).to_numpy()
        per_class[str(int(class_id))] = {
            "n": int(class_mask.sum()),
            "positives": int(y_c.sum()),
            "ap_base": safe_ap(y_c, b_c),
            "ap_rerank": safe_ap(y_c, r_c),
            "best_f1_base": best_f1(y_c, b_c),
            "best_f1_rerank": best_f1(y_c, r_c),
        }
    result["per_class"] = per_class

    per_model: dict[str, dict[str, float | int]] = {}
    for model_tag in sorted(df["model_tag"].unique()):
        model_mask = df["model_tag"] == model_tag
        y_m = to_numeric(df.loc[model_mask, target_col], default=0).astype(int).to_numpy()
        b_m = to_numeric(df.loc[model_mask, base_col], default=0.0).to_numpy()
        r_m = to_numeric(df.loc[model_mask, rerank_col], default=0.0).to_numpy()
        per_model[model_tag] = {
            "n": int(model_mask.sum()),
            "positives": int(y_m.sum()),
            "ap_base": safe_ap(y_m, b_m),
            "ap_rerank": safe_ap(y_m, r_m),
        }
    result["per_model"] = per_model
    return result


def train_crossfold(
    df: pd.DataFrame,
    target_col: str,
    fold_col: str,
    c_value: float,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    if fold_col not in df.columns:
        raise ValueError(f"Missing fold column: {fold_col}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    fold_values = sorted(to_numeric(df[fold_col], default=-1).astype(int).unique().tolist())
    if len(fold_values) < 2:
        raise ValueError(f"Need at least 2 folds in column '{fold_col}'")

    oof_scores = np.full(len(df), np.nan, dtype=np.float64)
    fold_stats: list[dict[str, float | int | str]] = []
    base_model = make_pipeline(c_value=c_value, max_iter=max_iter, seed=seed)

    for fold in fold_values:
        train_mask = to_numeric(df[fold_col], default=-1).astype(int) != fold
        valid_mask = to_numeric(df[fold_col], default=-1).astype(int) == fold

        train_df = df.loc[train_mask]
        valid_df = df.loc[valid_mask]
        y_train = to_numeric(train_df[target_col], default=0).astype(int)
        y_valid = to_numeric(valid_df[target_col], default=0).astype(int)

        if len(train_df) == 0 or len(valid_df) == 0:
            continue

        if y_train.nunique() < 2:
            oof_scores[valid_df.index] = to_numeric(valid_df["confidence"], default=0.0).to_numpy()
            fold_stats.append(
                {
                    "fold": int(fold),
                    "n_valid": int(len(valid_df)),
                    "positives_valid": int(y_valid.sum()),
                    "ap_base": safe_ap(y_valid.to_numpy(), to_numeric(valid_df["confidence"], default=0.0).to_numpy()),
                    "ap_rerank": safe_ap(y_valid.to_numpy(), to_numeric(valid_df["confidence"], default=0.0).to_numpy()),
                    "note": "fallback_confidence_only",
                }
            )
            continue

        model = clone(base_model)
        model.fit(train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], y_train)
        scores_valid = model.predict_proba(valid_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
        oof_scores[valid_df.index] = scores_valid

        fold_stats.append(
            {
                "fold": int(fold),
                "n_valid": int(len(valid_df)),
                "positives_valid": int(y_valid.sum()),
                "ap_base": safe_ap(y_valid.to_numpy(), to_numeric(valid_df["confidence"], default=0.0).to_numpy()),
                "ap_rerank": safe_ap(y_valid.to_numpy(), scores_valid),
                "note": "ok",
            }
        )

    return oof_scores, fold_stats


def score_external_csv(
    model: Pipeline,
    source_csv: Path,
    out_csv: Path,
    blend_alpha: float,
) -> None:
    df = pd.read_csv(source_csv)
    df = ensure_features(df)
    df["rerank_score_raw"] = model.predict_proba(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    df["rerank_score"] = blend_alpha * df["confidence"] + (1.0 - blend_alpha) * df["rerank_score_raw"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def main() -> None:
    args = parse_args()

    oof_dir = Path(args.oof_dir)
    predictions_csv = Path(args.predictions_csv) if args.predictions_csv else oof_dir / "oof_predictions.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_csv.exists():
        raise FileNotFoundError(f"OOF predictions CSV not found: {predictions_csv}")

    df = pd.read_csv(predictions_csv)
    if len(df) == 0:
        raise ValueError(f"No rows in {predictions_csv}")

    df = ensure_features(df)
    if args.target_col not in df.columns:
        raise ValueError(f"Missing target column: {args.target_col}")
    if args.fold_col not in df.columns:
        raise ValueError(f"Missing fold column: {args.fold_col}")
    if args.score_col not in df.columns:
        raise ValueError(f"Missing score column: {args.score_col}")

    if args.score_col != "confidence":
        df["confidence"] = to_numeric(df[args.score_col], default=0.0).clip(0.0, 1.0)

    df[args.target_col] = to_numeric(df[args.target_col], default=0).astype(int)
    df[args.fold_col] = to_numeric(df[args.fold_col], default=-1).astype(int)

    if args.min_confidence > 0:
        df = df[df["confidence"] >= args.min_confidence].copy()

    if len(df) == 0:
        raise ValueError("No rows left after filtering")

    oof_scores, fold_stats = train_crossfold(
        df=df,
        target_col=args.target_col,
        fold_col=args.fold_col,
        c_value=args.c,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    confidence_fallback = to_numeric(df["confidence"], default=0.0).to_numpy()
    rerank_raw = np.where(np.isnan(oof_scores), confidence_fallback, oof_scores)
    df["rerank_score_raw"] = rerank_raw

    y_true = to_numeric(df[args.target_col], default=0).astype(int).to_numpy()
    ap_base = safe_ap(y_true, confidence_fallback)
    ap_raw = safe_ap(y_true, rerank_raw)

    if args.blend_alpha is not None:
        blend_alpha = float(np.clip(args.blend_alpha, 0.0, 1.0))
    elif args.auto_blend:
        best_alpha = 1.0
        best_ap = ap_base
        for alpha in np.linspace(0.0, 1.0, 101):
            blended = alpha * confidence_fallback + (1.0 - alpha) * rerank_raw
            ap = safe_ap(y_true, blended)
            if np.isnan(ap):
                continue
            if ap > best_ap:
                best_ap = ap
                best_alpha = float(alpha)
        blend_alpha = float(best_alpha)
    else:
        blend_alpha = 0.0

    df["rerank_score"] = blend_alpha * confidence_fallback + (1.0 - blend_alpha) * rerank_raw

    metrics = evaluate_scores(df=df, target_col=args.target_col, base_col="confidence", rerank_col="rerank_score")
    metrics["settings"] = {
        "predictions_csv": str(predictions_csv),
        "rows_used": int(len(df)),
        "min_confidence": float(args.min_confidence),
        "target_col": args.target_col,
        "fold_col": args.fold_col,
        "score_col": args.score_col,
        "c": float(args.c),
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "ap_base": ap_base,
        "ap_rerank_raw": ap_raw,
        "blend_alpha_selected": blend_alpha,
        "auto_blend": bool(args.auto_blend),
    }

    final_model = make_pipeline(c_value=args.c, max_iter=args.max_iter, seed=args.seed)
    final_model.fit(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[args.target_col].astype(int))

    model_path = out_dir / "reranker_model.joblib"
    scored_oof_path = out_dir / "reranker_oof_scored.csv"
    metrics_path = out_dir / "reranker_metrics.json"
    fold_metrics_path = out_dir / "reranker_fold_metrics.csv"

    joblib.dump(final_model, model_path)
    df.to_csv(scored_oof_path, index=False)
    pd.DataFrame(fold_stats).to_csv(fold_metrics_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model: {model_path}")
    print(f"Saved OOF scored rows: {scored_oof_path}")
    print(f"Saved fold metrics: {fold_metrics_path}")
    print(f"Saved metrics json: {metrics_path}")

    overall = metrics["overall"]
    ap_base_final = overall["ap_base"]
    ap_rerank = overall["ap_rerank"]
    print(f"Blend alpha selected={blend_alpha:.3f}")
    print(f"Overall AP base={ap_base_final:.6f} rerank={ap_rerank:.6f} delta={(ap_rerank - ap_base_final):.6f}")
    print(f"Raw rerank AP (before blend)={ap_raw:.6f}")

    if args.score_csv:
        score_source = Path(args.score_csv)
        score_dest = Path(args.score_out_csv) if args.score_out_csv else out_dir / "reranker_scored_external.csv"
        if not score_source.exists():
            raise FileNotFoundError(f"--score-csv not found: {score_source}")
        score_external_csv(final_model, source_csv=score_source, out_csv=score_dest, blend_alpha=blend_alpha)
        print(f"Saved scored external CSV: {score_dest}")


if __name__ == "__main__":
    main()
