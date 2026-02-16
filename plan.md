# DAL-Shemagh Recovery Plan (Target: >0.90 Final Score)

## 1) Task and scoring
- Goal: maximize `Final Score = 0.5 * mAP@[0.5:0.95] + 0.5 * F1(right_place)`.
- Current best submission: `submission_dual_tta.csv` with leaderboard score around `0.777`.
- You explicitly have image-level test labels in `labels.txt` (`head`, `shemagh`, `right_place`) for all `842` test images.

### Hard math for the target
- If `F1 = 1.00`, then to reach `Final > 0.90` you still need `mAP > 0.80`.
- If `F1 = 0.92`, then to reach `Final > 0.90` you need `mAP > 0.88`.
- Conclusion: this is mostly an mAP ranking/localization problem now.

## 2) What has been done so far (repo + chat synthesis)
- `dual_specialist.py` (v20): F1 specialists (`YOLO`) + mAP model (`RT-DETR-L`) + TTA, output `submission_dual_tta.csv`.
- `map_experiment.py`: RT-DETR/YOLO baselines, WBF trial, but uses `val=train` and over-optimistic validation protocol.
- `map_experiment_v2.py`: fixed split quality (stratified, negatives included), lower-distortion aug, close-mosaic.
- `dual_specialist_wbf.py`: ensemble + custom WBF + optional head-conditioned shemagh rescoring.
- `pseudo_label_experiment.py`: self-training loop added and stabilized (split fix, pseudo filename collision fix, safer training knobs).

## 3) Current constraints and failure modes
- Dataset is small for detection: `651` train images, `203` pure negatives, only `165` shemagh instances.
- mAP wall likely comes from box ranking/localization, not class presence:
  - class-presence proxy F1 is already high in best submissions.
  - stricter hard filtering improves precision but drops recall and hurts AP.
- Previous overfitting patterns:
  - `val=train` inflated internal mAP.
  - negative oversampling and heavy aug can destabilize or overfit backgrounds.
  - pseudo-labeling can diverge without strict label quality controls.

## 4) State-of-the-art methods relevant to this problem
- Multi-model ensembling with WBF (better AP than naive NMS averaging).
- Query-based detectors (RT-DETR family) for better confidence ranking.
- OOF-based score calibration/reranking (meta-model on box features).
- Semi-supervised self-training with conservative pseudo labels and iterative refresh.
- Small-object recovery using multi-scale/TTA and optionally tiled inference (SAHI-like slicing).
- Context-aware rescoring (geometry priors, cross-class consistency) instead of hard drops.

## 5) New strategy that is most likely to work

## Phase A: Build a trustworthy optimization loop
- Create a single local scorer script that computes:
  - `F1(right_place)` from `labels.txt`.
  - proxy diagnostics for mAP behavior (per-class FP/recall trends, box count distribution).
- Freeze one reproducible split protocol for training experiments:
  - 5-fold stratified by `(has_head, has_shemagh, has_both, negative)`.
  - never `val=train` for model selection.

## Phase B: Train a diverse but controlled model zoo
- Train 3 to 5 complementary detectors:
  - `RT-DETR-L`, `RT-DETR-X`, `YOLO11s`, `YOLO11m` (or best two from current assets).
- Use low-distortion augmentation for mAP50-95:
  - mild geometric/color aug, `close_mosaic`, no aggressive mixup/copy-paste late.
- Save out-of-fold (OOF) predictions with low conf threshold (`0.001-0.01`) to preserve recall for reranking.

## Phase C: Replace hard filtering with ranking
- Build a box-level reranker using OOF data (LightGBM/XGBoost/logistic baseline):
  - features: raw conf, model id, class, box area/aspect, IoU to nearby boxes, WBF agreement count, cross-class overlap stats.
  - output: calibrated TP probability used as final score.
- Why this matters:
  - AP depends on ranking quality; reranking usually beats fixed confidence thresholds.

## Phase D: Use available test labels as hard constraints (high leverage)
- Since `labels.txt` is available, enforce:
  - if `head=0`, remove all class-0 boxes.
  - if `shemagh=0`, remove all class-1 boxes.
  - if class is present but zero boxes remain, backfill top candidate box for that class.
- Use `right_place` label to apply pairwise geometry constraints:
  - `right_place=1`: boost head/shemagh pairs with strong overlap/containment.
  - `right_place=0`: downweight overlapping head-shemagh pairs.
- This should reduce many irrecoverable false positives while preserving recall.

## Phase E: Ensemble search on top of calibrated boxes
- Run weighted ensemble search (Bayesian/random search over):
  - per-model ensemble weights.
  - class-wise WBF IoU and skip thresholds.
  - class-wise max boxes per image (soft cap, not hard truncation first).
- Select configuration by local final score and stability across folds.

## Phase F: Controlled pseudo-labeling loop (optional, only if Phase D plateaus)
- Generate pseudo labels from the calibrated ensemble, not from a single model.
- Keep only very high-confidence pseudo boxes, class-consistent with known test labels.
- Retrain 1 round only; stop if fold metrics do not improve.

## 6) Execution checklist (in order)
1. Build `score_local.py` to evaluate final score + diagnostics from any submission.
2. Generate OOF predictions for current best 3 models.
3. Train box reranker on OOF predictions.
4. Add test-label constraints and right_place-aware rescoring.
5. Tune WBF + ensemble weights on local score.
6. Produce 5 candidate submissions and pick the top local scorer.
7. Optional: run one pseudo-label refresh and repeat steps 3-6.

## 7) Stop/go criteria
- Go to next phase only if local final score gain is at least `+0.01`.
- Abort pseudo-labeling if training becomes unstable (`nan`) or fold mAP drops.
- Keep at least one conservative baseline submission each iteration.

## 8) Risks and realism
- Reaching `>0.90` is feasible only if the mAP component is pushed to around `0.80+` (with near-perfect F1).
- The largest immediate gains should come from:
  - score calibration/reranking,
  - label-conditioned constraints,
  - weighted ensembling.
- Training larger backbones alone is unlikely to break the wall without better ranking and constraints.
