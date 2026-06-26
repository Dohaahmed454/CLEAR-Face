# FER Model Report

> Numeric fields marked `<fill>` are populated from the result files produced by
> the pipeline (paths in parentheses). Run `notebooks/milestone_2/kaggle_runner.py`
> on Kaggle, then transcribe the printed metrics / open the CSVs.

## Primary model
- Architecture: MobileNetV3-Large (`timm: mobilenetv3_large_100`), ImageNet-pretrained, lightweight for the later real-time prototype
- Dataset: AffectNet (`antruong2477/affectnet-dataset`)
- Labels: 7-class CLEAR-Face space — angry, disgust, fear, happy, neutral, sad, surprise (AffectNet "contempt" dropped)
- Training split: 60% (stratified, image-level disjoint)
- Validation split: 15%
- Test split: 25%
- Seed: 42
- Best checkpoint: `models/fer_primary/best_model.pth` (selected on best val macro-F1)
- Accuracy: `<fill>` (`experiments/milestone_2/results/primary_internal_eval.metrics.json`)
- Macro F1: `<fill>` (same file)
- Main failure classes: `<fill>` (lowest per-class recall in the metrics JSON / confusion matrix)

Training config and loss curves: `models/fer_primary/config.json`,
`models/fer_primary/training_log.csv`.
Training used AdamW + cosine schedule, inverse-frequency class weights (AffectNet
is imbalanced), light augmentation (flip / color jitter / small rotation), and
mixed precision on GPU.

## Comparison model
- Architecture: ResNet-18 (`timm: resnet18`), conventional CNN baseline
- Dataset: AffectNet (same splits as primary, for a fair comparison)
- Accuracy: `<fill>` (`experiments/milestone_2/results/comparison_internal_eval.metrics.json`)
- Macro F1: `<fill>` (same file)

## Internal evaluation
Evaluated on the held-out AffectNet **test** split.

| Model | Accuracy | Macro F1 | Mean confidence |
|-------|----------|----------|-----------------|
| Primary (MobileNetV3-Large) | `<fill>` | `<fill>` | `<fill>` |
| Comparison (ResNet-18)      | `<fill>` | `<fill>` | `<fill>` |

Per-class precision/recall and the confusion matrix are in the `*.metrics.json`
and `*.confusion_matrix.csv` files alongside each eval CSV.

## External evaluation
Cross-dataset generalization on RAF-DB (external manifest), if available.

| Model | Dataset | Accuracy | Macro F1 |
|-------|---------|----------|----------|
| Primary | RAF-DB | `<fill>` | `<fill>` (`primary_external_eval.metrics.json`) |

> If no external manifest is provided, state "external evaluation deferred —
> RAF-DB manifest not yet built" here.

## Calibration, latency, memory, model size
- Temperature (temperature scaling on val): `<fill>` (`calibration_results.csv`)
- ECE before / after: `<fill>` / `<fill>`
- Mean / median / p95 latency: `<fill>` / `<fill>` / `<fill>` ms (`primary_profile.csv`)
- FPS: `<fill>` · Params: `<fill>` M · Model size: `<fill>` MB · Peak memory: `<fill>` MB

## Decision
Selected FER checkpoint: **`models/fer_primary/best_model.pth`** (MobileNetV3-Large).
Rationale: lightweight (low latency / high FPS for the real-time path) while
remaining competitive with the conventional ResNet-18 baseline on the same
splits. Confirm with the filled metrics above before proceeding.
