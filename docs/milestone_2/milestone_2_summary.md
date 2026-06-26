# Milestone 2 Summary

Milestone 2 builds the perception and synthetic-benchmark foundation for CARE.

## How to reproduce
Everything runs on Kaggle with the AffectNet dataset attached (no local
download). See `notebooks/milestone_2/README.md`:

```bash
python notebooks/milestone_2/kaggle_runner.py            # full pipeline
python notebooks/milestone_2/kaggle_runner.py --quick    # fast sanity pass
```

## Completed components
- Primary FER model trained/fine-tuned — MobileNetV3-Large (`src/fer/train_primary_fer.py`).
- Conventional comparison FER model trained — ResNet-18 (`src/fer/train_comparison_fer.py`).
- Internal (and optional external RAF-DB) FER evaluation (`src/fer/evaluate_fer.py`).
- Calibration (temperature scaling), latency / memory / model-size profiling
  (`src/fer/calibrate_fer.py`, `src/fer/profile_fer.py`).
- Image-quality analysis module — brightness, contrast, blur, face size, head
  pose, occlusion proxy, detector confidence (`src/quality/`).
- Synthetic degradations for lighting, blur/motion, pose, distance at mild /
  medium / severe levels (`src/synthetic/`).
- Synthetic pre/post reacquisition pairs + CLEAR-Face-R records, validated
  against quality metrics (`src/clear_face_r/`).
- Preliminary XAI-derived evidence signals + stability/feasibility check
  (`src/evidence_signals/`).

## Main outputs
- Selected FER checkpoint: `models/fer_primary/best_model.pth` (local/Kaggle only).
- FER comparison report: `docs/milestone_2/reports/fer_model_report.md`.
- Quality-analysis module + outputs (`experiments/milestone_2/results/quality_*.csv`).
- Synthetic degradation scripts + provenance manifest
  (`data/milestone_2/metadata/synthetic_degradation_manifest.csv`).
- Validation gallery / local gallery manifest (images kept local per policy).
- Synthetic CLEAR-Face-R records
  (`data/milestone_2/clear_face_r/synthetic_clear_face_r_records_validated.csv`).
- Preliminary evidence-signal module + stability table.
- Milestone 2 exit-gate report.

## Design decisions (locked in Milestone 1)
- 7-class space: angry, disgust, fear, happy, neutral, sad, surprise
  (AffectNet "contempt" dropped to align with RAF-DB).
- Splits: stratified 60/15/25, image-level disjoint (AffectNet has no
  participant IDs); synthetic images inherit their original's split; the
  benchmark degrades only held-out test images.
- Seed 42 throughout; deterministic, reproducible scripts.

## Exit-gate decision
Proceed to Milestone 3 only if each retained failure condition reliably harms
performance and its intended correction creates measurable improvement.
Preliminary evidence signals are retained for CARE only if reproducible,
computationally feasible, and stable enough for controller feature construction.
See `docs/milestone_2/reports/milestone_2_exit_gate_report.md` for the filled
decision. **Status: `<fill after run>`.**

## Repository policy reminder
Pushed: source code, reports, metadata manifests (paths only), result CSVs/JSON,
model configs/logs. Kept local: raw + synthetic face images, `best_model.pth`
checkpoints, `calibration.pkl`, validation-gallery face images.
