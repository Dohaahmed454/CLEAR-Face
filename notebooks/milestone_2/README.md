# Milestone 2 — Running on Kaggle (no local dataset download)

The AffectNet dataset is large, so we **never download it locally**. Instead we
attach it to a Kaggle Notebook (where it lives at `/kaggle/input/...`), train on
Kaggle's free GPU, and download only the small, GitHub-safe result CSVs and
reports.

## 1. Create the Kaggle Notebook

1. Go to <https://www.kaggle.com/code> → **New Notebook**.
2. **Add data** → search `antruong2477/affectnet-dataset` → **Add**.
   It mounts at `/kaggle/input/affectnet-dataset/`.
3. In the right panel set **Accelerator = GPU** and **Internet = On**
   (Internet is needed once to download the ImageNet-pretrained weights).

## 2. Set up the code

```python
!git clone https://github.com/<YOUR_USERNAME>/CLEAR-Face.git
%cd CLEAR-Face
!pip install -q timm grad-cam mediapipe torchmetrics albumentations
```

## 3. Run the whole pipeline

Quick end-to-end sanity pass first (a few minutes), then the full run:

```python
# Fast: tiny subset, 1 epoch — confirms every stage works
!python notebooks/milestone_2/kaggle_runner.py --quick

# Full run (tune epochs / limits to fit the GPU quota)
!python notebooks/milestone_2/kaggle_runner.py --epochs 20 --bench_limit 600
```

If auto-detection of the dataset folder fails, pass it explicitly:

```python
!python notebooks/milestone_2/kaggle_runner.py \
    --data_root /kaggle/input/affectnet-dataset
```

The runner executes, in order: build manifest → train primary
(MobileNetV3-Large) + comparison (ResNet-18) → internal/external eval →
calibration → profiling → synthetic degradations → degraded eval → quality
analysis → CLEAR-Face-R records → record validation → evidence signals →
stability. All GitHub-safe outputs are copied to
`/kaggle/working/clear_face_outputs/`.

## 4. Get the results out

`/kaggle/working/` files appear under the notebook's **Output** tab — download
the CSVs/JSON/reports from `clear_face_outputs/`. Commit those to the repo (see
the repo policy below). The trained `best_model.pth` checkpoints stay on Kaggle
unless you deliberately export them via Git LFS or an institutional store.

## What is safe to commit

| Commit to GitHub                                  | Keep local / Kaggle only            |
|---------------------------------------------------|-------------------------------------|
| `src/`, scripts, this runner                      | `data/milestone_2/synthetic/*` imgs |
| `data/milestone_2/metadata/*.csv` (paths only)    | raw AffectNet / RAF-DB images       |
| `data/milestone_2/clear_face_r/*.csv`             | `models/*/best_model.pth`           |
| `experiments/milestone_2/results/*.csv` / `*.json`| `calibration.pkl`                   |
| `docs/milestone_2/reports/*.md`                   | validation-gallery face images      |

The manifests store **paths** to images (e.g. `/kaggle/input/...`), never the
images themselves, so they are safe to push.

## Running individual steps

Every stage is also a standalone script with the exact commands in the roadmap,
e.g.:

```python
!python src/utils/build_manifest.py --data_root /kaggle/input/affectnet-dataset \
    --output data/milestone_2/metadata/dataset_manifest.csv
!python src/fer/train_primary_fer.py \
    --manifest data/milestone_2/metadata/dataset_manifest.csv \
    --output_dir models/fer_primary --epochs 30 --batch_size 32 --image_size 224
```

## Reproducibility notes

- All randomness is seeded (`--seed`, default 42); splits are deterministic and
  stratified (60/15/25), per the Milestone 1 split plan.
- AffectNet has no participant IDs, so splits are image-level disjoint (the same
  original image never appears in two splits; synthetic images inherit their
  original's split, and the benchmark only degrades held-out **test** images).
- `contempt` (AffectNet's 8th class) is dropped to keep a unified 7-class space
  shared with RAF-DB.
