"""Build the complete 5-way CLEAR-Face dataset manifest from AffectNet
(or any class-folder dataset).

=========================================================================
Split strategy  (Milestone 1 split plan + leakage-prevention rules)
=========================================================================

  Split         │ Fraction of ALL data │ Purpose
  ──────────────┼──────────────────────┼──────────────────────────────────
  final_test    │ 25 %                 │ Strictly held-out; never used for
                │                      │ tuning, thresholds, or CARE design
  care_dev      │ 15 %                 │ CARE policy development and
                │                      │ threshold tuning only
  train         │ 60 % × 60 % = 36 %  │ FER model training
  val           │ 60 % × 15 % =  9 %  │ FER validation + HP search
  test          │ 60 % × 25 % = 15 %  │ FER internal test
  ──────────────┼──────────────────────┼──────────────────────────────────
  TOTAL         │         100 %        │

All fractions are computed within each class (stratified), so every split
has a balanced label distribution.

=========================================================================
Leakage-prevention guarantees enforced by this script
=========================================================================

  ✓  No participant appears in more than one top-level split.
  ✓  AffectNet has no participant IDs → each image is its own participant.
  ✓  Splitting is deterministic and seeded (default seed = 42).
  ✓  Synthetic images (generated later) inherit the split of their source
     image via the ``synthetic_degradation_manifest.csv`` provenance field.
  ✓  Pre/post pairs share the same source image, so they are always in the
     same split.
  ✓  final_test images must NEVER be used for HP tuning, threshold setting,
     or CARE design decisions.

=========================================================================
Output CSV columns  (roadmap section 7 + 5-way split extension)
=========================================================================

  image_id, image_path, participant_id, source_dataset, split, label

  split values:  train | val | test | care_dev | final_test

The format is fully compatible with the existing FER training, evaluation,
and quality-analysis scripts, which filter by ``--split train/val/test``
and naturally ignore care_dev / final_test rows.

=========================================================================
Usage examples
=========================================================================

  # Local – AffectNet extracted to data/external/affectnet-dataset/
  python src/utils/build_manifest_full.py \\
      --data_root data/external/affectnet-dataset \\
      --output    data/milestone_2/metadata/dataset_manifest.csv

  # Kaggle environment
  python src/utils/build_manifest_full.py \\
      --data_root /kaggle/input/affectnet-dataset \\
      --output    data/milestone_2/metadata/dataset_manifest.csv

  # Limit to 500 images per class for a fast smoke test
  python src/utils/build_manifest_full.py \\
      --data_root data/external/affectnet-dataset \\
      --limit_per_class 500

  # View split counts only, do not write CSV
  python src/utils/build_manifest_full.py \\
      --data_root data/external/affectnet-dataset \\
      --dry_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Make ``src`` importable when script is run from the repo root ──────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.labels import is_valid_label, normalize_label  # noqa: E402
from src.utils.paths import METADATA_DIR, ensure_dir          # noqa: E402
from src.utils.seed import set_seed                            # noqa: E402

# ── Supported image extensions ─────────────────────────────────────────────
IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

# ── Split fractions (of ALL data, per class) ───────────────────────────────
FINAL_TEST_FRAC: float = 0.25   # strictly held-out
CARE_DEV_FRAC:   float = 0.15   # CARE policy development
# Remaining 60 % → FER model split
_FER_PORTION:    float = 1.0 - FINAL_TEST_FRAC - CARE_DEV_FRAC  # 0.60
FER_TRAIN_FRAC:  float = 0.60 * _FER_PORTION                    # 0.36
FER_VAL_FRAC:    float = 0.15 * _FER_PORTION                     # 0.09
FER_TEST_FRAC:   float = 0.25 * _FER_PORTION                     # 0.15

# ── Allowed values in the output ``split`` column ─────────────────────────
SPLIT_VALUES = ("train", "val", "test", "care_dev", "final_test")

# ── AffectNet folder-name → canonical split (ignored; we re-split) ─────────
_FOLDER_SPLIT_ALIASES: Dict[str, str] = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val", "dev": "val",
    "test": "test", "testing": "test", "eval": "test",
}


# ══════════════════════════════════════════════════════════════════════════
# Image scanning
# ══════════════════════════════════════════════════════════════════════════

def _label_from_path(path: Path, root: Path) -> Optional[str]:
    """Return the canonical label inferred from the nearest ancestor folder."""
    for parent in path.parents:
        if parent == root.parent:
            break
        label = normalize_label(parent.name)
        if label is not None and is_valid_label(parent.name):
            return label
    return None


def scan_images(data_root: Path) -> pd.DataFrame:
    """Recursively find all labelled images under *data_root*.

    Returns a DataFrame with columns: image_path, label.
    Images whose folder name cannot be mapped to a valid CLEAR-Face class
    are silently skipped (e.g. AffectNet's ``contempt`` / index 7).
    """
    rows: List[Dict] = []
    for path in sorted(data_root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
            continue
        label = _label_from_path(path, data_root)
        if label is None:
            continue
        rows.append(
            {
                "image_path": str(path).replace("\\", "/"),
                "label": label,
            }
        )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 5-way stratified split
# ══════════════════════════════════════════════════════════════════════════

def five_way_split(df: pd.DataFrame, seed: int) -> pd.Series:
    """Assign each row to one of the five CLEAR-Face splits.

    The assignment is done per-class to preserve class balance in every
    split.  Each class is shuffled independently with the same RNG so the
    overall ratio is respected even for minority classes.

    Returns a Series of split labels aligned with ``df.index``.
    """
    rng = np.random.default_rng(seed)
    result = pd.Series(index=df.index, dtype=object)

    for label, group in df.groupby("label"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)

        # Carve out from the end → deterministic ordering.
        # (Taking from the end means the first images go to train, making
        #  the slices easy to audit manually.)
        n_final_test = max(1, round(n * FINAL_TEST_FRAC))
        n_care_dev   = max(1, round(n * CARE_DEV_FRAC))
        n_fer_test   = max(1, round(n * FER_TEST_FRAC))
        n_fer_val    = max(1, round(n * FER_VAL_FRAC))
        # train gets whatever is left
        n_train      = n - n_final_test - n_care_dev - n_fer_test - n_fer_val

        if n_train < 1:
            # Class too small to honour all fractions; give the single image
            # to train so model training always has at least one example.
            result[idx] = "train"
            continue

        ptr = 0
        result[idx[ptr : ptr + n_train]]      = "train";      ptr += n_train
        result[idx[ptr : ptr + n_fer_val]]    = "val";        ptr += n_fer_val
        result[idx[ptr : ptr + n_fer_test]]   = "test";       ptr += n_fer_test
        result[idx[ptr : ptr + n_care_dev]]   = "care_dev";   ptr += n_care_dev
        result[idx[ptr : ptr + n_final_test]] = "final_test"

    return result


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def limit_per_class(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Cap the number of images per class (useful for smoke runs)."""
    if not n or n <= 0:
        return df
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(n=min(n, len(g)), random_state=seed))
        .reset_index(drop=True)
    )


def print_split_summary(df: pd.DataFrame) -> None:
    """Pretty-print split × class counts."""
    print("\n[manifest] ── Split × class breakdown ─────────────────────────")
    table = (
        df.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(list(SPLIT_VALUES))
    )
    print(table.to_string())
    print()
    print("[manifest] ── Split totals ─────────────────────────────────────")
    totals = df["split"].value_counts().reindex(list(SPLIT_VALUES))
    pct    = (totals / len(df) * 100).round(1)
    for s in SPLIT_VALUES:
        print(f"  {s:<15} {totals[s]:>7,}  ({pct[s]:>5.1f} %)")
    print(f"  {'TOTAL':<15} {len(df):>7,}  (100.0 %)")
    print()


def validate_leakage(df: pd.DataFrame) -> None:
    """Assert that no image_id / participant_id appears in multiple splits."""
    for col in ("image_id", "participant_id"):
        dup = (
            df.groupby(col)["split"]
            .nunique()
            .pipe(lambda s: s[s > 1])
        )
        if not dup.empty:
            raise RuntimeError(
                f"LEAKAGE DETECTED: {len(dup)} {col}(s) appear in multiple "
                f"splits.  First 5: {list(dup.index[:5])}"
            )
    print("[manifest] ✓  Leakage check passed – no participant in multiple splits.")


# ══════════════════════════════════════════════════════════════════════════
# Main build function
# ══════════════════════════════════════════════════════════════════════════

def build(args: argparse.Namespace) -> Optional[Path]:
    set_seed(args.seed)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"data_root does not exist: {data_root}\n"
            "  • On Kaggle: /kaggle/input/<dataset-slug>/\n"
            "  • Locally:   CLEAR-Face/data/external/affectnet-dataset/\n"
            "Download the dataset and point --data_root at the extracted folder."
        )

    print(f"[manifest] Scanning {data_root} ...")
    df = scan_images(data_root)

    if df.empty:
        raise RuntimeError(
            "No labelled images found under the given data_root.\n"
            "Expected class folders named like: happy/ sad/ angry/ 0/ 1/ 2/ ...\n"
            f"Check the folder structure of: {data_root}"
        )

    print(
        f"[manifest] Found {len(df):,} labelled images across "
        f"{df['label'].nunique()} classes."
    )
    print(f"[manifest] Class distribution (raw):\n"
          f"{df['label'].value_counts().to_string()}\n")

    # Optional cap
    df = df.reset_index(drop=True)
    if args.limit_per_class and args.limit_per_class > 0:
        before = len(df)
        df = limit_per_class(df, args.limit_per_class, args.seed)
        print(
            f"[manifest] Capped at {args.limit_per_class} per class: "
            f"{before:,} → {len(df):,} images."
        )

    # 5-way split
    df["split"] = five_way_split(df, seed=args.seed)

    # Stable sort so image IDs are deterministic regardless of OS / FS order
    df = df.sort_values(["label", "image_path"]).reset_index(drop=True)

    # Assign image IDs (zero-padded, 1-indexed)
    width = max(6, len(str(len(df))))
    df["image_id"]       = [f"img_{i:0{width}d}" for i in range(1, len(df) + 1)]
    df["participant_id"] = df["image_id"]   # AffectNet: no participant IDs
    df["source_dataset"] = args.source_name

    # Final column order (roadmap section 7)
    out = df[["image_id", "image_path", "participant_id",
              "source_dataset", "split", "label"]]

    print_split_summary(out)
    validate_leakage(out)

    if args.dry_run:
        print("[manifest] --dry_run: manifest NOT written to disk.")
        return None

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    out.to_csv(output_path, index=False)
    print(f"[manifest] ✓  Wrote {len(out):,} rows → {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the 5-way CLEAR-Face dataset manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data_root",
        required=True,
        help=(
            "Root folder of the AffectNet dataset (contains class sub-folders).\n"
            "Local example: data/external/affectnet-dataset\n"
            "Kaggle example: /kaggle/input/affectnet-dataset"
        ),
    )
    p.add_argument(
        "--output",
        default=str(METADATA_DIR / "dataset_manifest.csv"),
        help="Output manifest CSV path (default: data/milestone_2/metadata/dataset_manifest.csv).",
    )
    p.add_argument(
        "--source_name",
        default="affectnet",
        help="Value written to the source_dataset column (default: affectnet).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    p.add_argument(
        "--limit_per_class",
        type=int,
        default=0,
        help="Cap images per class (0 = use all). Useful for fast smoke tests.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print split statistics without writing any file.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
