"""Build a CLEAR-Face dataset manifest by scanning an image-folder dataset.

The manifest is the only thing downstream scripts read, so the actual images
never need to live in the repo. On Kaggle the AffectNet dataset is attached at
``/kaggle/input/...`` and this script writes a small CSV pointing at it.

Output columns (Milestone 2 roadmap, section 7):
    image_id, image_path, participant_id, source_dataset, split, label

Scanning strategy (robust to the various AffectNet folder layouts):
  * Recursively find image files under ``--data_root``.
  * The image's label is taken from the *nearest ancestor folder* whose name
    maps to a CLEAR-Face class (handles ``train/happy/x.jpg``,
    ``happy/x.jpg``, numeric ``6/x.jpg``, etc.).
  * Images whose folder maps to an excluded/unknown class (e.g. AffectNet
    "contempt") are skipped.
  * AffectNet has no participant IDs, so each image is its own participant
    (image-level disjoint split), per the Milestone 1 dataset-selection plan.

By default the script ignores any train/val/test folders in the source and
creates its own deterministic, stratified 60/15/25 split (Milestone 1 split
plan). Pass ``--keep_splits`` to honour folder-level splits when present.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.labels import is_valid_label, normalize_label  # noqa: E402
from src.utils.paths import METADATA_DIR, ensure_dir  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_FOLDER_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val", "dev": "val",
    "test": "test", "testing": "test", "eval": "test",
}


def _label_from_path(path: Path, root: Path):
    """Return the canonical label inferred from the nearest ancestor folder."""
    for parent in path.parents:
        if parent == root.parent:
            break
        label = normalize_label(parent.name)
        if label is not None and is_valid_label(parent.name):
            return label
    return None


def _split_hint_from_path(path: Path, root: Path):
    for parent in path.parents:
        if parent == root.parent:
            break
        hint = SPLIT_FOLDER_ALIASES.get(parent.name.strip().lower())
        if hint is not None:
            return hint
    return None


def scan_images(data_root: Path):
    rows = []
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
                "_split_hint": _split_hint_from_path(path, data_root),
            }
        )
    return pd.DataFrame(rows)


def stratified_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int):
    """Assign deterministic train/val/test labels, stratified by class."""
    rng = np.random.default_rng(seed)
    splits = np.empty(len(df), dtype=object)
    for label, group in df.groupby("label"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]
        train_idx = idx[n_test + n_val :]
        splits[df.index.get_indexer(test_idx)] = "test"
        splits[df.index.get_indexer(val_idx)] = "val"
        splits[df.index.get_indexer(train_idx)] = "train"
    return splits


def limit_per_class(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if not n or n <= 0:
        return df
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(n=min(n, len(g)), random_state=seed))
        .reset_index(drop=True)
    )


def build(args):
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"data_root does not exist: {data_root}\n"
            "On Kaggle this is usually /kaggle/input/<dataset-slug>/..."
        )

    print(f"[manifest] scanning {data_root} ...")
    df = scan_images(data_root)
    if df.empty:
        raise RuntimeError(
            "No labelled images found. Check that class folders are named like "
            "happy/sad/angry/... (or AffectNet numeric indices)."
        )
    print(f"[manifest] found {len(df)} labelled images across "
          f"{df['label'].nunique()} classes")

    df = df.reset_index(drop=True)
    df = limit_per_class(df, args.limit_per_class, args.seed)

    if args.keep_splits and df["_split_hint"].notna().any():
        df["split"] = df["_split_hint"].fillna("train")
        print("[manifest] honouring folder-level splits")
    else:
        df["split"] = stratified_split(df, args.val_frac, args.test_frac, args.seed)
        print(f"[manifest] generated stratified split "
              f"train/val/test = {1 - args.val_frac - args.test_frac:.0%}/"
              f"{args.val_frac:.0%}/{args.test_frac:.0%}")

    df = df.sort_values(["label", "image_path"]).reset_index(drop=True)
    width = max(6, len(str(len(df))))
    df["image_id"] = [f"img_{i:0{width}d}" for i in range(1, len(df) + 1)]
    df["participant_id"] = df["image_id"]  # AffectNet: no participant IDs
    df["source_dataset"] = args.source_name

    out = df[
        ["image_id", "image_path", "participant_id", "source_dataset", "split", "label"]
    ]
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    out.to_csv(output_path, index=False)

    print(f"[manifest] wrote {len(out)} rows -> {output_path}")
    print("[manifest] split counts:\n", out["split"].value_counts().to_string())
    print("[manifest] class counts:\n", out["label"].value_counts().to_string())
    return output_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build a CLEAR-Face dataset manifest.")
    p.add_argument(
        "--data_root",
        required=True,
        help="Root folder of the image dataset (e.g. /kaggle/input/affectnet-dataset).",
    )
    p.add_argument(
        "--output",
        default=str(METADATA_DIR / "dataset_manifest.csv"),
        help="Output manifest CSV path.",
    )
    p.add_argument("--source_name", default="affectnet", help="source_dataset value.")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--keep_splits",
        action="store_true",
        help="Honour existing train/val/test folders instead of resplitting.",
    )
    p.add_argument(
        "--limit_per_class",
        type=int,
        default=0,
        help="Cap images per class (0 = all). Useful for fast smoke runs.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
