"""Dataset, transforms, and manifest helpers for FER.

The FER scripts read a manifest CSV (clean or synthetic) and never assume a
fixed column layout, so the same code works for ``dataset_manifest.csv`` and
``synthetic_degradation_manifest.csv``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.utils.labels import CLASS_TO_IDX, normalize_label

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Manifest column auto-detection (clean vs synthetic manifests differ).
_ID_CANDIDATES = ["image_id", "synthetic_image_id", "id"]
_PATH_CANDIDATES = ["image_path", "synthetic_path", "path", "filepath"]
_LABEL_CANDIDATES = ["label", "ground_truth_label", "emotion", "class"]


def detect_columns(df: pd.DataFrame):
    """Return (id_col, path_col, label_col); label_col may be None."""

    def pick(cands, required):
        for c in cands:
            if c in df.columns:
                return c
        if required:
            raise ValueError(
                f"Manifest must contain one of {cands}; got {list(df.columns)}"
            )
        return None

    return (
        pick(_ID_CANDIDATES, True),
        pick(_PATH_CANDIDATES, True),
        pick(_LABEL_CANDIDATES, False),
    )


def build_transforms(image_size: int, train: bool):
    if train:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                T.RandomRotation(10),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_manifest(path, split: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str) == split].reset_index(drop=True)
    return df


def filter_labelled(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Keep only rows whose label maps to a canonical CLEAR-Face class."""
    mask = df[label_col].map(lambda x: normalize_label(x) in CLASS_TO_IDX)
    return df[mask].reset_index(drop=True)


class FERDataset(Dataset):
    """Returns (image_tensor, label_idx, image_id). label_idx == -1 if unknown."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        train: bool = False,
        label_to_idx: dict | None = None,
        root: str | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.transform = build_transforms(image_size, train)
        self.label_to_idx = label_to_idx or dict(CLASS_TO_IDX)
        self.id_col, self.path_col, self.label_col = detect_columns(self.df)
        self.root = Path(root) if root else None
        self.n_missing = 0

    def __len__(self):
        return len(self.df)

    def _resolve(self, raw_path) -> Path:
        p = Path(str(raw_path))
        if self.root is not None and not p.is_absolute():
            return self.root / p
        return p

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = self._resolve(row[self.path_col])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:  # noqa: BLE001 - tolerate missing/corrupt files
            self.n_missing += 1
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        x = self.transform(img)

        y = -1
        if self.label_col is not None:
            raw = row.get(self.label_col)
            if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
                canon = normalize_label(raw)
                y = self.label_to_idx.get(canon, -1)
        return x, y, str(row[self.id_col])


def compute_class_weights(df: pd.DataFrame, label_col: str, num_classes: int):
    """Inverse-frequency class weights for imbalanced FER training."""
    counts = np.ones(num_classes, dtype=np.float64)
    for raw, n in df[label_col].value_counts().items():
        idx = CLASS_TO_IDX.get(normalize_label(raw))
        if idx is not None:
            counts[idx] = n
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
