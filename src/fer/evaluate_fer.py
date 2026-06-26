"""Evaluate a FER model on a manifest (clean, external, or synthetic).

Writes a per-image prediction CSV (used downstream to build CLEAR-Face-R
records) and, when ground truth is present, a metrics summary + confusion
matrix next to it.

Per-image output columns:
    image_id, ground_truth, prediction, confidence, entropy, margin, correct

Example:
    python src/fer/evaluate_fer.py \
        --model_dir models/fer_primary \
        --manifest data/milestone_2/metadata/dataset_manifest.csv \
        --split test \
        --output experiments/milestone_2/results/primary_internal_eval.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.fer.dataset import FERDataset, load_manifest  # noqa: E402
from src.fer.model import (  # noqa: E402
    get_device,
    idx_to_class_from_label_map,
    load_checkpoint,
)


@torch.no_grad()
def collect_logits(model, loader, device):
    """Return (ids, logits, labels) as numpy arrays. labels == -1 if unknown."""
    ids, logits_all, labels_all = [], [], []
    for x, y, batch_ids in loader:
        x = x.to(device)
        logits = model(x).float().cpu().numpy()
        logits_all.append(logits)
        labels_all.append(np.asarray(y))
        ids.extend(batch_ids)
    logits = np.concatenate(logits_all) if logits_all else np.zeros((0, 1))
    labels = np.concatenate(labels_all) if labels_all else np.zeros((0,), dtype=int)
    return ids, logits, labels


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predictions_frame(ids, logits, labels, idx_to_class):
    probs = softmax(logits)
    pred_idx = probs.argmax(1)
    conf = probs.max(1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2] if probs.shape[1] > 1 else conf
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    gt = [idx_to_class.get(int(i)) if i >= 0 else None for i in labels]
    pred = [idx_to_class.get(int(i)) for i in pred_idx]
    correct = [
        (int(p == g) if g is not None else None) for p, g in zip(pred, gt)
    ]
    return pd.DataFrame(
        {
            "image_id": ids,
            "ground_truth": gt,
            "prediction": pred,
            "confidence": np.round(conf, 6),
            "entropy": np.round(entropy, 6),
            "margin": np.round(margin, 6),
            "correct": correct,
        }
    )


def compute_metrics(df, classes):
    labelled = df[df["ground_truth"].notna()].copy()
    if labelled.empty:
        return {"note": "no ground-truth labels in manifest", "n": int(len(df))}
    y_true = labelled["ground_truth"].to_numpy()
    y_pred = labelled["prediction"].to_numpy()
    acc = float((y_true == y_pred).mean())
    per_class, f1s = {}, []
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    cidx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        if t in cidx and p in cidx:
            cm[cidx[t], cidx[p]] += 1
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "support": int(np.sum(y_true == c)),
        }
        f1s.append(f1)
    return {
        "n": int(len(labelled)),
        "accuracy": round(acc, 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "mean_confidence": round(float(labelled["confidence"].mean()), 4),
        "per_class": per_class,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
    }


def evaluate(args):
    device = get_device(args.device)
    model, config, label_map = load_checkpoint(args.model_dir, device)
    idx_to_class = idx_to_class_from_label_map(label_map)
    classes = config.get("classes", list(idx_to_class.values()))
    image_size = args.image_size or config.get("image_size", 224)

    split = args.split if args.split != "all" else None
    df = load_manifest(args.manifest, split=split)
    if df.empty:
        raise RuntimeError(f"No rows for manifest={args.manifest} split={args.split}")

    ds = FERDataset(df, image_size, train=False,
                    label_to_idx={c: i for i, c in enumerate(classes)},
                    root=args.image_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    ids, logits, labels = collect_logits(model, loader, device)
    pred_df = predictions_frame(ids, logits, labels, idx_to_class)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    print(f"[eval] wrote {len(pred_df)} predictions -> {out_path}")
    if ds.n_missing:
        print(f"[eval] WARNING: {ds.n_missing} images could not be read (used black).")

    metrics = compute_metrics(pred_df, classes)
    metrics_path = out_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if "accuracy" in metrics:
        print(f"[eval] accuracy={metrics['accuracy']} "
              f"macro_f1={metrics['macro_f1']} (n={metrics['n']})")
        cm_df = pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes)
        cm_df.to_csv(out_path.with_suffix(".confusion_matrix.csv"))
    else:
        print(f"[eval] {metrics.get('note')}")
    print(f"[eval] metrics -> {metrics_path}")
    return metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a FER model on a manifest.")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default="all",
                   help="Filter on the manifest 'split' column, or 'all'.")
    p.add_argument("--output", required=True)
    p.add_argument("--image_size", type=int, default=0, help="0 = use model config.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--image_root", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
