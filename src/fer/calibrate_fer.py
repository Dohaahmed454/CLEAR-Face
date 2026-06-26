"""Calibrate FER confidence via temperature scaling and report ECE.

Fits a single temperature T on the validation split (NLL minimization),
reports Expected Calibration Error before/after, writes reliability-bin data,
and saves a calibration object to ``<model_dir>/calibration.pkl``.

Example:
    python src/fer/calibrate_fer.py \
        --model_dir models/fer_primary \
        --manifest data/milestone_2/metadata/dataset_manifest.csv \
        --split val \
        --output experiments/milestone_2/results/calibration_results.csv
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.fer.dataset import FERDataset, filter_labelled, load_manifest  # noqa: E402
from src.fer.evaluate_fer import collect_logits, softmax  # noqa: E402
from src.fer.model import get_device, load_checkpoint  # noqa: E402
from src.utils.labels import normalize_label  # noqa: E402


def expected_calibration_error(confidences, correct, n_bins=15):
    """ECE + per-bin reliability rows."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, rows = 0.0, []
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (
            (confidences >= lo) & (confidences <= hi)
        )
        count = int(mask.sum())
        if count:
            avg_conf = float(confidences[mask].mean())
            acc = float(correct[mask].mean())
            ece += (count / n) * abs(avg_conf - acc)
        else:
            avg_conf, acc = 0.0, 0.0
        rows.append({"bin_lower": round(lo, 4), "bin_upper": round(hi, 4),
                     "count": count, "avg_confidence": round(avg_conf, 4),
                     "accuracy": round(acc, 4)})
    return float(ece), rows


def fit_temperature(logits, labels, max_iter=200):
    """Optimize scalar T minimizing NLL with LBFGS."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_T = torch.zeros(1, requires_grad=True)  # parameterize as logT for positivity
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits_t / log_T.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_T.exp().item())


def calibrate(args):
    device = get_device(args.device)
    model, config, label_map = load_checkpoint(args.model_dir, device)
    classes = config.get("classes")
    image_size = args.image_size or config.get("image_size", 224)

    split = args.split if args.split != "all" else None
    df = load_manifest(args.manifest, split=split)
    if "label" in df.columns:
        df = filter_labelled(df, "label")
    if df.empty:
        raise RuntimeError("No labelled rows to calibrate on.")

    ds = FERDataset(df, image_size, train=False,
                    label_to_idx={c: i for i, c in enumerate(classes)},
                    root=args.image_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    ids, logits, labels = collect_logits(model, loader, device)
    keep = labels >= 0
    logits, labels = logits[keep], labels[keep]
    if len(labels) == 0:
        raise RuntimeError("No usable ground-truth labels for calibration.")

    # Before calibration.
    probs = softmax(logits)
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    ece_before, rows_before = expected_calibration_error(conf, correct, args.n_bins)

    # Fit temperature, recompute.
    temperature = fit_temperature(logits, labels)
    probs_cal = softmax(logits / temperature)
    conf_cal = probs_cal.max(1)
    ece_after, rows_after = expected_calibration_error(conf_cal, correct, args.n_bins)

    for r in rows_before:
        r["stage"] = "before"
    for r in rows_after:
        r["stage"] = "after"
    out_df = pd.DataFrame(rows_before + rows_after)
    out_df.insert(0, "temperature", round(temperature, 4))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    calib_obj = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "n_bins": args.n_bins,
        "n_samples": int(len(labels)),
        "split": args.split,
    }
    with open(Path(args.model_dir) / "calibration.pkl", "wb") as f:
        pickle.dump(calib_obj, f)

    print(f"[calib] temperature={temperature:.4f} "
          f"ECE before={ece_before:.4f} after={ece_after:.4f}")
    print(f"[calib] reliability bins -> {out_path}")
    print(f"[calib] calibration object -> {Path(args.model_dir) / 'calibration.pkl'}")
    return calib_obj


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Temperature-scale FER confidence.")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--output", required=True)
    p.add_argument("--n_bins", type=int, default=15)
    p.add_argument("--image_size", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--image_root", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    calibrate(parse_args())
