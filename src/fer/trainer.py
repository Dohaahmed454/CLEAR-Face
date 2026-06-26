"""Shared FER training loop used by the primary and comparison scripts."""

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.fer.dataset import (
    FERDataset,
    compute_class_weights,
    detect_columns,
    filter_labelled,
    load_manifest,
)
from src.fer.model import build_model, get_device, save_checkpoint
from src.utils.labels import CLEAR_FACE_CLASSES, label_map_dict
from src.utils.seed import set_seed, worker_init_fn


def _macro_f1(y_true, y_pred, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def _subsample(df, limit, seed):
    if not limit or limit <= 0:
        return df
    if len(df) <= limit:
        return df
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


@torch.no_grad()
def _evaluate(model, loader, device, num_classes):
    model.eval()
    all_true, all_pred = [], []
    total_loss, n = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        valid = y >= 0
        logits = model(x)
        if valid.any():
            total_loss += criterion(logits[valid], y[valid]).item() * valid.sum().item()
            n += valid.sum().item()
        preds = logits.argmax(1)
        all_true.append(y[valid].cpu().numpy())
        all_pred.append(preds[valid].cpu().numpy())
    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    f1 = _macro_f1(y_true, y_pred, num_classes) if len(y_true) else 0.0
    loss = total_loss / n if n else 0.0
    return acc, f1, loss


def train_model(
    *,
    arch: str,
    manifest: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 32,
    image_size: int = 224,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    num_workers: int = 2,
    device: str = "auto",
    limit: int = 0,
    pretrained: bool = True,
    use_class_weights: bool = True,
    image_root: str | None = None,
    model_label: str = "primary",
):
    """Train a FER model and write best_model.pth / config.json / label_map.json /
    training_log.csv into ``output_dir``. Returns a summary dict."""
    set_seed(seed)
    dev = get_device(device)
    num_classes = len(CLEAR_FACE_CLASSES)

    df = load_manifest(manifest)
    _, _, label_col = detect_columns(df)
    if label_col is None:
        raise ValueError("Training manifest must have a label column.")
    df = filter_labelled(df, label_col)
    if "split" not in df.columns:
        raise ValueError("Training manifest must have a 'split' column.")

    train_df = _subsample(df[df["split"] == "train"], limit, seed)
    val_df = _subsample(df[df["split"] == "val"], limit, seed)
    if len(train_df) == 0:
        raise RuntimeError("No training rows found (split == 'train').")
    if len(val_df) == 0:
        print("[train] WARNING: no validation rows; using train metrics for selection.")
        val_df = train_df

    print(f"[train:{model_label}] arch={arch} train={len(train_df)} val={len(val_df)} "
          f"device={dev}")

    train_ds = FERDataset(train_df, image_size, train=True, root=image_root)
    val_ds = FERDataset(val_df, image_size, train=False, root=image_root)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        worker_init_fn=worker_init_fn, pin_memory=(dev.type == "cuda"), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
    )

    model = build_model(arch, num_classes, pretrained=pretrained).to(dev)

    if use_class_weights:
        weights = compute_class_weights(train_df, label_col, num_classes).to(dev)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"
    log_rows = []
    best_f1, best_epoch = -1.0, -1
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(dev), y.to(dev)
            valid = y >= 0
            if not valid.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits[valid], y[valid])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * valid.sum().item()
            seen += valid.sum().item()
        scheduler.step()
        train_loss = running / seen if seen else 0.0
        val_acc, val_f1, val_loss = _evaluate(model, val_loader, dev, num_classes)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[train:{model_label}] epoch {epoch:>3}/{epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}")
        log_rows.append(
            {"epoch": epoch, "train_loss": round(train_loss, 6),
             "val_loss": round(val_loss, 6), "val_accuracy": round(val_acc, 6),
             "val_macro_f1": round(val_f1, 6), "lr": lr_now}
        )
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            config = {
                "arch": arch, "num_classes": num_classes, "image_size": image_size,
                "classes": CLEAR_FACE_CLASSES, "seed": seed, "epochs": epochs,
                "batch_size": batch_size, "lr": lr, "pretrained": pretrained,
                "model_label": model_label, "manifest": str(manifest),
                "best_epoch": best_epoch, "best_val_macro_f1": round(best_f1, 6),
                "best_val_accuracy": round(val_acc, 6),
            }
            save_checkpoint(model, out_dir, config, label_map_dict())

    # Persist the training log.
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    elapsed = time.time() - start
    summary = {
        "model_label": model_label, "arch": arch, "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_f1, 6), "train_seconds": round(elapsed, 1),
        "output_dir": str(out_dir), "n_train": len(train_df), "n_val": len(val_df),
    }
    print(f"[train:{model_label}] DONE best_epoch={best_epoch} "
          f"best_val_macro_f1={best_f1:.4f} in {elapsed:.0f}s -> {out_dir}")
    return summary
