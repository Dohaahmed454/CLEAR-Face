"""Profile FER inference: latency, FPS, model size, and memory.

Example:
    python src/fer/profile_fer.py \
        --model_dir models/fer_primary \
        --image_size 224 --runs 100 \
        --output experiments/milestone_2/results/primary_profile.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.fer.model import get_device, load_checkpoint  # noqa: E402


def profile(args):
    device = get_device(args.device)
    model, config, _ = load_checkpoint(args.model_dir, device)
    image_size = args.image_size or config.get("image_size", 224)

    n_params = sum(p.numel() for p in model.parameters())
    ckpt = Path(args.model_dir) / "best_model.pth"
    model_size_mb = ckpt.stat().st_size / (1024**2) if ckpt.exists() else float("nan")

    x = torch.randn(args.batch_size, 3, image_size, image_size, device=device)

    # Warmup.
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.no_grad():
        for _ in range(args.runs):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)  # ms

    times = np.array(times)
    per_image_ms = times / args.batch_size
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        try:
            import psutil

            peak_mem_mb = psutil.Process().memory_info().rss / (1024**2)
        except Exception:  # noqa: BLE001
            peak_mem_mb = float("nan")

    row = {
        "arch": config.get("arch"),
        "device": device.type,
        "image_size": image_size,
        "batch_size": args.batch_size,
        "runs": args.runs,
        "params_millions": round(n_params / 1e6, 4),
        "model_size_mb": round(model_size_mb, 4),
        "mean_latency_ms": round(float(times.mean()), 4),
        "median_latency_ms": round(float(np.median(times)), 4),
        "p95_latency_ms": round(float(np.percentile(times, 95)), 4),
        "per_image_ms": round(float(per_image_ms.mean()), 4),
        "fps": round(float(1000.0 / per_image_ms.mean()), 2),
        "peak_memory_mb": round(float(peak_mem_mb), 2),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print("[profile] " + "  ".join(f"{k}={v}" for k, v in row.items()))
    print(f"[profile] -> {out_path}")
    return row


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Profile FER inference cost.")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--image_size", type=int, default=0)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    profile(parse_args())
