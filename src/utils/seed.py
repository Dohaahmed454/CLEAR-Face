"""Deterministic seeding for reproducible Milestone 2 experiments."""

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and (if available) PyTorch RNGs.

    Returns the seed so callers can record it in configs/reports.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding so augmentation order is reproducible."""
    seed = (DEFAULT_SEED + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
