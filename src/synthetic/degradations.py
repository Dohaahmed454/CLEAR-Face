"""Controlled synthetic degradations + a shared runner.

Each failure condition has mild / medium / severe levels and records full
provenance so the degradation is reproducible and traceable back to its clean
original (Milestone 2 roadmap, sections 16-17).

Synthetic degradation manifest columns:
    synthetic_image_id, original_image_id, synthetic_path, original_path,
    split, label, failure_condition, severity, intended_action, method, seed
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.fer.dataset import detect_columns

SEVERITIES = ["mild", "medium", "severe"]

CONDITIONS = ["lighting", "blur_motion", "pose", "distance"]

INTENDED_ACTION = {
    "lighting": "improve_lighting",
    "blur_motion": "hold_still",
    "pose": "face_camera",
    "distance": "move_closer",
}

_ABBREV = {"lighting": "lig", "blur_motion": "blr", "pose": "pos", "distance": "dst"}

# Per-condition severity parameters.
_GAMMA = {"mild": 1.8, "medium": 2.6, "severe": 3.6}        # >1 darkens
_BLUR_K = {"mild": 5, "medium": 15}                          # gaussian kernel (odd)
_MOTION_LEN = {"severe": 25}                                 # motion-blur length
_POSE_SKEW = {"mild": 0.08, "medium": 0.16, "severe": 0.28}  # fraction of width
_DIST_SCALE = {"mild": 0.6, "medium": 0.42, "severe": 0.28}  # content shrink factor


def _gamma_darken(img, gamma):
    # gamma>1 with this LUT darkens midtones (out = (in/255)^gamma * 255).
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def _gaussian_blur(img, k):
    k = int(k) | 1  # force odd
    return cv2.GaussianBlur(img, (k, k), 0)


def _motion_blur(img, length):
    length = max(3, int(length))
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length  # horizontal motion
    return cv2.filter2D(img, -1, kernel)


def _perspective_skew(img, frac):
    h, w = img.shape[:2]
    dx = frac * w
    dy = frac * h * 0.5
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # Push the right edge inward/vertically -> simulates a yaw turn.
    dst = np.float32([[0, 0], [w - dx, dy], [w - dx, h - dy], [0, h]])
    mat = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _shrink_pad(img, scale):
    h, w = img.shape[:2]
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(
        small, (h - sh) // 2, h - sh - (h - sh) // 2,
        (w - sw) // 2, w - sw - (w - sw) // 2,
        cv2.BORDER_REPLICATE,
    )
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)


def degrade(condition, img, severity):
    """Return (degraded_image, method_string)."""
    if condition == "lighting":
        g = _GAMMA[severity]
        return _gamma_darken(img, g), f"gamma_darkening_g{g}"
    if condition == "blur_motion":
        if severity in _BLUR_K:
            k = _BLUR_K[severity]
            return _gaussian_blur(img, k), f"gaussian_blur_k{int(k) | 1}"
        length = _MOTION_LEN[severity]
        return _motion_blur(img, length), f"motion_blur_len{length}"
    if condition == "pose":
        frac = _POSE_SKEW[severity]
        return _perspective_skew(img, frac), f"perspective_warp_f{frac}"
    if condition == "distance":
        scale = _DIST_SCALE[severity]
        return _shrink_pad(img, scale), f"downscale_pad_s{scale}"
    raise ValueError(f"Unknown condition: {condition}")


def run_condition(
    condition,
    manifest_path,
    output_dir,
    severities=None,
    seed=42,
    limit=0,
    image_root=None,
    quality="95",
):
    """Generate degraded images for one condition; return a manifest DataFrame."""
    severities = severities or SEVERITIES
    df = pd.read_csv(manifest_path)
    id_col, path_col, label_col = detect_columns(df)
    has_split = "split" in df.columns
    if limit and limit > 0:
        df = df.head(limit)

    out_dir = Path(output_dir) / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(image_root) if image_root else None

    rows, n_missing = [], 0
    for _, r in df.iterrows():
        raw = str(r[path_col])
        src_path = root / raw if (root and not Path(raw).is_absolute()) else Path(raw)
        img = cv2.imread(str(src_path))
        if img is None:
            n_missing += 1
            continue
        orig_id = str(r[id_col])
        for sev in severities:
            deg, method = degrade(condition, img, sev)
            # NB: full severity name (not sev[0]) -- "mild"/"medium" share 'm'.
            syn_id = f"syn_{_ABBREV[condition]}_{sev}_{orig_id}"
            syn_path = out_dir / f"{syn_id}.jpg"
            cv2.imwrite(str(syn_path), deg,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            rows.append(
                {
                    "synthetic_image_id": syn_id,
                    "original_image_id": orig_id,
                    "synthetic_path": str(syn_path).replace("\\", "/"),
                    "original_path": str(src_path).replace("\\", "/"),
                    "split": (r["split"] if has_split else ""),
                    "label": (r[label_col] if label_col else ""),
                    "failure_condition": condition,
                    "severity": sev,
                    "intended_action": INTENDED_ACTION[condition],
                    "method": method,
                    "seed": seed,
                }
            )
    if n_missing:
        print(f"[synthetic:{condition}] WARNING: {n_missing} source images unreadable.")
    print(f"[synthetic:{condition}] generated {len(rows)} images -> {out_dir}")
    return pd.DataFrame(rows)
