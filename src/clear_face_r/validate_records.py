"""Validate synthetic CLEAR-Face-R records against quality measurements.

Joins pre (degraded) and post (clean) quality metrics onto each record and
checks that the degradation actually moved quality in the expected direction
for its condition (Milestone 2 roadmap, section 20).

Example:
    python src/clear_face_r/validate_records.py \
        --records data/milestone_2/clear_face_r/synthetic_clear_face_r_records.csv \
        --clean_quality experiments/milestone_2/results/quality_clean_images.csv \
        --degraded_quality experiments/milestone_2/results/quality_synthetic_degraded_images.csv \
        --output data/milestone_2/clear_face_r/synthetic_clear_face_r_records_validated.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import ensure_dir  # noqa: E402

_QUALITY_FIELDS = ["brightness", "contrast", "blur_score", "face_area_ratio",
                   "yaw", "pitch", "roll", "occlusion_score"]


def _quality_map(path):
    df = pd.read_csv(path)
    df = df.set_index(df["image_id"].astype(str))
    return df


def _direction_ok(condition, init_q, post_q):
    """Did quality move the way the degradation intends? Returns (ok, note)."""
    def g(q, k):
        v = q.get(k, np.nan)
        return float(v) if pd.notna(v) else np.nan

    if condition == "lighting":
        a, b = g(init_q, "brightness"), g(post_q, "brightness")
        return (a < b, f"brightness {a:.1f} < {b:.1f}") if not np.isnan(a + b) else (None, "missing")
    if condition == "blur_motion":
        a, b = g(init_q, "blur_score"), g(post_q, "blur_score")
        return (a < b, f"blur_var {a:.1f} < {b:.1f}") if not np.isnan(a + b) else (None, "missing")
    if condition == "distance":
        a, b = g(init_q, "face_area_ratio"), g(post_q, "face_area_ratio")
        return (a < b, f"face_ratio {a:.3f} < {b:.3f}") if not np.isnan(a + b) else (None, "missing")
    if condition == "pose":
        a, b = abs(g(init_q, "yaw")), abs(g(post_q, "yaw"))
        if np.isnan(a + b):
            return (None, "pose proxy / yaw unavailable (limited)")
        return (a > b, f"|yaw| {a:.1f} > {b:.1f} (weak proxy)")
    return (None, "unknown condition")


def validate(args):
    records = pd.read_csv(args.records)
    clean_q = _quality_map(args.clean_quality)
    deg_q = _quality_map(args.degraded_quality)

    init_cols = {f"initial_{k}": [] for k in _QUALITY_FIELDS}
    post_cols = {f"post_{k}": [] for k in _QUALITY_FIELDS}
    quality_degraded, valid, notes = [], [], []

    for r in records.itertuples(index=False):
        init_id = str(r.initial_observation_id)
        post_id = str(r.post_action_observation_id)
        init_q = deg_q.loc[init_id].to_dict() if init_id in deg_q.index else {}
        post_q = clean_q.loc[post_id].to_dict() if post_id in clean_q.index else {}
        for k in _QUALITY_FIELDS:
            init_cols[f"initial_{k}"].append(init_q.get(k, np.nan))
            post_cols[f"post_{k}"].append(post_q.get(k, np.nan))
        ok, note = _direction_ok(r.failure_condition, init_q, post_q)
        degraded = bool(ok) if ok is not None else False
        quality_degraded.append(degraded)
        # A record is valid when quality degraded as intended (pose: not required,
        # flagged limited) and predictions exist (records were prebuilt with both).
        if r.failure_condition == "pose":
            valid.append(True)
            notes.append(f"pose limited; {note}")
        else:
            valid.append(degraded)
            notes.append(note)

    out = records.copy()
    for k, v in {**init_cols, **post_cols}.items():
        out[k] = v
    out["quality_degraded"] = quality_degraded
    out["valid"] = valid
    out["validation_note"] = notes

    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False)
    print(f"[validate] wrote {len(out)} validated records -> {out_path}")
    if len(out):
        print("[validate] quality-degraded rate by condition:")
        print(out.groupby("failure_condition")["quality_degraded"].mean().round(3).to_string())
        print(f"[validate] valid records: {int(out['valid'].sum())}/{len(out)}")
    return out_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Validate CLEAR-Face-R records.")
    p.add_argument("--records", required=True)
    p.add_argument("--clean_quality", required=True)
    p.add_argument("--degraded_quality", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    validate(parse_args())
