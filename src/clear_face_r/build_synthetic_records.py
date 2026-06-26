"""Build synthetic CLEAR-Face-R pre/post records.

For each synthetic degraded image: ``pre`` = the degraded observation, ``post``
= the original clean image (the synthetic post-correction observation). Joins
FER predictions before/after correction into one record per pair
(Milestone 2 roadmap, section 19; schema in docs/milestone_1/schemas).

Example:
    python src/clear_face_r/build_synthetic_records.py \
        --clean_manifest data/milestone_2/metadata/dataset_manifest.csv \
        --synthetic_manifest data/milestone_2/metadata/synthetic_degradation_manifest.csv \
        --pre_predictions experiments/milestone_2/results/primary_synthetic_degraded_eval.csv \
        --post_predictions experiments/milestone_2/results/primary_internal_eval.csv \
        --output data/milestone_2/clear_face_r/synthetic_clear_face_r_records.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.labels import normalize_label  # noqa: E402
from src.utils.paths import ensure_dir  # noqa: E402

RECORD_COLUMNS = [
    "record_id",
    "original_image_id",
    "initial_observation_id",
    "post_action_observation_id",
    "source_dataset",
    "split",
    "observation_type",
    "failure_condition",
    "severity",
    "intended_action",
    "selected_action",
    "ground_truth_label",
    "initial_prediction",
    "post_prediction",
    "initial_confidence",
    "post_confidence",
    "initial_uncertainty",
    "post_uncertainty",
    "recovery",
    "risk_change",
]


def _pred_lookup(pred_csv):
    """Map image_id -> prediction row (deduplicated, last write wins)."""
    df = pd.read_csv(pred_csv)
    df = df.set_index(df["image_id"].astype(str))
    df = df[~df.index.duplicated(keep="last")]
    return df


def build(args):
    syn = pd.read_csv(args.synthetic_manifest)
    pre = _pred_lookup(args.pre_predictions)
    post = _pred_lookup(args.post_predictions)
    clean = pd.read_csv(args.clean_manifest)
    src_by_id = (
        clean.assign(_id=clean["image_id"].astype(str))
        .set_index("_id")["source_dataset"]
        .to_dict()
        if "source_dataset" in clean.columns
        else {}
    )

    rows, missing = [], 0
    for i, r in enumerate(syn.itertuples(index=False), start=1):
        syn_id = str(r.synthetic_image_id)
        orig_id = str(r.original_image_id)
        if syn_id not in pre.index or orig_id not in post.index:
            missing += 1
            continue
        pre_row = pre.loc[syn_id]
        post_row = post.loc[orig_id]
        gt = normalize_label(getattr(r, "label", None))
        init_pred = str(pre_row["prediction"])
        post_pred = str(post_row["prediction"])
        init_conf = float(pre_row["confidence"])
        post_conf = float(post_row["confidence"])
        recovery = bool((post_pred == gt) and (init_pred != gt)) if gt else False
        rows.append(
            {
                "record_id": f"rec_{i:06d}",
                "original_image_id": orig_id,
                "initial_observation_id": syn_id,
                "post_action_observation_id": orig_id,
                "source_dataset": src_by_id.get(orig_id, ""),
                "split": getattr(r, "split", ""),
                "observation_type": "synthetic",
                "failure_condition": r.failure_condition,
                "severity": r.severity,
                "intended_action": r.intended_action,
                "selected_action": r.intended_action,
                "ground_truth_label": gt,
                "initial_prediction": init_pred,
                "post_prediction": post_pred,
                "initial_confidence": round(init_conf, 6),
                "post_confidence": round(post_conf, 6),
                "initial_uncertainty": round(float(pre_row.get("entropy", np.nan)), 6),
                "post_uncertainty": round(float(post_row.get("entropy", np.nan)), 6),
                "recovery": recovery,
                "risk_change": round(post_conf - init_conf, 6),
            }
        )

    if missing:
        print(f"[records] WARNING: {missing} synthetic rows had no matching "
              "pre/post prediction (skipped).")
    out = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False)
    print(f"[records] wrote {len(out)} records -> {out_path}")
    if len(out):
        rate = out["recovery"].mean()
        print(f"[records] overall recovery rate = {rate:.3f}")
    return out_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build synthetic CLEAR-Face-R records.")
    p.add_argument("--clean_manifest", required=True)
    p.add_argument("--synthetic_manifest", required=True)
    p.add_argument("--pre_predictions", required=True,
                   help="evaluate_fer output on the degraded manifest.")
    p.add_argument("--post_predictions", required=True,
                   help="evaluate_fer output on the clean internal test split.")
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
