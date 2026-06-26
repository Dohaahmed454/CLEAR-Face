"""Assess which preliminary evidence signals are feasible and stable enough.

Aggregates the per-record signal table into one row per signal and classifies
each as ``usable`` / ``experimental`` / ``exclude_from_CARE`` (Milestone 2
roadmap, section 24).

Heuristic (transparent, tunable via flags):
  * feasible       = computed on >= ``min_computed_frac`` of records
  * non_degenerate = signal has real spread (std > eps) and isn't constant
  * usable         = feasible AND non_degenerate
  * experimental   = feasible but degenerate, OR partially computed
  * exclude        = essentially never computed

Note: full stability would require re-extraction under perturbation. Here
"stable" means reproducibly computable + non-degenerate distribution; this is
documented as a limitation in the evidence-signals report.

Example:
    python src/evidence_signals/validate_signal_stability.py \
        --signals experiments/milestone_2/results/preliminary_evidence_signals.csv \
        --output experiments/milestone_2/results/evidence_signal_stability.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evidence_signals.extract_preliminary_signals import SIGNAL_COLUMNS  # noqa: E402


def classify(computed_frac, std, min_frac, eps):
    feasible = computed_frac >= min_frac
    non_degenerate = (not np.isnan(std)) and (std > eps)
    if feasible and non_degenerate:
        return True, True, "usable", "Yes"
    if computed_frac > 0 and (feasible or non_degenerate):
        return non_degenerate, feasible, "experimental", "Maybe"
    return False, False, "exclude_from_CARE", "No"


def validate(args):
    df = pd.read_csv(args.signals)
    n = len(df)
    rows = []
    for sig in SIGNAL_COLUMNS:
        if sig not in df.columns:
            rows.append({"signal": sig, "computed": 0, "computed_frac": 0.0,
                         "mean": np.nan, "std": np.nan, "stable": False,
                         "feasible": False, "classification": "exclude_from_CARE",
                         "use_in_care": "No"})
            continue
        col = pd.to_numeric(df[sig], errors="coerce")
        valid = col.dropna()
        computed = int(len(valid))
        frac = computed / n if n else 0.0
        mean = float(valid.mean()) if computed else np.nan
        std = float(valid.std(ddof=0)) if computed else np.nan
        stable, feasible, classification, use = classify(
            frac, std, args.min_computed_frac, args.std_eps
        )
        rows.append({
            "signal": sig,
            "computed": computed,
            "computed_frac": round(frac, 4),
            "mean": round(mean, 4) if not np.isnan(mean) else np.nan,
            "std": round(std, 4) if not np.isnan(std) else np.nan,
            "stable": stable,
            "feasible": feasible,
            "classification": classification,
            "use_in_care": use,
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[stability] evaluated {len(out)} signals over {n} records -> {out_path}")
    print(out[["signal", "computed_frac", "std", "classification"]].to_string(index=False))
    return out_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Validate evidence-signal stability.")
    p.add_argument("--signals", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min_computed_frac", type=float, default=0.8)
    p.add_argument("--std_eps", type=float, default=1e-4)
    return p.parse_args(argv)


if __name__ == "__main__":
    validate(parse_args())
