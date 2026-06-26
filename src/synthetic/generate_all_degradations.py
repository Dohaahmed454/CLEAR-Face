"""Run all degradation conditions and write the unified synthetic manifest.

Produces images under ``<output_dir>/<condition>/`` for every condition and a
single provenance manifest at
``data/milestone_2/metadata/synthetic_degradation_manifest.csv``
(Milestone 2 roadmap, sections 16.5 and 17).

Example:
    python src/synthetic/generate_all_degradations.py \
        --manifest data/milestone_2/metadata/dataset_manifest.csv \
        --output_dir data/milestone_2/synthetic
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.synthetic.degradations import CONDITIONS, SEVERITIES, run_condition  # noqa: E402
from src.utils.paths import METADATA_DIR, ensure_dir  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate all synthetic degradations.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", default="data/milestone_2/synthetic")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    p.add_argument("--severities", nargs="+", default=SEVERITIES, choices=SEVERITIES)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap source images (per condition) for fast runs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_root", default=None)
    p.add_argument("--output_manifest",
                   default=str(METADATA_DIR / "synthetic_degradation_manifest.csv"))
    args = p.parse_args(argv)

    frames = []
    for cond in args.conditions:
        df = run_condition(cond, args.manifest, args.output_dir,
                           args.severities, args.seed, args.limit, args.image_root)
        if not df.empty:
            df.to_csv(Path(args.output_dir) / cond / f"{cond}_manifest.csv", index=False)
            frames.append(df)

    if not frames:
        print("[synthetic] no degraded images produced; check the manifest paths.")
        return
    combined = pd.concat(frames, ignore_index=True)
    out_path = Path(args.output_manifest)
    ensure_dir(out_path.parent)
    combined.to_csv(out_path, index=False)
    print(f"[synthetic] unified manifest: {len(combined)} rows -> {out_path}")
    print("[synthetic] counts by condition/severity:")
    print(combined.groupby(["failure_condition", "severity"]).size().to_string())


if __name__ == "__main__":
    main()
