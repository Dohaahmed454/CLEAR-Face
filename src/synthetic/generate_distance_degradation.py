"""Generate distance degradations (shrink the face within the frame).

Simulates a small/distant face by shrinking image content onto a same-size
canvas with added background, then resampling (resolution loss).

Example:
    python src/synthetic/generate_distance_degradation.py \
        --manifest data/milestone_2/metadata/dataset_manifest.csv \
        --output_dir data/milestone_2/synthetic
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.synthetic.degradations import SEVERITIES, run_condition  # noqa: E402

CONDITION = "distance"


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate distance degradations.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", default="data/milestone_2/synthetic")
    p.add_argument("--severities", nargs="+", default=SEVERITIES, choices=SEVERITIES)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_root", default=None)
    args = p.parse_args(argv)

    df = run_condition(CONDITION, args.manifest, args.output_dir,
                       args.severities, args.seed, args.limit, args.image_root)
    out = Path(args.output_dir) / CONDITION / f"{CONDITION}_manifest.csv"
    df.to_csv(out, index=False)
    print(f"[synthetic:{CONDITION}] manifest -> {out}")


if __name__ == "__main__":
    main()
