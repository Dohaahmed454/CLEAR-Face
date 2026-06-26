"""Generate pose degradations (perspective warp simulating a yaw turn).

LIMITATION: realistic out-of-plane pose cannot be synthesized from a single
2D image. This perspective warp is a weak proxy and is flagged as limited in
the synthetic-benchmark report; pose experiments should be treated cautiously
until real trials (Milestone 2 roadmap, section 16.3).

Example:
    python src/synthetic/generate_pose_degradation.py \
        --manifest data/milestone_2/metadata/dataset_manifest.csv \
        --output_dir data/milestone_2/synthetic
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.synthetic.degradations import SEVERITIES, run_condition  # noqa: E402

CONDITION = "pose"


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate pose degradations (limited).")
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
    print(f"[synthetic:{CONDITION}] NOTE: perspective warp is a weak pose proxy.")


if __name__ == "__main__":
    main()
