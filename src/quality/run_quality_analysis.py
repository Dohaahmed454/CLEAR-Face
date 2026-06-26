"""Run image-quality analysis over a manifest and write per-image metrics.

Works on any manifest with an id + path column (clean or synthetic).

Example:
    python src/quality/run_quality_analysis.py \
        --manifest data/milestone_2/metadata/dataset_manifest.csv \
        --output experiments/milestone_2/results/quality_clean_images.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.fer.dataset import detect_columns  # noqa: E402
from src.quality.quality_metrics import QUALITY_COLUMNS, QualityAnalyzer  # noqa: E402

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    def tqdm(it, **kw):  # type: ignore
        return it


def run(args):
    df = pd.read_csv(args.manifest)
    id_col, path_col, _ = detect_columns(df)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    analyzer = QualityAnalyzer(use_mediapipe=not args.no_mediapipe)
    print(f"[quality] backend={analyzer.backend} rows={len(df)}")

    rows = []
    root = Path(args.image_root) if args.image_root else None
    for _, r in tqdm(df.iterrows(), total=len(df), desc="quality"):
        raw = str(r[path_col])
        path = root / raw if (root and not Path(raw).is_absolute()) else Path(raw)
        metrics = analyzer.analyze_path(path)
        rows.append({"image_id": r[id_col], **metrics})
    analyzer.close()

    out = pd.DataFrame(rows, columns=["image_id", *QUALITY_COLUMNS])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[quality] wrote {len(out)} rows -> {out_path}")
    n_faces = (out["face_area_ratio"] > 0).sum()
    print(f"[quality] faces detected in {n_faces}/{len(out)} images")
    return out_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run image-quality analysis.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--image_root", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no_mediapipe", action="store_true",
                   help="Force the OpenCV Haar fallback (no pose/occlusion).")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
