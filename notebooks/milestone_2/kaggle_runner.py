"""End-to-end Milestone 2 pipeline driver (designed for Kaggle).

Runs the whole roadmap on Kaggle where the AffectNet dataset is attached at
``/kaggle/input/...`` -- so the images never need to be downloaded locally.
Each step shells out to the corresponding ``src/`` script (the same commands
documented in the roadmap), then copies GitHub-safe outputs to
``/kaggle/working/clear_face_outputs`` for download.

Usage on Kaggle (Notebook, GPU on, Internet on for pretrained weights):
    !git clone https://github.com/<you>/CLEAR-Face.git
    %cd CLEAR-Face
    !pip install -q timm grad-cam mediapipe
    !python notebooks/milestone_2/kaggle_runner.py --quick      # fast sanity pass
    !python notebooks/milestone_2/kaggle_runner.py               # fuller run

Key flags:
    --data_root       dataset root (auto-detected under /kaggle/input if omitted)
    --epochs          training epochs per model
    --train_limit     cap rows per split during training (0 = all)
    --bench_limit     cap test-split images used for the synthetic benchmark
    --signals_limit   cap records used for (slow) XAI signal extraction
    --quick           tiny end-to-end pass (few images, 1 epoch)
    --skip_train      reuse existing checkpoints in models/
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.utils.labels import is_valid_label  # noqa: E402
from src.utils.paths import (  # noqa: E402
    CLEAR_FACE_R_DIR,
    FER_COMPARISON_DIR,
    FER_PRIMARY_DIR,
    METADATA_DIR,
    RESULTS_DIR,
    SYNTHETIC_DIR,
    ensure_dir,
)

PY = sys.executable


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(REPO), check=True)


def autodetect_data_root():
    base = Path("/kaggle/input")
    if not base.exists():
        return None
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        for sub in child.rglob("*"):
            if sub.is_dir() and is_valid_label(sub.name):
                return child
    return None


def make_bench_manifest(full_manifest, out_path, limit):
    """Test-split subset used for the synthetic benchmark (held-out images)."""
    df = pd.read_csv(full_manifest)
    test = df[df["split"].astype(str) == "test"].copy()
    if limit and limit > 0 and len(test) > limit:
        test = (test.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), max(1, limit // test["label"].nunique())),
                                          random_state=42)))
    ensure_dir(Path(out_path).parent)
    test.to_csv(out_path, index=False)
    print(f"[runner] bench manifest: {len(test)} test images -> {out_path}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description="Milestone 2 end-to-end runner.")
    p.add_argument("--data_root", default=None)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--train_limit", type=int, default=0)
    p.add_argument("--bench_limit", type=int, default=400)
    p.add_argument("--signals_limit", type=int, default=200)
    p.add_argument("--primary_arch", default="mobilenetv3_large_100")
    p.add_argument("--comparison_arch", default="resnet18")
    p.add_argument("--external_manifest", default=None,
                   help="Optional RAF-DB external manifest for generalization eval.")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output_dir", default="/kaggle/working/clear_face_outputs")
    args = p.parse_args(argv)

    if args.quick:
        args.epochs = 1
        args.train_limit = args.train_limit or 200
        args.bench_limit = min(args.bench_limit, 40)
        args.signals_limit = min(args.signals_limit, 20)

    data_root = args.data_root or autodetect_data_root()
    if not data_root:
        raise SystemExit("Could not auto-detect dataset root; pass --data_root.")
    print(f"[runner] data_root = {data_root}")

    manifest = METADATA_DIR / "dataset_manifest.csv"
    bench_manifest = METADATA_DIR / "bench_test_manifest.csv"
    syn_manifest = METADATA_DIR / "synthetic_degradation_manifest.csv"
    ensure_dir(RESULTS_DIR)

    # 1. Build manifest from the attached dataset.
    run([PY, "src/utils/build_manifest.py", "--data_root", data_root,
         "--output", manifest])
    make_bench_manifest(manifest, bench_manifest, args.bench_limit)

    # 2. Train primary + comparison FER models.
    if not args.skip_train:
        run([PY, "src/fer/train_primary_fer.py", "--manifest", manifest,
             "--output_dir", FER_PRIMARY_DIR, "--arch", args.primary_arch,
             "--epochs", args.epochs, "--batch_size", args.batch_size,
             "--image_size", args.img_size, "--limit", args.train_limit,
             "--device", args.device])
        run([PY, "src/fer/train_comparison_fer.py", "--manifest", manifest,
             "--output_dir", FER_COMPARISON_DIR, "--arch", args.comparison_arch,
             "--epochs", args.epochs, "--batch_size", args.batch_size,
             "--image_size", args.img_size, "--limit", args.train_limit,
             "--device", args.device])

    # 3. Internal evaluation (test split).
    primary_internal = RESULTS_DIR / "primary_internal_eval.csv"
    run([PY, "src/fer/evaluate_fer.py", "--model_dir", FER_PRIMARY_DIR,
         "--manifest", manifest, "--split", "test", "--output", primary_internal,
         "--device", args.device])
    run([PY, "src/fer/evaluate_fer.py", "--model_dir", FER_COMPARISON_DIR,
         "--manifest", manifest, "--split", "test",
         "--output", RESULTS_DIR / "comparison_internal_eval.csv",
         "--device", args.device])

    # 3b. External evaluation (optional).
    if args.external_manifest:
        run([PY, "src/fer/evaluate_fer.py", "--model_dir", FER_PRIMARY_DIR,
             "--manifest", args.external_manifest, "--split", "test",
             "--output", RESULTS_DIR / "primary_external_eval.csv",
             "--device", args.device])

    # 4. Calibration + profiling.
    run([PY, "src/fer/calibrate_fer.py", "--model_dir", FER_PRIMARY_DIR,
         "--manifest", manifest, "--split", "val",
         "--output", RESULTS_DIR / "calibration_results.csv", "--device", args.device])
    run([PY, "src/fer/profile_fer.py", "--model_dir", FER_PRIMARY_DIR,
         "--image_size", args.img_size, "--runs", 100,
         "--output", RESULTS_DIR / "primary_profile.csv", "--device", args.device])

    # 5. Synthetic degradations (from held-out test images).
    run([PY, "src/synthetic/generate_all_degradations.py",
         "--manifest", bench_manifest, "--output_dir", SYNTHETIC_DIR,
         "--output_manifest", syn_manifest])

    # 6. Evaluate FER on degraded images.
    primary_degraded = RESULTS_DIR / "primary_synthetic_degraded_eval.csv"
    run([PY, "src/fer/evaluate_fer.py", "--model_dir", FER_PRIMARY_DIR,
         "--manifest", syn_manifest, "--split", "all",
         "--output", primary_degraded, "--device", args.device])
    run([PY, "src/fer/evaluate_fer.py", "--model_dir", FER_COMPARISON_DIR,
         "--manifest", syn_manifest, "--split", "all",
         "--output", RESULTS_DIR / "comparison_synthetic_degraded_eval.csv",
         "--device", args.device])

    # 7. Quality analysis before/after correction.
    quality_clean = RESULTS_DIR / "quality_clean_images.csv"
    quality_degraded = RESULTS_DIR / "quality_synthetic_degraded_images.csv"
    run([PY, "src/quality/run_quality_analysis.py", "--manifest", bench_manifest,
         "--output", quality_clean])
    run([PY, "src/quality/run_quality_analysis.py", "--manifest", syn_manifest,
         "--output", quality_degraded])

    # 8. Build + validate CLEAR-Face-R records.
    records = CLEAR_FACE_R_DIR / "synthetic_clear_face_r_records.csv"
    records_valid = CLEAR_FACE_R_DIR / "synthetic_clear_face_r_records_validated.csv"
    run([PY, "src/clear_face_r/build_synthetic_records.py",
         "--clean_manifest", bench_manifest, "--synthetic_manifest", syn_manifest,
         "--pre_predictions", primary_degraded, "--post_predictions", primary_internal,
         "--output", records])
    run([PY, "src/clear_face_r/validate_records.py", "--records", records,
         "--clean_quality", quality_clean, "--degraded_quality", quality_degraded,
         "--output", records_valid])

    # 9. Preliminary evidence signals + stability.
    signals = RESULTS_DIR / "preliminary_evidence_signals.csv"
    run([PY, "src/evidence_signals/extract_preliminary_signals.py",
         "--records", records_valid, "--model_dir", FER_PRIMARY_DIR,
         "--synthetic_manifest", syn_manifest, "--output", signals,
         "--limit", args.signals_limit, "--device", args.device])
    run([PY, "src/evidence_signals/validate_signal_stability.py",
         "--signals", signals,
         "--output", RESULTS_DIR / "evidence_signal_stability.csv"])

    # 10. Collect GitHub-safe outputs (CSVs + reports + configs, no images).
    out_dir = ensure_dir(Path(args.output_dir))
    for src_dir, pattern in [(RESULTS_DIR, "*.csv"), (RESULTS_DIR, "*.json"),
                             (METADATA_DIR, "*.csv"), (CLEAR_FACE_R_DIR, "*.csv"),
                             (REPO / "docs/milestone_2/reports", "*.md")]:
        for f in Path(src_dir).glob(pattern):
            shutil.copy2(f, out_dir / f.name)
    for model_dir in (FER_PRIMARY_DIR, FER_COMPARISON_DIR):
        for name in ("config.json", "training_log.csv", "label_map.json"):
            f = Path(model_dir) / name
            if f.exists():
                shutil.copy2(f, out_dir / f"{Path(model_dir).name}_{name}")
    print(f"\n[runner] DONE. GitHub-safe outputs copied to {out_dir}")
    print("[runner] Download checkpoints (best_model.pth) separately if needed.")


if __name__ == "__main__":
    main()
