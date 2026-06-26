"""Train the CONVENTIONAL comparison FER model.

Not the main contribution -- it exists to show the primary model is reasonable
and fairly compared (Milestone 2 roadmap, section 9). Default architecture:
ResNet-18 (a conventional CNN baseline).

Example:
    python src/fer/train_comparison_fer.py \
        --manifest data/milestone_2/metadata/dataset_manifest.csv \
        --output_dir models/fer_comparison \
        --epochs 30 --batch_size 32 --image_size 224
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.fer.trainer import train_model  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the comparison FER model.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", default="models/fer_comparison")
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--limit", type=int, default=0, help="Cap rows per split (smoke run).")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--no_class_weights", action="store_true")
    p.add_argument("--image_root", default=None, help="Prefix for relative image paths.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_model(
        arch=args.arch,
        manifest=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        limit=args.limit,
        pretrained=not args.no_pretrained,
        use_class_weights=not args.no_class_weights,
        image_root=args.image_root,
        model_label="comparison",
    )


if __name__ == "__main__":
    main()
