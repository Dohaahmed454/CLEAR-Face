"""FER model factory and checkpoint I/O (shared by all FER scripts)."""

import json
from pathlib import Path

import torch

from src.utils.labels import CLEAR_FACE_CLASSES, label_map_dict


def get_device(prefer: str = "auto") -> "torch.device":
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def build_model(arch: str, num_classes: int, pretrained: bool = True):
    """Create a timm model. Falls back to random init if weights can't download
    (e.g. Kaggle without internet)."""
    import timm

    try:
        return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
    except Exception as exc:  # noqa: BLE001 - network/weight issues -> degrade gracefully
        if pretrained:
            print(f"[model] pretrained weights for '{arch}' unavailable ({exc}); "
                  "using random initialization.")
            return timm.create_model(arch, pretrained=False, num_classes=num_classes)
        raise


def save_checkpoint(model, output_dir, config: dict, label_map: dict | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "best_model.pth")
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map or label_map_dict(), f, indent=2)


def load_checkpoint(model_dir, device=None):
    """Load model + config + label_map saved by ``save_checkpoint``."""
    model_dir = Path(model_dir)
    with open(model_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    label_path = model_dir / "label_map.json"
    if label_path.exists():
        with open(label_path, encoding="utf-8") as f:
            label_map = json.load(f)
    else:
        label_map = label_map_dict()

    device = device or get_device()
    model = build_model(config["arch"], config["num_classes"], pretrained=False)
    state = torch.load(model_dir / "best_model.pth", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, config, label_map


def idx_to_class_from_label_map(label_map: dict) -> dict:
    """Invert a {class_name: idx} map into {idx: class_name}."""
    return {int(v): k for k, v in label_map.items()}


def default_label_map() -> dict:
    return {name: i for i, name in enumerate(CLEAR_FACE_CLASSES)}
