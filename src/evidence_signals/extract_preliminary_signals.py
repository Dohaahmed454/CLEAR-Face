"""Extract preliminary XAI-derived evidence signals on degraded observations.

These numeric signals are candidate CARE inputs (Milestone 2 roadmap, section
23). Each signal is computed independently and degrades gracefully to NaN if a
dependency (grad-cam / mediapipe) is unavailable, so a partial environment
still produces a usable table.

Signals (one row per record):
    au_prediction_consistency  - AU vs FER agreement (experimental: no AU model,
                                 emitted as NaN unless a detector is wired in)
    landmark_reliability       - fraction of FaceMesh landmarks detected
    attribution_concentration  - fraction of Grad-CAM mass on the face region
    background_attribution_ratio - 1 - attribution_concentration
    occlusion_support          - prob drop when the face centre is occluded
    pseudo_temporal_stability  - prediction agreement under small augmentations
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.fer.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.fer.model import (  # noqa: E402
    get_device,
    idx_to_class_from_label_map,
    load_checkpoint,
)
from src.quality.quality_metrics import QualityAnalyzer  # noqa: E402

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    def tqdm(it, **kw):  # type: ignore
        return it

SIGNAL_COLUMNS = [
    "au_prediction_consistency",
    "landmark_reliability",
    "attribution_concentration",
    "background_attribution_ratio",
    "occlusion_support",
    "pseudo_temporal_stability",
]

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(IMAGENET_STD, dtype=np.float32)


def _to_tensor(rgb_uint8, device):
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0).to(device)


class CamHelper:
    """Thin Grad-CAM wrapper that no-ops if pytorch-grad-cam is missing."""

    def __init__(self, model):
        self.ok = False
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            target = self._last_conv(model)
            self.cam = GradCAM(model=model, target_layers=[target])
            self.Target = ClassifierOutputTarget
            self.ok = True
        except Exception as exc:  # noqa: BLE001
            print(f"[signals] Grad-CAM unavailable ({exc}); attribution -> NaN")

    @staticmethod
    def _last_conv(model):
        last = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last = m
        if last is None:
            raise RuntimeError("model has no Conv2d layer for Grad-CAM")
        return last

    def heatmap(self, input_tensor, class_idx):
        grayscale = self.cam(input_tensor=input_tensor,
                             targets=[self.Target(class_idx)])
        return grayscale[0]  # (H, W) in [0, 1]


@torch.no_grad()
def _prob(model, input_tensor):
    logits = model(input_tensor)
    return torch.softmax(logits, dim=1)[0].cpu().numpy()


def _attribution_on_face(cam_helper, input_tensor, class_idx, box, size):
    if not cam_helper.ok:
        return float("nan"), float("nan")
    try:
        heat = cam_helper.heatmap(input_tensor, class_idx)
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")
    total = float(heat.sum())
    if total <= 0:
        return float("nan"), float("nan")
    if box is None:
        return 0.0, 1.0  # no face found -> all attribution counts as background
    x, y, w, h, _ = box
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(size, int(x + w)), min(size, int(y + h))
    face_mass = float(heat[y0:y1, x0:x1].sum())
    concentration = max(0.0, min(1.0, face_mass / total))
    return round(concentration, 4), round(1.0 - concentration, 4)


def _occlusion_support(model, rgb, device, class_idx, box, size):
    base = _prob(model, _to_tensor(rgb, device))[class_idx]
    occ = rgb.copy()
    if box is not None:
        x, y, w, h, _ = box
        cx, cy = int(x + w / 2), int(y + h / 2)
        half = int(0.2 * size)
    else:
        cx, cy, half = size // 2, size // 2, int(0.2 * size)
    occ[max(0, cy - half):cy + half, max(0, cx - half):cx + half] = 127
    occ_p = _prob(model, _to_tensor(occ, device))[class_idx]
    return round(float(base - occ_p), 4)


def _pseudo_temporal_stability(model, rgb, device, base_class, size):
    variants = [
        cv2.flip(rgb, 1),
        _rotate(rgb, 8),
        _rotate(rgb, -8),
        np.clip(rgb.astype(np.float32) * 1.1, 0, 255).astype(np.uint8),
        np.clip(rgb.astype(np.float32) * 0.9, 0, 255).astype(np.uint8),
    ]
    same = 0
    for v in variants:
        pred = int(_prob(model, _to_tensor(v, device)).argmax())
        same += int(pred == base_class)
    return round(same / len(variants), 4)


def _rotate(rgb, angle):
    h, w = rgb.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(rgb, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)


def extract(args):
    device = get_device(args.device)
    model, config, label_map = load_checkpoint(args.model_dir, device)
    idx_to_class = idx_to_class_from_label_map(label_map)
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    size = args.image_size or config.get("image_size", 224)

    records = pd.read_csv(args.records)
    syn = pd.read_csv(args.synthetic_manifest)
    path_by_id = dict(
        zip(syn["synthetic_image_id"].astype(str), syn["synthetic_path"].astype(str))
    )
    if args.limit and args.limit > 0:
        records = records.head(args.limit)

    qa = QualityAnalyzer(use_mediapipe=not args.no_mediapipe)
    cam_helper = CamHelper(model)
    print(f"[signals] records={len(records)} backend={qa.backend} "
          f"gradcam={'on' if cam_helper.ok else 'off'}")

    rows = []
    for r in tqdm(records.itertuples(index=False), total=len(records), desc="signals"):
        init_id = str(r.initial_observation_id)
        img_path = path_by_id.get(init_id)
        row = {"record_id": r.record_id, "initial_observation_id": init_id,
               "predicted_class": None, "confidence": np.nan}
        row.update({c: np.nan for c in SIGNAL_COLUMNS})

        bgr = cv2.imread(str(img_path)) if img_path else None
        if bgr is None:
            rows.append(row)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

        probs = _prob(model, _to_tensor(rgb, device))
        base_class = int(probs.argmax())
        row["predicted_class"] = idx_to_class.get(base_class)
        row["confidence"] = round(float(probs[base_class]), 4)

        box = qa.face_box(rgb)
        row["landmark_reliability"] = qa.landmark_reliability(rgb)
        conc, bg = _attribution_on_face(
            cam_helper, _to_tensor(rgb, device), base_class, box, size
        )
        row["attribution_concentration"] = conc
        row["background_attribution_ratio"] = bg
        try:
            row["occlusion_support"] = _occlusion_support(
                model, rgb, device, base_class, box, size
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            row["pseudo_temporal_stability"] = _pseudo_temporal_stability(
                model, rgb, device, base_class, size
            )
        except Exception:  # noqa: BLE001
            pass
        # AU consistency: no AU detector wired in -> experimental NaN.
        row["au_prediction_consistency"] = np.nan
        rows.append(row)

    qa.close()
    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[signals] wrote {len(out)} rows -> {out_path}")
    return out_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract preliminary evidence signals.")
    p.add_argument("--records", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--synthetic_manifest",
                   default="data/milestone_2/metadata/synthetic_degradation_manifest.csv",
                   help="Maps initial_observation_id -> degraded image path.")
    p.add_argument("--output", required=True)
    p.add_argument("--image_size", type=int, default=0)
    p.add_argument("--limit", type=int, default=300,
                   help="Cap records (XAI is slow); 0 = all.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no_mediapipe", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    extract(parse_args())
