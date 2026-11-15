"""Choose a continuous shadow-confidence bias on validation scenes only."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from models.baseline import select_device
from models.multiscale import ShadowRefinementRenderer
from training.paired import ShadowPairs
from training.train_pairs import build_model


@torch.inference_mode()
def collect(model, dataset, device):
    rows = []
    model.eval()
    for index, (condition, target) in enumerate(dataset):
        x = condition.unsqueeze(0).to(device)
        y = target.unsqueeze(0).to(device)
        features = model.features(x)
        logits = model.mask_head(features)
        strength = model.strength_head(features)
        correction = (model.max_delta * torch.tanh(strength) if isinstance(model, ShadowRefinementRenderer)
                      else -model.max_delta * torch.sigmoid(strength))
        rows.append((dataset.rows[index]["scene_id"], x[:, :3], y, correction, logits))
    return rows


def score(rows, bias):
    by_scene = defaultdict(list)
    improved = 0
    for scene, rgb, target, correction, logits in rows:
        calibrated_mask = torch.sigmoid(logits - bias)
        prediction = torch.clamp(rgb + calibrated_mask * correction, 0, 1)
        mae = (prediction - target).abs().mean().item()
        by_scene[scene].append(mae)
        improved += mae < (rgb - target).abs().mean().item()
    means = {str(scene): float(np.mean(values)) for scene, values in by_scene.items()}
    return {"mae": float(np.mean([value for values in by_scene.values() for value in values])),
            "scene_mae": means, "improved_pairs": improved}


def run(checkpoint: Path, device_name: str, output: Path,
        data_root: Path = Path("dataset/raw"), kind: str = "geometry_mask"):
    device = select_device(device_name)
    model = build_model(kind).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    validation = collect(model, ShadowPairs(data_root, "validation", include_geometry=True), device)
    test = collect(model, ShadowPairs(data_root, "test", include_geometry=True), device)
    candidates = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    validation_scores = {str(bias): score(validation, bias) for bias in candidates}
    chosen = min(candidates, key=lambda bias: validation_scores[str(bias)]["mae"])
    result = {"checkpoint": str(checkpoint), "data": str(data_root), "kind": kind,
              "selected_bias": chosen,
              "validation_scores": validation_scores,
              "test_at_zero_bias": score(test, 0.0),
              "test_at_selected_bias": score(test, chosen)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/shadow_mask_32.pt"))
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/mask_calibration_32.json"))
    parser.add_argument("--data", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--kind", choices=("geometry_mask", "shadow_refine_mask"), default="geometry_mask")
    args = parser.parse_args()
    run(args.checkpoint, args.device, args.output, args.data, args.kind)
