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


def score(rows, bias, hard_gate=False):
    by_scene = defaultdict(list)
    improved = 0
    unchanged_count = false_modify = false_darken = 0
    for scene, rgb, target, correction, logits in rows:
        calibrated_mask = torch.sigmoid(logits - bias)
        if hard_gate:
            calibrated_mask = calibrated_mask * (logits >= bias).to(calibrated_mask.dtype)
        prediction = torch.clamp(rgb + calibrated_mask * correction, 0, 1)
        mae = (prediction - target).abs().mean().item()
        by_scene[scene].append(mae)
        improved += mae < (rgb - target).abs().mean().item()
        unchanged = (rgb - target).abs().amax(dim=1, keepdim=True) <= (5 / 255)
        correction = prediction - rgb
        unchanged_count += unchanged.sum().item()
        false_modify += ((correction.abs().amax(dim=1, keepdim=True) > 2 / 255) & unchanged).sum().item()
        false_darken += (((-correction).amax(dim=1, keepdim=True) > 5 / 255) & unchanged).sum().item()
    means = {str(scene): float(np.mean(values)) for scene, values in by_scene.items()}
    return {"mae": float(np.mean([value for values in by_scene.values() for value in values])),
            "scene_mae": means, "improved_pairs": improved,
            "unchanged_modify_rate_gt_2_255": false_modify / unchanged_count,
            "unchanged_darken_rate_gt_5_255": false_darken / unchanged_count}


def run(checkpoint: Path, device_name: str, output: Path,
        data_root: Path = Path("dataset/raw"), kind: str = "geometry_mask",
        max_false_darken_rate: float | None = None, hard_gate: bool = False,
        test_root: Path | None = None):
    masked_kinds = {"shadow_refine_mask", "shadow_refine_rgb"}
    if hard_gate and kind not in masked_kinds:
        raise ValueError("hard gate is supported only for masked refinement models")
    device = select_device(device_name)
    model = build_model(kind).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    use_geometry = kind != "shadow_refine_rgb"
    validation = collect(model, ShadowPairs(data_root, "validation", include_geometry=use_geometry), device)
    test_root = test_root or data_root
    test = collect(model, ShadowPairs(test_root, "test", include_geometry=use_geometry), device)
    candidates = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]
    validation_scores = {str(bias): score(validation, bias, hard_gate) for bias in candidates}
    eligible = candidates
    if max_false_darken_rate is not None:
        if not 0 <= max_false_darken_rate <= 1:
            raise ValueError("max false-darkening rate must be in [0, 1]")
        eligible = [bias for bias in candidates
                    if validation_scores[str(bias)]["unchanged_darken_rate_gt_5_255"]
                    <= max_false_darken_rate]
        if not eligible:
            raise ValueError("no validation mask bias meets the false-darkening limit")
    chosen = min(eligible, key=lambda bias: validation_scores[str(bias)]["mae"])
    result = {"checkpoint": str(checkpoint), "validation_data": str(data_root),
              "test_data": str(test_root), "kind": kind,
              "selected_bias": chosen,
              "hard_gate": hard_gate,
              "selection": {"metric": "validation MAE",
                            "max_unchanged_darken_rate_gt_5_255": max_false_darken_rate},
              "validation_scores": validation_scores,
              "test_at_zero_bias": score(test, 0.0, hard_gate),
              "test_at_selected_bias": score(test, chosen, hard_gate)}
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
    parser.add_argument("--test-data", type=Path, default=None,
                        help="Optional separate held-out dataset root; its test split is never used for selection")
    parser.add_argument("--kind", choices=("geometry_mask", "shadow_refine_mask", "shadow_refine_rgb"), default="geometry_mask")
    parser.add_argument("--max-false-darken-rate", type=float, default=None,
                        help="Optional validation constraint for unchanged pixels darkened by >5/255")
    parser.add_argument("--hard-gate", action="store_true",
                        help="Set mask confidence below the bias to exactly zero at inference")
    args = parser.parse_args()
    run(args.checkpoint, args.device, args.output, args.data, args.kind,
        args.max_false_darken_rate, args.hard_gate, args.test_data)
