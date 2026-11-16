"""Scene-level check of a trained geometry-conditioned renderer."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from models.baseline import select_device
from training.paired import ShadowPairs, evaluate
from training.train_pairs import build_model


def run(checkpoint: Path, output: Path, device_name: str, data_root: Path = Path("dataset/raw"),
        kind: str = "geometry_mask", mask_bias: float = 0.0, hard_mask: bool = False):
    device = select_device(device_name)
    data = ShadowPairs(data_root, "test", include_geometry=kind in {"geometry", "geometry_mask", "shadow_refine_mask"})
    model = build_model(kind).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    if kind in {"shadow_refine_mask", "shadow_refine_rgb"}:
        model.mask_bias = mask_bias
        model.hard_mask = hard_mask
    elif mask_bias != 0 or hard_mask:
        raise ValueError("mask bias and hard mask require a masked refinement model")
    scene_scores = defaultdict(lambda: {"input": [], "model": []})
    with torch.inference_mode():
        for row, (condition, reference) in zip(data.rows, data):
            x = condition.unsqueeze(0).to(device)
            y = reference.unsqueeze(0).to(device)
            scene = row["scene_id"]
            scene_scores[scene]["input"].append((x[:, :3] - y).abs().mean().item())
            scene_scores[scene]["model"].append((model(x) - y).abs().mean().item())
    scenes = {str(scene): {"input_mae": float(np.mean(scores["input"])),
                           "model_mae": float(np.mean(scores["model"])),
                           "improved_frames": sum(m < i for m, i in zip(scores["model"], scores["input"]))}
              for scene, scores in scene_scores.items()}
    result = {"checkpoint": str(checkpoint), "device": str(device),
              "data": str(data_root), "kind": kind, "mask_bias": mask_bias,
              "hard_mask": hard_mask,
              "input": evaluate(None, data, device), "model": evaluate(model, data, device),
              "scenes": scenes, "improved_scenes": sum(v["model_mae"] < v["input_mae"] for v in scenes.values()),
              "improved_frames": sum(v["improved_frames"] for v in scenes.values())}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/shadow_mask_32.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/final_holdout_40.json"))
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--data", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--kind", choices=("geometry", "geometry_mask", "shadow_refine_mask", "shadow_refine_rgb"), default="geometry_mask")
    parser.add_argument("--mask-bias", type=float, default=0.0)
    parser.add_argument("--hard-mask", action="store_true")
    args = parser.parse_args()
    run(args.checkpoint, args.output, args.device, args.data, args.kind,
        args.mask_bias, args.hard_mask)
