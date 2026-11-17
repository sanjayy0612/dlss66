"""Evaluate full recurrent scene rollouts, including feedback drift."""

import argparse
import json
from pathlib import Path

import torch

from models.baseline import select_device
from models.temporal import TemporalShadowRefinementRenderer
from training.paired import ShadowPairs
from training.train_recurrent import evaluate_rollouts


def run(checkpoint: Path, data_root: Path, split: str, output: Path,
        device_name="mps", mask_bias=4.0, hard_mask=True,
        history_blend_alpha=0.0):
    device = select_device(device_name)
    dataset = ShadowPairs(data_root, split, include_geometry=True)
    model = TemporalShadowRefinementRenderer().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    metrics = evaluate_rollouts(model, dataset, device, mask_bias, hard_mask,
                                history_blend_alpha=history_blend_alpha)
    result = {"checkpoint": str(checkpoint), "data": str(data_root), "split": split,
              "device": str(device), "mask_bias": mask_bias,
              "hard_mask": hard_mask, "history_blend_alpha": history_blend_alpha,
              "metrics": metrics,
              "note": "Prior predicted RGB corrections are reprojected and fed back frame by frame. "
                      "Motion uses dataset metadata and depth-tested reprojection, not engine motion vectors."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/shadow_refine_recurrent_32.pt"))
    parser.add_argument("--data", type=Path, default=Path("dataset/motion_test"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/recurrent_motion_test.json"))
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--mask-bias", type=float, default=4.0)
    parser.add_argument("--history-blend-alpha", type=float, default=0.0)
    parser.add_argument("--no-hard-mask", action="store_true")
    args = parser.parse_args()
    run(args.checkpoint, args.data, args.split, args.output, args.device,
        args.mask_bias, not args.no_hard_mask, args.history_blend_alpha)
