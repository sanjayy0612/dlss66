"""Offline depth-rejected history baseline, selected on validation scenes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluation.temporal import reprojection, temporal_mae
from models.baseline import select_device
from training.paired import ShadowPairs
from training.train_pairs import build_model


@torch.inference_mode()
def prepare(root, split, model, device, kind):
    data = ShadowPairs(root, split, include_geometry=True)
    scenes = defaultdict(list)
    for row, (condition, reference) in zip(data.rows, data):
        model_condition = condition[:3] if kind == "shadow_refine_rgb" else condition
        predicted = model(model_condition[None].to(device))[0].cpu()
        scenes[row["scene_id"]].append((row, condition[:3], reference, predicted))
    for frames in scenes.values():
        frames.sort(key=lambda item: item[0]["frame_id"])
    return scenes


def score(root, scenes, alpha, mode="rgb"):
    if mode not in {"rgb", "residual"}:
        raise ValueError(mode)
    frame_mae = []
    temporal_errors = []
    unchanged_pixels = false_modify = false_darken = 0
    scene_results = {}
    for scene, frames in scenes.items():
        outputs = []
        for index, (row, cheap, reference, predicted) in enumerate(frames):
            output = predicted
            if index:
                grid, visible = reprojection(root, row, frames[index - 1][0])
                previous = outputs[-1] if mode == "rgb" else outputs[-1] - frames[index - 1][1]
                history = F.grid_sample(previous[None], grid, mode="bilinear",
                                        padding_mode="zeros", align_corners=False)[0]
                current = predicted if mode == "rgb" else predicted - cheap
                blended = (1 - alpha) * current + alpha * history
                output = torch.where(visible[None], blended if mode == "rgb" else
                                     torch.clamp(cheap + blended, 0, 1), predicted)
            outputs.append(output)
        scene_mae = []
        scene_temporal = []
        for index, (row, _cheap, reference, _predicted) in enumerate(frames):
            mae = (outputs[index] - reference).abs().mean().item()
            frame_mae.append(mae)
            scene_mae.append(mae)
            unchanged = (_cheap - reference).abs().amax(dim=0) <= (5 / 255)
            correction = outputs[index] - _cheap
            unchanged_pixels += unchanged.sum().item()
            false_modify += ((correction.abs().amax(dim=0) > 2 / 255) & unchanged).sum().item()
            false_darken += (((-correction).amax(dim=0) > 5 / 255) & unchanged).sum().item()
            if index + 1 < len(frames):
                next_row, _, next_reference, _ = frames[index + 1]
                grid, visible = reprojection(root, row, next_row)
                temporal = temporal_mae(outputs[index] - reference,
                                        outputs[index + 1] - next_reference, grid, visible)
                if temporal is not None:
                    temporal_errors.append(temporal)
                    scene_temporal.append(temporal)
        scene_results[str(scene)] = {"mae": float(np.mean(scene_mae)),
                                     "temporal_error": float(np.mean(scene_temporal))}
    return {"mae": float(np.mean(frame_mae)), "temporal_error": float(np.mean(temporal_errors)),
            "unchanged_modify_rate_gt_2_255": false_modify / unchanged_pixels,
            "unchanged_darken_rate_gt_5_255": false_darken / unchanged_pixels,
            "scenes": scene_results}


def run(validation_root, test_root, checkpoint, device_name, mask_bias, hard_mask=False,
        kind="shadow_refine_mask"):
    device = select_device(device_name)
    model = build_model(kind).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    model.mask_bias = mask_bias
    model.hard_mask = hard_mask
    validation = prepare(validation_root, "validation", model, device, kind)
    fresh_test = prepare(test_root, "test", model, device, kind)
    candidates = [("rgb", 0.0)] + [(mode, alpha) for mode in ("rgb", "residual")
                                   for alpha in (0.25, 0.5, 0.75)]
    validation_scores = {f"{mode}:{alpha}": score(validation_root, validation, alpha, mode)
                         for mode, alpha in candidates}
    # Equal weight on image error and reprojected error is a declared choice, not a universal metric.
    selected = min(candidates, key=lambda item: validation_scores[f"{item[0]}:{item[1]}"]["mae"] +
                   validation_scores[f"{item[0]}:{item[1]}"]["temporal_error"])
    result = {"device": str(device), "mask_bias": mask_bias,
              "hard_mask": hard_mask, "kind": kind,
              "selected_mode": selected[0], "selected_alpha": selected[1],
              "validation": validation_scores,
              "fresh_test_no_history": score(test_root, fresh_test, 0.0),
              "fresh_test_selected": score(test_root, fresh_test, selected[1], selected[0])}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-data", type=Path, default=Path("dataset/shadow_resolution"))
    parser.add_argument("--test-data", type=Path, default=Path("dataset/shadow_resolution_fresh"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/shadow_refine_mask_32.pt"))
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--mask-bias", type=float, default=1.5)
    parser.add_argument("--hard-mask", action="store_true")
    parser.add_argument("--kind", choices=("shadow_refine_mask", "shadow_refine_rgb"),
                        default="shadow_refine_mask")
    args = parser.parse_args()
    run(args.validation_data, args.test_data, args.checkpoint, args.device,
        args.mask_bias, args.hard_mask, args.kind)
