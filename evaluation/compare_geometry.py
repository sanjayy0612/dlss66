"""Held-out image and model-call timing for matched RGB/geometry models."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluation.compare_shadow import as_image, benchmark
from models.baseline import select_device
from training.paired import ShadowPairs
from training.train_pairs import build_model


def run(device_name: str, output: Path, geometry_checkpoint: Path = Path("artifacts/shadow_geometry_16.pt"),
        model_kind: str = "geometry") -> dict:
    device = select_device(device_name)
    rgb_data = ShadowPairs(Path("dataset/raw"), "test")
    geometry_data = ShadowPairs(Path("dataset/raw"), "test", include_geometry=True)
    rgb, target = rgb_data[0]
    condition, _ = geometry_data[0]
    rgb, target, condition = (tensor.unsqueeze(0).to(device) for tensor in (rgb, target, condition))
    rgb_model = build_model("multiscale").to(device).eval()
    geometry_model = build_model(model_kind).to(device).eval()
    rgb_model.load_state_dict(torch.load("artifacts/shadow_rgb_16.pt", map_location=device, weights_only=True)["model"])
    geometry_model.load_state_dict(torch.load(geometry_checkpoint, map_location=device, weights_only=True)["model"])
    calls = {"rgb_only": lambda: rgb_model(rgb), "geometry": lambda: geometry_model(condition)}
    timings = {name: benchmark(call, device) for name, call in calls.items()}
    with torch.inference_mode():
        rgb_output, geometry_output = (call() for call in calls.values())
    frames = [as_image(tensor) for tensor in (rgb, target, rgb_output, geometry_output)]
    for prediction in (rgb_output, geometry_output):
        error = (prediction - target).abs().mul(3).clamp(0, 1)
        frames.append(as_image(error))
    output.mkdir(parents=True, exist_ok=True)
    montage = Image.new("RGB", (frames[0].width * len(frames), frames[0].height))
    for index, frame in enumerate(frames):
        montage.paste(frame, (index * frame.width, 0))
    scene_id = rgb_data.rows[0]["scene_id"]
    montage.save(output / f"test_scene_{scene_id}_frame_0.png")
    by_scene = defaultdict(lambda: {"input": [], "rgb_only": [], "geometry": []})
    improved = 0
    with torch.inference_mode():
        for index in range(len(rgb_data)):
            x_i, y_i = rgb_data[index]
            condition_i, _ = geometry_data[index]
            x_i, y_i, condition_i = (tensor.unsqueeze(0).to(device) for tensor in (x_i, y_i, condition_i))
            scores = {"input": (x_i - y_i).abs().mean().item(),
                      "rgb_only": (rgb_model(x_i) - y_i).abs().mean().item(),
                      "geometry": (geometry_model(condition_i) - y_i).abs().mean().item()}
            scene = rgb_data.rows[index]["scene_id"]
            for name, value in scores.items():
                by_scene[scene][name].append(value)
            improved += scores["geometry"] < scores["input"]
    scene_means = {str(scene): {name: float(np.mean(values)) for name, values in scores.items()}
                   for scene, scores in by_scene.items()}
    result = {"device": str(device), "resolution": [rgb.shape[-1], rgb.shape[-2]],
              "geometry_checkpoint": str(geometry_checkpoint),
              "model_kind": model_kind,
              "montage_order": ["cheap", "reference", "rgb_model", "geometry_model",
                                "rgb_error_x3", "geometry_error_x3"], "model_only": timings,
              "test_scene_mae": scene_means, "geometry_improved_pairs": improved,
              "test_pairs": len(rgb_data)}
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/geometry_comparison"))
    parser.add_argument("--geometry-checkpoint", type=Path, default=Path("artifacts/shadow_geometry_16.pt"))
    parser.add_argument("--model-kind", choices=("geometry", "geometry_mask"), default="geometry")
    args = parser.parse_args()
    run(args.device, args.output, args.geometry_checkpoint, args.model_kind)
