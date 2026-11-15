"""Measure the exact live-server tensor path without concurrent Godot rendering."""

import argparse
import json
import statistics
import struct
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluation.benchmark import synchronize
from evaluation.serve_live import condition_tensor
from models.baseline import select_device
from training.train_pairs import build_model


def summary(values):
    return {"mean_ms": statistics.mean(values), "p50_ms": statistics.median(values),
            "p95_ms": float(np.percentile(values, 95))}


def run(device_name: str, samples: int):
    root = Path("dataset/shadow_resolution")
    rows = (json.loads(line) for line in (root / "manifest_cheap.jsonl").read_text().splitlines())
    row = next(row for row in rows if row["scene_id"] == 28 and row["frame_id"] == 0)
    rgb = np.asarray(Image.open(root / row["image"]).convert("RGB")).tobytes()
    geometry = (root / row["geometry"]).read_bytes()
    light = struct.pack("<fff", *row["light_direction_view"])
    device = select_device(device_name)
    model = build_model("shadow_refine_mask").to(device).eval()
    model.load_state_dict(torch.load("artifacts/shadow_refine_mask_32.pt", map_location=device,
                                     weights_only=True)["model"])
    model.mask_bias = 1.5
    times = {"prepare": [], "transfer_model": [], "output": [], "total": []}
    with torch.inference_mode():
        for sample in range(samples + 10):
            start = time.perf_counter()
            x = condition_tensor(rgb, geometry, light)
            prepared = time.perf_counter()
            y = model(x.to(device))
            synchronize(device)
            inferred = time.perf_counter()
            _output = (y[0].cpu().permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            ended = time.perf_counter()
            if sample >= 10:
                for key, value in (("prepare", prepared - start), ("transfer_model", inferred - prepared),
                                   ("output", ended - inferred), ("total", ended - start)):
                    times[key].append(value * 1000)
    result = {"device": str(device), "samples": samples,
              "timings": {key: summary(values) for key, values in times.items()}}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    run(args.device, args.samples)
