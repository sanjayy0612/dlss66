"""Measure model calls and host-tensor round trips; excludes Godot rendering."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.benchmark import synchronize
from models.baseline import select_device
from training.paired import ShadowPairs
from training.train_pairs import build_model


def summarize(values):
    return {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
            "p95_ms": float(np.percentile(values, 95)), "min_ms": min(values), "max_ms": max(values)}


def time_calls(call, device, warmup, samples):
    for _ in range(warmup):
        call()
    synchronize(device)
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        call()
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
    return summarize(times)


def run(data_root: Path, checkpoint: Path, device_name: str, warmup: int, samples: int):
    device = select_device(device_name)
    dataset = ShadowPairs(data_root, "test", include_geometry=True)
    cpu_input = dataset[0][0].unsqueeze(0)
    model = build_model("shadow_refine_mask").to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    gpu_input = cpu_input.to(device)
    synchronize(device)

    def model_call():
        return model(gpu_input)

    def round_trip():
        return model(cpu_input.to(device)).cpu()

    with torch.inference_mode():
        result = {"device": str(device), "resolution": [cpu_input.shape[-1], cpu_input.shape[-2]],
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "warmup": warmup, "samples": samples,
                  "model_only": time_calls(model_call, device, warmup, samples),
                  "host_tensor_to_device_model_to_host_tensor": time_calls(round_trip, device, warmup, samples)}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/shadow_resolution"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/shadow_refine_mask_32.pt"))
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup and samples must be valid")
    run(args.data, args.checkpoint, args.device, args.warmup, args.samples)
