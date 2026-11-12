"""Model-only forward latency; no render, transfer, or presentation."""

import argparse
import json
import statistics
import time

import torch

from models.baseline import ResidualRenderer, select_device


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(height: int, width: int, device_name: str, warmup: int, samples: int) -> dict:
    device = select_device(device_name)
    model = ResidualRenderer().to(device).eval()
    x = torch.rand(1, 3, height, width, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        synchronize(device)
        timings = []
        for _ in range(samples):
            start = time.perf_counter()
            model(x)
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    result = {"device": str(device), "resolution": [width, height], "warmup": warmup,
              "samples": samples, "parameters": sum(p.numel() for p in model.parameters()),
              "mean_ms": statistics.mean(timings), "median_ms": statistics.median(timings),
              "p95_ms": timings[min(samples - 1, int(0.95 * samples))]}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if min(args.height, args.width, args.samples) < 1 or args.warmup < 0:
        parser.error("dimensions and samples must be positive")
    benchmark(args.height, args.width, args.device, args.warmup, args.samples)
