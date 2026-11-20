"""Measure recurrent model latency and CPU/MPS handoff at renderer resolution."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.baseline import select_device
from models.temporal import (ConfidenceGatedTemporalRenderer, TemporalShadowRefinementRenderer,
                             fuse_history)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean_ms": float(values.mean()), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "min_ms": float(values.min()),
            "max_ms": float(values.max())}


def run(checkpoint: Path, device_name="mps", width=256, height=144,
        warmup=10, samples=100, output=Path("artifacts/recurrent_mps_benchmark.json")):
    if width < 8 or height < 8 or warmup < 0 or samples < 1:
        raise ValueError("resolution must be >=8; warmup non-negative; samples positive")
    device = select_device(device_name)
    torch.manual_seed(7)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model_type = (ConfidenceGatedTemporalRenderer
                  if state.get("kind") == "shadow_refine_temporal_confidence"
                  else TemporalShadowRefinementRenderer)
    model = model_type().to(device).eval()
    model.load_state_dict(state["model"])
    model.mask_bias = 4.0
    model.hard_mask = True

    condition_cpu = torch.rand(1, 11, height, width)
    previous_rgb_cpu = torch.rand(1, 3, height, width)
    previous_output_cpu = torch.rand(1, 3, height, width)
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    grid_cpu = torch.stack((2 * (xx.float() + 0.5) / width - 1,
                            2 * (yy.float() + 0.5) / height - 1), dim=-1)[None]
    valid_cpu = torch.ones(1, 1, height, width)
    inputs = (condition_cpu.to(device), previous_rgb_cpu.to(device),
              previous_output_cpu.to(device), grid_cpu.to(device), valid_cpu.to(device))
    condition, previous_rgb, previous_output, grid, valid = inputs
    with torch.inference_mode():
        history = F.grid_sample(previous_output - previous_rgb, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False) * valid
        for _ in range(warmup):
            model.forward_temporal(condition, history, valid)
        synchronize(device)

        model_only = []
        for _ in range(samples):
            synchronize(device)
            start = time.perf_counter()
            result = model.forward_temporal(condition, history, valid)
            if len(result) == 3:
                allowed = (result[1] >= model.mask_bias).to(result[2].dtype)
                fuse_history(condition, result[0], history, valid, result[2], allowed)
            synchronize(device)
            model_only.append((time.perf_counter() - start) * 1000)

        transfer_ms, warp_model_ms, readback_ms, total_ms = [], [], [], []
        for index in range(warmup + samples):
            start = time.perf_counter()
            condition_device = condition_cpu.to(device)
            previous_rgb_device = previous_rgb_cpu.to(device)
            previous_output_device = previous_output_cpu.to(device)
            grid_device = grid_cpu.to(device)
            valid_device = valid_cpu.to(device)
            synchronize(device)
            transferred = time.perf_counter()
            history_device = F.grid_sample(previous_output_device - previous_rgb_device,
                                           grid_device, mode="bilinear",
                                           padding_mode="zeros", align_corners=False) * valid_device
            result = model.forward_temporal(condition_device, history_device, valid_device)
            if len(result) == 2:
                prediction = result[0]
            else:
                allowed = (result[1] >= model.mask_bias).to(result[2].dtype)
                prediction = fuse_history(condition_device, result[0], history_device,
                                          valid_device, result[2], allowed)
            synchronize(device)
            computed = time.perf_counter()
            prediction_cpu = prediction.cpu()
            synchronize(device)
            finished = time.perf_counter()
            if index >= warmup:
                transfer_ms.append((transferred - start) * 1000)
                warp_model_ms.append((computed - transferred) * 1000)
                readback_ms.append((finished - computed) * 1000)
                total_ms.append((finished - start) * 1000)
    result = {"device": str(device), "torch_version": torch.__version__,
              "platform": platform.platform(), "resolution": [width, height],
              "parameters": sum(p.numel() for p in model.parameters()),
              "warmup": warmup, "samples": samples,
              "model_only": summary(model_only),
              "cpu_to_device": summary(transfer_ms),
              "history_warp_plus_model": summary(warp_model_ms),
              "device_to_cpu": summary(readback_ms),
              "transfer_warp_model_readback": summary(total_ms),
              "note": "Synthetic CPU buffers and identity reprojection grid; excludes Godot rendering, geometry readback, IPC, and frame presentation."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/shadow_refine_recurrent_m2_mps_temporal10.pt"))
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/recurrent_mps_benchmark.json"))
    args = parser.parse_args()
    run(args.checkpoint, args.device, args.width, args.height, args.warmup,
        args.samples, args.output)
