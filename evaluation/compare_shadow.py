"""Compare trained shadow models on one held-out frame and time model calls."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluation.benchmark import synchronize
from models.baseline import select_device
from models.one_step import ConditionalNet, cosine_alphas, ddim_sample
from training.paired import ShadowPairs
from training.train_pairs import build_model


def benchmark(call, device: torch.device, warmup: int = 10, samples: int = 30) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            call()
        synchronize(device)
        times = []
        for _ in range(samples):
            start = time.perf_counter()
            call()
            synchronize(device)
            times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return {"mean_ms": statistics.mean(times), "median_ms": statistics.median(times),
            "p95_ms": times[min(samples - 1, int(0.95 * samples))]}


def as_image(tensor: torch.Tensor) -> Image.Image:
    rgb = tensor.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.uint8(np.rint(rgb * 255)))


def run(device_name: str, output: Path) -> dict:
    device = select_device(device_name)
    data = ShadowPairs(Path("dataset/raw"), "test")
    x, y = data[0]
    x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
    z = torch.randn(x.shape, generator=torch.Generator().manual_seed(9000)).to(device)
    cnn = build_model("cnn").to(device).eval()
    multi = build_model("multiscale").to(device).eval()
    cnn.load_state_dict(torch.load("artifacts/shadow_cnn.pt", map_location=device, weights_only=True)["model"])
    multi.load_state_dict(torch.load("artifacts/shadow_multiscale.pt", map_location=device, weights_only=True)["model"])
    state = torch.load("artifacts/shadow_one_step.pt", map_location=device, weights_only=True)
    teacher = ConditionalNet(True).to(device).eval()
    student = ConditionalNet(False).to(device).eval()
    teacher.load_state_dict(state["teacher"])
    student.load_state_dict(state["student"])
    alphas = cosine_alphas(state["diffusion_steps"], device)
    calls = {"cnn": lambda: cnn(x), "multiscale": lambda: multi(x),
             "teacher_8_step": lambda: (x + ddim_sample(teacher, x, z, alphas)).clamp(0, 1),
             "student_1_step": lambda: (x + 0.6 * torch.tanh(student(z, x))).clamp(0, 1)}
    timings = {name: benchmark(call, device) for name, call in calls.items()}
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        frames = [as_image(x), as_image(y)] + [as_image(call()) for call in calls.values()]
    montage = Image.new("RGB", (frames[0].width * len(frames), frames[0].height))
    for index, frame in enumerate(frames):
        montage.paste(frame, (index * frame.width, 0))
    montage.save(output / "test_scene_7_frame_0.png")
    result = {"device": str(device), "resolution": [x.shape[-1], x.shape[-2]],
              "montage_order": ["cheap", "reference", *calls], "model_only": timings}
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/shadow_comparison"))
    args = parser.parse_args()
    run(args.device, args.output)
