"""Evaluate paired live Godot captures from ``--capture-gpu-pairs``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def evaluate(capture_dir: Path) -> dict:
    rows = [json.loads(line) for line in (capture_dir / "manifest.jsonl").read_text().splitlines()]
    if len(rows) < 2:
        raise ValueError("at least two captured frames are required")
    cheap_mae: list[float] = []
    model_mae: list[float] = []
    changed_cheap_mae: list[float] = []
    changed_model_mae: list[float] = []
    unchanged_cheap_mae: list[float] = []
    unchanged_model_mae: list[float] = []
    temporal_cheap_error: list[float] = []
    temporal_model_error: list[float] = []
    temporal_static_pixels = 0
    for index, row in enumerate(rows):
        cheap = load_rgb(capture_dir / row["cheap"])
        model = load_rgb(capture_dir / row["neural"])
        reference = load_rgb(capture_dir / row["reference"])
        cheap_error = np.abs(cheap - reference)
        model_error = np.abs(model - reference)
        cheap_mae.append(float(cheap_error.mean()))
        model_mae.append(float(model_error.mean()))
        changed = np.max(np.abs(cheap - reference), axis=2) > (5 / 255)
        if np.any(changed):
            changed_cheap_mae.append(float(cheap_error[changed].mean()))
            changed_model_mae.append(float(model_error[changed].mean()))
        unchanged = ~changed
        if np.any(unchanged):
            unchanged_cheap_mae.append(float(cheap_error[unchanged].mean()))
            unchanged_model_mae.append(float(model_error[unchanged].mean()))
        if index:
            previous = rows[index - 1]
            previous_reference = load_rgb(capture_dir / previous["reference"])
            previous_cheap = load_rgb(capture_dir / previous["cheap"])
            previous_model = load_rgb(capture_dir / previous["neural"])
            reference_delta = reference - previous_reference
            static = np.max(np.abs(reference_delta), axis=2) <= (2 / 255)
            temporal_static_pixels += int(static.sum())
            if np.any(static):
                cheap_delta_error = np.abs((cheap - previous_cheap) - reference_delta)
                model_delta_error = np.abs((model - previous_model) - reference_delta)
                temporal_cheap_error.append(float(cheap_delta_error[static].mean()))
                temporal_model_error.append(float(model_delta_error[static].mean()))
    cheap_mean = float(np.mean(cheap_mae))
    model_mean = float(np.mean(model_mae))
    return {
        "frames": len(rows),
        "resolution": list(load_rgb(capture_dir / rows[0]["cheap"]).shape[1::-1]),
        "frame_mae_rgb_0_1": {
            "cheap_to_reference": summarize(cheap_mae),
            "neural_to_reference": summarize(model_mae),
            "mean_relative_improvement_percent": 100.0 * (cheap_mean - model_mean) / max(cheap_mean, 1e-12),
        },
        "changed_shadow_region_mae_rgb_0_1": {
            "cheap": summarize(changed_cheap_mae),
            "neural": summarize(changed_model_mae),
        },
        "unchanged_region_mae_rgb_0_1": {
            "cheap": summarize(unchanged_cheap_mae),
            "neural": summarize(unchanged_model_mae),
        },
        "static_region_temporal_error_rgb_0_1": {
            "cheap": summarize(temporal_cheap_error),
            "neural": summarize(temporal_model_error),
            "static_pixel_samples": temporal_static_pixels,
            "definition": "mean abs((frame_t - frame_t-1) - reference_delta), only where reference change <= 2/255",
        },
        "limits": [
            "This evaluates a fixed camera and moving sphere in one procedural Godot scene.",
            "The neural model runs spatially each frame; it has no temporal history or motion reprojection.",
            "The reference changes the directional shadow atlas from 512 to 4096; it is a project-specific high-quality target.",
        ],
    }


def make_comparison_image(capture_dir: Path, output: Path, frame: int = 0) -> None:
    row = next(json.loads(line) for line in (capture_dir / "manifest.jsonl").read_text().splitlines()
               if json.loads(line)["frame"] == frame)
    tile_size = (512, 288)
    title_height = 48
    labels = ["Cheap 512 shadow", "Godot GPU model", "High quality 4096 reference", "Model error ×4"]
    canvas = Image.new("RGB", (tile_size[0] * 2, (tile_size[1] + title_height) * 2), (19, 23, 31))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    images = [Image.open(capture_dir / row["cheap"]).convert("RGB"),
              Image.open(capture_dir / row["neural"]).convert("RGB"),
              Image.open(capture_dir / row["reference"]).convert("RGB")]
    error = Image.fromarray(np.clip(np.abs(np.asarray(images[1], dtype=np.float32) -
                                            np.asarray(images[2], dtype=np.float32)) * 4,
                                    0, 255).astype(np.uint8))
    images.append(error)
    for index, (label, image) in enumerate(zip(labels, images)):
        x = (index % 2) * tile_size[0]
        y = (index // 2) * (tile_size[1] + title_height)
        draw.text((x + 10, y + 10), label, fill=(240, 244, 249), font=font)
        canvas.paste(image.resize(tile_size, Image.Resampling.NEAREST), (x, y + title_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=Path("artifacts/gpu_shadow_pairs"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gpu_shadow_pairs/metrics.json"))
    parser.add_argument("--comparison-image", type=Path,
                        default=Path("docs/images/neural-shadow-gpu-live-comparison.png"))
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    result = evaluate(args.capture_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    make_comparison_image(args.capture_dir, args.comparison_image, args.frame)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
