"""Save cheap/reference/model/error strips from an existing paired dataset."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from models.baseline import select_device
from training.paired import ShadowPairs
from training.train_pairs import build_model


def run(data_root: Path, checkpoint: Path, output: Path, scenes: list[int], device_name: str,
        mask_bias: float = 0.0):
    device = select_device(device_name)
    data = ShadowPairs(data_root, "test", include_geometry=True)
    model = build_model("shadow_refine_mask").to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    model.mask_bias = mask_bias
    indices = [i for i, row in enumerate(data.rows) if row["scene_id"] in scenes and row["frame_id"] == 0]
    if len(indices) != len(scenes):
        raise ValueError("one frame-0 pair is required for each requested test scene")
    labels = ("cheap", "reference", "model", "error x4")
    width, height = 256, 144
    canvas = Image.new("RGB", (width * 4, (height + 22) * len(indices)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    with torch.inference_mode():
        for row_index, index in enumerate(indices):
            condition, target = data[index]
            predicted = model(condition.unsqueeze(0).to(device)).squeeze(0).cpu()
            cheap = condition[:3]
            images = (cheap, target, predicted, (predicted - target).abs().mul(4).clamp(0, 1))
            for column, tensor in enumerate(images):
                pixels = (tensor.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
                canvas.paste(Image.fromarray(pixels), (column * width, row_index * (height + 22) + 22))
                draw.text((column * width + 4, row_index * (height + 22) + 4),
                          f"scene {data.rows[index]['scene_id']}: {labels[column]}", fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/shadow_resolution"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/shadow_refine_mask_32.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/shadow_refine_comparison.png"))
    parser.add_argument("--scenes", type=int, nargs="+", default=[28, 29, 30, 31])
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--mask-bias", type=float, default=0.0)
    args = parser.parse_args()
    run(args.data, args.checkpoint, args.output, args.scenes, args.device, args.mask_bias)
