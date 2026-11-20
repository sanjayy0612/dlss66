"""Render paired fresh frames with recurrent model output and amplified error."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from evaluation.temporal import reprojection
from models.temporal import ConfidenceGatedTemporalRenderer, TemporalShadowRefinementRenderer
from training.paired import ShadowPairs
from training.train_recurrent import temporal_forward


def render(data_root: Path, checkpoint: Path, output: Path, frames, mask_bias=5.0):
    data = ShadowPairs(data_root, "test", include_geometry=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_type = (ConfidenceGatedTemporalRenderer
                  if state.get("kind") == "shadow_refine_temporal_confidence"
                  else TemporalShadowRefinementRenderer)
    model = model_type().eval()
    model.load_state_dict(state["model"])
    model.mask_bias = mask_bias
    model.hard_mask = True
    selected = set(frames)
    samples = []
    previous = None
    with torch.inference_mode():
        for row, (condition, reference) in zip(data.rows, data):
            if previous is None or row["scene_id"] != previous[0]["scene_id"]:
                history = torch.zeros_like(condition[:3])[None]
                valid = torch.zeros_like(condition[:1])[None]
            else:
                old_row, old_condition, old_prediction = previous
                grid, visible = reprojection(data.root, row, old_row)
                valid = visible[None, None].float()
                history = F.grid_sample(
                    (old_prediction - old_condition[:3])[None], grid,
                    mode="bilinear", padding_mode="zeros", align_corners=False) * valid
            prediction, _, _, _ = temporal_forward(model, condition[None], history, valid)
            prediction = prediction[0]
            if (row["scene_id"], row["frame_id"]) in selected:
                samples.append((row, condition[:3], prediction, reference))
            previous = row, condition, prediction
    if len(samples) != len(selected):
        raise ValueError("a requested scene/frame was not in the test dataset")
    width, height = samples[0][1].shape[-1], samples[0][1].shape[-2]
    band = 24
    canvas = Image.new("RGB", (width * 4, (height + band) * len(samples)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for index, (row, cheap, prediction, reference) in enumerate(samples):
        images = (cheap, prediction, reference,
                  (prediction - reference).abs().mul(8).clamp(0, 1))
        labels = ("cheap input", "model output", "high-quality reference", "model error x8")
        for column, (tensor, label) in enumerate(zip(images, labels)):
            pixels = (tensor.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            x, y = column * width, index * (height + band)
            canvas.paste(Image.fromarray(pixels), (x, y + band))
            draw.text((x + 5, y + 5), f"scene {row['scene_id']} frame {row['frame_id']}: {label}",
                      fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/preservation_motion_fresh"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/shadow_refine_confidence_preserved_mps.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/current_recurrent_output.png"))
    parser.add_argument("--mask-bias", type=float, default=5.0)
    parser.add_argument("--frame", nargs=2, type=int, action="append", dest="frames",
                        metavar=("SCENE", "FRAME"), required=True)
    args = parser.parse_args()
    print(render(args.data, args.checkpoint, args.output,
                 [tuple(frame) for frame in args.frames], args.mask_bias))
