"""Create an offline side-by-side video from a held-out moving scene."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

from evaluation.temporal import reprojection
from models.temporal import ConfidenceGatedTemporalRenderer, TemporalShadowRefinementRenderer
from training.paired import ShadowPairs
from training.train_recurrent import temporal_forward


def load_font(size: int):
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def frame_image(row, cheap, prediction, reference, frame_number, frame_count):
    width, height = cheap.shape[-1], cheap.shape[-2]
    scale = 2
    gutter, title_h, footer_h = 18, 42, 48
    panel_w, panel_h = width * scale, height * scale
    canvas = Image.new("RGB", (panel_w * 2 + gutter * 3,
                                title_h + panel_h * 2 + gutter * 3 + footer_h),
                       (15, 19, 26))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(20)
    body_font = load_font(15)
    labels = (
        ("CHEAP SHADOW", cheap),
        ("NEURAL REFINEMENT", prediction),
        ("HIGH-QUALITY REFERENCE", reference),
        ("MODEL ERROR ×8", (prediction - reference).abs().mul(8).clamp(0, 1)),
    )
    positions = ((gutter, title_h), (gutter * 2 + panel_w, title_h),
                 (gutter, title_h * 2 + gutter + panel_h),
                 (gutter * 2 + panel_w, title_h * 2 + gutter + panel_h))
    for (label, tensor), (x, y) in zip(labels, positions):
        draw.text((x, y - title_h + 7), label, fill=(230, 235, 244), font=title_font)
        pixels = (tensor.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(pixels).resize((panel_w, panel_h), Image.Resampling.NEAREST)
        canvas.paste(image, (x, y))

    cheap_mae = float((cheap - reference).abs().mean())
    model_mae = float((prediction - reference).abs().mean())
    footer_y = canvas.height - footer_h + 8
    draw.text((gutter, footer_y),
              f"Scene {row['scene_id']}  •  Frame {frame_number + 1}/{frame_count}  •  "
              f"Frame MAE: cheap {cheap_mae:.4f}   model {model_mae:.4f}",
              fill=(230, 235, 244), font=body_font)
    draw.text((gutter, footer_y + 20),
              "Offline held-out sequence · known-scene reprojection · not live in-game inference",
              fill=(172, 187, 204), font=body_font)
    return np.asarray(canvas, dtype=np.uint8), cheap_mae, model_mae


def render(data_root: Path, checkpoint: Path, output: Path, scene_id: int,
           start_frame: int, count: int, fps: int, mask_bias: float):
    data = ShadowPairs(data_root, "test", include_geometry=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_type = (ConfidenceGatedTemporalRenderer
                  if state.get("kind") == "shadow_refine_temporal_confidence"
                  else TemporalShadowRefinementRenderer)
    model = model_type().eval()
    model.load_state_dict(state["model"])
    model.mask_bias = mask_bias
    model.hard_mask = True

    selected = [(row, pair) for row, pair in zip(data.rows, data.frames)
                if row["scene_id"] == scene_id and
                start_frame <= row["frame_id"] < start_frame + count]
    selected.sort(key=lambda item: item[0]["frame_id"])
    if not selected:
        raise ValueError(f"no held-out frames found for scene {scene_id}")
    if [row["frame_id"] for row, _ in selected] != list(
            range(start_frame, start_frame + len(selected))):
        raise ValueError("selected scene frames are not contiguous")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "{}x{}".format(
                   4 * 256 + 3 * 18, 42 + 2 * 288 + 3 * 18 + 48),
               "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
               "-crf", "18", "-pix_fmt", "yuv420p", str(output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    scores = []
    previous = None
    try:
        with torch.inference_mode():
            for frame_number, (row, (condition, reference)) in enumerate(selected):
                if previous is None:
                    history = torch.zeros_like(condition[:3])[None]
                    valid = torch.zeros_like(condition[:1])[None]
                else:
                    old_row, old_condition, old_prediction = previous
                    grid, visible = reprojection(data.root, row, old_row)
                    valid = visible[None, None].float()
                    history = F.grid_sample(
                        (old_prediction - old_condition[:3])[None], grid,
                        mode="bilinear", padding_mode="zeros", align_corners=False) * valid
                prediction, _, _, _ = temporal_forward(
                    model, condition[None], history, valid)
                prediction = prediction[0]
                rgb, cheap_mae, model_mae = frame_image(
                    row, condition[:3], prediction, reference, frame_number, len(selected))
                assert encoder.stdin is not None
                encoder.stdin.write(rgb.tobytes())
                scores.append({"frame": row["frame_id"], "cheap_mae": cheap_mae,
                               "model_mae": model_mae})
                previous = row, condition, prediction
                print(f"frame={row['frame_id']} cheap_mae={cheap_mae:.6f} "
                      f"model_mae={model_mae:.6f}", flush=True)
    finally:
        if encoder.stdin:
            encoder.stdin.close()
    code = encoder.wait()
    if code:
        raise RuntimeError(f"ffmpeg failed with exit code {code}")
    summary = {
        "data": str(data_root), "checkpoint": str(checkpoint), "scene_id": scene_id,
        "frames": len(scores), "fps": fps,
        "mean_cheap_frame_mae": float(np.mean([x["cheap_mae"] for x in scores])),
        "mean_model_frame_mae": float(np.mean([x["model_mae"] for x in scores])),
        "note": "Offline held-out sequence inference using recorded geometry and known-scene reprojection. Not live Godot inference.",
        "per_frame": scores,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_frame"},
                     indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/expanded_motion_test"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/shadow_refine_recurrent_m2_mps_temporal10.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/neural_shadow_demo.mp4"))
    parser.add_argument("--scene", type=int, default=48)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--mask-bias", type=float, default=4.0)
    args = parser.parse_args()
    render(args.data, args.checkpoint, args.output, args.scene,
           args.start_frame, args.count, args.fps, args.mask_bias)
