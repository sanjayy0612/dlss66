"""Offline error stability with camera and optional known sphere motion."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.baseline import select_device
from training.paired import ShadowPairs
from training.train_pairs import build_model


def camera_axes(row):
    position = np.asarray(row["camera_position"], dtype=np.float32)
    forward = np.asarray(row["look_at"], dtype=np.float32) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0, 1, 0], dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return position, forward, right, up


def view_depth(root: Path, row):
    height, width = row["resolution"][1], row["resolution"][0]
    raw = np.fromfile(root / row["geometry"], dtype="<f4").reshape(height, width, 4)
    z = raw[:, :, 0]
    near, far = row["camera_near"], row["camera_far"]
    distance = np.zeros_like(z)
    valid = z > 0
    distance[valid] = near * far / (near + z[valid] * (far - near))
    return distance


def reprojection(root: Path, source, target, depth_tolerance=0.05, include_moving=False):
    """Return target grid coordinates and conservative visible-surface mask."""
    height, width = source["resolution"][1], source["resolution"][0]
    if target["resolution"] != source["resolution"]:
        raise ValueError("frame resolutions differ")
    src_depth = view_depth(root, source)
    dst_depth = view_depth(root, target)
    pos, forward, right, up = camera_axes(source)
    next_pos, next_forward, next_right, next_up = camera_axes(target)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    tan_v = np.tan(np.deg2rad(source.get("camera_fov", 75.0)) / 2)
    tan_h = tan_v * width / height
    x_view = (2 * (xx + 0.5) / width - 1) * src_depth * tan_h
    y_view = (1 - 2 * (yy + 0.5) / height) * src_depth * tan_v
    world = (pos + src_depth[..., None] * forward +
             x_view[..., None] * right + y_view[..., None] * up)
    moving = np.zeros((height, width), dtype=bool)
    if "moving_center" in source and "moving_center" in target:
        source_center = np.asarray(source["moving_center"], dtype=np.float32)
        target_center = np.asarray(target["moving_center"], dtype=np.float32)
        radius = float(source["moving_radius"])
        moving = np.linalg.norm(world - source_center, axis=2) <= radius + 0.03
        world[moving] += target_center - source_center
    relative = world - next_pos
    projected_z = np.sum(relative * next_forward, axis=2)
    projected_x = np.sum(relative * next_right, axis=2)
    projected_y = np.sum(relative * next_up, axis=2)
    safe_z = np.maximum(projected_z, 1e-6)
    target_tan_v = np.tan(np.deg2rad(target.get("camera_fov", 75.0)) / 2)
    target_tan_h = target_tan_v * width / height
    grid_x = projected_x / (safe_z * target_tan_h)
    grid_y = -projected_y / (safe_z * target_tan_v)
    grid = torch.from_numpy(np.stack((grid_x, grid_y), axis=-1)).float().unsqueeze(0)
    sampled_depth = F.grid_sample(torch.from_numpy(dst_depth).float()[None, None], grid,
                                  mode="nearest", padding_mode="zeros", align_corners=False)[0, 0].numpy()
    visible = ((src_depth > 0) & (projected_z > 0) & (np.abs(grid_x) < 0.99) &
               (np.abs(grid_y) < 0.99) & (sampled_depth > 0) &
               (np.abs(sampled_depth - projected_z) < depth_tolerance))
    if include_moving:
        return grid, torch.from_numpy(visible), torch.from_numpy(moving)
    return grid, torch.from_numpy(visible)


def temporal_mae(source_error, target_error, grid, visible):
    warped = F.grid_sample(target_error[None], grid, mode="bilinear",
                           padding_mode="zeros", align_corners=False)[0]
    if not visible.any():
        return None
    return (source_error - warped).abs()[:, visible].mean().item()


def run(data_root: Path, checkpoint: Path, device_name: str, mask_bias: float):
    device = select_device(device_name)
    dataset = ShadowPairs(data_root, "test", include_geometry=True)
    model = build_model("shadow_refine_mask").to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    model.mask_bias = mask_bias
    frame = {}
    with torch.inference_mode():
        for row, (condition, reference) in zip(dataset.rows, dataset):
            predicted = model(condition[None].to(device))[0].cpu()
            frame[(row["scene_id"], row["frame_id"])] = (row, condition[:3] - reference,
                                                            predicted - reference)
    scores = defaultdict(lambda: {"cheap": [], "model": [], "valid_fraction": [],
                                  "cheap_moving": [], "model_moving": [],
                                  "cheap_static": [], "model_static": [],
                                  "moving_fraction": []})
    for (scene_id, frame_id), (row, cheap_error, model_error) in frame.items():
        successor = frame.get((scene_id, frame_id + 1))
        if successor is None:
            continue
        next_row, next_cheap_error, next_model_error = successor
        grid, visible, moving = reprojection(data_root, row, next_row, include_moving=True)
        if not visible.any():
            continue
        scores[scene_id]["cheap"].append(temporal_mae(cheap_error, next_cheap_error, grid, visible))
        scores[scene_id]["model"].append(temporal_mae(model_error, next_model_error, grid, visible))
        scores[scene_id]["valid_fraction"].append(visible.float().mean().item())
        if "moving_center" in row:
            scores[scene_id]["moving_fraction"].append((visible & moving).float().mean().item())
            for name, error, next_error in (("cheap", cheap_error, next_cheap_error),
                                            ("model", model_error, next_model_error)):
                for region, mask in (("moving", visible & moving), ("static", visible & ~moving)):
                    value = temporal_mae(error, next_error, grid, mask)
                    if value is not None:
                        scores[scene_id][f"{name}_{region}"].append(value)
    per_scene = {}
    for scene, values in scores.items():
        entry = {"cheap_temporal_error": float(np.mean(values["cheap"])),
                 "model_temporal_error": float(np.mean(values["model"])),
                 "mean_valid_fraction": float(np.mean(values["valid_fraction"])),
                 "transitions": len(values["cheap"])}
        if values["moving_fraction"]:
            entry["mean_moving_valid_fraction"] = float(np.mean(values["moving_fraction"]))
            for name in ("cheap", "model"):
                for region in ("moving", "static"):
                    samples = values[f"{name}_{region}"]
                    entry[f"{name}_{region}_temporal_error"] = float(np.mean(samples)) if samples else None
        per_scene[str(scene)] = entry
    result = {"data": str(data_root), "device": str(device), "mask_bias": mask_bias,
              "mean_cheap_temporal_error": float(np.mean([x for v in scores.values() for x in v["cheap"]])),
              "mean_model_temporal_error": float(np.mean([x for v in scores.values() for x in v["model"]])),
              "per_scene": per_scene,
              "note": ("Known-sphere translation and smooth camera, depth-tested reprojection; "
                       "not an engine motion-vector or real-time video test." if any("moving_center" in row for row, _, _ in frame.values())
                       else "Static scenes, 30-degree camera steps, depth-tested reprojection; not a real-time video test.")}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/shadow_resolution_fresh"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/shadow_refine_mask_32.pt"))
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--mask-bias", type=float, default=1.5)
    args = parser.parse_args()
    run(args.data, args.checkpoint, args.device, args.mask_bias)
