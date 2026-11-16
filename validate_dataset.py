"""Pair and validate deterministic Godot shadow captures."""

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent / os.getenv("DLSS66_OUTPUT", "dataset/raw")


def load(variant):
    records = [json.loads(line) for line in (ROOT / f"manifest_{variant}.jsonl").read_text().splitlines()]
    default_layouts = "32" if records[0]["effect"].startswith("Shadow atlas") else "40"
    assert len(records) == int(os.getenv("DLSS66_LAYOUTS", default_layouts)) * int(os.getenv("DLSS66_FRAMES", "12")), (
        variant, len(records)
    )
    by_key = {}
    for record in records:
        key = (record["scene_id"], record["frame_id"])
        assert key not in by_key, f"duplicate {variant} frame {key}"
        assert record["variant"] == variant
        by_key[key] = record
    return by_key


cheap = load("cheap")
reference = load("reference")
assert cheap.keys() == reference.keys(), "missing paired frames"
records = []
differences = []
changed_fractions = []
valid_fractions = []
depth_ranges = []
normal_errors = []
sky_disagreements = []
splits = Counter()
for key in sorted(cheap):
    a_meta, b_meta = cheap[key], reference[key]
    for field in ("seed", "camera_position", "look_at", "effect", "resolution",
                  "camera_near", "camera_far", "light_rotation_degrees", "light_direction_view"):
        assert a_meta[field] == b_meta[field], (key, field)
    for field in ("moving_center", "moving_radius", "camera_fov"):
        assert a_meta.get(field) == b_meta.get(field), (key, field)
    cheap_atlas = a_meta.get("shadow_atlas_size", 4096)
    reference_atlas = b_meta.get("shadow_atlas_size", 4096)
    assert cheap_atlas <= reference_atlas, key
    scene = key[0]
    if a_meta["effect"].startswith("Shadow atlas"):
        split = "train" if scene < 24 else "validation" if scene < 28 else "test"
    else:
        split = ("validation" if scene in (12, 13) else "development" if scene in (14, 15)
                 else "test" if scene >= 32 else "train")
    splits[split] += 1
    a = np.asarray(Image.open(ROOT / a_meta["image"]).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(ROOT / b_meta["image"]).convert("RGB"), dtype=np.int16)
    assert a.shape == b.shape == (144, 256, 3), (key, a.shape, b.shape)
    geometry_path = ROOT / a_meta["geometry"]
    geometry = np.fromfile(geometry_path, dtype="<f4")
    assert geometry.size == 144 * 256 * 4, (key, geometry.size)
    geometry = geometry.reshape(144, 256, 4)
    assert np.isfinite(geometry).all(), key
    depth, normal = geometry[:, :, 0], geometry[:, :, 1:4]
    valid = depth > 0
    assert valid.any(), key
    assert np.all((depth >= 0) & (depth <= 1)), key
    if (~valid).any():
        assert np.max(np.abs(normal[~valid])) < 1e-6, key
    lengths = np.linalg.norm(normal[valid], axis=1)
    normal_error = float(np.max(np.abs(lengths - 1)))
    assert normal_error < 2e-4, (key, normal_error)
    near, far = a_meta["camera_near"], a_meta["camera_far"]
    light_vector = np.asarray(a_meta["light_direction_view"], dtype=np.float32)
    assert abs(np.linalg.norm(light_vector) - 1) < 1e-5, key
    meters = near * far / (near + depth[valid] * (far - near))
    assert np.isfinite(meters).all() and np.all((meters >= near) & (meters <= far)), key
    # Sky color is fixed; matching masks test cheap/reference RGB registration.
    sky_color = np.array([20, 23, 28], dtype=np.int16)
    sky_a = np.all(np.abs(a - sky_color) <= 1, axis=2)
    sky_b = np.all(np.abs(b - sky_color) <= 1, axis=2)
    sky_disagreement = float(np.mean(sky_a != sky_b))
    assert sky_disagreement < 0.005, (key, sky_disagreement)
    assert np.mean(sky_a != ~valid) < 0.005, key
    valid_fractions.append(float(valid.mean()))
    depth_ranges.append((float(meters.min()), float(meters.max())))
    normal_errors.append(normal_error)
    sky_disagreements.append(sky_disagreement)
    difference = np.abs(a - b)
    differences.append(float(difference.mean()))
    changed_fractions.append(float(np.mean(np.any(difference > 5, axis=2))))
    records.append({"scene_id": scene, "frame_id": key[1], "split": split,
                    "seed": a_meta["seed"], "camera_position": a_meta["camera_position"],
                    "look_at": a_meta["look_at"], "cheap": a_meta["image"],
                    "reference": b_meta["image"], "effect": a_meta["effect"],
                    "resolution": a_meta["resolution"], "geometry": a_meta["geometry"],
                    "camera_near": near, "camera_far": far,
                    "light_rotation_degrees": a_meta["light_rotation_degrees"],
                    "light_direction_view": a_meta["light_direction_view"],
                    **({field: a_meta[field] for field in ("moving_center", "moving_radius", "camera_fov")
                       if field in a_meta}),
                    "cheap_shadow_atlas_size": cheap_atlas,
                    "reference_shadow_atlas_size": reference_atlas})
    if len(records) == 1:
        depth_vis = np.zeros((144, 256), dtype=np.uint8)
        depth_vis[valid] = np.uint8(np.clip(meters / np.percentile(meters, 95), 0, 1) * 255)
        Image.fromarray(depth_vis).save(ROOT / "sample_depth.png")
        normal_vis = np.uint8(np.clip((normal * 0.5 + 0.5) * 255, 0, 255))
        normal_vis[~valid] = 0
        Image.fromarray(normal_vis).save(ROOT / "sample_normal.png")
assert min(differences) > 0, "at least one pair has no visible rendering change"
(ROOT / "manifest.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
summary = {"pairs": len(records), "scenes": len({r["scene_id"] for r in records}),
           "splits": dict(splits), "mean_abs_rgb_8bit": float(np.mean(differences)),
           "min_pair_mae": min(differences), "max_pair_mae": max(differences),
           "mean_fraction_pixels_changed_gt5": float(np.mean(changed_fractions)),
           "mean_valid_depth_fraction": float(np.mean(valid_fractions)),
           "depth_meters_range": [min(x[0] for x in depth_ranges), max(x[1] for x in depth_ranges)],
           "max_normal_length_error": max(normal_errors),
           "max_cheap_reference_sky_mask_disagreement": max(sky_disagreements),
           "cheap_shadow_atlas_size": records[0]["cheap_shadow_atlas_size"],
           "reference_shadow_atlas_size": records[0]["reference_shadow_atlas_size"],
           "effect": records[0]["effect"]}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("DATASET_VALIDATION", json.dumps(summary))
