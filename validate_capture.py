"""Validate the matched RGB/geometry capture and write numerical/visual outputs."""

from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent / "captures"
HEIGHT, WIDTH = 144, 256
rgb = np.asarray(Image.open(ROOT / "scene.png").convert("RGB"))
raw = np.fromfile(ROOT / "geometry.rgba32f", dtype="<f4")
assert raw.size == HEIGHT * WIDTH * 4, raw.size
geometry = raw.reshape(HEIGHT, WIDTH, 4)
assert rgb.shape == (HEIGHT, WIDTH, 3), rgb.shape
assert np.isfinite(geometry).all()

depth = geometry[:, :, 0]
normal = geometry[:, :, 1:4]
valid = depth > 0
assert valid.any() and (~valid).any()
assert np.all((depth >= 0) & (depth <= 1))
assert np.allclose(normal[~valid], 0, atol=1e-6)
lengths = np.linalg.norm(normal[valid], axis=1)
assert np.max(np.abs(lengths - 1)) < 2e-4

# Godot Forward+ uses reversed device depth. Camera near/far are Godot defaults.
near, far = 0.05, 4000.0
linear_depth = np.zeros_like(depth)
linear_depth[valid] = near * far / (near + depth[valid] * (far - near))
assert np.isfinite(linear_depth).all()

# Sky is dark gray in RGB and invalid in depth; floor occupies the bottom center.
assert not valid[0, WIDTH // 2] and valid[-1, WIDTH // 2]
assert np.linalg.norm(normal[-1, WIDTH // 2]) > 0.99

np.save(ROOT / "depth_device.npy", depth)
np.save(ROOT / "depth_meters.npy", linear_depth)
np.save(ROOT / "normal_view.npy", normal)
np.save(ROOT / "valid.npy", valid)

depth_vis = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
scale = np.percentile(linear_depth[valid], 95)
depth_vis[valid] = np.uint8(np.clip(linear_depth[valid] / scale, 0, 1) * 255)
Image.fromarray(depth_vis).save(ROOT / "depth.png")
normal_vis = np.uint8(np.clip((normal * 0.5 + 0.5) * 255, 0, 255))
normal_vis[~valid] = 0
Image.fromarray(normal_vis).save(ROOT / "normal.png")

print(f"VALIDATION rgb={rgb.shape} geometry={geometry.shape} valid={valid.sum()}/{valid.size}")
print(f"DEPTH device=[{depth[valid].min():.6f},{depth[valid].max():.6f}] meters=[{linear_depth[valid].min():.3f},{linear_depth[valid].max():.3f}]")
print(f"NORMAL length=[{lengths.min():.6f},{lengths.max():.6f}] background_zero={np.all(normal[~valid] == 0)}")
