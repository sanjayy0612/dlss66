"""Small paired RGB dataset and held-out-scene evaluation."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ShadowPairs(Dataset):
    def __init__(self, root: Path, split: str, include_geometry: bool = False):
        self.root = Path(root)
        rows = [json.loads(line) for line in (self.root / "manifest.jsonl").read_text().splitlines()]
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"no {split} pairs in {root}")
        self.frames = []
        self.include_geometry = include_geometry
        for row in self.rows:
            pair = []
            for key in ("cheap", "reference"):
                rgb = np.asarray(Image.open(self.root / row[key]).convert("RGB"), dtype=np.float32)
                pair.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).div_(255))
            if include_geometry:
                raw = np.fromfile(self.root / row["geometry"], dtype="<f4")
                height, width = pair[0].shape[-2:]
                if raw.size != height * width * 4:
                    raise ValueError(f"invalid geometry size: {row['geometry']}")
                geometry = raw.reshape(height, width, 4)
                device_depth = geometry[:, :, 0]
                valid = (device_depth > 0).astype(np.float32)
                near, far = row["camera_near"], row["camera_far"]
                meters = np.zeros_like(device_depth)
                meters[valid > 0] = near * far / (near + device_depth[valid > 0] * (far - near))
                depth = torch.from_numpy(np.clip(meters / 10.0, 0, 1).copy()).unsqueeze(0)
                normal = torch.from_numpy(geometry[:, :, 1:4].copy()).permute(2, 0, 1)
                mask = torch.from_numpy(valid.copy()).unsqueeze(0)
                light = torch.tensor(row["light_direction_view"], dtype=torch.float32)[:, None, None]
                light = light.expand(3, *device_depth.shape)
                pair[0] = torch.cat((pair[0], depth, normal, mask, light), dim=0)
            self.frames.append(tuple(pair))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.frames[index]


@torch.inference_mode()
def evaluate(model, dataset: ShadowPairs, device: torch.device) -> dict:
    if model is not None:
        model.eval()
    absolute = squared = changed_absolute = unchanged_absolute = 0.0
    channels = changed_channels = unchanged_channels = 0
    for cheap, reference in dataset:
        x = cheap.unsqueeze(0).to(device)
        y = reference.unsqueeze(0).to(device)
        rgb = x[:, :3]
        prediction = rgb if model is None else model(x)
        error = (prediction - y).abs()
        changed = (rgb - y).abs().amax(dim=1, keepdim=True) > (5 / 255)
        changed = changed.expand_as(error)
        absolute += error.sum().item()
        squared += error.square().sum().item()
        changed_absolute += error[changed].sum().item()
        unchanged_absolute += error[~changed].sum().item()
        channels += error.numel()
        changed_channels += changed.sum().item()
        unchanged_channels += (~changed).sum().item()
    mse = squared / channels
    return {"mae": absolute / channels, "psnr_db": -10 * np.log10(mse),
            "changed_mae": changed_absolute / changed_channels,
            "unchanged_mae": unchanged_absolute / unchanged_channels,
            "pairs": len(dataset)}
