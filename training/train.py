"""Short synthetic-pair smoke test; not evidence of learned rendering."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.baseline import ResidualRenderer, select_device


def synthetic_pair(seed: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    rgb = np.full((3, height, width), 0.12, np.float32)
    for _ in range(5):
        x0 = int(rng.integers(0, width - 4))
        y0 = int(rng.integers(0, height - 4))
        x1 = min(width, x0 + int(rng.integers(4, max(5, width // 3))))
        y1 = min(height, y0 + int(rng.integers(4, max(5, height // 3))))
        rgb[:, y0:y1, x0:x1] = rng.uniform(0.25, 0.85, (3, 1, 1))
    x = torch.from_numpy(rgb)
    illumination = torch.linspace(0.06, 0.16, width).view(1, 1, width)
    target = (x * 0.93 + illumination).clamp(0, 1)
    return x, target


def train(steps: int, height: int, width: int, device_name: str, output: Path, seed: int = 7) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device_name)
    model = ResidualRenderer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    validation = synthetic_pair(100_001, height, width)
    initial = None
    for step in range(steps):
        batch = [synthetic_pair(seed + step * 4 + j, height, width) for j in range(4)]
        x = torch.stack([item[0] for item in batch]).to(device)
        y = torch.stack([item[1] for item in batch]).to(device)
        loss = F.l1_loss(model(x), y)
        if initial is None:
            initial = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        vx, vy = (t.unsqueeze(0).to(device) for t in validation)
        val_l1 = float(F.l1_loss(model(vx), vy).cpu())
        input_l1 = float(F.l1_loss(vx, vy).cpu())
    result = {"device": str(device), "steps": steps, "shape": [1, 3, height, width],
              "parameters": sum(p.numel() for p in model.parameters()),
              "initial_train_l1": initial, "validation_input_l1": input_l1,
              "validation_model_l1": val_l1, "seed": seed}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": {"channels": 16, "blocks": 2, "max_delta": 0.25}, "result": result}, output)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline.pt"))
    args = parser.parse_args()
    if args.steps < 1 or min(args.height, args.width) < 8:
        parser.error("steps must be positive and dimensions at least 8")
    train(args.steps, args.height, args.width, args.device, args.output)
