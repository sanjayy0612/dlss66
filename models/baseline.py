"""Deterministic RGB residual baseline. This is not a diffusion model."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualRenderer(nn.Module):
    """Map B×3×H×W RGB in [0,1] to a bounded RGB correction."""

    def __init__(self, channels: int = 16, blocks: int = 2, max_delta: float = 0.25):
        super().__init__()
        if channels < 1 or blocks < 0 or not 0 < max_delta <= 1:
            raise ValueError("invalid model configuration")
        self.max_delta = max_delta
        self.head = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("expected B×3×H×W RGB")
        correction = self.max_delta * torch.tanh(self.tail(self.blocks(self.head(rgb))))
        return torch.clamp(rgb + correction, 0.0, 1.0)


def select_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        preference = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    if preference not in {"cpu", "mps", "cuda"}:
        raise ValueError(preference)
    return torch.device(preference)
