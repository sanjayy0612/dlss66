"""Conditional pixel-space DDPM teacher and one-call distilled student."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.multiscale import conv_pair


class ConditionalNet(nn.Module):
    def __init__(self, with_time: bool):
        super().__init__()
        self.with_time = with_time
        self.enc = conv_pair(7 if with_time else 6, 16)
        self.down = conv_pair(16, 24)
        self.mid = conv_pair(24, 24)
        self.up = conv_pair(24 + 16, 16)
        self.tail = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, noisy: torch.Tensor, rgb: torch.Tensor,
                t: torch.Tensor | None = None) -> torch.Tensor:
        if noisy.shape != rgb.shape or noisy.ndim != 4 or noisy.shape[1] != 3:
            raise ValueError("expected matching B×3×H×W tensors")
        inputs = [noisy, rgb]
        if self.with_time:
            if t is None or t.shape != (rgb.shape[0],):
                raise ValueError("expected B time values")
            inputs.append(t[:, None, None, None].expand(-1, 1, *rgb.shape[-2:]))
        a = self.enc(torch.cat(inputs, dim=1))
        b = self.mid(self.down(F.avg_pool2d(a, 2)))
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.tail(self.up(torch.cat((a, b), dim=1)))


def cosine_alphas(steps: int, device: torch.device) -> torch.Tensor:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    indices = torch.arange(steps + 1, device=device, dtype=torch.float32)
    cumulative = torch.cos((indices / steps + 0.008) / 1.008 * math.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(0.0001, 0.999)
    return torch.cumprod(1 - betas, dim=0)


@torch.no_grad()
def ddim_sample(teacher: ConditionalNet, rgb: torch.Tensor, noise: torch.Tensor,
                alphas: torch.Tensor) -> torch.Tensor:
    """Deterministic DDIM path from noise to shadow residual, eta=0."""
    current = noise
    for index in reversed(range(len(alphas))):
        alpha = alphas[index]
        t = torch.full((rgb.shape[0],), index / (len(alphas) - 1), device=rgb.device)
        predicted_noise = teacher(current, rgb, t)
        clean = ((current - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)).clamp(-0.6, 0.6)
        previous_alpha = alphas[index - 1] if index else torch.ones_like(alpha)
        current = torch.sqrt(previous_alpha) * clean + torch.sqrt(1 - previous_alpha) * predicted_noise
    return current
