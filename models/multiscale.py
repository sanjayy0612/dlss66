"""Small multiscale RGB residual model for spatially extended shadows."""

import torch
from torch import nn
from torch.nn import functional as F


def conv_pair(source: int, target: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(source, target, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(target, target, 3, padding=1), nn.ReLU(inplace=True))


class MultiScaleRenderer(nn.Module):
    def __init__(self, max_delta: float = 0.6):
        super().__init__()
        if not 0 < max_delta <= 1:
            raise ValueError("invalid max_delta")
        self.max_delta = max_delta
        self.enc0 = conv_pair(3, 12)
        self.enc1 = conv_pair(12, 16)
        self.enc2 = conv_pair(16, 24)
        self.enc3 = conv_pair(24, 32)
        self.mid = conv_pair(32, 32)
        self.dec2 = conv_pair(32 + 24, 24)
        self.dec1 = conv_pair(24 + 16, 16)
        self.dec0 = conv_pair(16 + 12, 12)
        self.tail = nn.Conv2d(12, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("expected B×3×H×W RGB")
        a = self.enc0(rgb)
        b = self.enc1(F.avg_pool2d(a, 2))
        c = self.enc2(F.avg_pool2d(b, 2))
        d = self.mid(self.enc3(F.avg_pool2d(c, 2)))
        c = self.dec2(torch.cat((F.interpolate(d, size=c.shape[-2:], mode="bilinear", align_corners=False), c), 1))
        b = self.dec1(torch.cat((F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b), 1))
        a = self.dec0(torch.cat((F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False), a), 1))
        return torch.clamp(rgb + self.max_delta * torch.tanh(self.tail(a)), 0, 1)
