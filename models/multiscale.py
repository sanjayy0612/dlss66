"""Small multiscale RGB residual model for spatially extended shadows."""

import torch
from torch import nn
from torch.nn import functional as F


def conv_pair(source: int, target: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(source, target, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(target, target, 3, padding=1), nn.ReLU(inplace=True))


class MultiScaleRenderer(nn.Module):
    def __init__(self, max_delta: float = 0.6, condition_channels: int = 3):
        super().__init__()
        if not 0 < max_delta <= 1 or condition_channels < 3:
            raise ValueError("invalid model configuration")
        self.max_delta = max_delta
        self.condition_channels = condition_channels
        self.enc0 = conv_pair(condition_channels, 12)
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

    def features(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 4 or condition.shape[1] != self.condition_channels:
            raise ValueError("unexpected conditioning tensor")
        a = self.enc0(condition)
        b = self.enc1(F.avg_pool2d(a, 2))
        c = self.enc2(F.avg_pool2d(b, 2))
        d = self.mid(self.enc3(F.avg_pool2d(c, 2)))
        c = self.dec2(torch.cat((F.interpolate(d, size=c.shape[-2:], mode="bilinear", align_corners=False), c), 1))
        b = self.dec1(torch.cat((F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b), 1))
        a = self.dec0(torch.cat((F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False), a), 1))
        return a

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        rgb = condition[:, :3]
        return torch.clamp(rgb + self.max_delta * torch.tanh(self.tail(self.features(condition))), 0, 1)


class MaskedShadowRenderer(MultiScaleRenderer):
    """Predict shadow location and darkness separately; output can only darken RGB."""

    def __init__(self):
        super().__init__(max_delta=0.6, condition_channels=11)
        self.tail = nn.Identity()
        self.mask_head = nn.Conv2d(12, 1, 3, padding=1)
        self.strength_head = nn.Conv2d(12, 3, 3, padding=1)
        for head in (self.mask_head, self.strength_head):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -2.0)

    def forward_with_mask(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(condition)
        logits = self.mask_head(features)
        mask = torch.sigmoid(logits)
        strength = torch.sigmoid(self.strength_head(features))
        rgb = condition[:, :3]
        output = torch.clamp(rgb - self.max_delta * mask * strength, 0, 1)
        return output, logits

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.forward_with_mask(condition)[0]


class ShadowRefinementRenderer(MaskedShadowRenderer):
    """Gated signed RGB correction for an existing low-resolution shadow cue."""

    def __init__(self):
        super().__init__()
        nn.init.zeros_(self.strength_head.bias)
        self.mask_bias = 0.0

    def forward_with_mask(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(condition)
        logits = self.mask_head(features)
        mask = torch.sigmoid(logits - self.mask_bias)
        correction = self.max_delta * mask * torch.tanh(self.strength_head(features))
        return torch.clamp(condition[:, :3] + correction, 0, 1), logits
