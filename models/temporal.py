"""Renderer-conditioned shadow refinement with explicit reprojected history."""

import torch

from models.multiscale import ShadowRefinementRenderer


class TemporalShadowRefinementRenderer(ShadowRefinementRenderer):
    """Use current 11-channel geometry plus RGB history correction and validity.

    Input channels are current RGB/depth/normal/validity/light (11), previous
    correction reprojected to this frame (3), and a correspondence-valid mask (1).
    """

    condition_channels = 15

    def __init__(self):
        super().__init__(condition_channels=self.condition_channels)

    def forward_temporal(self, condition, history_correction, history_valid):
        if condition.ndim != 4 or condition.shape[1] != 11:
            raise ValueError("expected B×11×H×W current renderer condition")
        expected = (condition.shape[0], 3, *condition.shape[-2:])
        if history_correction.shape != expected:
            raise ValueError("history correction must be B×3×H×W")
        if history_valid.shape != (condition.shape[0], 1, *condition.shape[-2:]):
            raise ValueError("history validity must be B×1×H×W")
        temporal_condition = torch.cat((condition, history_correction, history_valid), dim=1)
        return self.forward_with_mask(temporal_condition)


def initialize_from_spatial(model: TemporalShadowRefinementRenderer, state_dict):
    """Copy spatial weights and zero the new history-input kernel slices."""
    target = model.state_dict()
    first_layer = "enc0.0.weight"
    for name, value in state_dict.items():
        if name == first_layer:
            if value.ndim != 4 or target[name].shape[0] != value.shape[0] or target[name].shape[2:] != value.shape[2:]:
                raise ValueError("spatial first-convolution shape is incompatible")
            if target[name].shape[1] < value.shape[1]:
                raise ValueError("temporal input has fewer channels than the spatial model")
            target[name].zero_()
            target[name][:, :value.shape[1]].copy_(value)
        else:
            if name not in target or target[name].shape != value.shape:
                raise ValueError(f"incompatible spatial checkpoint parameter: {name}")
            target[name].copy_(value)
    missing = set(target) - set(state_dict)
    if missing != {first_layer} and missing:
        raise ValueError(f"spatial checkpoint is missing parameters: {sorted(missing)}")
    model.load_state_dict(target)
