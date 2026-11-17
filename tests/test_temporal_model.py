import unittest

import torch

from models.multiscale import ShadowRefinementRenderer
from models.temporal import TemporalShadowRefinementRenderer, initialize_from_spatial
from training.train_recurrent import blend_history_correction, temporal_input


class TemporalModelTests(unittest.TestCase):
    def test_spatial_initialization_is_identical_without_history(self):
        torch.manual_seed(4)
        spatial = ShadowRefinementRenderer().eval()
        temporal = TemporalShadowRefinementRenderer().eval()
        initialize_from_spatial(temporal, spatial.state_dict())
        condition = torch.rand(1, 11, 16, 24)
        history = torch.rand(1, 3, 16, 24)
        valid = torch.ones(1, 1, 16, 24)
        spatial_output = spatial(condition)[0]
        temporal_output, _ = temporal.forward_temporal(condition, history, valid)
        self.assertTrue(torch.allclose(spatial_output, temporal_output, atol=1e-6))

    def test_input_shapes_are_explicit(self):
        model = TemporalShadowRefinementRenderer()
        condition = torch.rand(1, 11, 16, 24)
        history = torch.zeros(1, 3, 16, 24)
        valid = torch.zeros(1, 1, 16, 24)
        prediction, logits = model.forward_temporal(condition, history, valid)
        self.assertEqual(tuple(prediction.shape), (1, 3, 16, 24))
        self.assertEqual(tuple(logits.shape), (1, 1, 16, 24))
        with self.assertRaises(ValueError):
            model.forward_temporal(condition[:, :3], history, valid)

    def test_reprojected_history_is_masked_by_visibility(self):
        rgb = torch.full((1, 3, 4, 4), 0.4)
        previous_condition = torch.cat((rgb, torch.zeros(1, 8, 4, 4)), dim=1)
        previous_prediction = rgb + 0.1
        y, x = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        grid = torch.stack((2 * (x.float() + 0.5) / 4 - 1,
                            2 * (y.float() + 0.5) / 4 - 1), dim=-1)[None]
        visible = torch.ones(1, 1, 4, 4)
        visible[:, :, 0] = 0
        history, valid = temporal_input(previous_prediction, previous_condition, grid, visible)
        self.assertTrue(torch.allclose(history[:, :, 1:], torch.full_like(history[:, :, 1:], 0.1)))
        self.assertTrue(torch.equal(history[:, :, 0], torch.zeros_like(history[:, :, 0])))
        self.assertTrue(torch.equal(valid, visible))

    def test_history_blend_changes_visible_pixels_only(self):
        rgb = torch.full((1, 3, 4, 4), 0.5)
        prediction = torch.full_like(rgb, 0.4)
        history = torch.full_like(rgb, 0.2)
        visible = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        visible[:, :, 1:, :] = True
        blended = blend_history_correction(prediction, rgb, history, visible, 1.0)
        self.assertTrue(torch.equal(blended[:, :, 0], prediction[:, :, 0]))
        self.assertTrue(torch.allclose(blended[:, :, 1:], torch.full_like(blended[:, :, 1:], 0.7)))


if __name__ == "__main__":
    unittest.main()
