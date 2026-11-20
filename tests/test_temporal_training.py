import unittest

import torch

from training.train_temporal import temporal_residual_loss, weighted_l1
from training.train_recurrent import false_darkening_loss


class TemporalTrainingLossTests(unittest.TestCase):
    def setUp(self):
        y, x = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        self.grid = torch.stack(((2 * (x.float() + 0.5) / 4 - 1),
                                 (2 * (y.float() + 0.5) / 4 - 1)), dim=-1)[None]
        self.visible = torch.ones(1, 1, 4, 4)

    def test_temporal_loss_is_zero_for_equal_reprojected_residuals(self):
        reference = torch.zeros(1, 3, 4, 4)
        prediction = torch.full_like(reference, 0.25)
        loss = temporal_residual_loss(prediction, reference, prediction, reference,
                                      self.grid, self.visible)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_temporal_loss_detects_changed_residual(self):
        reference = torch.zeros(1, 3, 4, 4)
        source = torch.zeros_like(reference)
        target = torch.ones_like(reference) * 0.1
        loss = temporal_residual_loss(source, reference, target, reference,
                                      self.grid, self.visible)
        self.assertAlmostEqual(float(loss), 0.1, places=6)

    def test_weighted_l1_keeps_tensor_shape_and_is_finite(self):
        rgb = torch.zeros(1, 3, 4, 4)
        target = torch.zeros_like(rgb)
        target[:, :, 1, 1] = 1
        prediction = torch.full_like(rgb, 0.1)
        loss = weighted_l1(prediction, target, rgb)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_false_darkening_penalty_ignores_intended_shadow(self):
        condition = torch.full((1, 11, 1, 2), 0.5)
        reference = condition[:, :3].clone()
        reference[:, :, :, 1] = 0.3
        prediction = condition[:, :3].clone()
        prediction[:, :, :, 0] = 0.4
        prediction[:, :, :, 1] = 0.3
        self.assertAlmostEqual(float(false_darkening_loss(prediction, condition, reference)),
                               0.1 - 5 / 255, places=6)
        self.assertAlmostEqual(float(false_darkening_loss(reference, condition, reference)),
                               0.0, places=6)


if __name__ == "__main__":
    unittest.main()
