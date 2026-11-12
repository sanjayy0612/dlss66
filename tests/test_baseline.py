import tempfile
import unittest
from pathlib import Path

import torch

from models.baseline import ResidualRenderer, select_device
from training.train import synthetic_pair, train


class BaselineTests(unittest.TestCase):
    def test_shapes_identity_and_gradients(self):
        model = ResidualRenderer()
        for height, width in [(128, 128), (144, 256)]:
            x = torch.rand(2, 3, height, width)
            y = model(x)
            self.assertEqual(y.shape, x.shape)
            self.assertTrue(torch.equal(y, x))  # Zero-initialized tail.
            y.mean().backward()
            self.assertIsNotNone(model.tail.weight.grad)
            model.zero_grad(set_to_none=True)
        with self.assertRaises(ValueError):
            model(torch.rand(1, 4, 16, 16))

    def test_synthetic_pair_reproducible(self):
        x1, y1 = synthetic_pair(17, 32, 48)
        x2, y2 = synthetic_pair(17, 32, 48)
        self.assertTrue(torch.equal(x1, x2))
        self.assertTrue(torch.equal(y1, y2))
        self.assertFalse(torch.equal(x1, y1))

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            result = train(2, 16, 24, "cpu", path)
            state = torch.load(path, map_location="cpu", weights_only=True)
            restored = ResidualRenderer(**state["config"])
            restored.load_state_dict(state["model"])
            self.assertEqual(result["parameters"], sum(p.numel() for p in restored.parameters()))
            self.assertTrue(torch.isfinite(restored(torch.rand(1, 3, 16, 24))).all())
            self.assertEqual(select_device("cpu"), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
