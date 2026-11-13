import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training.paired import ShadowPairs, evaluate
from models.multiscale import MultiScaleRenderer
from models.one_step import ConditionalNet, cosine_alphas, ddim_sample


class PairedTests(unittest.TestCase):
    def test_geometry_channels_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = np.full((8, 12, 3), 64, dtype=np.uint8)
            Image.fromarray(rgb).save(root / "cheap.png")
            Image.fromarray(rgb).save(root / "reference.png")
            geometry = np.zeros((8, 12, 4), dtype="<f4")
            geometry[2:6, 3:9, 0] = 0.1
            geometry[2:6, 3:9, 2] = 1.0
            geometry.tofile(root / "geometry.rgba32f")
            row = {"split": "test", "cheap": "cheap.png", "reference": "reference.png",
                   "geometry": "geometry.rgba32f", "camera_near": 0.05, "camera_far": 100.0,
                   "light_direction_view": [0, -1, 0]}
            (root / "manifest.jsonl").write_text(json.dumps(row) + "\n")
            x, y = ShadowPairs(root, "test", include_geometry=True)[0]
            self.assertEqual(tuple(x.shape), (11, 8, 12))
            self.assertEqual(tuple(y.shape), (3, 8, 12))
            self.assertEqual(float(x[7, 3, 4]), 1.0)  # Valid mask.
            self.assertEqual(float(x[7, 0, 0]), 0.0)
            self.assertAlmostEqual(float(x[5, 3, 4]), 1.0)  # View normal Y.
            model = MultiScaleRenderer(condition_channels=11)
            self.assertTrue(torch.equal(model(x.unsqueeze(0)), x[:3].unsqueeze(0)))

    def test_diffusion_teacher_and_student_shapes(self):
        device = torch.device("cpu")
        teacher = ConditionalNet(with_time=True)
        student = ConditionalNet(with_time=False)
        x = torch.rand(2, 3, 16, 24)
        noise = torch.randn_like(x)
        alphas = cosine_alphas(8, device)
        self.assertTrue(torch.all(alphas[1:] < alphas[:-1]))
        self.assertEqual(teacher(noise, x, torch.zeros(2)).shape, x.shape)
        self.assertEqual(student(noise, x).shape, x.shape)
        self.assertEqual(ddim_sample(teacher, x, noise, alphas).shape, x.shape)
        with self.assertRaises(ValueError):
            teacher(noise, x)

    def test_multiscale_shape_identity_and_gradient(self):
        model = MultiScaleRenderer()
        x = torch.rand(1, 3, 144, 256)
        y = model(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.equal(y, x))
        y.mean().backward()
        self.assertIsNotNone(model.tail.weight.grad)

    def test_split_and_identity_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cheap = np.zeros((8, 12, 3), dtype=np.uint8)
            reference = cheap.copy()
            reference[2:4, 3:5] = 100
            Image.fromarray(cheap).save(root / "cheap.png")
            Image.fromarray(reference).save(root / "reference.png")
            row = {"split": "test", "cheap": "cheap.png", "reference": "reference.png"}
            (root / "manifest.jsonl").write_text(json.dumps(row) + "\n")
            dataset = ShadowPairs(root, "test")
            x, y = dataset[0]
            self.assertEqual(tuple(x.shape), (3, 8, 12))
            self.assertAlmostEqual(float(y.max()), 100 / 255)
            result = evaluate(None, dataset, torch.device("cpu"))
            self.assertEqual(result["pairs"], 1)
            self.assertAlmostEqual(result["changed_mae"], 100 / 255)
            self.assertAlmostEqual(result["unchanged_mae"], 0)
            with self.assertRaises(ValueError):
                ShadowPairs(root, "train")


if __name__ == "__main__":
    unittest.main()
