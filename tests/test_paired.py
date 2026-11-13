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
