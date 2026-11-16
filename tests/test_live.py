import struct
import unittest

import numpy as np

from evaluation.serve_live import GEOMETRY_BYTES, HEIGHT, RGB_BYTES, WIDTH, condition_tensor, rgb_tensor


class LiveProtocolTests(unittest.TestCase):
    def test_rgb_only_tensor_layout_and_range(self):
        rgb = bytes([128]) * RGB_BYTES
        x = rgb_tensor(rgb)
        self.assertEqual(tuple(x.shape), (1, 3, HEIGHT, WIDTH))
        self.assertTrue(x.is_contiguous())
        self.assertAlmostEqual(float(x[0, 0, 0, 0]), 128 / 255)

    def test_condition_tensor_layout_and_channels(self):
        rgb = bytes([128]) * RGB_BYTES
        geometry = np.zeros((HEIGHT, WIDTH, 4), dtype="<f4")
        geometry[0, 0, 0] = 0.1
        geometry[0, 0, 2] = 1.0
        self.assertEqual(len(geometry.tobytes()), GEOMETRY_BYTES)
        x = condition_tensor(rgb, geometry.tobytes(), struct.pack("<fff", 0, -1, 0))
        self.assertEqual(tuple(x.shape), (1, 11, HEIGHT, WIDTH))
        self.assertTrue(x.is_contiguous())
        self.assertAlmostEqual(float(x[0, 0, 0, 0]), 128 / 255)
        self.assertEqual(float(x[0, 7, 0, 0]), 1.0)
        self.assertEqual(float(x[0, 7, 1, 0]), 0.0)
        self.assertEqual(float(x[0, 9, 0, 0]), -1.0)


if __name__ == "__main__":
    unittest.main()
