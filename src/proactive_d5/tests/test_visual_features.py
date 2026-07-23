from __future__ import annotations

import unittest

import torch

from proactive_d5.run_visual_features import current_interval_vision_state


class VisualFeaturePoolingTest(unittest.TestCase):
    def test_current_tail_mean_is_l2_normalized(self) -> None:
        features = torch.zeros((3, 2, 4), dtype=torch.float32)
        features[0] = 100.0
        features[1, :, 0] = 2.0
        features[2, :, 1] = 2.0
        state = current_interval_vision_state(features, 2)
        expected = torch.tensor([2**-0.5, 2**-0.5, 0.0, 0.0])
        torch.testing.assert_close(state, expected)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(state)), 1.0, places=6)

    def test_rejects_invalid_current_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            current_interval_vision_state(torch.ones((2, 1, 4)), 3)


if __name__ == "__main__":
    unittest.main()
