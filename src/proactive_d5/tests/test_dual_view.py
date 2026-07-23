from __future__ import annotations

import unittest

import numpy as np

from proactive_d5.dual_view import build_dual_view_matrices


class DualViewFeaturesTest(unittest.TestCase):
    def test_shared_and_dialog_gated_differences(self) -> None:
        names = ("tag_margin", *(f"hidden_{index:04d}" for index in range(1024)), "assistant_added_since_previous")
        uniform = np.zeros((2, len(names)), dtype=np.float64)
        multiscale = uniform.copy()
        multiscale[:, :1025] = 2.0
        uniform[1, -1] = 1.0
        multiscale[1, -1] = 1.0
        matrices = build_dual_view_matrices(
            uniform, multiscale, names, gate_feature="assistant_added_since_previous"
        )
        shared, shared_names = matrices["shared_delta"]
        gated, gated_names = matrices["dialog_gated_delta"]
        self.assertEqual(shared.shape, (2, len(names) + 1025))
        self.assertEqual(gated.shape, (2, len(names) + 2050))
        np.testing.assert_array_equal(shared[:, -1025:], 2.0)
        np.testing.assert_array_equal(gated[0, len(names) : len(names) + 1025], 2.0)
        np.testing.assert_array_equal(gated[0, -1025:], 0.0)
        np.testing.assert_array_equal(gated[1, len(names) : len(names) + 1025], 0.0)
        np.testing.assert_array_equal(gated[1, -1025:], 2.0)
        self.assertEqual(len(shared_names), shared.shape[1])
        self.assertEqual(len(gated_names), gated.shape[1])

    def test_rejects_non_binary_gate(self) -> None:
        names = ("tag_margin", *(f"hidden_{index:04d}" for index in range(1024)), "gate")
        uniform = np.zeros((1, len(names)))
        uniform[0, -1] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            build_dual_view_matrices(uniform, uniform, names, gate_feature="gate")


if __name__ == "__main__":
    unittest.main()
