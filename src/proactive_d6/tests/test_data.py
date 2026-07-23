from __future__ import annotations

import unittest

from proactive_d6.data import select_uniform_causal_frames
from proactive_r0.core import subsample_frames


class DataTest(unittest.TestCase):
    def test_frame_selection_matches_d4_uniform_policy(self) -> None:
        groups = [[f"{group}:{index}" for index in range(16)] for group in range(4)]
        selected = select_uniform_causal_frames(groups, 32)
        expected = subsample_frames([value for group in groups for value in group], 32)
        self.assertEqual(list(selected.frames), expected)
        self.assertEqual(sum(selected.current_interval_mask), 8)
        self.assertTrue(all(index <= 3 for index, _ in selected.source_indices))

    def test_current_interval_is_required(self) -> None:
        groups = [[object() for _ in range(64)], [object()]]
        with self.assertRaises(ValueError):
            select_uniform_causal_frames(groups, 1)


if __name__ == "__main__":
    unittest.main()

