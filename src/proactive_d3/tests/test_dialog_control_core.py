from __future__ import annotations

import unittest

import numpy as np

from proactive_d1.core import LabelFreeChunk
from proactive_d3.dialog_control_core import (
    DIALOG_POLICY_NAMES,
    build_dialog_policy_features,
)


def _chunk(chunk_index: int) -> LabelFreeChunk:
    return LabelFreeChunk(
        input_index=0,
        video_path="v.mp4",
        domain="Chef",
        fold=0,
        chunk_index=chunk_index,
        total_chunks=3,
        interval=(float(chunk_index), float(chunk_index + 1)),
        raw_response="$silent$",
        values={},
    )


class DialogPolicyFeatureTest(unittest.TestCase):
    def test_increment_and_stage_features_are_strictly_prefix_based(self) -> None:
        rows = [
            {
                "video_intervals": [[0, 1], [1, 2], [2, 3]],
                "dialog": [
                    [{"role": "user", "text": "query"}],
                    [
                        {"role": "user", "text": "query"},
                        {"role": "assistant", "text": "$interrupt$Do A."},
                    ],
                    [
                        {"role": "user", "text": "query"},
                        {"role": "assistant", "text": "$interrupt$Do A."},
                    ],
                ],
            }
        ]
        matrix, audit = build_dialog_policy_features(rows, [_chunk(i) for i in range(3)])
        self.assertEqual(matrix.shape, (3, len(DIALOG_POLICY_NAMES)))
        self.assertEqual(matrix[:, 1].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(matrix[:, 2].tolist(), [0.0, 1.0, 0.0])
        self.assertAlmostEqual(matrix[2, 5], np.log1p(1))
        self.assertEqual(matrix[:, 7].tolist(), [0.0, 1.0, 1.0])
        self.assertFalse(audit["labels_read"])

    def test_rejects_answers_and_non_cumulative_assistant_count(self) -> None:
        rows = [
            {
                "answers": ["$silent$"],
                "video_intervals": [[0, 1]],
                "dialog": [[]],
            }
        ]
        with self.assertRaisesRegex(ValueError, "answer-stripped"):
            build_dialog_policy_features(rows, [_chunk(0)])
        decreasing = [
            {
                "video_intervals": [[0, 1], [1, 2]],
                "dialog": [
                    [{"role": "assistant", "text": "A"}],
                    [{"role": "user", "text": "query"}],
                ],
            }
        ]
        with self.assertRaisesRegex(ValueError, "decreased"):
            build_dialog_policy_features(decreasing, [_chunk(0), _chunk(1)])


if __name__ == "__main__":
    unittest.main()
