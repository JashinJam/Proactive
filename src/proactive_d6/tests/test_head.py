from __future__ import annotations

import unittest

import numpy as np

from proactive_d1.core import LabelFreeChunk
from proactive_d6.head import LabelFreeMatrix, fit_rotation_head


class HeadTest(unittest.TestCase):
    @staticmethod
    def _matrix() -> LabelFreeMatrix:
        chunks = []
        values = []
        for fold in range(5):
            for item in range(4):
                chunks.append(
                    LabelFreeChunk(
                        input_index=fold * 4 + item,
                        video_path=f"{fold}-{item}.mp4",
                        domain="D",
                        fold=fold,
                        chunk_index=0,
                        total_chunks=1,
                        interval=(0.0, 1.0),
                        raw_response="$silent$",
                        values={},
                    )
                )
                values.append([float(item), float(fold), float(item % 2)])
        return LabelFreeMatrix(
            chunks=tuple(chunks),
            values=np.asarray(values, dtype=np.float32),
            names=("a", "b", "c"),
            dialog_audit={},
        )

    def test_test_labels_are_rejected_before_prediction_freeze(self) -> None:
        matrix = self._matrix()
        labels = {
            (chunk.input_index, 0): index % 2
            for index, chunk in enumerate(matrix.chunks)
            if chunk.fold != 0
        }
        result = fit_rotation_head(
            matrix=matrix,
            labels=labels,
            test_fold=0,
            l2_weights=[0.01],
            seed=7,
            max_iterations=10,
        )
        self.assertEqual(result.test_chunks, 4)
        leaked = dict(labels)
        leaked[result.test_keys[0]] = 1
        with self.assertRaises(ValueError):
            fit_rotation_head(
                matrix=matrix,
                labels=leaked,
                test_fold=0,
                l2_weights=[0.01],
                seed=7,
                max_iterations=10,
            )


if __name__ == "__main__":
    unittest.main()

