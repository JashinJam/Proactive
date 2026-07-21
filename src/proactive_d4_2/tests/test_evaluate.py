from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from proactive_d4_2.evaluate import (
    feature_arrays,
    length_quartiles,
    records_for_scalar_features,
    write_feature_cache,
)
from proactive_d1.neural_core import NeuralFeatureCache


def records() -> list[dict[str, object]]:
    return [
        {
            "input_index": 0,
            "video_path": "a.mp4",
            "chunks": [
                {
                    "chunk_index": 0,
                    "interval": [0.0, 1.0],
                    "model_input_frames": 8,
                    "raw_response": "Do this",
                    "r0_answer": "$silent$",
                    "hidden_state": [1.0, 2.0],
                    "tag_margin": 0.3,
                    "silent_log_probability": -0.4,
                    "interrupt_log_probability": -0.1,
                    "prompt_tokens": 7,
                },
                {
                    "chunk_index": 1,
                    "interval": [1.0, 2.0],
                    "model_input_frames": 16,
                    "raw_response": "$interrupt$Next",
                    "r0_answer": "$interrupt$Next",
                    "hidden_state": [3.0, 4.0],
                    "tag_margin": 0.5,
                    "silent_log_probability": -0.7,
                    "interrupt_log_probability": -0.2,
                    "prompt_tokens": 9,
                },
            ],
        }
    ]


class FeatureAssemblyTest(unittest.TestCase):
    def test_d4_decisions_are_not_used_as_r0_scalar_answers(self) -> None:
        value = records()
        value[0]["prediction"] = {"video_path": "a.mp4", "answers": ["$interrupt$old-head", "$silent$"]}
        projected = records_for_scalar_features(value)
        self.assertEqual(projected[0]["prediction"]["answers"], ["$silent$", "$interrupt$Next"])

    def test_cache_arrays_are_aligned_and_round_trip(self) -> None:
        arrays = feature_arrays(records(), hidden_size=2)
        self.assertEqual(arrays["hidden_state"].shape, (2, 2))
        self.assertEqual(arrays["input_index"].tolist(), [0, 0])
        self.assertEqual(arrays["chunk_index"].tolist(), [0, 1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            write_feature_cache(path, arrays)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), set(NeuralFeatureCache.__dataclass_fields__))
                np.testing.assert_array_equal(archive["hidden_state"], arrays["hidden_state"])

    def test_length_quartiles_are_label_free_and_domain_local(self) -> None:
        rows = [
            {"video_path": f"v{i}.mp4", "domain": f"d{i % 2}", "video_intervals": [[0, 1]] * (i + 1)}
            for i in range(8)
        ]
        quartiles = length_quartiles(rows)
        self.assertEqual(set(quartiles.values()), {1, 2, 3, 4})
        rows[0]["answers"] = ["$silent$"]
        with self.assertRaisesRegex(ValueError, "answer-stripped"):
            length_quartiles(rows)


if __name__ == "__main__":
    unittest.main()
