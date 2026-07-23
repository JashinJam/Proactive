from __future__ import annotations

import unittest

import torch
import numpy as np

from proactive_d1.core import LabelFreeChunk, LabeledChunk
from proactive_d5.temporal import (
    VisualTemporalResidual,
    parameter_count,
    temporal_residual_oof,
)


class VisualTemporalResidualTest(unittest.TestCase):
    def test_parameter_count_is_frozen(self) -> None:
        self.assertEqual(parameter_count(VisualTemporalResidual()), 39073)

    def test_future_input_does_not_change_past_logits(self) -> None:
        torch.manual_seed(7)
        model = VisualTemporalResidual(input_width=4, hidden_width=3)
        torch.nn.init.normal_(model.output.weight)
        prefix = torch.randn(3, 4)
        first = model(prefix)
        extended = model(torch.cat([prefix, torch.randn(2, 4)], dim=0))
        torch.testing.assert_close(first, extended[:3])

    def test_small_session_oof_covers_every_chunk(self) -> None:
        examples = []
        for input_index in range(6):
            for chunk_index in range(2):
                examples.append(
                    LabeledChunk(
                        feature=LabelFreeChunk(
                            input_index=input_index,
                            video_path=f"v{input_index}.mp4",
                            domain="test",
                            fold=input_index % 3,
                            chunk_index=chunk_index,
                            total_chunks=2,
                            interval=(float(chunk_index), float(chunk_index + 1)),
                            raw_response="$silent$",
                            values={},
                        ),
                        gold_interrupt=chunk_index,
                    )
                )
        base = np.asarray(
            [[example.gold_interrupt, example.feature.chunk_index] for example in examples],
            dtype=np.float64,
        )
        rng = np.random.default_rng(7)
        vision = rng.normal(size=(len(examples), 1024)).astype(np.float32)
        vision /= np.linalg.norm(vision, axis=1, keepdims=True)
        decisions, logits, details = temporal_residual_oof(
            examples,
            base,
            vision,
            folds=3,
            calibration_fold_offset=1,
            base_config={
                "l2_weights": [0.01],
                "seed": 7,
                "max_iterations": 5,
                "l2_reduction": "sum",
            },
            temporal_config={
                "seed": 7,
                "parameters": 39073,
                "learning_rate": 0.0003,
                "weight_decay": 0.01,
                "max_epochs": 2,
                "calibration_loss_patience": 1,
                "gradient_norm_clip": 1.0,
            },
            device="cpu",
        )
        expected = {example.key for example in examples}
        self.assertEqual(set(decisions), expected)
        self.assertEqual(set(logits), expected)
        self.assertEqual(len(details), 3)


if __name__ == "__main__":
    unittest.main()
