from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from proactive_d1.core import LinearDecisionHead, LinearModel
from proactive_d1.internvl_features import NeuralDecisionFeatures
from proactive_d1.neural_core import NeuralFeatureCache
from proactive_d3.core import DYNAMIC_SCALAR_NAMES, build_causal_dynamics
from proactive_d3.deploy import (
    OnlineCausalDynamicsState,
    dynamics_feature_values,
    process_session_with_dynamics_head,
)
from proactive_r0.core import CausalInferenceConfig, StarterKitSymbols


def cache(
    hidden: list[list[float]],
    margins: list[float],
    inputs: list[int],
    chunks: list[int],
) -> NeuralFeatureCache:
    rows = len(hidden)
    return NeuralFeatureCache(
        hidden_state=np.asarray(hidden, dtype=np.float32),
        tag_margin=np.asarray(margins, dtype=np.float32),
        silent_log_probability=np.full(rows, -2.0, dtype=np.float32),
        interrupt_log_probability=np.full(rows, -1.0, dtype=np.float32),
        prompt_tokens=np.full(rows, 10, dtype=np.int32),
        input_index=np.asarray(inputs, dtype=np.int32),
        chunk_index=np.asarray(chunks, dtype=np.int32),
    )


class CausalDynamicsTest(unittest.TestCase):
    def test_exact_previous_and_history_dynamics(self) -> None:
        source = cache(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]],
            [1.0, 3.0, 2.0, 5.0],
            [0, 0, 0, 1],
            [0, 1, 2, 0],
        )
        result = build_causal_dynamics(source)
        self.assertEqual(result.scalar.shape, (4, len(DYNAMIC_SCALAR_NAMES)))
        np.testing.assert_array_equal(result.scalar[0], np.zeros(8, dtype=np.float32))
        np.testing.assert_array_equal(result.hidden_delta[0], np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(result.hidden_delta[1], [-1.0, 1.0])
        self.assertEqual(result.scalar[1, 0], 1.0)
        self.assertEqual(result.scalar[1, 1], 2.0)
        self.assertEqual(result.scalar[1, 2], 2.0)
        self.assertEqual(result.scalar[1, 3], 2.0)
        self.assertAlmostEqual(float(result.scalar[1, 4]), 0.0, places=6)
        self.assertAlmostEqual(float(result.scalar[1, 5]), 1.0, places=6)
        self.assertAlmostEqual(float(result.scalar[2, 3]), 0.0, places=6)
        self.assertAlmostEqual(float(result.scalar[2, 6]), 1.0, places=6)
        np.testing.assert_array_equal(result.scalar[3], np.zeros(8, dtype=np.float32))
        np.testing.assert_array_equal(result.hidden_delta[3], np.zeros(2, dtype=np.float32))

    def test_prefix_is_invariant_to_future_rows(self) -> None:
        full = cache(
            [[1.0, 0.0], [0.5, 0.5], [99.0, -99.0]],
            [1.0, 2.0, -20.0],
            [0, 0, 0],
            [0, 1, 2],
        )
        prefix = cache(
            [[1.0, 0.0], [0.5, 0.5]],
            [1.0, 2.0],
            [0, 0],
            [0, 1],
        )
        complete = build_causal_dynamics(full)
        causal = build_causal_dynamics(prefix)
        np.testing.assert_array_equal(complete.scalar[:2], causal.scalar)
        np.testing.assert_array_equal(complete.hidden_delta[:2], causal.hidden_delta)

    def test_rejects_noncontiguous_chunks(self) -> None:
        source = cache([[1.0], [2.0]], [0.0, 1.0], [0, 0], [0, 2])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            build_causal_dynamics(source)

    def test_online_state_matches_offline_sequence_exactly(self) -> None:
        source = cache(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [1.0, 3.0, 2.0],
            [0, 0, 0],
            [0, 1, 2],
        )
        offline = build_causal_dynamics(source)
        state = OnlineCausalDynamicsState(hidden_size=2)
        for index in range(3):
            online = state.consume(source.hidden_state[index], source.tag_margin[index])
            np.testing.assert_array_equal(online.scalar, offline.scalar[index])
            np.testing.assert_array_equal(
                online.hidden_delta, offline.hidden_delta[index]
            )

    def test_dynamic_scalar_names_are_not_hidden_vector_columns(self) -> None:
        names = (
            "is_first_chunk",
            "tag_margin",
            "hidden_0000",
            "hidden_0001",
            *DYNAMIC_SCALAR_NAMES,
            "hidden_delta_0000",
            "hidden_delta_0001",
        )
        head = LinearDecisionHead(
            feature_names=names,
            model=LinearModel(
                mean=(0.0,) * len(names),
                scale=(1.0,) * len(names),
                weight=(0.0,) * len(names),
                bias=0.0,
                train_loss=0.0,
            ),
            threshold_logit=0.0,
        )
        neural = NeuralDecisionFeatures(
            hidden_state=np.asarray([1.0, 2.0], dtype=np.float32),
            silent_log_probability=-2.0,
            interrupt_log_probability=-1.0,
            tag_margin=1.0,
            prompt_tokens=10,
            hidden_max_abs_difference=0.0,
            hidden_cosine_similarity=1.0,
        )
        values, _ = dynamics_feature_values(
            {"is_first_chunk": 1.0},
            neural,
            OnlineCausalDynamicsState(hidden_size=2),
            head,
        )
        self.assertEqual(values["hidden_0000"], 1.0)
        self.assertEqual(values["hidden_0001"], 2.0)
        self.assertEqual(values["hidden_cosine_previous"], 0.0)

    def test_online_session_updates_dynamics_after_each_chunk(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.responses = iter(["Start now", "Continue"])
                self.features = iter(
                    [
                        NeuralDecisionFeatures(
                            hidden_state=np.asarray([1.0, 0.0], dtype=np.float32),
                            silent_log_probability=-2.0,
                            interrupt_log_probability=-1.0,
                            tag_margin=1.0,
                            prompt_tokens=20,
                            hidden_max_abs_difference=0.0,
                            hidden_cosine_similarity=1.0,
                        ),
                        NeuralDecisionFeatures(
                            hidden_state=np.asarray([-1.0, 0.0], dtype=np.float32),
                            silent_log_probability=-1.0,
                            interrupt_log_probability=-2.0,
                            tag_margin=-1.0,
                            prompt_tokens=24,
                            hidden_max_abs_difference=0.0,
                            hidden_cosine_similarity=1.0,
                        ),
                    ]
                )

            def generate(self, frames, messages, max_new_tokens):
                return next(self.responses)

            def extract_decision_features(self, frames, messages):
                return next(self.features)

        starter = StarterKitSymbols(
            system_prompt="system",
            normalize_dialog_turns=lambda turns: [
                {"role": "assistant", "content": str(turn["text"])} for turn in turns
            ],
            extract_frames=lambda *args, **kwargs: [object()] * 16,
        )
        names = (
            "is_first_chunk",
            "domain=Chef",
            "domain=Tutorial",
            "tag_margin",
            "hidden_0000",
            "hidden_0001",
            *DYNAMIC_SCALAR_NAMES,
            "hidden_delta_0000",
            "hidden_delta_0001",
        )
        weights = [0.0] * len(names)
        weights[names.index("hidden_delta_0000")] = -1.0
        head = LinearDecisionHead(
            feature_names=names,
            model=LinearModel(
                mean=(0.0,) * len(names),
                scale=(1.0,) * len(names),
                weight=tuple(weights),
                bias=-0.5,
                train_loss=0.0,
            ),
            threshold_logit=0.0,
        )
        row = {
            "video_path": "video.mp4",
            "video_intervals": [[0.0, 2.0], [2.0, 10.0]],
            "query": "Help",
            "task": "Task",
            "domain": "Chef",
            "dialog": [
                [{"role": "user", "text": "Help"}],
                [
                    {"role": "user", "text": "Help"},
                    {"role": "assistant", "text": "Earlier guidance"},
                ],
            ],
        }
        result = process_session_with_dynamics_head(
            row=row,
            input_index=0,
            video_folder=Path("/unused"),
            model=FakeModel(),
            starter=starter,
            config=CausalInferenceConfig(
                frames_per_interval=16,
                max_frames=32,
                max_history_turns=4,
                max_new_tokens=8,
            ),
            head=head,
            record_hidden_state=True,
        )
        self.assertEqual(
            result["prediction"]["answers"],  # type: ignore[index]
            ["$silent$", "$interrupt$Continue"],
        )
        chunks = result["chunks"]
        self.assertEqual([chunk["has_previous_chunk"] for chunk in chunks], [0.0, 1.0])
        self.assertEqual(chunks[1]["hidden_delta"], [-2.0, 0.0])


if __name__ == "__main__":
    unittest.main()
