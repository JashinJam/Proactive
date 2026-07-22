from __future__ import annotations

import copy
import unittest

import numpy as np

from proactive_d1.core import LabelFreeChunk, LabeledChunk
from proactive_d1.neural_core import NeuralFeatureCache
from proactive_d3.core import build_causal_dynamics
from proactive_d3.dialog_control_core import build_dialog_policy_features
from proactive_d5.core import (
    ACTION_HISTORY_NAMES,
    D5_VARIANTS,
    OnlineActionHistoryState,
    action_history_values,
    build_action_history_features,
    d5_matrix,
)


def _chunk(index: int, fold: int = 0) -> LabelFreeChunk:
    return LabelFreeChunk(
        input_index=0,
        video_path="video.mp4",
        domain="Cooking",
        fold=fold,
        chunk_index=index,
        total_chunks=5,
        interval=(float(index), float(index + 1)),
        raw_response="",
        values={"is_first_chunk": float(index == 0)},
    )


def _rows() -> list[dict[str, object]]:
    user = {"role": "user", "text": "help"}
    assistant_a = {"role": "assistant", "text": "$interrupt$A"}
    assistant_b = {"role": "assistant", "text": "$interrupt$B"}
    return [
        {
            "video_path": "video.mp4",
            "video_intervals": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
            "query": "help",
            "domain": "Cooking",
            "dialog": [
                [user],
                [user, assistant_a],
                [user, assistant_a],
                [user, assistant_a],
                [user, assistant_a, assistant_b],
            ],
        }
    ]


def _cache() -> NeuralFeatureCache:
    rows = 5
    return NeuralFeatureCache(
        hidden_state=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]],
            dtype=np.float32,
        ),
        tag_margin=np.asarray([0.0, 1.0, -1.0, 2.0, 1.0], dtype=np.float32),
        silent_log_probability=np.full(rows, -2.0, dtype=np.float32),
        interrupt_log_probability=np.full(rows, -1.0, dtype=np.float32),
        prompt_tokens=np.full(rows, 10, dtype=np.int32),
        input_index=np.zeros(rows, dtype=np.int32),
        chunk_index=np.arange(rows, dtype=np.int32),
    )


class ActionHistoryTest(unittest.TestCase):
    def test_frozen_action_history_values(self) -> None:
        values = action_history_values([1, 0, 0, 1])
        by_name = dict(zip(ACTION_HISTORY_NAMES, values.tolist()))
        self.assertEqual(by_name["action_lag2_interrupt"], 0.0)
        self.assertEqual(by_name["action_lag3_interrupt"], 0.0)
        self.assertEqual(by_name["action_lag4_interrupt"], 1.0)
        self.assertEqual(by_name["action_lag4_available"], 1.0)
        self.assertAlmostEqual(by_name["action_interrupt_rate_last2"], 0.5)
        self.assertAlmostEqual(by_name["action_interrupt_rate_last4"], 0.5)
        self.assertAlmostEqual(
            by_name["action_log1p_consecutive_interrupts"], np.log(2.0)
        )
        self.assertEqual(by_name["action_log1p_consecutive_silents"], 0.0)
        self.assertAlmostEqual(
            by_name["action_log1p_chunks_since_silent"], np.log(2.0)
        )
        self.assertEqual(by_name["action_last2_si"], 1.0)
        self.assertAlmostEqual(by_name["action_transition_rate_last4"], 2 / 3)

    def test_offline_builder_matches_visible_previous_actions(self) -> None:
        chunks = [_chunk(index) for index in range(5)]
        matrix, previous, audit = build_action_history_features(_rows(), chunks)
        np.testing.assert_array_equal(previous, [0, 1, 0, 0, 1])
        np.testing.assert_array_equal(matrix[0], np.zeros(len(ACTION_HISTORY_NAMES)))
        names = {name: index for index, name in enumerate(ACTION_HISTORY_NAMES)}
        self.assertEqual(matrix[2, names["action_lag2_interrupt"]], 1.0)
        self.assertEqual(matrix[3, names["action_last2_ss"]], 1.0)
        self.assertEqual(matrix[4, names["action_last2_si"]], 1.0)
        self.assertEqual(audit["assistant_additions"], 2)
        self.assertFalse(audit["labels_read"])

    def test_prefix_is_invariant_to_future_dialog(self) -> None:
        rows = _rows()
        chunks = [_chunk(index) for index in range(5)]
        original, _, _ = build_action_history_features(rows, chunks)
        changed = copy.deepcopy(rows)
        changed[0]["dialog"][3] = changed[0]["dialog"][3] + [  # type: ignore[index,operator]
            {"role": "assistant", "text": "future mutation"}
        ]
        changed[0]["dialog"][4] = changed[0]["dialog"][4] + [  # type: ignore[index,operator]
            {"role": "assistant", "text": "later mutation"}
        ]
        mutated, _, _ = build_action_history_features(changed, chunks)
        np.testing.assert_array_equal(original[:3], mutated[:3])

    def test_rejects_answers_and_decreasing_prefix(self) -> None:
        rows = _rows()
        rows[0]["answers"] = ["$silent$"] * 5
        with self.assertRaisesRegex(ValueError, "answer-stripped"):
            build_action_history_features(rows, [_chunk(index) for index in range(5)])
        state = OnlineActionHistoryState()
        state.consume([{"role": "assistant", "text": "A"}], 0)
        with self.assertRaisesRegex(ValueError, "decreased"):
            state.consume([{"role": "user", "text": "help"}], 1)

    def test_online_state_requires_contiguous_chunks(self) -> None:
        state = OnlineActionHistoryState()
        with self.assertRaisesRegex(ValueError, "expected chunk 0"):
            state.consume([], 1)


class D5MatrixTest(unittest.TestCase):
    def test_all_variants_have_unique_expected_columns(self) -> None:
        rows = _rows()
        chunks = [_chunk(index) for index in range(5)]
        examples = [LabeledChunk(chunk, index % 2) for index, chunk in enumerate(chunks)]
        cache = _cache()
        dialog, _ = build_dialog_policy_features(rows, chunks)
        action, previous, _ = build_action_history_features(rows, chunks)
        np.testing.assert_array_equal(previous, dialog[:, 1].astype(np.int8))
        dynamics = build_causal_dynamics(cache)
        expected = {
            "d4_replay": 1 + 1 + 2 + 8,
            "d4_plus_dynamic_scalar": 1 + 1 + 2 + 8 + 7,
            "d4_plus_full_dynamics": 1 + 1 + 2 + 8 + 7 + 2,
            "d4_plus_action_history": 1 + 1 + 2 + 8 + 18,
            "d4_plus_full_dynamics_history": 1 + 1 + 2 + 8 + 7 + 2 + 18,
        }
        for variant in D5_VARIANTS:
            values, names = d5_matrix(
                examples,
                cache,
                ("is_first_chunk",),
                dialog,
                dynamics,
                action,
                variant,
            )
            self.assertEqual(values.shape, (5, expected[variant]))
            self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
