from __future__ import annotations

import unittest

import numpy as np

from proactive_d1.core import LabelFreeChunk, LinearDecisionHead, LinearModel
from proactive_d1.internvl_features import NeuralDecisionFeatures
from proactive_d3.dialog_control_core import (
    DIALOG_POLICY_NAMES,
    build_dialog_policy_features,
)
from proactive_d4.deploy import (
    OnlineDialogPolicyState,
    dialog_stage_feature_values,
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


class OnlineDialogPolicyTest(unittest.TestCase):
    def test_matches_frozen_offline_features(self) -> None:
        dialog = [
            [{"role": "user", "text": "query"}],
            [
                {"role": "user", "text": "query"},
                {"role": "assistant", "text": "$interrupt$Do A."},
            ],
            [
                {"role": "user", "text": "query"},
                {"role": "assistant", "text": "$interrupt$Do A."},
            ],
        ]
        rows = [{"video_intervals": [[0, 1], [1, 2], [2, 3]], "dialog": dialog}]
        offline, _ = build_dialog_policy_features(rows, [_chunk(i) for i in range(3)])
        state = OnlineDialogPolicyState()
        online = np.stack([state.consume(value, index) for index, value in enumerate(dialog)])
        self.assertTrue(np.array_equal(online, offline))

    def test_feature_assembly_preserves_canonical_order(self) -> None:
        names = (
            "is_first_chunk",
            "domain=Chef",
            "tag_margin",
            "hidden_0000",
            "hidden_0001",
            *DIALOG_POLICY_NAMES,
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
            hidden_state=np.asarray([1.5, -2.0], dtype=np.float32),
            silent_log_probability=-1.0,
            interrupt_log_probability=-0.5,
            tag_margin=0.5,
            prompt_tokens=10,
            hidden_max_abs_difference=0.0,
            hidden_cosine_similarity=1.0,
        )
        dialog = np.arange(len(DIALOG_POLICY_NAMES), dtype=np.float32)
        values = dialog_stage_feature_values(
            {"is_first_chunk": 1.0, "domain=Chef": 1.0},
            neural,
            dialog,
            head,
        )
        self.assertEqual(values["tag_margin"], 0.5)
        self.assertEqual(values["hidden_0001"], -2.0)
        self.assertEqual(values[DIALOG_POLICY_NAMES[-1]], 7.0)

    def test_rejects_non_contiguous_online_chunks(self) -> None:
        state = OnlineDialogPolicyState()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            state.consume([], 1)


if __name__ == "__main__":
    unittest.main()
