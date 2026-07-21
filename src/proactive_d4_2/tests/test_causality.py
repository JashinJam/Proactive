from __future__ import annotations

import unittest

import numpy as np

from proactive_d1.internvl_features import NeuralDecisionFeatures
from proactive_d4_2.run_features import ReusedComponentModel, sanitize_inference_rows


class _Model:
    device = "cpu"

    def __init__(self) -> None:
        self.generations = 0
        self.features = 0

    def generate(self, frames, messages, max_new_tokens):
        self.generations += 1
        return "$interrupt$new"

    def extract_decision_features(self, frames, messages):
        self.features += 1
        return NeuralDecisionFeatures(
            hidden_state=np.asarray([5.0, 6.0], dtype=np.float32),
            silent_log_probability=-0.4,
            interrupt_log_probability=-0.1,
            tag_margin=0.3,
            prompt_tokens=9,
            hidden_max_abs_difference=0.0,
            hidden_cosine_similarity=1.0,
            extraction_mode="shared_vision",
            candidate_forward_passes=2,
        )


def _reference() -> dict[str, object]:
    return {
        "chunks": [
            {
                "raw_response": "$silent$cached",
                "hidden_state": [1.0, 2.0],
                "silent_log_probability": -0.6,
                "interrupt_log_probability": -0.2,
                "tag_margin": 0.4,
                "prompt_tokens": 7,
                "hidden_max_abs_difference": 0.0,
                "hidden_cosine_similarity": 1.0,
                "decision_feature_mode": "shared_vision",
                "candidate_forward_passes": 2,
            }
        ]
    }


class CausalityTest(unittest.TestCase):
    def test_answers_are_removed_without_mutating_source(self) -> None:
        source = [{"video_path": "x.mp4", "answers": ["$interrupt$gold"]}]
        sanitized = sanitize_inference_rows(source)
        self.assertNotIn("answers", sanitized[0])
        self.assertIn("answers", source[0])

    def test_cached_generation_still_runs_decision_feature_forward(self) -> None:
        model = _Model()
        reused = ReusedComponentModel(
            model, "cached_generation_recomputed_decision_features"  # type: ignore[arg-type]
        )
        reused.begin_session(_reference())
        self.assertEqual(reused.generate([], [], 64), "$silent$cached")
        neural = reused.extract_decision_features([], [])
        reused.finish_session()
        self.assertEqual(model.generations, 0)
        self.assertEqual(model.features, 1)
        self.assertEqual(neural.hidden_state.tolist(), [5.0, 6.0])

    def test_tokens16_runs_generation_and_reuses_baseline_neural(self) -> None:
        model = _Model()
        reused = ReusedComponentModel(
            model, "generation_with_d4_2_baseline_decision_features"  # type: ignore[arg-type]
        )
        reused.begin_session(_reference())
        self.assertEqual(reused.generate([], [], 16), "$interrupt$new")
        neural = reused.extract_decision_features([], [])
        reused.finish_session()
        self.assertEqual(model.generations, 1)
        self.assertEqual(model.features, 0)
        self.assertEqual(neural.hidden_state.tolist(), [1.0, 2.0])
        self.assertEqual(neural.tag_margin, 0.4)


if __name__ == "__main__":
    unittest.main()
