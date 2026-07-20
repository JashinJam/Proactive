from __future__ import annotations

import unittest

from proactive_state_s0.core import (
    CANDIDATE_LABELS,
    messages_from_sample,
    multiclass_metrics,
    prediction_from_scores,
    probabilities,
    state_question_messages,
)


class StateS0CoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = {
            "query": "How do I do this?",
            "task": "Example task",
            "goal": "Finish the example.",
            "prior_dialog": [
                {"role": "user", "text": "How do I do this?"},
                {"role": "assistant", "text": "Start with preparation."},
            ],
            "steps": [
                {"id": f"s{index}", "text": f"Perform step {index}."}
                for index in range(1, 5)
            ],
        }

    @staticmethod
    def normalize(turns):
        return [
            {"role": str(turn["role"]), "content": str(turn["text"])}
            for turn in turns
        ]

    def test_no_assistant_view_removes_only_assistant_history(self) -> None:
        official = messages_from_sample(
            self.sample, "system", self.normalize, 4, "official_dialog"
        )
        diagnostic = messages_from_sample(
            self.sample, "system", self.normalize, 4, "no_assistant_history"
        )
        self.assertEqual(len(official), 3)
        self.assertEqual(len(diagnostic), 2)

    def test_question_contains_plan_but_no_dynamic_target(self) -> None:
        base = messages_from_sample(
            self.sample, "system", self.normalize, 4, "official_dialog"
        )
        result = state_question_messages(base, self.sample, "step")
        system = result[0]["content"]
        self.assertIn("Perform step 4", system)
        self.assertIn("Return only the option digit", system)
        self.assertNotIn("current_step_id", system)

    def test_score_softmax_and_frozen_tie_break(self) -> None:
        posterior = probabilities([-2.0, -1.0])
        self.assertAlmostEqual(sum(posterior), 1.0)
        result = prediction_from_scores("error", [-1.0, -1.0])
        self.assertEqual(result["label"], "absent")

    def test_multiclass_macro_includes_unsupported_predictions(self) -> None:
        metrics = multiclass_metrics(
            ["s1", "s2", "s3", "s4"],
            ["s1", "s1", "s3", "s3"],
            CANDIDATE_LABELS["step"],
        )
        self.assertEqual(metrics["support"], 4)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertLess(metrics["macro_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()

