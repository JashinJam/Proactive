from __future__ import annotations

import copy
import unittest

from proactive_u0.ratings import analyze_ratings, validate_ratings


def fixtures():
    blind = [
        {
            "review_id": "U0-0001",
            "domain": "Chef",
            "task": "Task 1",
            "chunk_index": 1,
            "model_action": "spoke",
            "candidate_utterance": "Continue.",
        },
        {
            "review_id": "U0-0002",
            "domain": "Tutorial",
            "task": "Task 2",
            "chunk_index": 3,
            "model_action": "spoke",
            "candidate_utterance": "Tighten the screw.",
        },
        {
            "review_id": "U0-0003",
            "domain": "Handyman",
            "task": "Task 3",
            "chunk_index": 6,
            "model_action": "silent",
            "candidate_utterance": None,
        },
    ]
    key = [
        {
            "review_id": "U0-0001",
            "input_index": 0,
            "chunk_index": 1,
            "stratum": "tp_fallback",
            "confusion": "tp",
            "is_fallback": True,
        },
        {
            "review_id": "U0-0002",
            "input_index": 1,
            "chunk_index": 3,
            "stratum": "fp_nonfallback",
            "confusion": "fp",
            "is_fallback": False,
        },
        {
            "review_id": "U0-0003",
            "input_index": 2,
            "chunk_index": 6,
            "stratum": "fn_silent",
            "confusion": "fn",
            "is_fallback": False,
        },
    ]
    ratings = []
    for item_index, item in enumerate(blind):
        for slot in ("A", "B"):
            spoke = item["model_action"] == "spoke"
            base = 1 + item_index + (2 if slot == "B" and item_index == 0 else 0)
            row = {
                "review_id": item["review_id"],
                "reviewer_slot": slot,
                "should_interrupt": "yes" if slot == "A" else "no",
                "decision_confidence_1_5": "4",
                "timeliness_1_5": str(min(base, 5)),
                "correctness_1_5": str(min(base, 5)) if spoke else "",
                "specificity_1_5": str(min(base, 5)) if spoke else "",
                "actionability_1_5": str(min(base, 5)) if spoke else "",
                "groundedness_1_5": str(min(base, 5)) if spoke else "",
                "plan_consistency_1_5": str(min(base, 5)) if spoke else "",
                "conciseness_1_5": str(min(base, 5)) if spoke else "",
                "safety_1_5": "5" if spoke else "",
                "generic_flag": ("yes" if item_index == 0 else "no") if spoke else "",
                "hallucination_flag": "no" if spoke else "",
                "unsafe_flag": "no" if spoke else "",
                "primary_error_type": ("generic" if item_index == 0 else "none")
                if spoke
                else "",
                "notes": "",
            }
            ratings.append(row)
    return ratings, blind, key


class RatingsValidationTest(unittest.TestCase):
    def test_validates_conditional_schema_and_complete_coverage(self) -> None:
        ratings, blind, key = fixtures()
        parsed = validate_ratings(ratings, blind, key, expected_items=3)
        self.assertEqual(len(parsed), 6)
        silent = [row for row in parsed if row["model_action"] == "silent"]
        self.assertEqual(len(silent), 2)
        self.assertTrue(all(row["content_composite"] is None for row in silent))

    def test_rejects_content_scores_on_silent_item(self) -> None:
        ratings, blind, key = fixtures()
        changed = copy.deepcopy(ratings)
        changed[-1]["correctness_1_5"] = "3"
        with self.assertRaisesRegex(ValueError, "populates content fields"):
            validate_ratings(changed, blind, key, expected_items=3)

    def test_rejects_duplicate_and_missing_rows(self) -> None:
        ratings, blind, key = fixtures()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_ratings(
                [*ratings, dict(ratings[0])], blind, key, expected_items=3
            )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_ratings(ratings[:-1], blind, key, expected_items=3)


class RatingsAnalysisTest(unittest.TestCase):
    def test_builds_agreement_strata_bootstrap_and_disagreement_cases(self) -> None:
        ratings, blind, key = fixtures()
        parsed = validate_ratings(ratings, blind, key, expected_items=3)
        analysis, items, disagreements = analyze_ratings(
            parsed, bootstrap_seed=7, bootstrap_samples=100
        )
        self.assertEqual(analysis["validation"]["items"], 3)
        self.assertEqual(analysis["validation"]["spoken_items"], 2)
        self.assertEqual(analysis["overall"]["content_composite"]["items"], 2)
        self.assertEqual(set(analysis["stratified"]["by_fallback_status"]), {
            "fallback",
            "nonfallback",
            "silent",
        })
        self.assertEqual(len(items), 3)
        self.assertTrue(disagreements)
        self.assertIn(
            "score_gap:timeliness_1_5", disagreements[0]["disagreement_triggers"]
        )
        self.assertEqual(
            analysis["overall"]["scores"]["decision_confidence_1_5"]
            ["pair_average"]["estimate"],
            4.0,
        )


if __name__ == "__main__":
    unittest.main()
