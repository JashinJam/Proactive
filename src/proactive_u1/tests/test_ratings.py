from __future__ import annotations

import unittest

from proactive_u1.ratings import analyze_ratings, validate_ratings


class RatingsTest(unittest.TestCase):
    def _fixture(self):
        domains = ["Arts and Crafts", "Chef", "Handyman", "Tutorial"]
        samples = []
        blind = []
        key = []
        ratings = []
        for index, domain in enumerate(domains):
            pair_id = f"sample-{index}"
            samples.append(
                {
                    "sample_id": pair_id,
                    "input_index": index,
                    "domain": domain,
                    "position_bin": "2-4",
                }
            )
            for candidate, variant, score in (
                ("A", "current_fallback", 3),
                ("B", "forced_no_state", 4),
            ):
                review_id = f"{pair_id}-{candidate}"
                blind.append(
                    {"review_id": review_id, "pair_id": pair_id, "candidate": candidate}
                )
                key.append(
                    {
                        "review_id": review_id,
                        "pair_id": pair_id,
                        "candidate": candidate,
                        "variant": variant,
                    }
                )
                for reviewer in ("A", "B"):
                    ratings.append(
                        {
                            "review_id": review_id,
                            "pair_id": pair_id,
                            "candidate": candidate,
                            "reviewer_slot": reviewer,
                            "correctness_1_5": str(score),
                            "specificity_1_5": str(score),
                            "actionability_1_5": str(score),
                            "groundedness_1_5": str(score),
                            "plan_consistency_1_5": str(score),
                            "conciseness_1_5": str(score),
                            "safety_1_5": "5",
                            "generic_flag": "0",
                            "hallucination_flag": "0",
                            "premature_completion_flag": "0",
                        }
                    )
        return ratings, blind, key, samples

    def test_validates_and_passes_constant_positive_gate(self) -> None:
        ratings, blind, key, samples = self._fixture()
        parsed, _ = validate_ratings(
            ratings, blind, key, samples, expected_pairs=4, expected_sessions=4
        )
        result = analyze_ratings(parsed, bootstrap_seed=7, bootstrap_samples=200)
        comparison = result["paired_comparison"]
        self.assertEqual(comparison["content_composite_delta"], 1.0)
        self.assertEqual(comparison["session_bootstrap"]["ci95_low"], 1.0)
        self.assertTrue(comparison["promotion_gate"]["passed"])
        self.assertEqual(
            result["agreement"]["scores"]["correctness_1_5"][
                "quadratic_weighted_kappa"
            ],
            1.0,
        )

    def test_ignores_blank_rows_but_rejects_missing_coverage(self) -> None:
        ratings, blind, key, samples = self._fixture()
        blank = dict(ratings[0])
        for field in (
            "correctness_1_5",
            "specificity_1_5",
            "actionability_1_5",
            "groundedness_1_5",
            "plan_consistency_1_5",
            "conciseness_1_5",
            "safety_1_5",
            "generic_flag",
            "hallucination_flag",
            "premature_completion_flag",
        ):
            blank[field] = ""
        validate_ratings(
            [*ratings, blank], blind, key, samples, expected_pairs=4, expected_sessions=4
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_ratings(
                ratings[:-1], blind, key, samples, expected_pairs=4, expected_sessions=4
            )

    def test_rejects_duplicate_populated_rating(self) -> None:
        ratings, blind, key, samples = self._fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_ratings(
                [*ratings, dict(ratings[0])],
                blind,
                key,
                samples,
                expected_pairs=4,
                expected_sessions=4,
            )

    def test_hallucination_increase_fails_gate(self) -> None:
        ratings, blind, key, samples = self._fixture()
        for row in ratings:
            if row["candidate"] == "B":
                row["hallucination_flag"] = "1"
        parsed, _ = validate_ratings(
            ratings, blind, key, samples, expected_pairs=4, expected_sessions=4
        )
        result = analyze_ratings(parsed, bootstrap_seed=7, bootstrap_samples=50)
        gate = result["paired_comparison"]["promotion_gate"]
        self.assertFalse(gate["checks"]["hallucination_rate_increase_at_most_0_02"])
        self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
