from __future__ import annotations

import unittest

from proactive_u1.ratings import validate_ratings
from proactive_u1.state_ratings import analyze_state_ratings
from proactive_u1.state_review import STATE_VARIANTS


class StateRatingsTest(unittest.TestCase):
    def _parsed(self, full_score: int):
        domains = ["Arts and Crafts", "Chef", "Handyman", "Tutorial"]
        scores = {
            "forced_no_state": 3,
            "forced_oracle_step": 4,
            "forced_oracle_full": full_score,
        }
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
                    "position_bin": "5-9",
                }
            )
            for candidate_index, variant in enumerate(STATE_VARIANTS):
                candidate = chr(ord("A") + candidate_index)
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
                    row = {
                        "review_id": review_id,
                        "pair_id": pair_id,
                        "candidate": candidate,
                        "reviewer_slot": reviewer,
                        "generic_flag": "0",
                        "hallucination_flag": "0",
                        "premature_completion_flag": "0",
                    }
                    for field in (
                        "correctness_1_5",
                        "specificity_1_5",
                        "actionability_1_5",
                        "groundedness_1_5",
                        "plan_consistency_1_5",
                        "conciseness_1_5",
                    ):
                        row[field] = str(scores[variant])
                    row["safety_1_5"] = "5"
                    ratings.append(row)
        parsed, _ = validate_ratings(
            ratings,
            blind,
            key,
            samples,
            expected_pairs=4,
            expected_sessions=4,
            allowed_variants=STATE_VARIANTS,
        )
        return parsed

    def test_prefers_compact_step_when_full_adds_nothing(self) -> None:
        result = analyze_state_ratings(
            self._parsed(full_score=4), bootstrap_seed=3, bootstrap_samples=100
        )
        self.assertTrue(result["decision"]["state_promoted"])
        self.assertFalse(result["decision"]["full_over_step_justified"])
        self.assertEqual(
            result["decision"]["preferred_state_representation"],
            "forced_oracle_step",
        )

    def test_prefers_full_when_increment_is_clear(self) -> None:
        result = analyze_state_ratings(
            self._parsed(full_score=5), bootstrap_seed=3, bootstrap_samples=100
        )
        self.assertTrue(result["decision"]["full_over_step_justified"])
        self.assertEqual(
            result["decision"]["preferred_state_representation"],
            "forced_oracle_full",
        )


if __name__ == "__main__":
    unittest.main()
