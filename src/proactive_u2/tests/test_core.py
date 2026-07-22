from __future__ import annotations

import unittest

from proactive_u2.core import (
    VIEWS,
    analyze_records,
    build_review_packages,
    fact_extraction_messages,
    normalize_visual_facts,
    remove_assistant_history,
    visual_fact_block,
)


class PromptControlTest(unittest.TestCase):
    def test_removes_assistant_history_and_builds_fact_controls(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "old step"},
        ]
        result = remove_assistant_history(messages)
        self.assertEqual([row["role"] for row in result], ["system", "user"])
        self.assertEqual(fact_extraction_messages("make tea")[1]["role"], "user")
        normalized = normalize_visual_facts("$interrupt$ a red cup is visible")
        self.assertEqual(normalized["visual_facts"], "a red cup is visible")
        self.assertTrue(normalized["facts_removed_decision_tag"])
        self.assertIn("a red cup", visual_fact_block(normalized["visual_facts"]))


def fixture_records():
    samples = []
    facts = []
    records = []
    for index in range(2):
        sample_id = f"U2-{index:04d}"
        samples.append(
            {
                "sample_id": sample_id,
                "video_path": f"video-{index}.mp4",
                "interval": [2.0, 10.0],
                "observed_through_sec": 10.0,
                "video_intervals_so_far": [[0.0, 2.0], [2.0, 10.0]],
                "query": "Tighten the screw",
                "task": "Task",
                "domain": "Handyman",
                "chunk_index": 1,
                "position_bin": "1:second",
                "prior_dialog": [{"role": "user", "text": "q"}],
            }
        )
        facts.append(
            {
                "sample_id": sample_id,
                "input_index": index,
                "chunk_index": 1,
                "visual_facts": "a screw and screwdriver are visible",
                "facts_used_unclear": False,
                "facts_removed_decision_tag": False,
            }
        )
        for view in VIEWS:
            fallback = view == "query_only_current_video"
            content = "" if fallback else "Tighten the visible screw."
            records.append(
                {
                    "sample_id": sample_id,
                    "input_index": index,
                    "chunk_index": 1,
                    "domain": "Handyman",
                    "position_bin": "1:second",
                    "view": view,
                    "content": content,
                    "answer": "$interrupt$Please continue with the next step."
                    if fallback
                    else f"$interrupt${content}",
                    "used_fallback": fallback,
                    "raw_continuation": content,
                    "visual_facts": "a screw and screwdriver are visible",
                    "model_input_frames": 16,
                    "assistant_history_turns": 0,
                }
            )
    return records, facts, samples


class AnalysisTest(unittest.TestCase):
    def test_validates_views_and_detects_fact_cold_start_rescue(self) -> None:
        records, facts, _ = fixture_records()
        analysis, discordant = analyze_records(
            records,
            facts,
            {
                "facts_non_unclear_rate_at_least": 0.8,
                "fact_query_nonempty_gain_at_least": 0.2,
            },
        )
        self.assertEqual(analysis["samples"], 2)
        self.assertEqual(
            analysis["paired_contrasts"]["fact_query_cold_start"]
            ["nonempty_rate_delta_candidate_minus_reference"],
            1.0,
        )
        self.assertTrue(all(analysis["review_priority_checks"].values()))
        self.assertTrue(discordant)

    def test_rejects_incomplete_view_coverage(self) -> None:
        records, facts, _ = fixture_records()
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            analyze_records(records[:-1], facts, {})

    def test_blind_review_package_does_not_expose_variants_or_facts(self) -> None:
        records, facts, samples = fixture_records()
        blind, key, fact_blind, summary = build_review_packages(
            records, facts, samples, seed="test"
        )
        self.assertEqual(len(blind), 2 * len(VIEWS))
        self.assertEqual(len(key), len(blind))
        self.assertEqual(len(fact_blind), 2)
        self.assertFalse(any("view" in row or "visual_facts" in row for row in blind))
        self.assertFalse(summary["variant_exposed_in_blind_package"])


if __name__ == "__main__":
    unittest.main()
