from __future__ import annotations

import unittest

from proactive_u2.prepare import prepare_sample


class PrepareSampleTest(unittest.TestCase):
    def test_selects_complete_boolean_intersection_without_answers(self) -> None:
        sources = []
        oof = []
        final = []
        items = []
        for index in range(700):
            source = {
                "video_path": f"video-{index}.mp4",
                "video_intervals": [[0.0, 2.0], [2.0, 10.0]],
                "dialog": [[{"role": "user", "text": "q"}]] * 2,
                "query": "q",
                "task": "task",
                "domain": "Chef",
            }
            sources.append(source)
            answer = ["$silent$", "$interrupt$Do it."]
            oof.append({"video_path": source["video_path"], "answers": answer})
            final.append({"video_path": source["video_path"], "answers": answer})
        items.append(
            {
                "review_id": "U0-0001",
                "input_index": 3,
                "chunk_index": 1,
                "position_bin": "1:second",
                "fallback_status": "fallback",
                "pair_content_composite": 1.5,
                "disagreement_triggers": [],
                "reviewers": {
                    "A": {"should_interrupt": "yes"},
                    "B": {"should_interrupt": "yes"},
                },
            }
        )
        samples, key, summary = prepare_sample(
            items, oof, final, sources, seed="test"
        )
        self.assertEqual(len(samples), 1)
        self.assertEqual(key[0]["u0_fallback_status"], "fallback")
        self.assertNotIn("answers", samples[0])
        self.assertEqual(summary["by_position"], {"1:second": 1})

    def test_rejects_target_bearing_source_rows(self) -> None:
        sources = [
            {
                "video_path": f"video-{index}.mp4",
                "answers": ["$silent$"],
            }
            for index in range(700)
        ]
        predictions = [
            {"video_path": f"video-{index}.mp4", "answers": ["$silent$"]}
            for index in range(700)
        ]
        with self.assertRaisesRegex(ValueError, "answer-stripped"):
            prepare_sample([], predictions, predictions, sources, seed="test")


if __name__ == "__main__":
    unittest.main()
