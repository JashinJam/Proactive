from __future__ import annotations

import unittest

from PIL import Image

from proactive_u1.visual_audit import (
    analyze_records,
    mask_frames,
    remove_assistant_history,
)


class ViewConstructionTest(unittest.TestCase):
    def test_removes_only_assistant_history(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "old guidance"},
            {"role": "user", "content": "follow-up"},
        ]
        result = remove_assistant_history(messages)
        self.assertEqual([row["role"] for row in result], ["system", "user", "user"])
        self.assertEqual(len(messages), 4)

    def test_mask_preserves_count_and_dimensions(self) -> None:
        frames = [Image.new("RGB", (7, 5), (1, 2, 3)), Image.new("L", (3, 2), 9)]
        result = mask_frames(frames, 127)
        self.assertEqual([frame.size for frame in result], [(7, 5), (3, 2)])
        self.assertEqual([frame.mode for frame in result], ["RGB", "RGB"])
        self.assertEqual(result[0].getpixel((0, 0)), (127, 127, 127))


class VisualAuditAnalysisTest(unittest.TestCase):
    def _row(
        self,
        view: str,
        content: str,
        fallback: bool,
        sample: int,
    ) -> dict[str, object]:
        return {
            "view": view,
            "sample_id": f"sample-{sample}",
            "input_index": sample,
            "chunk_index": 1,
            "domain": "Chef",
            "position_bin": "1:second",
            "content": content,
            "answer": "$interrupt$Please continue with the next step."
            if fallback
            else f"$interrupt${content}",
            "used_fallback": fallback,
        }

    def test_paired_metrics_and_frozen_gates(self) -> None:
        records: list[dict[str, object]] = []
        for sample in range(2):
            records.extend(
                [
                    self._row("full", "Tighten the screw.", False, sample),
                    self._row("no_assistant_history", "", True, sample),
                    self._row(
                        "no_current_interval_video", "Tighten the screw.", False, sample
                    ),
                    self._row("masked_video", "You are all set.", False, sample),
                ]
            )
        thresholds = {
            "history_fallback_rate_increase": 0.2,
            "current_video_fallback_rate_increase": 0.1,
            "current_video_mean_similarity_below": 0.7,
            "masked_video_fallback_rate_increase": 0.15,
            "masked_video_mean_similarity_below": 0.65,
        }
        result = analyze_records(records, thresholds)
        self.assertTrue(result["diagnostic_gates"]["history_necessary"])
        self.assertFalse(result["diagnostic_gates"]["current_visual_material"])
        self.assertTrue(result["diagnostic_gates"]["any_visual_material"])
        self.assertEqual(
            result["views"]["masked_video"]["overall"]["completion_claims"], 2
        )
        self.assertEqual(result["discordant_cases"], 4)

    def test_rejects_incomplete_view_coverage(self) -> None:
        records = [self._row("full", "A", False, 0)]
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            analyze_records(records, {})


if __name__ == "__main__":
    unittest.main()
