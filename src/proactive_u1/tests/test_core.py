from __future__ import annotations

import unittest

import torch

from proactive_u1.core import (
    controlled_messages,
    normalize_continuation,
    oracle_state_block,
    validate_oracle_annotations,
    validate_decision_invariance,
)
from proactive_u1.internvl import append_text_prefix
from proactive_u1.analyze import (
    analyze_content,
    build_blind_multivariant,
    build_blind_pairs,
)


class PrefixTest(unittest.TestCase):
    def test_appends_input_mask_and_positions(self) -> None:
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "position_ids": torch.tensor([[0, 1, 2]]),
            "pixel_values_videos": torch.zeros((2, 3, 4, 4)),
        }
        result, prompt_length, prefix_length = append_text_prefix(
            inputs, torch.tensor([[8, 9]])
        )
        self.assertEqual(prompt_length, 3)
        self.assertEqual(prefix_length, 2)
        self.assertEqual(result["input_ids"].tolist(), [[1, 2, 3, 8, 9]])
        self.assertEqual(result["attention_mask"].tolist(), [[1, 1, 1, 1, 1]])
        self.assertEqual(result["position_ids"].tolist(), [[0, 1, 2, 3, 4]])


class PromptAndOutputTest(unittest.TestCase):
    def test_controlled_prompt_does_not_mutate_input(self) -> None:
        messages = [
            {"role": "system", "content": "Base"},
            {"role": "user", "content": "Task"},
        ]
        result = controlled_messages(messages, "Current step: attach the part")
        self.assertEqual(messages[0]["content"], "Base")
        self.assertIn("already been made", result[0]["content"])
        self.assertIn("Current step", result[0]["content"])

    def test_normalization_preserves_failure_diagnostics(self) -> None:
        normal = normalize_continuation("Tighten the screw.")
        self.assertEqual(normal["answer"], "$interrupt$Tighten the screw.")
        repeated = normalize_continuation("$interrupt$Tighten it.")
        self.assertTrue(repeated["extra_interrupt_tag"])
        self.assertEqual(repeated["answer"], "$interrupt$Tighten it.")
        silent = normalize_continuation("$silent$")
        self.assertTrue(silent["generated_silent_tag"])
        self.assertTrue(silent["used_fallback"])

    def test_oracle_variants_have_strict_fields(self) -> None:
        annotation = {
            "status": "complete",
            "steps": [
                {"id": "s1", "text": "Attach the part."},
                {"id": "s2", "text": "Tighten the screw."},
            ],
            "sampled_chunk_states": [
                {
                    "sample_id": "sample",
                    "current_step_id": "s1",
                    "next_step_id": "s2",
                    "progress": "ongoing",
                    "completion_evidence": ["Part is aligned."],
                    "incompletion_or_error_evidence": ["Screw is loose."],
                    "recovery_action": "Realign before tightening.",
                }
            ],
        }
        step = oracle_state_block(annotation, "sample", "forced_oracle_step")
        full = oracle_state_block(annotation, "sample", "forced_oracle_full")
        self.assertNotIn("Screw is loose", step)
        self.assertIn("Screw is loose", full)
        self.assertIn("Realign", full)

    def test_oracle_validator_enforces_time_and_target_blinding(self) -> None:
        sample = {
            "sample_id": "sample",
            "input_index": 1,
            "chunk_index": 2,
            "observed_through_sec": 18.0,
        }
        annotation = {
            "status": "complete",
            "input_index": 1,
            "steps": [{"id": "s1", "text": "Attach the part."}],
            "sampled_chunk_states": [
                {
                    "sample_id": "sample",
                    "chunk_index": 2,
                    "observed_through_sec": 18.0,
                    "current_step_id": "s1",
                    "next_step_id": None,
                    "progress": "ongoing",
                    "completion_evidence": [],
                    "incompletion_or_error_evidence": [],
                    "recovery_action": "",
                    "confidence": 0.8,
                }
            ],
        }
        result = validate_oracle_annotations([annotation], [sample])
        self.assertEqual(result["sampled_states"], 1)
        annotation["sampled_chunk_states"][0]["recovery_action"] = "$silent$"
        with self.assertRaisesRegex(ValueError, "Forbidden target marker"):
            validate_oracle_annotations([annotation], [sample])
        annotation["sampled_chunk_states"][0]["recovery_action"] = "The model should speak now."
        with self.assertRaisesRegex(ValueError, "Forbidden target marker"):
            validate_oracle_annotations([annotation], [sample])

    def test_decision_invariance_ignores_content_only(self) -> None:
        left = [{"video_path": "v", "answers": ["$interrupt$A", "$silent$"]}]
        right = [{"video_path": "v", "answers": ["$interrupt$B", "$silent$"]}]
        validate_decision_invariance(left, right)
        right[0]["answers"][1] = "$interrupt$C"
        with self.assertRaisesRegex(ValueError, "decision changed"):
            validate_decision_invariance(left, right)


class AnalysisTest(unittest.TestCase):
    def test_paired_review_is_blind_and_deterministic(self) -> None:
        samples = [
            {
                "sample_id": "sample",
                "input_index": 1,
                "chunk_index": 2,
                "domain": "Chef",
                "position_bin": "2-4",
                "video_path": "v.mp4",
                "interval": [1.0, 2.0],
                "observed_through_sec": 2.0,
                "video_intervals_so_far": [[1.0, 2.0]],
                "query": "Make food",
                "task": "Food",
                "prior_dialog": [],
            }
        ]
        content = [
            {
                "sample_id": "sample",
                "input_index": 1,
                "chunk_index": 2,
                "domain": "Chef",
                "position_bin": "2-4",
                "variant": "current_fallback",
                "content": "Please continue with the next step.",
                "used_fallback": True,
            },
            {
                "sample_id": "sample",
                "input_index": 1,
                "chunk_index": 2,
                "domain": "Chef",
                "position_bin": "2-4",
                "variant": "forced_no_state",
                "content": "Heat the pan.",
                "used_fallback": False,
            },
        ]
        analysis = analyze_content(samples, content)
        self.assertEqual(analysis["overall"]["nonempty"], 1)
        blind, key = build_blind_pairs(samples, content, "seed")
        repeated, repeated_key = build_blind_pairs(samples, content, "seed")
        self.assertEqual((blind, key), (repeated, repeated_key))
        self.assertEqual(len(blind), 2)
        self.assertNotIn("variant", blind[0])
        self.assertEqual({row["variant"] for row in key}, {"current_fallback", "forced_no_state"})

    def test_three_way_state_review_is_blind_and_complete(self) -> None:
        samples = [
            {
                "sample_id": "sample",
                "video_path": "v.mp4",
                "interval": [1.0, 2.0],
                "observed_through_sec": 2.0,
                "video_intervals_so_far": [[1.0, 2.0]],
                "query": "Do it",
                "task": "Task",
                "domain": "Tutorial",
                "chunk_index": 1,
                "prior_dialog": [],
            }
        ]
        variants = ("forced_no_state", "forced_oracle_step", "forced_oracle_full")
        content = [
            {
                "sample_id": "sample",
                "variant": variant,
                "content": variant,
                "used_fallback": False,
            }
            for variant in variants
        ]
        blind, key = build_blind_multivariant(samples, content, variants, "seed")
        self.assertEqual(len(blind), 3)
        self.assertEqual({row["candidate"] for row in blind}, {"A", "B", "C"})
        self.assertTrue(all("variant" not in row for row in blind))
        self.assertEqual({row["variant"] for row in key}, set(variants))


if __name__ == "__main__":
    unittest.main()
