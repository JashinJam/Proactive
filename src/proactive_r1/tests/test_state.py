from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from proactive_r0.core import CausalInferenceConfig, StarterKitSymbols
from proactive_r1.core import process_session_variants
from proactive_r1.constrained import (
    ForcedPrefixConstraint,
    TagPrefixConstraint,
    TagPrefixLogitsProcessor,
)
from proactive_r1.format_core import FORMAT_VARIANTS, process_format_session
from proactive_r1.prepare import sanitized_chunk_contexts
from proactive_r1.state import (
    STATE_VARIANTS,
    build_state_messages,
    render_state,
    text_sha256,
    validate_annotations,
)


def normalize_dialog(turns: list[dict[str, object]]) -> list[dict[str, str]]:
    return [
        {"role": str(turn["role"]), "content": str(turn["text"])}
        for turn in turns
        if turn.get("text")
    ]


def source_row() -> dict[str, object]:
    return {
        "video_path": "sample.mp4",
        "video_intervals": [[0.0, 2.0], [2.0, 10.0]],
        "query": "How do I make tea?",
        "task": "Making tea",
        "domain": "Chef",
        "dialog": [
            [{"role": "user", "text": "How do I make tea?"}],
            [
                {"role": "user", "text": "How do I make tea?"},
                {"role": "assistant", "text": "Start by heating water."},
            ],
        ],
        "answers": ["hidden", "hidden"],
    }


def annotation() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "complete",
        "input_index": 4,
        "video_path": "sample.mp4",
        "query_sha256": text_sha256("How do I make tea?"),
        "provenance": {
            "plan_inputs": ["task", "query"],
            "chunk_inputs": ["task", "query", "dialog_at_chunk", "video_through_interval_end"],
            "excluded_inputs": ["answers", "future_dialog", "future_video"],
            "annotation_type": "evaluation_only_oracle_non_deployable",
        },
        "goal": "How do I make tea?",
        "steps": [
            {
                "id": "s1",
                "text": "Heat the water.",
                "completion_cues": ["Water is hot."],
                "incompletion_cues": ["Water is still cold."],
            },
            {
                "id": "s2",
                "text": "Steep the tea.",
                "completion_cues": ["Tea has steeped."],
                "incompletion_cues": ["Tea is not in the water."],
            },
        ],
        "chunk_states": [
            {
                "chunk_index": 0,
                "observed_through_sec": 2.0,
                "current_step_id": "s1",
                "progress": "not_started",
                "completion_evidence": [],
                "incompletion_or_error_evidence": ["Kettle has not been switched on."],
                "next_step_id": "s2",
                "confidence": 0.9,
                "last_update_chunk": 0,
            },
            {
                "chunk_index": 1,
                "observed_through_sec": 10.0,
                "current_step_id": "s1",
                "progress": "ongoing",
                "completion_evidence": ["Kettle is switched on."],
                "incompletion_or_error_evidence": [],
                "next_step_id": "s2",
                "confidence": 0.8,
                "last_update_chunk": 1,
            },
        ],
    }


class AnnotationValidationTest(unittest.TestCase):
    def test_valid_annotation(self) -> None:
        row = source_row()
        result = validate_annotations([annotation()], [(4, row)])
        self.assertEqual(set(result), {4})

    def test_future_timestamp_is_rejected(self) -> None:
        value = annotation()
        value["chunk_states"][0]["observed_through_sec"] = 2.1  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "interval end"):
            validate_annotations([value], [(4, source_row())])

    def test_target_marker_is_rejected(self) -> None:
        value = annotation()
        value["steps"][0]["text"] = "$interrupt$ boil water"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "forbidden target marker"):
            validate_annotations([value], [(4, source_row())])

    def test_undeclared_state_change_is_rejected(self) -> None:
        value = annotation()
        value["chunk_states"][1]["last_update_chunk"] = 0  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "changed without an update"):
            validate_annotations([value], [(4, source_row())])


class RenderingTest(unittest.TestCase):
    def test_variants_expose_only_authorized_fields(self) -> None:
        value = annotation()
        null = render_state(value, 0, "null")
        step = render_state(value, 0, "step")
        cues = render_state(value, 0, "cues")
        full = render_state(value, 0, "full")
        self.assertIn("status: unavailable", null)
        self.assertIn("current_step: Heat the water.", step)
        self.assertNotIn("completion_cues", step)
        self.assertIn("completion_cues", cues)
        self.assertNotIn("progress", cues)
        self.assertIn("progress: not_started", full)
        self.assertIn("next_step: Steep the tea.", full)

    def test_official_system_prompt_is_an_unchanged_prefix(self) -> None:
        messages = build_state_messages(
            source_row(), 0, "official prompt", normalize_dialog, 4, "<state />"
        )
        self.assertEqual(messages[0]["content"], "official prompt\n\n<state />")
        self.assertEqual(messages[1]["content"], "How do I make tea?")


class SanitizationTest(unittest.TestCase):
    def test_context_has_no_answers_or_future_dialog(self) -> None:
        contexts = sanitized_chunk_contexts([(4, source_row())])
        self.assertEqual(len(contexts), 2)
        self.assertTrue(all("answers" not in context for context in contexts))
        self.assertEqual(len(contexts[0]["dialog_at_chunk"]), 1)


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[object], list[dict[str, str]], int]] = []

    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> str:
        self.calls.append((list(frames), copy.deepcopy(messages), max_new_tokens))
        return "$silent$"


class MultiVariantFlowTest(unittest.TestCase):
    def test_frames_are_extracted_once_per_chunk(self) -> None:
        extraction_calls: list[tuple[float, float]] = []

        def extract_frames(
            video_path: str,
            intervals: list[tuple[float, float]],
            frames_per_interval: int,
        ) -> list[object]:
            extraction_calls.append(intervals[0])
            return [(intervals[0], index) for index in range(frames_per_interval)]

        model = FakeModel()
        starter = StarterKitSymbols("system", normalize_dialog, extract_frames)
        with tempfile.TemporaryDirectory() as directory:
            video_folder = Path(directory)
            (video_folder / "sample.mp4").touch()
            result = process_session_variants(
                source_row(),
                4,
                annotation(),
                video_folder,
                model,
                starter,
                CausalInferenceConfig(2, 3, 4, 8),
            )
        self.assertEqual(extraction_calls, [(0.0, 2.0), (2.0, 10.0)])
        self.assertEqual(len(model.calls), len(STATE_VARIANTS) * 2)
        self.assertEqual(set(result["variants"]), set(STATE_VARIANTS))
        first_variant_frames = [call[0] for call in model.calls[: len(STATE_VARIANTS)]]
        self.assertTrue(all(frames == first_variant_frames[0] for frames in first_variant_frames))

    def test_format_factorial_shares_frames_and_preserves_exact_r0_context(self) -> None:
        extraction_calls: list[tuple[float, float]] = []

        def extract_frames(
            video_path: str,
            intervals: list[tuple[float, float]],
            frames_per_interval: int,
        ) -> list[object]:
            extraction_calls.append(intervals[0])
            return [(intervals[0], index) for index in range(frames_per_interval)]

        model = FakeModel()
        starter = StarterKitSymbols("official", normalize_dialog, extract_frames)
        with tempfile.TemporaryDirectory() as directory:
            video_folder = Path(directory)
            (video_folder / "sample.mp4").touch()
            result = process_format_session(
                source_row(),
                4,
                annotation(),
                video_folder,
                model,
                starter,
                CausalInferenceConfig(2, 3, 4, 8),
            )
        self.assertEqual(extraction_calls, [(0.0, 2.0), (2.0, 10.0)])
        self.assertEqual(len(model.calls), len(FORMAT_VARIANTS) * 2)
        self.assertEqual(set(result["variants"]), set(FORMAT_VARIANTS))
        r0_chunk = result["variants"]["r0_format"]["chunks"][0]
        null_chunk = result["variants"]["null"]["chunks"][0]
        self.assertIsNone(r0_chunk["state_block"])
        self.assertIn("status: unavailable", null_chunk["state_block"])
        self.assertEqual(model.calls[0][1][0]["content"], "official")


class TagPrefixConstraintTest(unittest.TestCase):
    def setUp(self) -> None:
        self.constraint = TagPrefixConstraint([10, 11, 12], [20, 21, 12], 99)

    def test_prefix_transitions(self) -> None:
        self.assertEqual(self.constraint.allowed_next([]), (10, 20))
        self.assertEqual(self.constraint.allowed_next([10]), (11,))
        self.assertEqual(self.constraint.allowed_next([10, 11]), (12,))
        self.assertEqual(self.constraint.allowed_next([10, 11, 12]), (99,))
        self.assertEqual(self.constraint.allowed_next([20]), (21,))
        self.assertEqual(self.constraint.allowed_next([20, 21]), (12,))
        self.assertIsNone(self.constraint.allowed_next([20, 21, 12]))
        self.assertIsNone(self.constraint.allowed_next([20, 21, 12, 5, 6]))

    def test_invalid_prefix_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "escaped"):
            self.constraint.allowed_next([30])

    def test_logits_processor_masks_disallowed_tokens(self) -> None:
        import torch

        scores = torch.arange(0, 32, dtype=torch.float32).unsqueeze(0)
        processor = TagPrefixLogitsProcessor(2, self.constraint)
        output = processor(torch.tensor([[7, 8]]), scores)
        finite = torch.isfinite(output[0]).nonzero().flatten().tolist()
        self.assertEqual(finite, [10, 20])
        output = processor(torch.tensor([[7, 8, 20, 21, 12]]), scores)
        self.assertTrue(torch.equal(output, scores))

    def test_forced_prefix_releases_after_complete_prefix(self) -> None:
        constraint = ForcedPrefixConstraint([20, 21, 12])
        self.assertEqual(constraint.allowed_next([]), (20,))
        self.assertEqual(constraint.allowed_next([20]), (21,))
        self.assertEqual(constraint.allowed_next([20, 21]), (12,))
        self.assertIsNone(constraint.allowed_next([20, 21, 12]))
        self.assertIsNone(constraint.allowed_next([20, 21, 12, 7]))


if __name__ == "__main__":
    unittest.main()
