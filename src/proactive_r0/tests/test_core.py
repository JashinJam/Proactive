from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from proactive_r0.core import (
    CausalInferenceConfig,
    StarterKitSymbols,
    build_messages,
    canonicalize_response,
    process_session,
    subsample_frames,
    validate_prediction_rows,
    validate_source_rows,
)
from proactive_r0.internvl import resolve_physical_cuda_identifier
from proactive_r0.run import contiguous_shard_bounds


def normalize_dialog(turns: list[dict[str, object]]) -> list[dict[str, str]]:
    return [
        {"role": str(turn["role"]), "content": str(turn["text"])}
        for turn in turns
        if turn.get("text")
    ]


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[object], list[dict[str, str]], int]] = []

    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> str:
        self.calls.append((list(frames), messages, max_new_tokens))
        return "$interrupt$ next" if len(self.calls) == 1 else "not a tag"


def source_row() -> dict[str, object]:
    return {
        "video_path": "sample.mp4",
        "video_intervals": [[0.0, 2.0], [3.5, 11.5], [11.5, 19.5]],
        "query": "Help me do this",
        "dialog": [
            [{"role": "user", "text": "Help me do this"}],
            [
                {"role": "user", "text": "Help me do this"},
                {"role": "assistant", "text": "$interrupt$ first"},
            ],
            [
                {"role": "user", "text": "Help me do this"},
                {"role": "assistant", "text": "$interrupt$ first"},
                {"role": "assistant", "text": "$silent$"},
            ],
        ],
        "answers": ["must", "not", "be read"],
    }


class CanonicalizeResponseTest(unittest.TestCase):
    def test_response_cases(self) -> None:
        cases = [
            ("$silent$", "$silent$", None),
            ("  $silent$ extra", "$silent$", "trimmed_silent_suffix"),
            ("$interrupt$Do it", "$interrupt$Do it", None),
            (
                "$interrupt$",
                "$interrupt$Please continue with the next step.",
                "empty_interrupt_utterance",
            ),
            ("analysis first", "$silent$", "malformed_response_scored_as_silent"),
        ]
        for raw, expected, reason in cases:
            with self.subTest(raw=raw):
                self.assertEqual(canonicalize_response(raw), (expected, reason))


class CausalCoreTest(unittest.TestCase):
    def test_subsample_frames_matches_official_stride(self) -> None:
        self.assertEqual(subsample_frames(list(range(10)), 4), [0, 2, 5, 7])
        self.assertEqual(subsample_frames([1, 2], 4), [1, 2])

    def test_build_messages_uses_only_current_dialog_history(self) -> None:
        messages = build_messages(
            source_row(),
            chunk_index=1,
            system_prompt="system",
            normalize_dialog_turns=normalize_dialog,
            max_history_turns=1,
        )
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Help me do this"},
                {"role": "assistant", "content": "$interrupt$ first"},
            ],
        )

    def test_process_session_is_causal_and_preserves_order(self) -> None:
        row = source_row()
        with tempfile.TemporaryDirectory() as directory:
            video_folder = Path(directory)
            (video_folder / "sample.mp4").touch()
            events: list[str] = []

            def extract_frames(
                video_path: str,
                intervals: list[tuple[float, float]],
                frames_per_interval: int,
            ) -> list[object]:
                start = intervals[0][0]
                events.append(f"extract:{start}")
                return [(start, frame) for frame in range(frames_per_interval)]

            class OrderedFakeModel(FakeModel):
                def generate(
                    self,
                    frames: list[object],
                    messages: list[dict[str, str]],
                    max_new_tokens: int,
                ) -> str:
                    events.append(f"generate:{len(self.calls)}")
                    return super().generate(frames, messages, max_new_tokens)

            model = OrderedFakeModel()
            starter = StarterKitSymbols(
                system_prompt="system",
                normalize_dialog_turns=normalize_dialog,
                extract_frames=extract_frames,
            )
            result = process_session(
                row=row,
                input_index=0,
                video_folder=video_folder,
                model=model,
                starter=starter,
                config=CausalInferenceConfig(
                    frames_per_interval=2,
                    max_frames=3,
                    max_history_turns=1,
                    max_new_tokens=8,
                ),
            )

        self.assertEqual([len(call[0]) for call in model.calls], [2, 3, 3])
        self.assertEqual(
            events,
            [
                "extract:0.0",
                "generate:0",
                "extract:3.5",
                "generate:1",
                "extract:11.5",
                "generate:2",
            ],
        )
        self.assertTrue(all(frame[0] <= 0.0 for frame in model.calls[0][0]))
        self.assertTrue(all(frame[0] <= 3.5 for frame in model.calls[1][0]))
        self.assertEqual(
            result["prediction"],
            {
                "video_path": "sample.mp4",
                "answers": ["$interrupt$next", "$silent$", "$silent$"],
            },
        )
        self.assertEqual(
            result["chunks"][1]["normalization"],
            "malformed_response_scored_as_silent",
        )


class ValidationTest(unittest.TestCase):
    def test_validation_never_requires_gold_answers(self) -> None:
        row = source_row()
        row.pop("answers")
        with tempfile.TemporaryDirectory() as directory:
            video_folder = Path(directory)
            (video_folder / "sample.mp4").touch()
            summary = validate_source_rows([row], video_folder)
        self.assertEqual(
            summary, {"sessions": 1, "chunks": 3, "missing_videos": 0}
        )
        predictions = [
            {
                "video_path": "sample.mp4",
                "answers": ["$silent$", "$interrupt$Now act", "$silent$"],
            }
        ]
        self.assertEqual(
            validate_prediction_rows([row], predictions),
            {"sessions": 1, "chunks": 3, "interrupts": 1},
        )

    def test_prediction_validation_rejects_order_change(self) -> None:
        row = source_row()
        row.pop("answers")
        with self.assertRaisesRegex(ValueError, "order mismatch"):
            validate_prediction_rows(
                [row],
                [{"video_path": "other.mp4", "answers": ["$silent$"] * 3}],
            )


class GpuSafetyMappingTest(unittest.TestCase):
    def test_numeric_visible_device_is_mapped_to_physical_index(self) -> None:
        self.assertEqual(resolve_physical_cuda_identifier("cuda:0", "3"), 3)
        self.assertEqual(resolve_physical_cuda_identifier("cuda:1", "3,7"), 7)

    def test_uuid_visible_device_is_preserved(self) -> None:
        self.assertEqual(
            resolve_physical_cuda_identifier("cuda:0", "GPU-example"),
            "GPU-example",
        )

    def test_unmapped_logical_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "absent"):
            resolve_physical_cuda_identifier("cuda:1", "3")


class ShardBoundsTest(unittest.TestCase):
    def test_even_four_way_split(self) -> None:
        self.assertEqual(
            [contiguous_shard_bounds(700, 4, index) for index in range(4)],
            [(0, 175), (175, 350), (350, 525), (525, 700)],
        )

    def test_remainder_is_assigned_to_early_shards(self) -> None:
        self.assertEqual(
            [contiguous_shard_bounds(5, 3, index) for index in range(3)],
            [(0, 2), (2, 4), (4, 5)],
        )

    def test_eight_way_full_validation_split(self) -> None:
        self.assertEqual(
            [contiguous_shard_bounds(700, 8, index) for index in range(8)],
            [
                (0, 88),
                (88, 176),
                (176, 264),
                (264, 352),
                (352, 439),
                (439, 526),
                (526, 613),
                (613, 700),
            ],
        )


if __name__ == "__main__":
    unittest.main()
