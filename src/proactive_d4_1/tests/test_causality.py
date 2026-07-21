from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from proactive_d1.core import LinearDecisionHead, LinearModel
from proactive_d1.internvl_features import NeuralDecisionFeatures
from proactive_d3.dialog_control_core import DIALOG_POLICY_NAMES
from proactive_d4.deploy import process_session_with_dialog_stage_head
from proactive_d4_1.run_variant import (
    TimedDecisionModel,
    _write_command,
    sanitize_inference_rows,
)
from proactive_r0.core import CausalInferenceConfig, StarterKitSymbols


class _FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[object], list[dict[str, str]], int | None]] = []

    def generate(self, frames, messages, max_new_tokens):
        self.calls.append(("generate", list(frames), list(messages), max_new_tokens))
        return "$silent$"

    def extract_decision_features(self, frames, messages):
        self.calls.append(("features", list(frames), list(messages), None))
        return NeuralDecisionFeatures(
            hidden_state=np.asarray([0.0], dtype=np.float32),
            silent_log_probability=-0.1,
            interrupt_log_probability=-0.2,
            tag_margin=-0.1,
            prompt_tokens=3,
            hidden_max_abs_difference=0.0,
            hidden_cosine_similarity=1.0,
            extraction_mode="shared_vision",
            candidate_forward_passes=2,
        )


def _head() -> LinearDecisionHead:
    names = ("domain=Chef", "tag_margin", "hidden_0000", *DIALOG_POLICY_NAMES)
    return LinearDecisionHead(
        feature_names=names,
        model=LinearModel(
            mean=(0.0,) * len(names),
            scale=(1.0,) * len(names),
            weight=(0.0,) * len(names),
            bias=-1.0,
            train_loss=0.0,
        ),
        threshold_logit=0.0,
    )


class CausalityTest(unittest.TestCase):
    def test_answers_are_removed_without_mutating_source(self) -> None:
        source = [{"video_path": "x.mp4", "answers": ["$interrupt$gold"]}]
        sanitized = sanitize_inference_rows(source)
        self.assertNotIn("answers", sanitized[0])
        self.assertIn("answers", source[0])

    def test_reproduction_command_preserves_physical_gpu_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command_path = Path(directory) / "command.sh"
            with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "4"}):
                _write_command(command_path, ["--device", "cuda:0"])
            command = command_path.read_text(encoding="utf-8")
        self.assertIn("export CUDA_VISIBLE_DEVICES=4", command)
        self.assertIn("--device cuda:0", command)

    def test_d4_calls_only_current_or_past_frames_and_current_dialog_prefix(self) -> None:
        row = {
            "video_path": "x.mp4",
            "video_intervals": [[0.0, 1.0], [5.0, 6.0], [9.0, 10.0]],
            "query": "help",
            "domain": "Chef",
            "dialog": [
                [{"role": "user", "text": "help"}],
                [
                    {"role": "user", "text": "help"},
                    {"role": "assistant", "text": "$interrupt$one"},
                ],
                [
                    {"role": "user", "text": "help"},
                    {"role": "assistant", "text": "$interrupt$one"},
                    {"role": "assistant", "text": "$interrupt$future only at chunk 2"},
                ],
            ],
        }
        extracted: list[tuple[float, float]] = []

        def extract_frames(path, *, intervals, frames_per_interval):
            interval = tuple(intervals[0])
            extracted.append(interval)
            return [interval] * frames_per_interval

        starter = StarterKitSymbols(
            system_prompt="system",
            normalize_dialog_turns=lambda turns: [
                {"role": str(turn["role"]), "content": str(turn["text"])}
                for turn in turns
            ],
            extract_frames=extract_frames,
        )
        fake = _FakeModel()
        record = process_session_with_dialog_stage_head(
            row,
            0,
            Path("/unused"),
            fake,
            starter,
            CausalInferenceConfig(1, 2, 16, 8),
            _head(),
        )
        self.assertEqual(extracted, [(0.0, 1.0), (5.0, 6.0), (9.0, 10.0)])
        generate_calls = [call for call in fake.calls if call[0] == "generate"]
        self.assertEqual(generate_calls[0][1], [(0.0, 1.0)])
        self.assertEqual(generate_calls[1][1], [(0.0, 1.0), (5.0, 6.0)])
        self.assertNotIn("future only", str(generate_calls[1][2]))
        self.assertIn("future only", str(generate_calls[2][2]))
        self.assertEqual(len(record["prediction"]["answers"]), 3)  # type: ignore[index]

    def test_timing_proxy_attaches_per_chunk_and_session_totals(self) -> None:
        fake = _FakeModel()
        timed = TimedDecisionModel(fake)  # type: ignore[arg-type]
        timed.generate([], [], 8)
        timed.extract_decision_features([], [])
        record = {"chunks": [{}]}
        timed.attach_timing(record, 0.1)
        timing = record["timing"]
        self.assertGreaterEqual(timing["model_inference_seconds"], 0.0)  # type: ignore[index]
        self.assertIn("generation_seconds", record["chunks"][0])  # type: ignore[index]
        self.assertIn("decision_feature_seconds", record["chunks"][0])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
