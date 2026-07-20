from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from proactive_d4.submission import (
    DEFAULT_CONFIG,
    DEFAULT_HEAD,
    DEFAULT_MANIFEST,
    build_runtime_config,
    publish_predictions,
    validate_bundle_manifest,
    validate_submission_input_rows,
)
from proactive_r0.artifacts import sha256_file
from proactive_r0.core import load_jsonl, write_jsonl


def _row(*, include_answers: bool = False) -> dict[str, object]:
    value: dict[str, object] = {
        "video_path": "sample.mp4",
        "video_intervals": [[0.0, 2.0], [2.0, 10.0]],
        "query": "Help me",
        "domain": "Chef",
        "dialog": [
            [{"role": "user", "text": "Help me"}],
            [
                {"role": "user", "text": "Help me"},
                {"role": "assistant", "text": "$interrupt$Start."},
            ],
        ],
    }
    if include_answers:
        value["answers"] = ["$interrupt$gold", "$silent$"]
    return value


class SubmissionInputTest(unittest.TestCase):
    def test_hidden_contract_rejects_answers(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain answers"):
            validate_submission_input_rows([_row(include_answers=True)])

    def test_local_audit_can_explicitly_allow_answers(self) -> None:
        result = validate_submission_input_rows(
            [_row(include_answers=True)],
            allow_answers_for_local_audit=True,
        )
        self.assertEqual(result["rows_with_answers"], 1)
        self.assertFalse(result["hidden_input_contract"])

    def test_dialog_must_align_with_intervals(self) -> None:
        row = _row()
        row["dialog"] = row["dialog"][:1]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "must align"):
            validate_submission_input_rows([row])


class RuntimeConfigTest(unittest.TestCase):
    def test_runtime_paths_replace_pinned_development_paths(self) -> None:
        frozen = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        original = copy.deepcopy(frozen)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            write_jsonl(input_path, [_row()])
            runtime = build_runtime_config(
                frozen,
                input_path=input_path,
                video_dir=root / "videos",
                model_dir=root / "model",
                head_path=root / "head.json",
                starter_kit_dir=root / "starter",
                input_contains_answers=False,
            )
            input_sha256 = sha256_file(input_path)
            runtime_input = str(input_path)
        self.assertEqual(frozen, original)
        self.assertEqual(runtime["data"]["input"], runtime_input)
        self.assertEqual(runtime["data"]["input_sha256"], input_sha256)
        self.assertEqual(runtime["decision_head"]["feature_variant"], "dialog_stage_fused")
        self.assertFalse(runtime["submission_runtime"]["scorer_invoked"])

    def test_manifest_matches_frozen_parameter_accounting(self) -> None:
        frozen = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        audit = validate_bundle_manifest(manifest, frozen)
        self.assertEqual(audit["total_parameters"], 1_060_898_844)
        self.assertEqual(audit["active_parameters"], 1_060_898_844)
        self.assertTrue(audit["small_eligible_by_parameter_count"])

    def test_bundled_head_matches_manifest(self) -> None:
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        self.assertTrue(DEFAULT_HEAD.is_file())
        self.assertEqual(
            sha256_file(DEFAULT_HEAD), manifest["decision_head"]["sha256"]
        )


class PredictionPublicationTest(unittest.TestCase):
    def test_only_validated_prediction_fields_are_published(self) -> None:
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            internal = root / "internal.jsonl"
            output = root / "predictions.jsonl"
            receipt_path = root / "receipt.json"
            rows = [_row()]
            write_jsonl(input_path, rows)
            write_jsonl(
                internal,
                [
                    {
                        "video_path": "sample.mp4",
                        "answers": ["$interrupt$Do it.", "$silent$"],
                    }
                ],
            )
            receipt = publish_predictions(
                source_rows=rows,
                internal_predictions_path=internal,
                output_path=output,
                receipt_path=receipt_path,
                manifest=manifest,
                frozen_config=json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8")),
                input_path=input_path,
            )
            published = load_jsonl(output)
        self.assertEqual(
            published,
            [
                {
                    "video_path": "sample.mp4",
                    "answers": ["$interrupt$Do it.", "$silent$"],
                }
            ],
        )
        self.assertEqual(receipt["validation"]["chunks"], 2)
        self.assertFalse(receipt["answers_read_by_inference"])


if __name__ == "__main__":
    unittest.main()
