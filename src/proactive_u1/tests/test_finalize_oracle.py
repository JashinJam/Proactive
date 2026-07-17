from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from proactive_u1.finalize_oracle import finalize_oracle


class FinalizeOracleTest(unittest.TestCase):
    def test_merges_disjoint_formal_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_paths = []
            annotation_paths = []
            for index in (1, 2):
                sample = {
                    "sample_id": f"sample-{index}",
                    "input_index": index,
                    "chunk_index": index,
                    "observed_through_sec": float(index),
                }
                sample_path = root / f"samples-{index}.jsonl"
                sample_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
                sample_paths.append(sample_path)
                annotation = {
                    "status": "complete",
                    "input_index": index,
                    "provenance": {
                        "annotation_type": "formal_blind_evaluation_only_oracle_non_deployable",
                        "excluded_inputs": [
                            "answers",
                            "future_dialog",
                            "future_video",
                            "model_outputs",
                            "R0/D1 errors",
                        ],
                    },
                    "steps": [{"id": "s1", "text": "Do the task."}],
                    "sampled_chunk_states": [
                        {
                            "sample_id": f"sample-{index}",
                            "chunk_index": index,
                            "observed_through_sec": float(index),
                            "current_step_id": "s1",
                            "next_step_id": None,
                            "progress": "ongoing",
                            "completion_evidence": [],
                            "incompletion_or_error_evidence": [],
                            "recovery_action": "Continue the task.",
                            "confidence": 0.5,
                        }
                    ],
                }
                annotation_path = root / f"annotations-{index}.json"
                annotation_path.write_text(json.dumps([annotation]), encoding="utf-8")
                annotation_paths.append(annotation_path)
            rows, validation = finalize_oracle(
                sample_paths,
                annotation_paths,
                expected_sessions=2,
                expected_states=2,
            )
            self.assertEqual([row["input_index"] for row in rows], [1, 2])
            self.assertTrue(validation["formal_blind_provenance"])

    def test_rejects_nonblind_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "samples.jsonl"
            sample_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "input_index": 1,
                        "chunk_index": 1,
                        "observed_through_sec": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            annotation_path = root / "annotation.json"
            annotation_path.write_text(
                json.dumps(
                    [
                        {
                            "status": "complete",
                            "input_index": 1,
                            "provenance": {
                                "annotation_type": "engineering_smoke_nonblind_diagnostic_only",
                                "excluded_inputs": [],
                            },
                            "steps": [{"id": "s1", "text": "Do it."}],
                            "sampled_chunk_states": [],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "wrong provenance"):
                finalize_oracle(
                    [sample_path], [annotation_path], expected_sessions=1, expected_states=1
                )


if __name__ == "__main__":
    unittest.main()
