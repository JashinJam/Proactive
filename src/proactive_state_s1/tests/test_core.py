from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from proactive_state_s1.core import validate_annotation, validate_collection
from proactive_state_s1.materialize import materialize
from proactive_state_s1.model import classification_metrics, fit_head, predict_head
from proactive_state_s1.evaluate import session_bootstrap_delta
from proactive_state_s1.record import append_record, materialize_complete


def session() -> dict[str, object]:
    return {
        "input_index": 3,
        "video_path": "video.mp4",
        "query": "How?",
        "task": "Task",
        "domain": "Tutorial",
        "length_band": "short",
        "state_split": "train",
        "chunks": [
            {"chunk_index": 0, "observed_through_sec": 2.0},
            {"chunk_index": 1, "observed_through_sec": 10.0},
            {"chunk_index": 2, "observed_through_sec": 18.0},
        ],
    }


def annotation() -> dict[str, object]:
    row = session()
    return {
        "schema_version": 1,
        "status": "complete",
        **{key: row[key] for key in (
            "input_index", "video_path", "query", "task", "domain",
            "length_band", "state_split",
        )},
        "provenance": {
            "plan_inputs": ["task", "query"],
            "chunk_inputs": [
                "task", "query", "dialog_at_chunk", "video_intervals_so_far"
            ],
            "excluded_inputs": [
                "answers", "future_dialog", "future_video", "model_outputs",
                "R0/D1/D3 errors", "ratings", "existing_oracle_states",
            ],
            "annotation_type": "s1_training_or_heldout_causal_state_supervision",
        },
        "goal": "Finish the task.",
        "steps": [
            {
                "id": f"s{i}",
                "text": f"Step {i}",
                "completion_cues": [f"Step {i} done"],
                "incompletion_cues": [f"Step {i} unfinished"],
            }
            for i in range(1, 5)
        ],
        "chunk_states": [
            {
                "chunk_index": 0,
                "observed_through_sec": 2.0,
                "current_step_id": "s1",
                "progress": "ongoing",
                "completion_evidence": ["Materials visible"],
                "incompletion_or_error_evidence": ["Step unfinished"],
                "next_step_id": "s2",
                "recovery_action": "Continue step one.",
                "confidence": 0.8,
            },
            {
                "chunk_index": 1,
                "observed_through_sec": 10.0,
                "current_step_id": "s1",
                "progress": "deviated",
                "completion_evidence": [],
                "incompletion_or_error_evidence": ["Wrong action"],
                "next_step_id": "s2",
                "recovery_action": "Return to step one.",
                "confidence": 0.9,
            },
            {
                "chunk_index": 2,
                "observed_through_sec": 18.0,
                "current_step_id": "s1",
                "progress": "recovered",
                "completion_evidence": ["Correct action resumed"],
                "incompletion_or_error_evidence": ["Step still unfinished"],
                "next_step_id": "s2",
                "recovery_action": "Finish step one.",
                "confidence": 0.9,
            },
        ],
    }


class StateS1ValidationTest(unittest.TestCase):
    def test_valid_annotation(self) -> None:
        self.assertEqual(
            validate_annotation(annotation(), session()),
            {"states": 3, "error_present": 3},
        )

    def test_collection_requires_exact_split_coverage(self) -> None:
        self.assertEqual(
            validate_collection([annotation()], [session()], expected_split="train")["sessions"],
            1,
        )
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_collection([], [session()], expected_split="train")

    def test_complete_cannot_have_error_evidence(self) -> None:
        value = copy.deepcopy(annotation())
        value["chunk_states"][0]["progress"] = "complete"
        with self.assertRaisesRegex(ValueError, "complete has error evidence"):
            validate_annotation(value, session())

    def test_recovered_must_follow_deviated(self) -> None:
        value = copy.deepcopy(annotation())
        value["chunk_states"][1]["progress"] = "ongoing"
        with self.assertRaisesRegex(ValueError, "recovered must follow deviated"):
            validate_annotation(value, session())

    def test_target_markers_are_forbidden(self) -> None:
        value = copy.deepcopy(annotation())
        value["chunk_states"][0]["recovery_action"] = "$interrupt$ do this"
        with self.assertRaisesRegex(ValueError, "forbidden marker"):
            validate_annotation(value, session())

    def test_materialize_physically_separates_splits(self) -> None:
        heldout = copy.deepcopy(session())
        heldout["input_index"] = 4
        heldout["video_path"] = "heldout.mp4"
        heldout["state_split"] = "heldout"
        train_annotation = annotation()
        heldout_annotation = copy.deepcopy(annotation())
        heldout_annotation.update({
            "input_index": 4,
            "video_path": "heldout.mp4",
            "state_split": "heldout",
        })
        plans = [
            {"input_index": row["input_index"], "goal": row["goal"], "steps": row["steps"]}
            for row in (train_annotation, heldout_annotation)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions_path = root / "sessions.jsonl"
            sessions_path.write_text(
                "\n".join(json.dumps(row) for row in (session(), heldout)) + "\n",
                encoding="utf-8",
            )
            template_path = root / "template.json"
            template_path.write_text(
                json.dumps([train_annotation, heldout_annotation]), encoding="utf-8"
            )
            plans_path = root / "plans.json"
            plans_path.write_text(json.dumps(plans), encoding="utf-8")
            result = materialize(sessions_path, template_path, plans_path, root / "work")
            self.assertEqual(result["artifacts"]["train"]["sessions"], 1)
            self.assertEqual(result["artifacts"]["heldout"]["sessions"], 1)
            self.assertTrue((root / "work/train/annotations.json").exists())
            self.assertTrue((root / "work/heldout/annotations.json").exists())

    def test_record_is_append_only_and_materializes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions_path = root / "sessions.jsonl"
            sessions_path.write_text(json.dumps(session()) + "\n", encoding="utf-8")
            work_path = root / "annotations.json"
            work = annotation()
            work["status"] = "incomplete"
            work["chunk_states"] = [
                {"chunk_index": i, "observed_through_sec": value}
                for i, value in enumerate((2.0, 10.0, 18.0))
            ]
            work_path.write_text(json.dumps([work]), encoding="utf-8")
            records_path = root / "records.jsonl"
            append_record(
                sessions_path, records_path, 3, 0, "s1", "ongoing",
                ["Started"], ["Not done"], "Continue.", 0.8,
            )
            with self.assertRaisesRegex(ValueError, "next record must be chunk 1"):
                append_record(
                    sessions_path, records_path, 3, 0, "s1", "ongoing",
                    ["Started"], ["Not done"], "Continue.", 0.8,
                )
            append_record(
                sessions_path, records_path, 3, 1, "s1", "deviated",
                [], ["Wrong action"], "Return.", 0.9,
            )
            append_record(
                sessions_path, records_path, 3, 2, "s1", "recovered",
                ["Resumed"], ["Not done"], "Finish.", 0.9,
            )
            output = root / "complete.json"
            summary = materialize_complete(
                sessions_path, work_path, records_path, "train", output
            )
            self.assertEqual(summary, {"sessions": 1, "states": 3, "error_present": 3})
            self.assertTrue(output.exists())

    def test_standardized_linear_head_round_trip(self) -> None:
        import numpy as np

        values = np.asarray(
            [[-2.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
            dtype=np.float64,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        model = fit_head(values, labels, 0.1)
        predicted = predict_head(model, values)
        self.assertEqual(predicted.tolist(), labels.tolist())
        self.assertEqual(classification_metrics(labels, predicted, 2)["macro_f1"], 1.0)

    def test_session_bootstrap_uses_paired_sessions(self) -> None:
        import numpy as np

        result = session_bootstrap_delta(
            np.asarray([1, 1, 2, 2]),
            np.asarray([1.0, 1.0, 0.5, 0.5]),
            np.asarray([0.0, 0.0, 0.5, 0.5]),
            repetitions=1000,
            seed=7,
        )
        self.assertEqual(result["observed_session_mean_delta"], 0.5)
        self.assertGreaterEqual(result["ci95_low"], 0.0)


if __name__ == "__main__":
    unittest.main()
