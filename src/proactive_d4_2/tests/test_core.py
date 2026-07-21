from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from proactive_d4_1.core import prepare_task_directory, task_should_run
from proactive_d4_2.core import (
    BASELINE,
    build_source_manifest,
    feature_task_hash,
    load_candidates,
    partition_indices,
    rank_summaries,
    recover_duplicate_feature_prefix,
    stable_candidate_id,
    validate_feature_records,
)
from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import write_jsonl
from proactive_d4_2.run import experiment_lock, select_gpus
from proactive_d4_2.run_features import _recover_duplicate_records


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def full_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(700):
        chunks = 15 if index < 135 else 14
        rows.append(
            {
                "video_path": f"video-{index}.mp4",
                "domain": f"domain-{index % 4}",
                "task": f"task-{index % 9}",
                "query": "help",
                "video_intervals": [[float(i), float(i + 1)] for i in range(chunks)],
                "dialog": [[] for _ in range(chunks)],
            }
        )
    return rows


def feature_record(row: dict[str, object], input_index: int) -> dict[str, object]:
    chunks = []
    for chunk_index, interval in enumerate(row["video_intervals"]):  # type: ignore[index]
        chunks.append(
            {
                "chunk_index": chunk_index,
                "interval": interval,
                "model_input_frames": 2,
                "hidden_state": [0.1, -0.2],
                "tag_margin": 0.3,
                "silent_log_probability": -0.4,
                "interrupt_log_probability": -0.1,
                "prompt_tokens": 8,
                "raw_response": "$silent$",
                "r0_answer": "$silent$",
                "model_inference_seconds": 0.01,
            }
        )
    return {
        "input_index": input_index,
        "video_path": row["video_path"],
        "chunks": chunks,
    }


class CandidateProtocolTest(unittest.TestCase):
    def test_exact_four_mechanism_candidates_and_stable_ids(self) -> None:
        config = json.loads(
            (
                PROJECT_ROOT
                / "configs/d4_2_internvl35_1b_adapted_input_policy_oof_v1.json"
            ).read_text(encoding="utf-8")
        )
        candidates = load_candidates(config)
        self.assertEqual([item["name"] for item in candidates], ["baseline", "history8", "frames16", "tokens16"])
        self.assertEqual(sum(bool(item["is_baseline"]) for item in candidates), 1)
        self.assertEqual(candidates[0]["parameters"], BASELINE.to_dict())
        self.assertEqual(candidates[0]["candidate_id"], stable_candidate_id(BASELINE))

    def test_ranking_uses_metric_time_and_stable_id(self) -> None:
        summaries = [
            {"candidate_id": "b", "overall": {"macro_f1": 0.7, "gmean_f1": 0.6}, "timing": {"total_model_inference_seconds": 8.0}},
            {"candidate_id": "a", "overall": {"macro_f1": 0.7, "gmean_f1": 0.6}, "timing": {"total_model_inference_seconds": 8.0}},
            {"candidate_id": "c", "overall": {"macro_f1": 0.7, "gmean_f1": 0.61}, "timing": {"total_model_inference_seconds": 20.0}},
        ]
        self.assertEqual([row["candidate_id"] for row in rank_summaries(summaries)], ["c", "a", "b"])


class ManifestAndResumeTest(unittest.TestCase):
    def test_label_free_manifest_and_shards_cover_exact_dataset(self) -> None:
        rows = full_rows()
        manifest = build_source_manifest(rows)
        self.assertFalse(manifest["labels_used"])
        self.assertEqual(sum(item["chunks"] for item in manifest["sessions"]), 9935)  # type: ignore[union-attr]
        shards = partition_indices(manifest, 7)
        flattened = [index for shard in shards for index in shard]
        self.assertEqual(sorted(flattened), list(range(700)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(all(shard == sorted(shard) for shard in shards))

    def test_manifest_refuses_labels(self) -> None:
        rows = full_rows()
        rows[0]["answers"] = ["$silent$"]
        with self.assertRaisesRegex(ValueError, "answers stripped"):
            build_source_manifest(rows)

    def test_feature_task_hash_includes_hidden_recording_and_policy(self) -> None:
        candidate = {
            "candidate_id": stable_candidate_id(BASELINE),
            "parameters": BASELINE.to_dict(),
        }
        first = feature_task_hash(
            experiment_config_sha256="x",
            candidate=candidate,
            shard_id=0,
            session_indices=[0, 2],
        )
        changed = dict(candidate)
        changed["parameters"] = {**BASELINE.to_dict(), "max_new_tokens": 16}
        second = feature_task_hash(
            experiment_config_sha256="x",
            candidate=changed,
            shard_id=0,
            session_indices=[0, 2],
        )
        self.assertNotEqual(first, second)

    def test_resume_retries_failure_skips_complete_and_rejects_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            task = {"task_hash": "one", "stage": "features"}
            prepare_task_directory(task_dir, task)
            self.assertTrue(task_should_run(task_dir, "one"))
            write_json(task_dir / "status.json", {"status": "failed", "task_hash": "one"})
            self.assertTrue(task_should_run(task_dir, "one"))
            write_json(task_dir / "status.json", {"status": "complete", "task_hash": "one"})
            self.assertFalse(task_should_run(task_dir, "one"))
            with self.assertRaisesRegex(ValueError, "hash changed"):
                task_should_run(task_dir, "two")

    def test_partial_feature_records_must_be_exact_prefix_with_full_hidden(self) -> None:
        rows = full_rows()
        expected = [0, 3]
        records = [feature_record(rows[0], 0)]
        audit = validate_feature_records(
            records, expected, rows, hidden_size=2, require_complete=False
        )
        self.assertEqual(audit["sessions"], 1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_feature_records(
                records, expected, rows, hidden_size=2, require_complete=True
            )
        records[0]["chunks"][0]["hidden_state"] = [0.1]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "hidden-state"):
            validate_feature_records(
                records, expected, rows, hidden_size=2, require_complete=False
            )

    def test_identical_interleaved_duplicates_recover_to_exact_prefix(self) -> None:
        rows = full_rows()
        first = feature_record(rows[0], 0)
        second = feature_record(rows[3], 3)
        first_retry = copy.deepcopy(first)
        second_retry = copy.deepcopy(second)
        first_retry["chunks"][0]["model_inference_seconds"] = 9.0  # type: ignore[index]
        second_retry["timing"] = {"session_wall_seconds": 99.0}
        recovered, audit = recover_duplicate_feature_prefix(
            [first, second, first_retry, second_retry],
            [0, 3, 4],
            rows,
            hidden_size=2,
        )
        self.assertEqual([record["input_index"] for record in recovered], [0, 3])
        self.assertTrue(audit["recovered"])
        self.assertEqual(audit["discarded_duplicates"], 2)

    def test_semantically_different_duplicate_is_rejected(self) -> None:
        rows = full_rows()
        first = feature_record(rows[0], 0)
        changed = copy.deepcopy(first)
        changed["chunks"][0]["tag_margin"] = 0.4  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "semantically different"):
            recover_duplicate_feature_prefix(
                [first, changed], [0, 3], rows, hidden_size=2
            )

    def test_non_prefix_duplicates_are_rejected(self) -> None:
        rows = full_rows()
        record = feature_record(rows[3], 3)
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            recover_duplicate_feature_prefix(
                [record, copy.deepcopy(record)], [0, 3], rows, hidden_size=2
            )

    def test_experiment_lock_rejects_second_master(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment_dir = Path(directory) / "experiment"
            with experiment_lock(experiment_dir):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with experiment_lock(experiment_dir):
                        self.fail("concurrent master lock unexpectedly acquired")

    def test_duplicate_recovery_preserves_verified_backup_and_status(self) -> None:
        rows = full_rows()
        first = feature_record(rows[0], 0)
        retry = copy.deepcopy(first)
        retry["chunks"][0]["model_inference_seconds"] = 2.0  # type: ignore[index]
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            records_path = task_dir / "session_records.jsonl"
            write_jsonl(records_path, [first, retry])
            write_json(
                task_dir / "status.json",
                {"status": "failed", "task_hash": "task", "attempt": 2},
            )
            original_sha256 = sha256_file(records_path)
            recovered, audit = _recover_duplicate_records(
                task_dir=task_dir,
                records_path=records_path,
                records=[first, retry],
                expected_indices=[0, 3],
                rows=rows,
                hidden_size=2,
            )
            backup = task_dir / "session_records.pre_duplicate_recovery_001.jsonl"
            self.assertEqual(len(recovered), 1)
            self.assertEqual(sha256_file(backup), original_sha256)
            self.assertEqual(audit["previous_status"]["attempt"], 2)  # type: ignore[index]
            self.assertTrue((task_dir / "duplicate_recovery_001.json").exists())


class GpuSelectionTest(unittest.TestCase):
    @mock.patch("proactive_d4_2.run.subprocess.run")
    def test_explicit_shared_mode_accepts_all_existing_gpu_ids(
        self, run: mock.Mock
    ) -> None:
        run.return_value = subprocess.CompletedProcess(
            [], 0, stdout="0\n1\n2\n3\n4\n5\n6\n7\n", stderr=""
        )
        self.assertEqual(
            select_gpus(8, "0,1,2,3,4,5,6,7", allow_shared_gpu=True),
            [str(index) for index in range(8)],
        )

    @mock.patch("proactive_d4_2.run.discover_idle_gpus")
    def test_exclusive_mode_delegates_to_idle_only_selector(
        self, discover: mock.Mock
    ) -> None:
        discover.return_value = ["4", "6", "7"]
        self.assertEqual(
            select_gpus(3, "4,6,7", allow_shared_gpu=False), ["4", "6", "7"]
        )
        discover.assert_called_once_with(3, "4,6,7")


if __name__ == "__main__":
    unittest.main()
