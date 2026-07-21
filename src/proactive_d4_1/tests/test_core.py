from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from proactive_d4_1.core import (
    BASELINE,
    InferenceParameters,
    build_sample_manifest,
    compose_joint_variant,
    default_variants,
    partition_session_indices,
    prepare_task_directory,
    rank_summaries,
    stable_variant_id,
    task_should_run,
    validate_sample_manifest,
    validate_shard_records,
    validate_variants,
)


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_index in range(4):
        for session_index in range(44):
            chunks = 4 + session_index
            rows.append(
                {
                    "video_path": f"d{domain_index}_s{session_index}.mp4",
                    "domain": f"domain_{domain_index}",
                    "task": f"task_{session_index % 3}",
                    "query": "help",
                    "video_intervals": [[float(i), float(i + 1)] for i in range(chunks)],
                    "dialog": [[] for _ in range(chunks)],
                }
            )
    return rows


class VariantGridTest(unittest.TestCase):
    def test_grid_has_one_baseline_and_stable_ids(self) -> None:
        variants = default_variants()
        by_id = validate_variants(variants, expected_count=16)
        baselines = [variant for variant in variants if variant["is_baseline"]]
        self.assertEqual(len(baselines), 1)
        self.assertEqual(baselines[0]["parameters"], BASELINE.to_dict())
        self.assertEqual(set(baselines[0]["families"]), {"visual", "history", "generation"})
        self.assertEqual(stable_variant_id(BASELINE), baselines[0]["variant_id"])
        self.assertEqual(len(by_id), 16)

    def test_joint_uses_one_component_from_each_axis(self) -> None:
        variants = default_variants()
        visual = next(v for v in variants if v["parameters"]["max_frames"] == 64)  # type: ignore[index]
        history = next(v for v in variants if v["parameters"]["max_history_turns"] == 16)  # type: ignore[index]
        generation = next(v for v in variants if v["parameters"]["max_new_tokens"] == 96)  # type: ignore[index]
        joint = compose_joint_variant(visual, history, generation)
        self.assertEqual(
            joint["parameters"],
            InferenceParameters(64, 16, 16, 96).to_dict(),
        )
        self.assertEqual(joint["variant_id"], stable_variant_id(joint["parameters"]))

    def test_ranking_tie_break_is_stable(self) -> None:
        summaries = [
            {
                "variant_id": "b",
                "overall": {"macro_f1": 0.7, "gmean_f1": 0.6},
                "timing": {"total_model_inference_seconds": 9.0},
            },
            {
                "variant_id": "a",
                "overall": {"macro_f1": 0.7, "gmean_f1": 0.6},
                "timing": {"total_model_inference_seconds": 9.0},
            },
            {
                "variant_id": "c",
                "overall": {"macro_f1": 0.7, "gmean_f1": 0.61},
                "timing": {"total_model_inference_seconds": 20.0},
            },
        ]
        self.assertEqual(
            [item["variant_id"] for item in rank_summaries(summaries)],
            ["c", "a", "b"],
        )


class SamplingAndShardingTest(unittest.TestCase):
    def test_samples_are_disjoint_and_cover_every_stratum(self) -> None:
        rows = _rows()
        manifest = build_sample_manifest(rows)
        audit = validate_sample_manifest(manifest, rows)
        self.assertEqual(audit["search"], 80)
        self.assertEqual(audit["confirmation"], 80)
        self.assertFalse(
            set(manifest["search"]["indices"]) & set(manifest["confirmation"]["indices"])  # type: ignore[index]
        )
        smoke_index = manifest["smoke"]["indices"][0]  # type: ignore[index]
        search_chunks = [
            manifest["all_sessions"][index]["chunks"]  # type: ignore[index]
            for index in manifest["search"]["indices"]  # type: ignore[index]
        ]
        self.assertEqual(manifest["all_sessions"][smoke_index]["chunks"], min(search_chunks))  # type: ignore[index]
        for split in ("search", "confirmation"):
            for quartiles in audit["coverage"][split].values():  # type: ignore[index,union-attr]
                self.assertEqual(quartiles, {1: 5, 2: 5, 3: 5, 4: 5})

    def test_sampling_refuses_target_bearing_rows(self) -> None:
        rows = _rows()
        rows[0]["answers"] = ["$silent$"]
        with self.assertRaisesRegex(ValueError, "answers stripped"):
            build_sample_manifest(rows)

    def test_shards_exactly_cover_stage_and_preserve_local_order(self) -> None:
        rows = _rows()
        manifest = build_sample_manifest(rows)
        indices = manifest["search"]["indices"]  # type: ignore[index]
        shards = partition_session_indices(indices, manifest, 4)
        self.assertEqual(sorted(index for shard in shards for index in shard), indices)
        self.assertEqual(len({index for shard in shards for index in shard}), len(indices))
        self.assertTrue(all(shard == sorted(shard) for shard in shards))


class ResumeInvariantTest(unittest.TestCase):
    def test_completed_skips_failed_retries_and_hash_change_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            task = {"task_hash": "abc", "stage": "search"}
            prepare_task_directory(path, task)
            self.assertTrue(task_should_run(path, "abc"))
            (path / "status.json").write_text(
                json.dumps({"status": "failed", "task_hash": "abc"}), encoding="utf-8"
            )
            self.assertTrue(task_should_run(path, "abc"))
            (path / "status.json").write_text(
                json.dumps({"status": "complete", "task_hash": "abc"}), encoding="utf-8"
            )
            self.assertFalse(task_should_run(path, "abc"))
            with self.assertRaisesRegex(ValueError, "hash changed"):
                task_should_run(path, "different")
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                prepare_task_directory(path, {"task_hash": "changed", "stage": "search"})

    def test_partial_records_resume_only_as_exact_prefix(self) -> None:
        rows = _rows()
        expected = [0, 2, 4]
        records = [
            {
                "input_index": index,
                "video_path": rows[index]["video_path"],
                "prediction": {
                    "video_path": rows[index]["video_path"],
                    "answers": ["$silent$"] * len(rows[index]["video_intervals"]),
                },
            }
            for index in expected[:2]
        ]
        audit = validate_shard_records(
            records, expected, rows, require_complete=False
        )
        self.assertEqual(audit["sessions"], 2)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_shard_records(records, expected, rows, require_complete=True)
        records[1]["input_index"] = 4
        with self.assertRaisesRegex(ValueError, "order changed"):
            validate_shard_records(records, expected, rows, require_complete=False)


if __name__ == "__main__":
    unittest.main()
