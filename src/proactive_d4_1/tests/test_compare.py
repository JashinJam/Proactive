from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from proactive_d4_1.compare import (
    decision_change_statistics,
    flatten_decisions,
    load_official_scorer,
    merge_variant,
    official_score,
    paired_session_bootstrap,
    stratified_statistics,
    timing_statistics,
)
from proactive_r0.artifacts import write_json
from proactive_r0.core import write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class CompareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = load_official_scorer(
            PROJECT_ROOT / "starter_kit",
            "072301da6c65b3e30c7581920d178c6d5136305f2db26914df4785f47d809ee1",
        )

    def _data(self):
        golden = [
            {
                "video_path": "a.mp4",
                "domain": "A",
                "task": "one",
                "answers": ["$interrupt$x", "$silent$"],
            },
            {
                "video_path": "b.mp4",
                "domain": "B",
                "task": "two",
                "answers": ["$silent$", "$interrupt$x"],
            },
        ]
        baseline = [
            {"video_path": "a.mp4", "answers": ["$silent$", "$silent$"]},
            {"video_path": "b.mp4", "answers": ["$silent$", "$silent$"]},
        ]
        candidate = [
            {"video_path": "a.mp4", "answers": ["$interrupt$y", "$silent$"]},
            {"video_path": "b.mp4", "answers": ["$silent$", "$interrupt$y"]},
        ]
        return golden, baseline, candidate

    def test_official_aggregation_and_strata(self) -> None:
        golden, _, candidate = self._data()
        metrics = official_score(self.scorer, golden, candidate)
        self.assertEqual(metrics["overall"]["macro_f1"], 1.0)
        flattened = flatten_decisions(golden, candidate, [4, 7], {4: 1, 7: 4})
        strata = stratified_statistics(self.scorer, flattened)
        self.assertEqual(strata["domain"]["A"]["macro_f1"], 1.0)
        self.assertEqual(strata["session_length_quartile"]["Q4"]["support"], 2)
        self.assertIn("0:first", strata["chunk_position"])

    def test_change_counts_and_bootstrap_are_session_paired(self) -> None:
        golden, baseline, candidate = self._data()
        baseline_rows = flatten_decisions(golden, baseline, [4, 7], {4: 1, 7: 4})
        candidate_rows = flatten_decisions(golden, candidate, [4, 7], {4: 1, 7: 4})
        changes = decision_change_statistics(candidate_rows, baseline_rows)
        self.assertEqual(changes["changed_decisions"], 2)
        self.assertEqual(changes["corrected_errors"], 2)
        self.assertEqual(changes["new_errors"], 0)
        bootstrap = paired_session_bootstrap(
            candidate_rows, baseline_rows, repetitions=200, seed=3
        )
        self.assertGreater(bootstrap["delta_macro_f1_p2_5"], 0.0)
        self.assertEqual(bootstrap["unit"], "session")

    def test_timing_reports_limit_and_peak(self) -> None:
        records = [
            {
                "input_index": 2,
                "timing": {
                    "generation_seconds": 10.0,
                    "decision_feature_seconds": 20.0,
                    "model_inference_seconds": 30.0,
                    "session_wall_seconds": 35.0,
                },
            },
            {
                "input_index": 5,
                "timing": {
                    "generation_seconds": 100.0,
                    "decision_feature_seconds": 205.0,
                    "model_inference_seconds": 305.0,
                    "session_wall_seconds": 310.0,
                },
            },
        ]
        runtimes = [
            {"wall_time_seconds_this_attempt": 400.0, "peak_gpu_memory_bytes": 12},
            {"wall_time_seconds_this_attempt": 300.0, "peak_gpu_memory_bytes": 17},
        ]
        timing = timing_statistics(records, runtimes, session_limit_seconds=300.0)
        self.assertFalse(timing["deployable"])
        self.assertEqual(timing["sessions_over_limit"], [5])
        self.assertEqual(timing["peak_gpu_memory_bytes"], 17)
        self.assertEqual(timing["model_inference_seconds"]["max"], 305.0)

    def test_merge_variant_rebuilds_source_order_and_official_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {
                "video_path": "a.mp4",
                "video_intervals": [[0.0, 1.0], [1.0, 2.0]],
                "answers": ["$interrupt$x", "$silent$"],
                "domain": "A",
                "task": "one",
            }
            write_json(
                root / "config.json",
                {"starter_kit": {"path": str(PROJECT_ROOT / "starter_kit")}},
            )
            shard = root / "runs" / "search" / "v" / "shard_000"
            write_json(shard / "status.json", {"status": "complete"})
            record = {
                "input_index": 0,
                "video_path": "a.mp4",
                "prediction": {
                    "video_path": "a.mp4",
                    "answers": ["$interrupt$y", "$silent$"],
                },
                "chunks": [{}, {}],
                "timing": {
                    "generation_seconds": 1.0,
                    "decision_feature_seconds": 2.0,
                    "model_inference_seconds": 3.0,
                    "session_wall_seconds": 4.0,
                },
            }
            write_jsonl(shard / "session_records.jsonl", [record])
            write_json(
                shard / "runtime.json",
                {
                    "wall_time_seconds_this_attempt": 5.0,
                    "peak_gpu_memory_bytes": 99,
                },
            )
            summary, flattened = merge_variant(
                experiment_dir=root,
                stage_plan={
                    "stage": "search",
                    "indices": [0],
                    "shards": [[0]],
                },
                variant={
                    "variant_id": "v",
                    "parameters": {
                        "max_frames": 32,
                        "frames_per_interval": 16,
                        "max_history_turns": 4,
                        "max_new_tokens": 64,
                    },
                    "is_baseline": True,
                    "origin": "predefined",
                },
                source_rows=[source],
                scorer=self.scorer,
                quartile_by_index={0: 1},
                session_limit_seconds=300.0,
            )
            self.assertEqual(summary["overall"]["macro_f1"], 1.0)
            self.assertEqual(len(flattened), 2)
            self.assertTrue((root / "runs" / "search" / "v" / "metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
