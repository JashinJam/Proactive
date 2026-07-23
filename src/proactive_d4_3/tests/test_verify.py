from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from proactive_d3.dialog_control_core import DIALOG_POLICY_NAMES
from proactive_d4_3.verify import verify
from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import write_jsonl


class History8VerificationTest(unittest.TestCase):
    def test_exact_synthetic_record_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deployment = root / "deployment"
            deployment.mkdir()
            generation_path = root / "generation.jsonl"
            final_records_path = root / "final_records.jsonl"
            final_predictions_path = root / "final_predictions.jsonl"
            cache_path = root / "features.npz"
            dialog = {name: float(index) for index, name in enumerate(DIALOG_POLICY_NAMES)}
            chunk = {
                "chunk_index": 0,
                "raw_response": "ok",
                "prompt_tokens": 10,
                "model_input_frames": 2,
                "tag_margin": 0.5,
                "silent_log_probability": -1.0,
                "interrupt_log_probability": -0.5,
                "hidden_state": [1.0, 2.0],
                "decision_logit": 0.7,
                "decision_interrupt": 1,
                "answer": "$interrupt$ok",
                **dialog,
            }
            write_jsonl(
                deployment / "session_records.jsonl",
                [
                    {
                        "input_index": 0,
                        "session_wall_time_seconds": 1.0,
                        "chunks": [chunk],
                    }
                ],
            )
            write_jsonl(deployment / "predictions.jsonl", [{"video_path": "a.mp4", "answers": ["$interrupt$ok"]}])
            write_json(
                deployment / "runtime.json",
                {"total_parameters": 100, "peak_gpu_memory_bytes": 10},
            )
            write_jsonl(
                generation_path,
                [{"input_index": 0, "chunks": [{**chunk, **dialog}]}],
            )
            write_jsonl(
                final_records_path,
                [{"input_index": 0, "chunk_index": 0, "predicted_interrupt": 1, "logit": 0.7}],
            )
            write_jsonl(
                final_predictions_path,
                [{"video_path": "a.mp4", "answers": ["$interrupt$ok"]}],
            )
            np.savez_compressed(
                cache_path,
                hidden_state=np.asarray([[1.0, 2.0]], dtype=np.float32),
                tag_margin=np.asarray([0.5], dtype=np.float32),
                silent_log_probability=np.asarray([-1.0], dtype=np.float32),
                interrupt_log_probability=np.asarray([-0.5], dtype=np.float32),
                prompt_tokens=np.asarray([10], dtype=np.int32),
                input_index=np.asarray([0], dtype=np.int32),
                chunk_index=np.asarray([0], dtype=np.int32),
            )
            specs = {}
            for name, path in {
                "reference_generation_records": generation_path,
                "reference_neural_cache": cache_path,
                "reference_final_records": final_records_path,
                "reference_final_predictions": final_predictions_path,
            }.items():
                specs[name] = {"path": str(path), "sha256": sha256_file(path)}
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "smoke": {
                            "numeric_abs_tolerance": 1e-6,
                            "session_indices": [0],
                            "expected_sessions": 1,
                            "expected_chunks": 1,
                            "max_session_seconds": 300,
                            **specs,
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = verify(deployment, config_path)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["exact_match_counts"]["answer"], 1)  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
