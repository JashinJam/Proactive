from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from proactive_d2.merge_final_mlp_cache import (
    atomic_output_directory,
    combine_record_aggregates,
    validate_code_states,
    validate_data_manifest,
    validate_merged_against_d1,
    validate_record_chunks,
    validate_session_arrays,
    validate_shard_summary,
)
from proactive_d2.final_mlp_cache import STATE_ARRAY_NAMES
from proactive_r0.artifacts import sha256_file


def _cache_config() -> dict[str, object]:
    return {
        "frames_per_interval": 2,
        "max_frames": 4,
        "tag_tokens_each": 1,
        "hidden_size": 2,
        "required_zero_adapter_checks": {
            "max_hidden_abs_difference_vs_full_base": 0.0,
            "max_logit_abs_difference_vs_full_base": 0.0,
            "max_margin_abs_difference_vs_d1_cache": 1e-6,
        },
    }


def _source() -> dict[str, object]:
    return {
        "video_path": "video.mp4",
        "video_intervals": [[0.0, 1.0], [1.0, 2.0]],
    }


def _record(d1_margin_difference: float = 5e-7) -> dict[str, object]:
    margins = [float(np.float32(0.25)), float(np.float32(-0.5))]
    chunks = []
    for index, interval in enumerate(_source()["video_intervals"]):  # type: ignore[index]
        chunks.append(
            {
                "chunk_index": index,
                "interval": interval,
                "current_interval_frames": 2,
                "model_input_frames": 2 + index * 2,
                "prompt_tokens": 10 + index,
                "base_tag_margin": margins[index],
                "corrected_hidden_max_abs_difference": 0.0,
                "corrected_logit_max_abs_difference": 0.0,
                "candidate_hidden_max_abs_difference": 0.0,
                "d1_hidden_max_abs_difference": 0.0,
                "d1_margin_abs_difference_float32": d1_margin_difference,
                "d1_prompt_tokens_match": True,
            }
        )
    return {
        "input_index": 0,
        "video_path": "video.mp4",
        "source_chunks": 2,
        "extracted_chunks": 2,
        "complete_session": True,
        "state_shape": [2, 1, 2],
        "state_bytes_uncompressed": 96,
        "chunks": chunks,
    }


def _diagnostics() -> dict[str, np.ndarray]:
    return {
        "prompt_tokens": np.asarray([10, 11], dtype=np.int32),
        "base_tag_margin": np.asarray([0.25, -0.5], dtype=np.float32),
    }


class MergeFinalMLPCacheTest(unittest.TestCase):
    def test_session_archive_hash_shape_and_dtype_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session_dir = Path(temporary) / "sessions"
            session_dir.mkdir()
            feature_path = session_dir / "session_0000.npz"
            arrays = {
                name: np.zeros((2, 1, 2), dtype=np.uint16)
                for name in STATE_ARRAY_NAMES
            }
            arrays.update(
                {
                    "base_hidden_state": np.zeros((2, 2), dtype=np.float32),
                    "base_tag_margin": np.zeros(2, dtype=np.float32),
                    "base_silent_log_probability": np.zeros(2, dtype=np.float32),
                    "base_interrupt_log_probability": np.zeros(2, dtype=np.float32),
                    "prompt_tokens": np.asarray([10, 11], dtype=np.int32),
                    "input_index": np.asarray(0, dtype=np.int32),
                    "chunk_index": np.asarray([0, 1], dtype=np.int32),
                }
            )
            np.savez_compressed(feature_path, **arrays)
            record = {
                "input_index": 0,
                "source_chunks": 2,
                "extracted_chunks": 2,
                "complete_session": True,
                "feature_path": str(feature_path),
                "feature_sha256": sha256_file(feature_path),
                "state_bytes_uncompressed": 96,
            }
            loaded = validate_session_arrays(
                record,
                hidden_size=2,
                tag_length=1,
                bytes_per_chunk=48,
                expected_session_dir=session_dir,
            )
            self.assertEqual(loaded["prompt_tokens"].dtype, np.dtype(np.int32))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_session_arrays(
                    {**record, "feature_sha256": "bad"},
                    hidden_size=2,
                    tag_length=1,
                    bytes_per_chunk=48,
                    expected_session_dir=session_dir,
                )

    def test_record_gate_uses_configured_d1_margin_tolerance(self) -> None:
        aggregate = validate_record_chunks(
            _record(5e-7), _source(), _diagnostics(), _cache_config()
        )
        self.assertEqual(aggregate["max_d1_margin_abs_difference_float32"], 5e-7)
        with self.assertRaisesRegex(ValueError, "exactness gate"):
            validate_record_chunks(
                _record(2e-6), _source(), _diagnostics(), _cache_config()
            )

    def test_record_metadata_must_match_source_and_arrays(self) -> None:
        record = _record()
        record["chunks"][1]["interval"] = [1.0, 3.0]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "interval differs"):
            validate_record_chunks(record, _source(), _diagnostics(), _cache_config())

        record = _record()
        record["chunks"][1]["prompt_tokens"] = 99  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "prompt metadata differs"):
            validate_record_chunks(record, _source(), _diagnostics(), _cache_config())

    def test_summary_and_runtime_must_pin_records(self) -> None:
        aggregate = combine_record_aggregates(
            [validate_record_chunks(_record(), _source(), _diagnostics(), _cache_config())]
        )
        summary = {
            "status": "complete final-MLP cache extraction",
            **aggregate,
            "all_sessions_complete": True,
            "hidden_size": 2,
            "tag_length": 1,
            "storage_dtype": "uint16_bfloat16_bits",
            "records_sha256": "abc",
            "labels_read_or_stored": False,
        }
        runtime = {**summary, "wall_time_seconds": 1.0}
        validate_shard_summary(
            summary,
            runtime,
            aggregate,
            records_sha256="abc",
            hidden_size=2,
            tag_length=1,
        )
        broken = {**summary, "records_sha256": "tampered"}
        with self.assertRaisesRegex(ValueError, "records fingerprint"):
            validate_shard_summary(
                broken,
                {**broken},
                aggregate,
                records_sha256="abc",
                hidden_size=2,
                tag_length=1,
            )
        with self.assertRaisesRegex(ValueError, "runtime and summary"):
            validate_shard_summary(
                summary,
                {**runtime, "chunks": 99},
                aggregate,
                records_sha256="abc",
                hidden_size=2,
                tag_length=1,
            )

    def test_code_state_and_data_manifest_consistency(self) -> None:
        validate_code_states([{"file_sha256": {"extract.py": "a"}}] * 2)
        with self.assertRaisesRegex(ValueError, "code states differ"):
            validate_code_states(
                [
                    {"file_sha256": {"extract.py": "a"}},
                    {"file_sha256": {"extract.py": "b"}},
                ]
            )

        config = {
            "data": {"input": "/tmp/source.jsonl", "input_sha256": "source"},
            "model": {"weights_sha256": "model", "total_parameters": 10},
            "starter_kit": {
                "model_py_sha256": "m",
                "proactive_py_sha256": "p",
                "scorer_py_sha256": "s",
            },
            "cache": _cache_config(),
            "d1_neural_cache_reference": {
                "features_sha256": "d1f",
                "records_sha256": "d1r",
            },
        }
        runtime = {"selection_start": 0, "selection_stop": 1}
        record = {"source_chunks": 2}
        manifest = {
            "source": {
                "path": "/tmp/source.jsonl",
                "sha256": "source",
                "sessions_selected": 1,
                "chunks_selected": 2,
                "answers_present_in_generation_rows": False,
                "selection_start": 0,
                "selection_stop": 1,
            },
            "model": {"weights_sha256": "model", "stored_unique_parameters": 10},
            "starter_kit_sha256": {
                "input_sha256": "source",
                "model_py_sha256": "m",
                "proactive_py_sha256": "p",
                "scorer_py_sha256": "s",
            },
            "d1_neural_reference_sha256": {
                "features_sha256": "d1f",
                "records_sha256": "d1r",
            },
            "cache": _cache_config(),
            "supervision": {
                "answers_removed_before_extraction": True,
                "labels_read_or_stored": False,
            },
        }
        validate_data_manifest(
            manifest, config=config, runtime_config=runtime, records=[record]
        )
        broken_manifest = copy.deepcopy(manifest)
        d1_hashes = broken_manifest["d1_neural_reference_sha256"]
        assert isinstance(d1_hashes, dict)
        d1_hashes["features_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "D1 fingerprints"):
            validate_data_manifest(
                broken_manifest,
                config=config,
                runtime_config=runtime,
                records=[record],
            )

    def test_direct_d1_validation_checks_arrays_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reference_dir = Path(temporary)
            features_path = reference_dir / "features.npz"
            records_path = reference_dir / "records.jsonl"
            summary_path = reference_dir / "summary.json"
            np.savez_compressed(
                features_path,
                hidden_state=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                tag_margin=np.asarray([0.0, 0.0], dtype=np.float32),
                prompt_tokens=np.asarray([10, 11], dtype=np.int32),
                input_index=np.asarray([0, 0], dtype=np.int32),
                chunk_index=np.asarray([0, 1], dtype=np.int32),
            )
            records_path.write_text("{}\n", encoding="utf-8")
            summary_path.write_text(
                json.dumps({"labels_read_or_stored": False}), encoding="utf-8"
            )
            config = {
                "cache": _cache_config(),
                "d1_neural_cache_reference": {
                    "path": str(reference_dir),
                    "features_sha256": sha256_file(features_path),
                    "records_sha256": sha256_file(records_path),
                },
            }
            merged = {
                "base_hidden_state": np.asarray(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
                ),
                "base_tag_margin": np.asarray([5e-7, 0.0], dtype=np.float32),
                "prompt_tokens": np.asarray([10, 11], dtype=np.int32),
                "input_index": np.asarray([0, 0], dtype=np.int32),
                "chunk_index": np.asarray([0, 1], dtype=np.int32),
            }
            keys = np.asarray([[0, 0], [0, 1]], dtype=np.int32)
            result = validate_merged_against_d1(
                merged, config=config, expected_keys=keys
            )
            self.assertLess(float(result["margin_max_abs_difference_float32"]), 1e-6)
            merged["base_tag_margin"][0] = np.float32(2e-6)
            with self.assertRaisesRegex(ValueError, "margins differ"):
                validate_merged_against_d1(merged, config=config, expected_keys=keys)

    def test_atomic_output_failure_never_exposes_final_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "merged"
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with atomic_output_directory(output_dir) as staging_dir:
                    (staging_dir / "features.npz").write_bytes(b"partial")
                    raise RuntimeError("injected")
            self.assertFalse(output_dir.exists())
            self.assertEqual(list(Path(temporary).glob(".merged.tmp-*")), [])

            with atomic_output_directory(output_dir) as staging_dir:
                (staging_dir / "features.npz").write_bytes(b"complete")
            self.assertEqual((output_dir / "features.npz").read_bytes(), b"complete")


if __name__ == "__main__":
    unittest.main()
