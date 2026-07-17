"""Validate and merge complete label-free final-MLP cache shards for D2."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

from proactive_d1.core import strip_answers
from proactive_r0.artifacts import code_snapshot, environment_snapshot, sha256_file, write_json
from proactive_r0.core import load_jsonl, write_jsonl
from proactive_r0.run import contiguous_shard_bounds

from .final_mlp_cache import STATE_ARRAY_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ID = "20260716_internvl35_1b_final_mlp_cache_d2_v2"
DEFAULT_BASE = f"output/features/{DEFAULT_CACHE_ID}"

DIFFERENCE_THRESHOLDS: dict[str, str | None] = {
    "corrected_hidden_max_abs_difference": "max_hidden_abs_difference_vs_full_base",
    "corrected_logit_max_abs_difference": "max_logit_abs_difference_vs_full_base",
    "candidate_hidden_max_abs_difference": None,
    "d1_hidden_max_abs_difference": None,
    "d1_margin_abs_difference_float32": "max_margin_abs_difference_vs_d1_cache",
}
SUMMARY_DIFFERENCE_FIELDS: dict[str, str] = {
    "corrected_hidden_max_abs_difference": "max_corrected_hidden_abs_difference",
    "corrected_logit_max_abs_difference": "max_corrected_logit_abs_difference",
    "candidate_hidden_max_abs_difference": "max_candidate_hidden_abs_difference",
    "d1_hidden_max_abs_difference": "max_d1_hidden_abs_difference",
    "d1_margin_abs_difference_float32": "max_d1_margin_abs_difference_float32",
}


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join(
        [sys.executable, "-m", "proactive_d2.merge_final_mlp_cache", *argv]
    )
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


@contextmanager
def atomic_output_directory(output_dir: Path) -> Iterator[Path]:
    """Expose the final directory only after every artifact is complete."""
    if output_dir.exists():
        raise FileExistsError(f"Merged final-MLP cache already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        yield staging_dir
        if output_dir.exists():
            raise FileExistsError(f"Merged final-MLP cache appeared during merge: {output_dir}")
        staging_dir.rename(output_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def validate_code_states(code_states: list[dict[str, object]]) -> None:
    if not code_states:
        raise ValueError("Final-MLP merge received no code states")
    if any(value != code_states[0] for value in code_states[1:]):
        raise ValueError("Final-MLP shard code states differ")


def validate_data_manifest(
    manifest: dict[str, object],
    *,
    config: dict[str, object],
    runtime_config: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    data_config = dict(config["data"])  # type: ignore[arg-type]
    model_config = dict(config["model"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    cache_config = dict(config["cache"])  # type: ignore[arg-type]
    d1_reference = dict(config["d1_neural_cache_reference"])  # type: ignore[arg-type]
    source = dict(manifest.get("source", {}))  # type: ignore[arg-type]
    model = dict(manifest.get("model", {}))  # type: ignore[arg-type]
    starter = dict(manifest.get("starter_kit_sha256", {}))  # type: ignore[arg-type]
    d1_hashes = dict(manifest.get("d1_neural_reference_sha256", {}))  # type: ignore[arg-type]
    supervision = dict(manifest.get("supervision", {}))  # type: ignore[arg-type]

    start = int(runtime_config["selection_start"])
    stop = int(runtime_config["selection_stop"])
    if _resolve(source.get("path")) != _resolve(data_config["input"]):
        raise ValueError("Final-MLP shard manifest source path differs")
    expected_source = {
        "sha256": str(data_config["input_sha256"]),
        "sessions_selected": stop - start,
        "chunks_selected": sum(int(record["source_chunks"]) for record in records),
        "answers_present_in_generation_rows": False,
        "selection_start": start,
        "selection_stop": stop,
    }
    if any(source.get(name) != value for name, value in expected_source.items()):
        raise ValueError("Final-MLP shard manifest source metadata differs")
    if model.get("weights_sha256") != model_config["weights_sha256"]:
        raise ValueError("Final-MLP shard manifest model fingerprint differs")
    if int(model.get("stored_unique_parameters", -1)) != int(
        model_config["total_parameters"]
    ):
        raise ValueError("Final-MLP shard manifest model parameter count differs")
    expected_starter = {
        "input_sha256": str(data_config["input_sha256"]),
        "model_py_sha256": str(starter_config["model_py_sha256"]),
        "proactive_py_sha256": str(starter_config["proactive_py_sha256"]),
        "scorer_py_sha256": str(starter_config["scorer_py_sha256"]),
    }
    if starter != expected_starter:
        raise ValueError("Final-MLP shard manifest starter-kit fingerprints differ")
    expected_d1 = {
        "features_sha256": str(d1_reference["features_sha256"]),
        "records_sha256": str(d1_reference["records_sha256"]),
    }
    if d1_hashes != expected_d1:
        raise ValueError("Final-MLP shard manifest D1 fingerprints differ")
    if manifest.get("cache") != cache_config:
        raise ValueError("Final-MLP shard manifest cache configuration differs")
    if supervision != {
        "answers_removed_before_extraction": True,
        "labels_read_or_stored": False,
    }:
        raise ValueError("Final-MLP shard manifest supervision declaration differs")


def validate_record_chunks(
    record: dict[str, object],
    source: dict[str, object],
    arrays: dict[str, np.ndarray],
    cache_config: dict[str, object],
) -> dict[str, object]:
    """Validate record metadata and derive its exactness aggregate."""
    input_index = int(record["input_index"])
    chunks = int(record["extracted_chunks"])
    source_intervals = source["video_intervals"]  # type: ignore[index]
    if not isinstance(source_intervals, list):
        raise ValueError(f"Final-MLP source intervals are invalid for {input_index}")
    if (
        not bool(record.get("complete_session"))
        or chunks != int(record["source_chunks"])
        or chunks != len(source_intervals)
    ):
        raise ValueError(f"Final-MLP record is incomplete for session {input_index}")
    expected_shape = [
        chunks,
        int(cache_config["tag_tokens_each"]),
        int(cache_config["hidden_size"]),
    ]
    if record.get("state_shape") != expected_shape:
        raise ValueError(f"Final-MLP record state shape differs for session {input_index}")
    chunk_records = record.get("chunks")
    if not isinstance(chunk_records, list) or len(chunk_records) != chunks:
        raise ValueError(f"Final-MLP chunk metadata count differs for session {input_index}")

    gates = dict(cache_config["required_zero_adapter_checks"])  # type: ignore[arg-type]
    thresholds = {
        name: 0.0 if gate_name is None else float(gates[gate_name])
        for name, gate_name in DIFFERENCE_THRESHOLDS.items()
    }
    if any(not np.isfinite(value) or value < 0.0 for value in thresholds.values()):
        raise ValueError("Final-MLP exactness thresholds must be finite and non-negative")
    maxima = {name: 0.0 for name in DIFFERENCE_THRESHOLDS}
    prompt_match = True
    for chunk_index, (chunk, source_interval) in enumerate(
        zip(chunk_records, source_intervals)
    ):
        if not isinstance(chunk, dict) or int(chunk.get("chunk_index", -1)) != chunk_index:
            raise ValueError(
                f"Final-MLP record chunk order differs at {(input_index, chunk_index)}"
            )
        interval = chunk.get("interval")
        if not isinstance(interval, list) or [float(value) for value in interval] != [
            float(value) for value in source_interval
        ]:
            raise ValueError(
                f"Final-MLP interval differs at {(input_index, chunk_index)}"
            )
        current_frames = int(chunk.get("current_interval_frames", -1))
        model_frames = int(chunk.get("model_input_frames", -1))
        if (
            current_frames < 0
            or current_frames > int(cache_config["frames_per_interval"])
            or model_frames <= 0
            or model_frames > int(cache_config["max_frames"])
        ):
            raise ValueError(
                f"Final-MLP frame metadata differs at {(input_index, chunk_index)}"
            )
        if int(chunk.get("prompt_tokens", -1)) != int(arrays["prompt_tokens"][chunk_index]):
            raise ValueError(
                f"Final-MLP prompt metadata differs at {(input_index, chunk_index)}"
            )
        margin = float(chunk.get("base_tag_margin", float("nan")))
        if not np.isfinite(margin) or float(np.float32(margin)) != float(
            arrays["base_tag_margin"][chunk_index]
        ):
            raise ValueError(
                f"Final-MLP margin metadata differs at {(input_index, chunk_index)}"
            )
        for name, threshold in thresholds.items():
            difference = float(chunk.get(name, float("nan")))
            if not np.isfinite(difference) or difference < 0.0 or difference > threshold:
                raise ValueError(
                    f"Final-MLP exactness gate failed for {name} at "
                    f"{(input_index, chunk_index)}: {difference} > {threshold}"
                )
            maxima[name] = max(maxima[name], difference)
        if chunk.get("d1_prompt_tokens_match") is not True:
            prompt_match = False
            raise ValueError(
                f"Final-MLP D1 prompt check failed at {(input_index, chunk_index)}"
            )
    return {
        "sessions": 1,
        "chunks": chunks,
        "state_bytes_uncompressed": int(record["state_bytes_uncompressed"]),
        **{
            summary_name: maxima[record_name]
            for record_name, summary_name in SUMMARY_DIFFERENCE_FIELDS.items()
        },
        "all_d1_prompt_tokens_match": prompt_match,
    }


def combine_record_aggregates(values: list[dict[str, object]]) -> dict[str, object]:
    if not values:
        raise ValueError("Final-MLP shard contains no records")
    return {
        "sessions": sum(int(value["sessions"]) for value in values),
        "chunks": sum(int(value["chunks"]) for value in values),
        "state_bytes_uncompressed": sum(
            int(value["state_bytes_uncompressed"]) for value in values
        ),
        **{
            summary_name: max(float(value[summary_name]) for value in values)
            for summary_name in SUMMARY_DIFFERENCE_FIELDS.values()
        },
        "all_d1_prompt_tokens_match": all(
            bool(value["all_d1_prompt_tokens_match"]) for value in values
        ),
    }


def validate_shard_metadata_fingerprints(
    summary: dict[str, object],
    runtime: dict[str, object],
    *,
    records_sha256: str,
) -> None:
    if summary.get("status") != "complete final-MLP cache extraction":
        raise ValueError("Final-MLP shard summary status differs")
    if any(runtime.get(name) != value for name, value in summary.items()):
        raise ValueError("Final-MLP shard runtime and summary differ")
    if summary.get("records_sha256") != records_sha256:
        raise ValueError("Final-MLP shard summary records fingerprint differs")
    if runtime.get("records_sha256") != records_sha256:
        raise ValueError("Final-MLP shard runtime records fingerprint differs")


def validate_shard_summary(
    summary: dict[str, object],
    runtime: dict[str, object],
    aggregate: dict[str, object],
    *,
    records_sha256: str,
    hidden_size: int,
    tag_length: int,
) -> None:
    validate_shard_metadata_fingerprints(
        summary,
        runtime,
        records_sha256=records_sha256,
    )
    expected = {
        **aggregate,
        "all_sessions_complete": True,
        "hidden_size": hidden_size,
        "tag_length": tag_length,
        "storage_dtype": "uint16_bfloat16_bits",
        "labels_read_or_stored": False,
    }
    if any(summary.get(name) != value for name, value in expected.items()):
        raise ValueError("Final-MLP shard summary does not match its records")


def validate_session_arrays(
    record: dict[str, object],
    *,
    hidden_size: int,
    tag_length: int,
    bytes_per_chunk: int,
    expected_session_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    feature_path = Path(str(record.get("feature_path"))).resolve()
    if expected_session_dir is not None and feature_path.parent != expected_session_dir.resolve():
        raise ValueError(f"Final-MLP session cache escapes its shard: {feature_path}")
    if not feature_path.is_file():
        raise FileNotFoundError(f"Missing final-MLP session cache: {feature_path}")
    if sha256_file(feature_path) != record.get("feature_sha256"):
        raise ValueError(f"Final-MLP session cache hash mismatch: {feature_path}")
    with np.load(feature_path, allow_pickle=False) as archive:
        required = {
            *STATE_ARRAY_NAMES,
            "base_hidden_state",
            "base_tag_margin",
            "base_silent_log_probability",
            "base_interrupt_log_probability",
            "prompt_tokens",
            "input_index",
            "chunk_index",
        }
        if set(archive.files) != required:
            raise ValueError(f"Final-MLP cache keys differ for {feature_path}")
        arrays = {name: archive[name].copy() for name in archive.files}
    chunks = int(record["extracted_chunks"])
    input_index = int(record["input_index"])
    if not bool(record.get("complete_session")):
        raise ValueError(f"Final-MLP merge rejects partial session {input_index}")
    if chunks <= 0 or chunks != int(record["source_chunks"]):
        raise ValueError(f"Final-MLP chunk count mismatch for session {input_index}")
    for name in STATE_ARRAY_NAMES:
        if arrays[name].shape != (chunks, tag_length, hidden_size):
            raise ValueError(f"Final-MLP state shape mismatch for {name}/{input_index}")
        if arrays[name].dtype != np.uint16:
            raise ValueError(f"Final-MLP state dtype mismatch for {name}/{input_index}")
    state_bytes = sum(arrays[name].nbytes for name in STATE_ARRAY_NAMES)
    expected_bytes = chunks * bytes_per_chunk
    if state_bytes != expected_bytes or state_bytes != int(
        record["state_bytes_uncompressed"]
    ):
        raise ValueError(f"Final-MLP state byte count mismatch for session {input_index}")
    if arrays["base_hidden_state"].shape != (chunks, hidden_size):
        raise ValueError(f"Final-MLP base hidden shape mismatch for session {input_index}")
    expected_dtypes = {
        "base_hidden_state": np.dtype(np.float32),
        "base_tag_margin": np.dtype(np.float32),
        "base_silent_log_probability": np.dtype(np.float32),
        "base_interrupt_log_probability": np.dtype(np.float32),
        "prompt_tokens": np.dtype(np.int32),
        "input_index": np.dtype(np.int32),
        "chunk_index": np.dtype(np.int32),
    }
    for name, expected_dtype in expected_dtypes.items():
        if arrays[name].dtype != expected_dtype:
            raise ValueError(f"Final-MLP diagnostic dtype mismatch for {name}/{input_index}")
    for name in (
        "base_tag_margin",
        "base_silent_log_probability",
        "base_interrupt_log_probability",
        "prompt_tokens",
        "chunk_index",
    ):
        if arrays[name].shape != (chunks,):
            raise ValueError(f"Final-MLP diagnostic shape mismatch for {name}/{input_index}")
    if arrays["input_index"].shape != () or int(arrays["input_index"]) != input_index:
        raise ValueError(f"Final-MLP cached input index mismatch for session {input_index}")
    if not np.array_equal(arrays["chunk_index"], np.arange(chunks)):
        raise ValueError(f"Final-MLP cached chunk indices are not contiguous for {input_index}")
    for name in (
        "base_hidden_state",
        "base_tag_margin",
        "base_silent_log_probability",
        "base_interrupt_log_probability",
    ):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Final-MLP diagnostic {name} is non-finite for {input_index}")
    return arrays


def validate_merged_against_d1(
    merged: dict[str, np.ndarray],
    *,
    config: dict[str, object],
    expected_keys: np.ndarray,
) -> dict[str, object]:
    reference = dict(config["d1_neural_cache_reference"])  # type: ignore[arg-type]
    cache_config = dict(config["cache"])  # type: ignore[arg-type]
    reference_dir = _resolve(reference["path"])
    features_path = reference_dir / "features.npz"
    records_path = reference_dir / "records.jsonl"
    actual_hashes = {
        "features_sha256": sha256_file(features_path),
        "records_sha256": sha256_file(records_path),
    }
    expected_hashes = {
        "features_sha256": str(reference["features_sha256"]),
        "records_sha256": str(reference["records_sha256"]),
    }
    if actual_hashes != expected_hashes:
        raise ValueError("Final-MLP merge D1 reference fingerprint differs")
    reference_summary = _load_json(reference_dir / "summary.json")
    if reference_summary.get("labels_read_or_stored") is not False:
        raise ValueError("Final-MLP merge D1 reference is not label-free")
    required = (
        "hidden_state",
        "tag_margin",
        "prompt_tokens",
        "input_index",
        "chunk_index",
    )
    with np.load(features_path, allow_pickle=False) as archive:
        d1 = {name: archive[name].copy() for name in required}
    rows = expected_keys.shape[0]
    hidden_size = int(cache_config["hidden_size"])
    if d1["hidden_state"].shape != (rows, hidden_size):
        raise ValueError("Final-MLP D1 hidden-state shape differs")
    for name in ("tag_margin", "prompt_tokens", "input_index", "chunk_index"):
        if d1[name].shape != (rows,):
            raise ValueError(f"Final-MLP D1 diagnostic shape differs: {name}")
    d1_keys = np.stack([d1["input_index"], d1["chunk_index"]], axis=1)
    merged_keys = np.stack([merged["input_index"], merged["chunk_index"]], axis=1)
    if not np.array_equal(d1_keys, expected_keys) or not np.array_equal(
        merged_keys, d1_keys
    ):
        raise ValueError("Final-MLP merged keys differ from the fixed D1 cache")
    if not np.array_equal(merged["prompt_tokens"], d1["prompt_tokens"]):
        raise ValueError("Final-MLP merged prompt lengths differ from the fixed D1 cache")
    if not np.isfinite(d1["hidden_state"]).all() or not np.isfinite(
        d1["tag_margin"]
    ).all():
        raise ValueError("Final-MLP D1 reference contains non-finite diagnostics")
    hidden_difference = float(
        np.max(
            np.abs(
                merged["base_hidden_state"].astype(np.float32)
                - d1["hidden_state"].astype(np.float32)
            )
        )
    )
    margin_difference = float(
        np.max(
            np.abs(
                merged["base_tag_margin"].astype(np.float32)
                - d1["tag_margin"].astype(np.float32)
            )
        )
    )
    gates = dict(cache_config["required_zero_adapter_checks"])  # type: ignore[arg-type]
    margin_threshold = float(gates["max_margin_abs_difference_vs_d1_cache"])
    if hidden_difference != 0.0:
        raise ValueError(
            f"Final-MLP merged hidden states differ from D1: {hidden_difference}"
        )
    if margin_difference > margin_threshold:
        raise ValueError(
            f"Final-MLP merged margins differ from D1: {margin_difference}"
        )
    return {
        **actual_hashes,
        "hidden_max_abs_difference": hidden_difference,
        "margin_max_abs_difference_float32": margin_difference,
        "prompt_tokens_exact_match": True,
        "keys_exact_match": True,
    }


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", action="append", dest="shard_dirs")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--output-dir", default=DEFAULT_BASE)
    args = parser.parse_args(raw_argv)
    if args.num_shards < 2:
        parser.error("--num-shards must be at least two")
    started_at = time.monotonic()
    shard_dirs = (
        [_resolve(value) for value in args.shard_dirs]
        if args.shard_dirs
        else [
            _resolve(f"{DEFAULT_BASE}_shard{index}of{args.num_shards}")
            for index in range(args.num_shards)
        ]
    )
    if len(shard_dirs) != args.num_shards or len(set(shard_dirs)) != len(shard_dirs):
        raise ValueError("Final-MLP merge requires every distinct shard directory")
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Merged final-MLP cache already exists: {output_dir}")

    configs: list[dict[str, object]] = []
    runtimes: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    code_states: list[dict[str, object]] = []
    record_entries: list[tuple[dict[str, object], Path]] = []
    shard_artifacts: list[dict[str, object]] = []
    shard_records: dict[Path, list[dict[str, object]]] = {}
    ranges: list[tuple[int, int, int, int]] = []
    for shard_dir in shard_dirs:
        required_paths = {
            "config": shard_dir / "config.json",
            "runtime": shard_dir / "runtime.json",
            "records": shard_dir / "records.jsonl",
            "summary": shard_dir / "summary.json",
            "code_state": shard_dir / "code_state.txt",
            "data_manifest": shard_dir / "data_manifest.json",
        }
        for path in required_paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Incomplete final-MLP shard: {path}")
        config = _load_json(required_paths["config"])
        runtime = _load_json(required_paths["runtime"])
        summary = _load_json(required_paths["summary"])
        code_state = _load_json(required_paths["code_state"])
        data_manifest = _load_json(required_paths["data_manifest"])
        if runtime.get("status") != "complete final-MLP cache extraction":
            raise ValueError(f"Final-MLP shard did not complete: {shard_dir}")
        current_records = load_jsonl(required_paths["records"])
        runtime_config = dict(config["runtime"])  # type: ignore[arg-type]
        records_hash = sha256_file(required_paths["records"])
        validate_shard_metadata_fingerprints(
            summary,
            runtime,
            records_sha256=records_hash,
        )
        validate_data_manifest(
            data_manifest,
            config=config,
            runtime_config=runtime_config,
            records=current_records,
        )
        ranges.append(
            (
                int(runtime_config["selection_start"]),
                int(runtime_config["selection_stop"]),
                int(runtime_config["shard_index"]),
                int(runtime_config["num_shards"]),
            )
        )
        configs.append(config)
        runtimes.append(runtime)
        summaries.append(summary)
        code_states.append(code_state)
        shard_records[shard_dir] = current_records
        record_entries.extend((record, shard_dir) for record in current_records)
        shard_artifacts.append(
            {
                "path": str(shard_dir),
                "config_sha256": sha256_file(required_paths["config"]),
                "runtime_sha256": sha256_file(required_paths["runtime"]),
                "summary_sha256": sha256_file(required_paths["summary"]),
                "records_sha256": records_hash,
                "code_state_sha256": sha256_file(required_paths["code_state"]),
                "data_manifest_sha256": sha256_file(required_paths["data_manifest"]),
                "wall_time_seconds": runtime["wall_time_seconds"],
                "peak_gpu_memory_bytes": runtime["peak_gpu_memory_bytes"],
            }
        )

    validate_code_states(code_states)
    frozen_configs: list[dict[str, object]] = []
    for config in configs:
        copied = json.loads(json.dumps(config))
        copied.pop("runtime")
        frozen_configs.append(copied)
    if any(value != frozen_configs[0] for value in frozen_configs[1:]):
        raise ValueError("Final-MLP shard frozen configs differ")
    expected_ranges = [
        (*contiguous_shard_bounds(700, args.num_shards, index), index, args.num_shards)
        for index in range(args.num_shards)
    ]
    if sorted(ranges) != expected_ranges:
        raise ValueError(f"Final-MLP shard ranges differ: {sorted(ranges)}")
    records_in_shard_order = [shard_records[path] for path in shard_dirs]
    for shard_dir, config, current_records in zip(
        shard_dirs, configs, records_in_shard_order
    ):
        runtime_config = dict(config["runtime"])  # type: ignore[arg-type]
        start = int(runtime_config["selection_start"])
        stop = int(runtime_config["selection_stop"])
        if [int(record["input_index"]) for record in current_records] != list(
            range(start, stop)
        ):
            raise ValueError(f"Final-MLP records escape shard range: {shard_dir}")

    record_entries.sort(key=lambda value: int(value[0]["input_index"]))
    records = [record for record, _shard_dir in record_entries]
    if [int(record["input_index"]) for record in records] != list(range(700)):
        raise ValueError("Final-MLP shards do not cover sessions 0--699 exactly once")
    data_config = dict(frozen_configs[0]["data"])  # type: ignore[arg-type]
    source_path = _resolve(data_config["input"])
    if sha256_file(source_path) != data_config["input_sha256"]:
        raise ValueError("Final-MLP merge source fingerprint mismatch")
    source_rows = strip_answers(load_jsonl(source_path))
    if len(source_rows) != 700:
        raise ValueError("Final-MLP merge source does not contain 700 sessions")
    for input_index, (record, source) in enumerate(zip(records, source_rows)):
        if record.get("video_path") != source.get("video_path"):
            raise ValueError(f"Final-MLP merge video mismatch at {input_index}")
        if int(record["source_chunks"]) != len(source["video_intervals"]):  # type: ignore[arg-type]
            raise ValueError(f"Final-MLP merge source chunk mismatch at {input_index}")

    cache_config = dict(frozen_configs[0]["cache"])  # type: ignore[arg-type]
    hidden_size = int(cache_config["hidden_size"])
    tag_length = int(cache_config["tag_tokens_each"])
    bytes_per_chunk = int(cache_config["bytes_per_chunk_uncompressed"])
    parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            *STATE_ARRAY_NAMES,
            "base_hidden_state",
            "base_tag_margin",
            "base_silent_log_probability",
            "base_interrupt_log_probability",
            "prompt_tokens",
            "input_index",
            "chunk_index",
        )
    }
    aggregate_parts: dict[Path, list[dict[str, object]]] = {
        shard_dir: [] for shard_dir in shard_dirs
    }
    for record, shard_dir in record_entries:
        input_index = int(record["input_index"])
        arrays = validate_session_arrays(
            record,
            hidden_size=hidden_size,
            tag_length=tag_length,
            bytes_per_chunk=bytes_per_chunk,
            expected_session_dir=shard_dir / "sessions",
        )
        aggregate_parts[shard_dir].append(
            validate_record_chunks(
                record,
                source_rows[input_index],
                arrays,
                cache_config,
            )
        )
        chunks = int(record["extracted_chunks"])
        for name in STATE_ARRAY_NAMES:
            parts[name].append(arrays[name])
        for name in (
            "base_hidden_state",
            "base_tag_margin",
            "base_silent_log_probability",
            "base_interrupt_log_probability",
            "prompt_tokens",
            "chunk_index",
        ):
            parts[name].append(arrays[name])
        parts["input_index"].append(
            np.full(chunks, input_index, dtype=np.int32)
        )
    shard_aggregates = [
        combine_record_aggregates(aggregate_parts[shard_dir])
        for shard_dir in shard_dirs
    ]
    for summary, runtime, aggregate, artifact in zip(
        summaries, runtimes, shard_aggregates, shard_artifacts
    ):
        validate_shard_summary(
            summary,
            runtime,
            aggregate,
            records_sha256=str(artifact["records_sha256"]),
            hidden_size=hidden_size,
            tag_length=tag_length,
        )
        artifact["sessions"] = aggregate["sessions"]
        artifact["chunks"] = aggregate["chunks"]
    merged_aggregate = combine_record_aggregates(
        [value for values in aggregate_parts.values() for value in values]
    )
    merged = {name: np.concatenate(value) for name, value in parts.items()}
    expected_rows = sum(
        len(row["video_intervals"]) for row in source_rows  # type: ignore[arg-type]
    )
    if expected_rows != 9935:
        raise ValueError(f"Final-MLP source chunk total differs: {expected_rows}")
    for name in STATE_ARRAY_NAMES:
        if merged[name].shape != (expected_rows, tag_length, hidden_size):
            raise ValueError(f"Merged final-MLP state shape mismatch: {name}")
        if merged[name].dtype != np.uint16:
            raise ValueError(f"Merged final-MLP state dtype mismatch: {name}")
    if merged["base_hidden_state"].shape != (expected_rows, hidden_size):
        raise ValueError("Merged final-MLP base hidden shape mismatch")
    expected_keys = np.asarray(
        [
            (input_index, chunk_index)
            for input_index, row in enumerate(source_rows)
            for chunk_index in range(len(row["video_intervals"]))  # type: ignore[arg-type]
        ],
        dtype=np.int32,
    )
    actual_keys = np.stack([merged["input_index"], merged["chunk_index"]], axis=1)
    if not np.array_equal(actual_keys, expected_keys):
        raise ValueError("Merged final-MLP cache order differs from the source")
    d1_validation = validate_merged_against_d1(
        merged,
        config=frozen_configs[0],
        expected_keys=expected_keys,
    )
    if float(merged_aggregate["max_d1_hidden_abs_difference"]) != float(
        d1_validation["hidden_max_abs_difference"]
    ) or float(merged_aggregate["max_d1_margin_abs_difference_float32"]) != float(
        d1_validation["margin_max_abs_difference_float32"]
    ):
        raise ValueError("Final-MLP record aggregates differ from direct D1 validation")
    state_bytes = sum(merged[name].nbytes for name in STATE_ARRAY_NAMES)
    expected_state_bytes = int(cache_config["projected_bytes_for_9935_chunks_uncompressed"])
    if state_bytes != expected_state_bytes:
        raise ValueError(f"Merged final-MLP byte count differs: {state_bytes}")

    with atomic_output_directory(output_dir) as staging_dir:
        features_path = staging_dir / "features.npz"
        np.savez_compressed(features_path, **merged)
        records_path = staging_dir / "records.jsonl"
        write_jsonl(records_path, records)
        write_json(staging_dir / "config.json", frozen_configs[0])
        _write_command(staging_dir / "command.sh", raw_argv)
        write_json(staging_dir / "environment.txt", environment_snapshot())
        write_json(
            staging_dir / "code_state.txt",
            code_snapshot(
                PROJECT_ROOT,
                [
                    *sorted((PROJECT_ROOT / "src/proactive_d2").glob("*.py")),
                    *sorted((PROJECT_ROOT / "src/proactive_d2/tests").glob("*.py")),
                    PROJECT_ROOT / "configs/d2_internvl35_1b_final_mlp_lora_oof.json",
                    PROJECT_ROOT / "CURRENT_ROUTE.md",
                    PROJECT_ROOT / "Agent.md",
                ],
            ),
        )
        data_manifest = {
            "shards": shard_artifacts,
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "d1_reference_validation": d1_validation,
            "features_sha256": sha256_file(features_path),
            "records_sha256": sha256_file(records_path),
            "labels_read_or_stored": False,
            "state_bytes_uncompressed": state_bytes,
            "features_file_bytes_compressed": features_path.stat().st_size,
        }
        write_json(staging_dir / "data_manifest.json", data_manifest)
        summary = {
            "status": "complete merged final-MLP cache",
            "sessions": int(merged_aggregate["sessions"]),
            "chunks": int(merged_aggregate["chunks"]),
            "hidden_size": hidden_size,
            "tag_length": tag_length,
            "state_dtype": "uint16_bfloat16_bits",
            "state_bytes_uncompressed": state_bytes,
            "features_file_bytes_compressed": features_path.stat().st_size,
            "features_sha256": data_manifest["features_sha256"],
            "records_sha256": data_manifest["records_sha256"],
            **{
                name: merged_aggregate[name]
                for name in SUMMARY_DIFFERENCE_FIELDS.values()
            },
            "all_d1_prompt_tokens_match": bool(
                merged_aggregate["all_d1_prompt_tokens_match"]
            ),
            "d1_direct_validation": d1_validation,
            "labels_read_or_stored": False,
        }
        write_json(staging_dir / "summary.json", summary)
        write_json(
            staging_dir / "runtime.json",
            {
                **summary,
                "completed_at": datetime.now().astimezone().isoformat(),
                "merge_wall_time_seconds": round(time.monotonic() - started_at, 3),
                "shard_wall_time_seconds": [
                    runtime["wall_time_seconds"] for runtime in runtimes
                ],
                "shard_peak_gpu_memory_bytes": [
                    runtime["peak_gpu_memory_bytes"] for runtime in runtimes
                ],
            },
        )
        (staging_dir / "README.md").write_text(
            "\n".join(
                [
                    "# D2 merged final-MLP cache",
                    "",
                    "Status: **complete, label-free, and bit-exact**",
                    "",
                    f"- Sessions: `{summary['sessions']}`",
                    f"- Chunks: `{summary['chunks']}`",
                    f"- State bytes uncompressed: `{state_bytes}`",
                    f"- Feature SHA256: `{data_manifest['features_sha256']}`",
                    "- D1 hidden/prompt/key validation: exact",
                    "- D1 margin maximum difference: "
                    f"`{d1_validation['margin_max_abs_difference_float32']}`",
                    "- Gold labels read or stored: `False`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
