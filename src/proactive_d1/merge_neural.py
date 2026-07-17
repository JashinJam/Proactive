"""Validate and merge complete D1 neural feature shards without reading labels."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from proactive_r0.artifacts import code_snapshot, environment_snapshot, sha256_file, write_json
from proactive_r0.core import load_jsonl, write_jsonl

from .core import strip_answers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = "output/features/20260714_internvl35_1b_d1_neural_feature_cache_v1"


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d1.merge_neural", *argv])
    content = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n"
    )
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def validate_session_arrays(
    record: dict[str, object], hidden_size: int
) -> dict[str, np.ndarray]:
    feature_path = Path(str(record.get("feature_path")))
    if not feature_path.is_file():
        raise FileNotFoundError(f"Missing D1 neural session cache: {feature_path}")
    if sha256_file(feature_path) != record.get("feature_sha256"):
        raise ValueError(f"D1 neural session cache hash mismatch: {feature_path}")
    with np.load(feature_path, allow_pickle=False) as archive:
        required = {
            "hidden_state",
            "tag_margin",
            "silent_log_probability",
            "interrupt_log_probability",
            "prompt_tokens",
            "input_index",
            "chunk_index",
        }
        if set(archive.files) != required:
            raise ValueError(f"D1 neural cache keys differ for {feature_path}")
        arrays = {name: archive[name].copy() for name in archive.files}
    chunks = int(record["extracted_chunks"])
    input_index = int(record["input_index"])
    if not bool(record.get("complete_session")):
        raise ValueError(f"D1 merge rejects partial session {input_index}")
    if chunks != int(record["source_chunks"]) or chunks <= 0:
        raise ValueError(f"D1 neural chunk count mismatch for session {input_index}")
    if arrays["hidden_state"].shape != (chunks, hidden_size):
        raise ValueError(f"D1 hidden shape mismatch for session {input_index}")
    for name in (
        "tag_margin",
        "silent_log_probability",
        "interrupt_log_probability",
        "prompt_tokens",
        "chunk_index",
    ):
        if arrays[name].shape != (chunks,):
            raise ValueError(f"D1 array {name} shape mismatch for session {input_index}")
    if arrays["input_index"].shape != () or int(arrays["input_index"]) != input_index:
        raise ValueError(f"D1 cached input index mismatch for session {input_index}")
    if not np.array_equal(arrays["chunk_index"], np.arange(chunks)):
        raise ValueError(f"D1 cached chunk indices are not contiguous for session {input_index}")
    for name in (
        "hidden_state",
        "tag_margin",
        "silent_log_probability",
        "interrupt_log_probability",
    ):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"D1 array {name} is non-finite for session {input_index}")
    return arrays


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir",
        action="append",
        dest="shard_dirs",
        help="Repeat for every completed shard; defaults to four frozen paths.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_BASE)
    args = parser.parse_args(raw_argv)
    started_at = time.monotonic()
    shard_dirs = (
        [_resolve(value) for value in args.shard_dirs]
        if args.shard_dirs
        else [_resolve(f"{DEFAULT_BASE}_shard{index}") for index in range(4)]
    )
    if len(shard_dirs) < 2 or len(set(shard_dirs)) != len(shard_dirs):
        raise ValueError("D1 merge requires distinct feature shards")
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"D1 merged feature output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    configs: list[dict[str, object]] = []
    runtimes: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    shard_manifest: list[dict[str, object]] = []
    for shard_dir in shard_dirs:
        config_path = shard_dir / "config.json"
        runtime_path = shard_dir / "runtime.json"
        records_path = shard_dir / "records.jsonl"
        summary_path = shard_dir / "summary.json"
        for required in (config_path, runtime_path, records_path, summary_path):
            if not required.is_file():
                raise FileNotFoundError(f"Incomplete D1 shard: {required}")
        config = _load_json(config_path)
        runtime = _load_json(runtime_path)
        summary = _load_json(summary_path)
        if runtime.get("status") != "complete neural feature extraction":
            raise ValueError(f"D1 shard did not complete: {shard_dir}")
        if not bool(summary.get("all_sessions_complete")):
            raise ValueError(f"D1 shard contains partial sessions: {shard_dir}")
        if float(summary.get("max_hidden_abs_difference_between_tag_candidates", 1.0)) > 1e-5:
            raise ValueError(f"D1 shard hidden state depends on forced tag: {shard_dir}")
        if float(summary.get("min_hidden_cosine_similarity_between_tag_candidates", 0.0)) < 0.99999:
            raise ValueError(f"D1 shard candidate-prefix hidden mismatch: {shard_dir}")
        shard_records = load_jsonl(records_path)
        configs.append(config)
        runtimes.append(runtime)
        records.extend(shard_records)
        shard_manifest.append(
            {
                "path": str(shard_dir),
                "config_sha256": sha256_file(config_path),
                "runtime_sha256": sha256_file(runtime_path),
                "summary_sha256": sha256_file(summary_path),
                "records_sha256": sha256_file(records_path),
                "sessions": len(shard_records),
                "chunks": sum(int(record["extracted_chunks"]) for record in shard_records),
                "max_candidate_prefix_hidden_difference": summary[
                    "max_hidden_abs_difference_between_tag_candidates"
                ],
                "min_candidate_prefix_hidden_cosine": summary[
                    "min_hidden_cosine_similarity_between_tag_candidates"
                ],
            }
        )

    frozen_configs = []
    ranges = []
    for config in configs:
        copied = json.loads(json.dumps(config))
        runtime = dict(copied.pop("runtime"))
        ranges.append(
            (
                int(runtime["selection_start"]),
                int(runtime["selection_stop"]),
                int(runtime["shard_index"]),
                int(runtime["num_shards"]),
            )
        )
        frozen_configs.append(copied)
    if any(value != frozen_configs[0] for value in frozen_configs[1:]):
        raise ValueError("D1 neural shard frozen configs differ")
    ranges.sort()
    if ranges != [(0, 175, 0, 4), (175, 350, 1, 4), (350, 525, 2, 4), (525, 700, 3, 4)]:
        raise ValueError(f"D1 neural shard ranges differ from frozen coverage: {ranges}")

    records.sort(key=lambda record: int(record["input_index"]))
    if [int(record["input_index"]) for record in records] != list(range(700)):
        raise ValueError("D1 neural shards do not cover sessions 0--699 exactly once")
    source_config = dict(frozen_configs[0]["data"])  # type: ignore[arg-type]
    source_path = _resolve(source_config["input"])
    if sha256_file(source_path) != source_config["input_sha256"]:
        raise ValueError("D1 neural merge source fingerprint mismatch")
    source_rows = strip_answers(load_jsonl(source_path))
    if len(source_rows) != len(records):
        raise ValueError("D1 neural merge source/session count mismatch")
    for input_index, (record, source) in enumerate(zip(records, source_rows)):
        if record.get("video_path") != source.get("video_path"):
            raise ValueError(f"D1 merge video identity mismatch at {input_index}")
        if int(record["source_chunks"]) != len(source["video_intervals"]):  # type: ignore[arg-type]
            raise ValueError(f"D1 merge source chunk mismatch at {input_index}")

    feature_config = dict(frozen_configs[0]["features"])  # type: ignore[arg-type]
    hidden_size = int(feature_config["hidden_size"])
    hidden_parts: list[np.ndarray] = []
    margin_parts: list[np.ndarray] = []
    silent_parts: list[np.ndarray] = []
    interrupt_parts: list[np.ndarray] = []
    prompt_parts: list[np.ndarray] = []
    input_parts: list[np.ndarray] = []
    chunk_parts: list[np.ndarray] = []
    for record in records:
        arrays = validate_session_arrays(record, hidden_size)
        chunks = int(record["extracted_chunks"])
        hidden_parts.append(arrays["hidden_state"].astype(np.float32, copy=False))
        margin_parts.append(arrays["tag_margin"].astype(np.float32, copy=False))
        silent_parts.append(arrays["silent_log_probability"].astype(np.float32, copy=False))
        interrupt_parts.append(arrays["interrupt_log_probability"].astype(np.float32, copy=False))
        prompt_parts.append(arrays["prompt_tokens"].astype(np.int32, copy=False))
        input_parts.append(np.full(chunks, int(record["input_index"]), dtype=np.int32))
        chunk_parts.append(arrays["chunk_index"].astype(np.int32, copy=False))
    merged = {
        "hidden_state": np.concatenate(hidden_parts),
        "tag_margin": np.concatenate(margin_parts),
        "silent_log_probability": np.concatenate(silent_parts),
        "interrupt_log_probability": np.concatenate(interrupt_parts),
        "prompt_tokens": np.concatenate(prompt_parts),
        "input_index": np.concatenate(input_parts),
        "chunk_index": np.concatenate(chunk_parts),
    }
    if merged["hidden_state"].shape != (9935, hidden_size):
        raise ValueError(f"Unexpected merged D1 hidden shape: {merged['hidden_state'].shape}")
    features_path = output_dir / "features.npz"
    np.savez_compressed(features_path, **merged)
    write_jsonl(output_dir / "records.jsonl", records)
    write_json(output_dir / "config.json", frozen_configs[0])
    _write_command(output_dir / "command.sh", raw_argv)
    write_json(output_dir / "environment.txt", environment_snapshot())
    write_json(
        output_dir / "code_state.txt",
        code_snapshot(
            PROJECT_ROOT,
            [
                *sorted((PROJECT_ROOT / "src" / "proactive_d1").glob("*.py")),
                *sorted((PROJECT_ROOT / "src" / "proactive_d1" / "tests").glob("*.py")),
                PROJECT_ROOT / "configs" / "d1_internvl35_1b_neural_features.json",
                PROJECT_ROOT / "CURRENT_ROUTE.md",
                PROJECT_ROOT / "Agent.md",
            ],
        ),
    )
    data_manifest = {
        "shards": shard_manifest,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "features_sha256": sha256_file(features_path),
        "records_sha256": sha256_file(output_dir / "records.jsonl"),
        "labels_read_or_stored": False,
        "max_candidate_prefix_hidden_difference": max(
            float(value["max_candidate_prefix_hidden_difference"])
            for value in shard_manifest
        ),
        "min_candidate_prefix_hidden_cosine": min(
            float(value["min_candidate_prefix_hidden_cosine"])
            for value in shard_manifest
        ),
    }
    write_json(output_dir / "data_manifest.json", data_manifest)
    summary = {
        "status": "complete merged neural feature cache",
        "sessions": 700,
        "chunks": 9935,
        "hidden_size": hidden_size,
        "hidden_dtype": str(merged["hidden_state"].dtype),
        "features_sha256": data_manifest["features_sha256"],
        "records_sha256": data_manifest["records_sha256"],
        "tag_margin_min": float(merged["tag_margin"].min()),
        "tag_margin_max": float(merged["tag_margin"].max()),
        "tag_margin_mean": float(merged["tag_margin"].mean()),
        "prompt_tokens_min": int(merged["prompt_tokens"].min()),
        "prompt_tokens_max": int(merged["prompt_tokens"].max()),
        "max_candidate_prefix_hidden_difference": data_manifest[
            "max_candidate_prefix_hidden_difference"
        ],
        "min_candidate_prefix_hidden_cosine": data_manifest[
            "min_candidate_prefix_hidden_cosine"
        ],
        "labels_read_or_stored": False,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "runtime.json",
        {
            **summary,
            "completed_at": datetime.now().astimezone().isoformat(),
            "merge_wall_time_seconds": round(time.monotonic() - started_at, 3),
            "shard_wall_time_seconds": [runtime["wall_time_seconds"] for runtime in runtimes],
            "shard_peak_gpu_memory_bytes": [
                runtime["peak_gpu_memory_bytes"] for runtime in runtimes
            ],
        },
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# D1 merged neural feature cache",
                "",
                "Status: **complete and label-free**",
                "",
                "- Sessions: `700`",
                "- Chunks: `9,935`",
                f"- Hidden shape: `[9935, {hidden_size}]`",
                f"- Feature SHA256: `{data_manifest['features_sha256']}`",
                "- Gold labels read or stored: `False`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
