"""Extract resumable, label-free, bit-exact final-MLP cache shards for D2."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from proactive_d1.core import strip_answers
from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    verify_model_snapshot,
    write_json,
)
from proactive_r0.core import (
    CausalInferenceConfig,
    build_messages,
    load_jsonl,
    load_starter_kit,
    subsample_frames,
    validate_source_rows,
)
from proactive_r0.run import _validate_static_files, contiguous_shard_bounds

from .final_mlp_cache import (
    CANDIDATE_NAMES,
    STATE_ARRAY_NAMES,
    InternVLFinalMLPCacheExtractor,
    state_to_bit_arrays,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/d2_internvl35_1b_final_mlp_lora_oof.json"
LOGGER = logging.getLogger("proactive_d2.extract_final_mlp_cache")


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _configure_logging(output_dir: Path, append: bool) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / "run.log", mode="a" if append else "w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join(
        [sys.executable, "-m", "proactive_d2.extract_final_mlp_cache", *argv]
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


def _tracked_paths(config_path: Path) -> list[Path]:
    return [
        *sorted((PROJECT_ROOT / "src" / "proactive_r0").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d1").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d2").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d2" / "tests").glob("*.py")),
        config_path,
        PROJECT_ROOT / "models/internvl35_1b_hf.json",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
        PROJECT_ROOT / "C1_SPEC.md",
        PROJECT_ROOT / "Agent.md",
    ]


def _write_npz_atomic(path: Path, arrays: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _validate_record_prefix(
    records: list[dict[str, object]],
    selected: list[tuple[int, dict[str, object]]],
) -> None:
    if len(records) > len(selected):
        raise ValueError("Final-MLP cache records exceed selected sessions")
    for position, record in enumerate(records):
        input_index, row = selected[position]
        if int(record.get("input_index", -1)) != input_index:
            raise ValueError(f"Final-MLP record {position} input index mismatch")
        if record.get("video_path") != row.get("video_path"):
            raise ValueError(f"Final-MLP record {position} video mismatch")
        feature_path = Path(str(record.get("feature_path")))
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing final-MLP session cache: {feature_path}")
        if sha256_file(feature_path) != record.get("feature_sha256"):
            raise ValueError(f"Final-MLP session cache hash mismatch: {feature_path}")


def _load_d1_reference(
    reference: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[tuple[int, int], int], dict[str, str]]:
    directory = _resolve(reference["path"])
    features_path = directory / "features.npz"
    records_path = directory / "records.jsonl"
    hashes = {
        "features_sha256": sha256_file(features_path),
        "records_sha256": sha256_file(records_path),
    }
    expected = {
        "features_sha256": str(reference["features_sha256"]),
        "records_sha256": str(reference["records_sha256"]),
    }
    if hashes != expected:
        raise ValueError(f"D1 neural cache fingerprint mismatch: {hashes} != {expected}")
    summary = _load_json(directory / "summary.json")
    if summary.get("labels_read_or_stored") is not False:
        raise ValueError("Final-MLP extraction requires a label-free D1 reference")
    with np.load(features_path, allow_pickle=False) as archive:
        arrays = {
            name: archive[name].copy()
            for name in ("hidden_state", "tag_margin", "prompt_tokens", "input_index", "chunk_index")
        }
    keys = np.stack([arrays["input_index"], arrays["chunk_index"]], axis=1)
    lookup = {
        (int(input_index), int(chunk_index)): position
        for position, (input_index, chunk_index) in enumerate(keys)
    }
    if len(lookup) != len(keys):
        raise ValueError("D1 neural reference contains duplicate chunk keys")
    return arrays, lookup, hashes


def _extract_session(
    row: dict[str, object],
    input_index: int,
    video_folder: Path,
    model: InternVLFinalMLPCacheExtractor,
    starter: object,
    inference: CausalInferenceConfig,
    cache_config: dict[str, object],
    d1_arrays: dict[str, np.ndarray],
    d1_lookup: dict[tuple[int, int], int],
    max_chunks: int | None,
    feature_path: Path,
) -> dict[str, object]:
    video_name = str(row["video_path"])
    intervals = [
        (float(value[0]), float(value[1]))
        for value in row["video_intervals"]  # type: ignore[index]
    ]
    if max_chunks is not None:
        intervals = intervals[:max_chunks]
    hidden_size = int(cache_config["hidden_size"])
    tag_length = int(cache_config["tag_tokens_each"])
    expected_bytes_per_chunk = int(cache_config["bytes_per_chunk_uncompressed"])
    gates = dict(cache_config["required_zero_adapter_checks"])  # type: ignore[arg-type]

    state_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in STATE_ARRAY_NAMES
    }
    hidden_rows: list[np.ndarray] = []
    tag_margin: list[float] = []
    silent_logp: list[float] = []
    interrupt_logp: list[float] = []
    prompt_tokens: list[int] = []
    chunks: list[dict[str, object]] = []
    cumulative_frames: list[object] = []
    for chunk_index, interval in enumerate(intervals):
        current_frames = starter.extract_frames(  # type: ignore[attr-defined]
            str(video_folder / video_name),
            intervals=[interval],
            frames_per_interval=inference.frames_per_interval,
        )
        cumulative_frames.extend(current_frames)
        model_frames = subsample_frames(cumulative_frames, inference.max_frames)
        messages = build_messages(
            row=row,
            chunk_index=chunk_index,
            system_prompt=starter.system_prompt,  # type: ignore[attr-defined]
            normalize_dialog_turns=starter.normalize_dialog_turns,  # type: ignore[attr-defined]
            max_history_turns=inference.max_history_turns,
        )
        features = model.extract_final_mlp_cache(model_frames, messages)
        corrected_hidden_difference = max(
            features.silent.hidden_max_abs_difference,
            features.interrupt.hidden_max_abs_difference,
        )
        corrected_logit_difference = max(
            features.silent.logit_max_abs_difference,
            features.interrupt.logit_max_abs_difference,
        )
        if corrected_hidden_difference > float(
            gates["max_hidden_abs_difference_vs_full_base"]
        ):
            raise RuntimeError(
                f"Corrected base hidden mismatch at {(input_index, chunk_index)}: "
                f"{corrected_hidden_difference}"
            )
        if corrected_logit_difference > float(
            gates["max_logit_abs_difference_vs_full_base"]
        ):
            raise RuntimeError(
                f"Corrected base logit mismatch at {(input_index, chunk_index)}: "
                f"{corrected_logit_difference}"
            )
        if features.candidate_hidden_max_abs_difference != 0.0:
            raise RuntimeError(
                f"Forced tag changes causal hidden at {(input_index, chunk_index)}"
            )

        for candidate in CANDIDATE_NAMES:
            state = getattr(features, candidate).state
            arrays = state_to_bit_arrays(state, remove_batch_dimension=True)
            for state_name, array in arrays.items():
                if array.shape != (tag_length, hidden_size) or array.dtype != np.uint16:
                    raise RuntimeError(
                        f"Invalid {candidate}/{state_name} cache shape or dtype at "
                        f"{(input_index, chunk_index)}"
                    )
                state_parts[f"{candidate}_{state_name}_bits"].append(array)

        hidden = features.hidden_state
        hidden_array = hidden.numpy().astype(np.float32, copy=False)  # type: ignore[union-attr]
        reference_position = d1_lookup.get((input_index, chunk_index))
        if reference_position is None:
            raise ValueError(f"D1 reference lacks chunk {(input_index, chunk_index)}")
        d1_hidden_difference = float(
            np.max(np.abs(hidden_array - d1_arrays["hidden_state"][reference_position]))
        )
        d1_margin_difference = abs(
            float(np.float32(features.tag_margin))
            - float(d1_arrays["tag_margin"][reference_position])
        )
        d1_prompt_match = features.prompt_tokens == int(
            d1_arrays["prompt_tokens"][reference_position]
        )
        if d1_hidden_difference != 0.0:
            raise RuntimeError(
                f"Final-MLP hidden differs from D1 at {(input_index, chunk_index)}: "
                f"{d1_hidden_difference}"
            )
        if d1_margin_difference > float(gates["max_margin_abs_difference_vs_d1_cache"]):
            raise RuntimeError(
                f"Final-MLP margin differs from D1 at {(input_index, chunk_index)}: "
                f"{d1_margin_difference}"
            )
        if not d1_prompt_match:
            raise RuntimeError(
                f"Final-MLP prompt length differs from D1 at {(input_index, chunk_index)}"
            )

        hidden_rows.append(hidden_array)
        tag_margin.append(features.tag_margin)
        silent_logp.append(features.silent_log_probability)
        interrupt_logp.append(features.interrupt_log_probability)
        prompt_tokens.append(features.prompt_tokens)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "interval": list(interval),
                "current_interval_frames": len(current_frames),
                "model_input_frames": len(model_frames),
                "prompt_tokens": features.prompt_tokens,
                "base_tag_margin": features.tag_margin,
                "corrected_hidden_max_abs_difference": corrected_hidden_difference,
                "corrected_logit_max_abs_difference": corrected_logit_difference,
                "candidate_hidden_max_abs_difference": (
                    features.candidate_hidden_max_abs_difference
                ),
                "d1_hidden_max_abs_difference": d1_hidden_difference,
                "d1_margin_abs_difference_float32": d1_margin_difference,
                "d1_prompt_tokens_match": d1_prompt_match,
            }
        )

    arrays_out: dict[str, object] = {
        name: np.stack(parts).astype(np.uint16, copy=False)
        for name, parts in state_parts.items()
    }
    state_bytes = sum(array.nbytes for array in arrays_out.values())  # type: ignore[union-attr]
    expected_state_bytes = len(chunks) * expected_bytes_per_chunk
    if state_bytes != expected_state_bytes:
        raise RuntimeError(
            f"Final-MLP state bytes differ: {state_bytes} != {expected_state_bytes}"
        )
    arrays_out.update(
        {
            "base_hidden_state": np.stack(hidden_rows).astype(np.float32, copy=False),
            "base_tag_margin": np.asarray(tag_margin, dtype=np.float32),
            "base_silent_log_probability": np.asarray(silent_logp, dtype=np.float32),
            "base_interrupt_log_probability": np.asarray(interrupt_logp, dtype=np.float32),
            "prompt_tokens": np.asarray(prompt_tokens, dtype=np.int32),
            "input_index": np.asarray(input_index, dtype=np.int32),
            "chunk_index": np.arange(len(chunks), dtype=np.int32),
        }
    )
    _write_npz_atomic(feature_path, arrays_out)
    return {
        "input_index": input_index,
        "video_path": video_name,
        "source_chunks": len(row["video_intervals"]),  # type: ignore[arg-type]
        "extracted_chunks": len(chunks),
        "complete_session": len(chunks) == len(row["video_intervals"]),  # type: ignore[arg-type]
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "state_shape": [len(chunks), tag_length, hidden_size],
        "state_bytes_uncompressed": state_bytes,
        "chunks": chunks,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--max-chunks-per-session", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.max_sessions is not None and args.max_sessions <= 0:
        parser.error("--max-sessions must be positive")
    if args.max_chunks_per_session is not None and args.max_chunks_per_session <= 0:
        parser.error("--max-chunks-per-session must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    started_at = time.monotonic()
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    model_config = dict(config["model"])  # type: ignore[arg-type]
    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    cache_config = dict(config["cache"])  # type: ignore[arg-type]
    adapter_config = dict(config["adapter"])  # type: ignore[arg-type]
    d1_reference = dict(config["d1_neural_cache_reference"])  # type: ignore[arg-type]
    input_path = _resolve(data_config["input"])
    video_folder = _resolve(data_config["video_folder"])
    starter_dir = _resolve(starter_config["path"])
    model_path = _resolve(args.model_path or model_config["default_local_path"])
    cache_id = str(cache_config["experiment_id"])
    default_output = f"output/features/{cache_id}"
    if args.num_shards > 1:
        default_output += f"_shard{args.shard_index}of{args.num_shards}"
    output_dir = _resolve(args.output_dir or default_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_dir = output_dir / "sessions"
    session_dir.mkdir(exist_ok=True)
    records_path = output_dir / "records.jsonl"
    if records_path.exists() and not args.resume:
        raise FileExistsError(f"Existing cache records require --resume: {records_path}")
    _configure_logging(output_dir, append=args.resume)

    fingerprints = _validate_static_files(config, input_path, starter_dir)
    model_audit = verify_model_snapshot(model_path, model_config)
    d1_arrays, d1_lookup, d1_hashes = _load_d1_reference(d1_reference)
    source_rows = load_jsonl(input_path)
    start, stop = contiguous_shard_bounds(
        len(source_rows), args.num_shards, args.shard_index
    )
    selected_rows = source_rows[start:stop]
    if args.max_sessions is not None:
        selected_rows = selected_rows[: args.max_sessions]
        stop = start + len(selected_rows)
    generation_rows = strip_answers(selected_rows)
    source_validation = validate_source_rows(generation_rows, video_folder)
    selected = [(start + index, row) for index, row in enumerate(generation_rows)]

    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "model_path": str(model_path),
        "input_path": str(input_path),
        "video_folder": str(video_folder),
        "output_dir": str(output_dir),
        "device": args.device,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selection_start": start,
        "selection_stop": stop,
        "max_sessions": args.max_sessions,
        "max_chunks_per_session": args.max_chunks_per_session,
        "audit_only": args.audit_only,
    }
    config_output = output_dir / "config.json"
    if args.resume and config_output.exists():
        if _load_json(config_output) != effective:
            raise ValueError("Effective final-MLP cache config differs on resume")
    else:
        write_json(config_output, effective)
        _write_command(output_dir / "command.sh", raw_argv)
        write_json(output_dir / "environment.txt", environment_snapshot())
        write_json(
            output_dir / "code_state.txt",
            code_snapshot(PROJECT_ROOT, _tracked_paths(config_path)),
        )
    write_json(
        output_dir / "data_manifest.json",
        {
            "source": {
                "path": str(input_path),
                "sha256": fingerprints["input_sha256"],
                "sessions_selected": source_validation["sessions"],
                "chunks_selected": source_validation["chunks"],
                "answers_present_in_generation_rows": False,
                "selection_start": start,
                "selection_stop": stop,
            },
            "model": {**model_audit, "path": str(model_path)},
            "starter_kit_sha256": fingerprints,
            "d1_neural_reference_sha256": d1_hashes,
            "cache": cache_config,
            "supervision": {
                "answers_removed_before_extraction": True,
                "labels_read_or_stored": False,
            },
        },
    )
    if args.audit_only:
        write_json(
            output_dir / "runtime.json",
            {
                "status": "audit_only",
                "wall_time_seconds": round(time.monotonic() - started_at, 3),
                "gpu_used": False,
            },
        )
        return

    records = load_jsonl(records_path) if records_path.exists() else []
    _validate_record_prefix(records, selected)
    starter = load_starter_kit(starter_dir)
    inference = CausalInferenceConfig(
        frames_per_interval=int(cache_config["frames_per_interval"]),
        max_frames=int(cache_config["max_frames"]),
        max_history_turns=int(cache_config["max_history_turns"]),
        max_new_tokens=1,
    )
    model: InternVLFinalMLPCacheExtractor | None = None
    if len(records) < len(selected):
        model = InternVLFinalMLPCacheExtractor(
            model_path=str(model_path),
            device=args.device,
            dtype_name=str(model_config["dtype"]),
            attention_implementation=str(model_config["attention_implementation"]),
            seed=int(cache_config["seed"]),
            require_exclusive_gpu=args.require_exclusive_gpu,
            video_frame_size=int(cache_config["video_frame_size"]),
            pad_token_id=int(cache_config["pad_token_id"]),
            language_layer_index=int(adapter_config["language_layer_index"]),
        )
        if model.parameter_count != int(model_config["total_parameters"]):
            raise ValueError("Loaded final-MLP cache model parameter count mismatch")
        if model.hidden_size != int(cache_config["hidden_size"]):
            raise ValueError("Loaded final-MLP cache hidden size mismatch")
        with records_path.open("a", encoding="utf-8") as handle:
            for position in range(len(records), len(selected)):
                session_started = time.monotonic()
                input_index, row = selected[position]
                feature_path = session_dir / f"session_{input_index:04d}.npz"
                if feature_path.exists():
                    raise FileExistsError(f"Untracked final-MLP cache file: {feature_path}")
                record = _extract_session(
                    row,
                    input_index,
                    video_folder,
                    model,
                    starter,
                    inference,
                    cache_config,
                    d1_arrays,
                    d1_lookup,
                    args.max_chunks_per_session,
                    feature_path,
                )
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                records.append(record)
                LOGGER.info(
                    "Session %d/%d complete: input=%d chunks=%d elapsed=%.2fs",
                    position + 1,
                    len(selected),
                    input_index,
                    record["extracted_chunks"],
                    time.monotonic() - session_started,
                )
    _validate_record_prefix(records, selected)
    all_chunks = [
        chunk
        for record in records
        for chunk in record["chunks"]  # type: ignore[union-attr]
    ]
    summary = {
        "status": "complete final-MLP cache extraction",
        "sessions": len(records),
        "chunks": sum(int(record["extracted_chunks"]) for record in records),
        "all_sessions_complete": all(bool(record["complete_session"]) for record in records),
        "hidden_size": int(cache_config["hidden_size"]),
        "tag_length": int(cache_config["tag_tokens_each"]),
        "storage_dtype": "uint16_bfloat16_bits",
        "max_corrected_hidden_abs_difference": max(
            float(chunk["corrected_hidden_max_abs_difference"]) for chunk in all_chunks
        ),
        "max_corrected_logit_abs_difference": max(
            float(chunk["corrected_logit_max_abs_difference"]) for chunk in all_chunks
        ),
        "max_candidate_hidden_abs_difference": max(
            float(chunk["candidate_hidden_max_abs_difference"]) for chunk in all_chunks
        ),
        "max_d1_hidden_abs_difference": max(
            float(chunk["d1_hidden_max_abs_difference"]) for chunk in all_chunks
        ),
        "max_d1_margin_abs_difference_float32": max(
            float(chunk["d1_margin_abs_difference_float32"]) for chunk in all_chunks
        ),
        "all_d1_prompt_tokens_match": all(
            bool(chunk["d1_prompt_tokens_match"]) for chunk in all_chunks
        ),
        "state_bytes_uncompressed": sum(
            int(record["state_bytes_uncompressed"]) for record in records
        ),
        "records_sha256": sha256_file(records_path),
        "labels_read_or_stored": False,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "runtime.json",
        {
            **summary,
            "completed_at": datetime.now().astimezone().isoformat(),
            "wall_time_seconds": round(time.monotonic() - started_at, 3),
            "peak_gpu_memory_bytes": model.peak_memory_bytes() if model else None,
            "preexisting_gpu_processes": model.preexisting_gpu_processes if model else [],
        },
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {cache_config['experiment_id']}",
                "",
                "Status: **complete label-free final-MLP cache shard**",
                "",
                f"- Sessions: `{summary['sessions']}`",
                f"- Chunks: `{summary['chunks']}`",
                f"- All sessions complete: `{summary['all_sessions_complete']}`",
                f"- Max corrected base hidden difference: `{summary['max_corrected_hidden_abs_difference']}`",
                f"- Max D1 margin difference: `{summary['max_d1_margin_abs_difference_float32']}`",
                "- Gold labels read or stored: `False`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
