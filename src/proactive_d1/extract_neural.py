"""Extract resumable, label-free D1 neural features from frozen InternVL."""

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

from .core import strip_answers
from .internvl_features import InternVLDecisionFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "d1_internvl35_1b_neural_features.json"
LOGGER = logging.getLogger("proactive_d1.extract_neural")


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
    command = shlex.join([sys.executable, "-m", "proactive_d1.extract_neural", *argv])
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


def _tracked_paths(config_path: Path) -> list[Path]:
    return [
        *sorted((PROJECT_ROOT / "src" / "proactive_r0").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d1").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d1" / "tests").glob("*.py")),
        config_path,
        PROJECT_ROOT / "models" / "internvl35_1b_hf.json",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
        PROJECT_ROOT / "C1_SPEC.md",
        PROJECT_ROOT / "Agent.md",
    ]


def _validate_record_prefix(
    records: list[dict[str, object]],
    selected: list[tuple[int, dict[str, object]]],
) -> None:
    if len(records) > len(selected):
        raise ValueError("Neural feature records exceed selected sessions")
    for position, record in enumerate(records):
        input_index, row = selected[position]
        if record.get("input_index") != input_index:
            raise ValueError(f"Neural record {position} input index mismatch")
        if record.get("video_path") != row.get("video_path"):
            raise ValueError(f"Neural record {position} video mismatch")
        feature_path = Path(str(record.get("feature_path")))
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing neural feature file: {feature_path}")
        if sha256_file(feature_path) != record.get("feature_sha256"):
            raise ValueError(f"Neural feature hash mismatch: {feature_path}")


def _write_npz_atomic(path: Path, arrays: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _extract_session(
    row: dict[str, object],
    input_index: int,
    video_folder: Path,
    model: InternVLDecisionFeatureExtractor,
    starter: object,
    config: CausalInferenceConfig,
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
    cumulative_frames: list[object] = []
    hidden_rows: list[np.ndarray] = []
    tag_margin: list[float] = []
    silent_logp: list[float] = []
    interrupt_logp: list[float] = []
    prompt_tokens: list[int] = []
    chunks: list[dict[str, object]] = []
    for chunk_index, interval in enumerate(intervals):
        current_frames = starter.extract_frames(  # type: ignore[attr-defined]
            str(video_folder / video_name),
            intervals=[interval],
            frames_per_interval=config.frames_per_interval,
        )
        cumulative_frames.extend(current_frames)
        model_frames = subsample_frames(cumulative_frames, config.max_frames)
        messages = build_messages(
            row=row,
            chunk_index=chunk_index,
            system_prompt=starter.system_prompt,  # type: ignore[attr-defined]
            normalize_dialog_turns=starter.normalize_dialog_turns,  # type: ignore[attr-defined]
            max_history_turns=config.max_history_turns,
        )
        features = model.extract_decision_features(model_frames, messages)
        hidden = features.hidden_state
        hidden_array = hidden.numpy().astype(np.float32, copy=False)  # type: ignore[union-attr]
        if hidden_array.shape != (model.hidden_size,):
            raise RuntimeError(f"Unexpected saved hidden shape: {hidden_array.shape}")
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
                "silent_log_probability": features.silent_log_probability,
                "interrupt_log_probability": features.interrupt_log_probability,
                "tag_margin": features.tag_margin,
                "hidden_max_abs_difference": features.hidden_max_abs_difference,
                "hidden_cosine_similarity": features.hidden_cosine_similarity,
            }
        )
    arrays: dict[str, object] = {
        "hidden_state": np.stack(hidden_rows).astype(np.float32, copy=False),
        "tag_margin": np.asarray(tag_margin, dtype=np.float32),
        "silent_log_probability": np.asarray(silent_logp, dtype=np.float32),
        "interrupt_log_probability": np.asarray(interrupt_logp, dtype=np.float32),
        "prompt_tokens": np.asarray(prompt_tokens, dtype=np.int32),
        "input_index": np.asarray(input_index, dtype=np.int32),
        "chunk_index": np.arange(len(chunks), dtype=np.int32),
    }
    _write_npz_atomic(feature_path, arrays)
    return {
        "input_index": input_index,
        "video_path": video_name,
        "source_chunks": len(row["video_intervals"]),  # type: ignore[arg-type]
        "extracted_chunks": len(chunks),
        "complete_session": len(chunks) == len(row["video_intervals"]),  # type: ignore[arg-type]
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "hidden_shape": [len(chunks), model.hidden_size],
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
    inference_config = dict(config["inference"])  # type: ignore[arg-type]
    feature_config = dict(config["features"])  # type: ignore[arg-type]
    input_path = _resolve(data_config["input"])
    video_folder = _resolve(data_config["video_folder"])
    starter_dir = _resolve(starter_config["path"])
    model_path = _resolve(args.model_path or model_config["default_local_path"])
    output_dir = _resolve(args.output_dir or f"output/features/{config['experiment_id']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = output_dir / "sessions"
    feature_dir.mkdir(exist_ok=True)
    records_path = output_dir / "records.jsonl"
    if records_path.exists() and not args.resume:
        raise FileExistsError(f"Existing neural records require --resume: {records_path}")
    _configure_logging(output_dir, append=args.resume)

    fingerprints = _validate_static_files(config, input_path, starter_dir)
    model_audit = verify_model_snapshot(model_path, model_config)
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
    existing_config = output_dir / "config.json"
    if args.resume and existing_config.exists():
        if _load_json(existing_config) != effective:
            raise ValueError("Effective neural extraction config differs on resume")
    else:
        write_json(existing_config, effective)
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
            "features": feature_config,
            "supervision": config["validation_policy"],
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
    causal_config = CausalInferenceConfig(
        frames_per_interval=int(inference_config["frames_per_interval"]),
        max_frames=int(inference_config["max_frames"]),
        max_history_turns=int(inference_config["max_history_turns"]),
        max_new_tokens=1,
    )
    model: InternVLDecisionFeatureExtractor | None = None
    if len(records) < len(selected):
        model = InternVLDecisionFeatureExtractor(
            model_path=str(model_path),
            device=args.device,
            dtype_name=str(model_config["dtype"]),
            attention_implementation=str(model_config["attention_implementation"]),
            seed=int(inference_config["seed"]),
            require_exclusive_gpu=args.require_exclusive_gpu,
            video_frame_size=int(inference_config["video_frame_size"]),
            pad_token_id=int(inference_config["pad_token_id"]),
        )
        if model.parameter_count != int(model_config["total_parameters"]):
            raise ValueError("Loaded D1 feature model parameter count mismatch")
        if model.hidden_size != int(feature_config["hidden_size"]):
            raise ValueError("Loaded D1 feature hidden size mismatch")
        with records_path.open("a", encoding="utf-8") as handle:
            for position in range(len(records), len(selected)):
                session_started = time.monotonic()
                input_index, row = selected[position]
                feature_path = feature_dir / f"session_{input_index:04d}.npz"
                if feature_path.exists():
                    raise FileExistsError(f"Untracked neural feature file: {feature_path}")
                record = _extract_session(
                    row,
                    input_index,
                    video_folder,
                    model,
                    starter,
                    causal_config,
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
    extracted_chunks = sum(int(record["extracted_chunks"]) for record in records)
    max_hidden_difference = max(
        float(chunk["hidden_max_abs_difference"])
        for record in records
        for chunk in record["chunks"]  # type: ignore[union-attr]
    )
    min_hidden_cosine = min(
        float(chunk["hidden_cosine_similarity"])
        for record in records
        for chunk in record["chunks"]  # type: ignore[union-attr]
    )
    summary = {
        "status": "complete neural feature extraction",
        "sessions": len(records),
        "chunks": extracted_chunks,
        "all_sessions_complete": all(
            bool(record["complete_session"]) for record in records
        ),
        "hidden_size": int(feature_config["hidden_size"]),
        "max_hidden_abs_difference_between_tag_candidates": max_hidden_difference,
        "min_hidden_cosine_similarity_between_tag_candidates": min_hidden_cosine,
        "records_sha256": sha256_file(records_path),
    }
    write_json(output_dir / "summary.json", summary)
    runtime = {
        **summary,
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started_at, 3),
        "peak_gpu_memory_bytes": model.peak_memory_bytes() if model else None,
        "preexisting_gpu_processes": model.preexisting_gpu_processes if model else [],
    }
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "Status: **complete label-free neural feature extraction**",
                "",
                f"- Sessions: `{summary['sessions']}`",
                f"- Chunks: `{summary['chunks']}`",
                f"- Complete sessions: `{summary['all_sessions_complete']}`",
                f"- Hidden size: `{summary['hidden_size']}`",
                f"- Max candidate-prefix hidden difference: `{max_hidden_difference}`",
                "- Gold labels read during extraction: `False`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
