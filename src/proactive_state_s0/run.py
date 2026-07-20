"""Run target-isolated S0 fixed-candidate state inference."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    verify_model_snapshot,
    write_json,
)
from proactive_r0.core import load_jsonl, load_starter_kit, subsample_frames
from proactive_r0.run import _validate_static_files
from proactive_state_s0.core import (
    STATE_TARGETS,
    STATE_VIEWS,
    messages_from_sample,
    prediction_from_scores,
    state_question_messages,
)
from proactive_state_s0.internvl import InternVLStateCandidateScorer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/s0_internvl35_1b_oracle_plan_state_inference.json"
LOGGER = logging.getLogger("proactive_state_s0.run")
FORBIDDEN_INPUT_KEYS = {
    "current_step_id",
    "next_step_id",
    "progress",
    "completion_evidence",
    "incompletion_or_error_evidence",
    "recovery_action",
    "confidence",
    "answers",
    "current_output",
    "frozen_decision",
}


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _check_hash(path: Path, expected: object) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"S0 frozen artifact fingerprint mismatch: {path}")
    return actual


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
    command = shlex.join([sys.executable, "-m", "proactive_state_s0.run", *argv])
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _validate_inputs(rows: list[dict[str, object]]) -> dict[str, int]:
    if len(rows) != 80:
        raise ValueError("S0 prepared input must contain exactly 80 states")
    seen: set[str] = set()
    sessions: set[int] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in seen:
            raise ValueError("S0 input sample IDs are empty or duplicated")
        seen.add(sample_id)
        sessions.add(int(row["input_index"]))
        forbidden = FORBIDDEN_INPUT_KEYS.intersection(row)
        if forbidden:
            raise ValueError(f"S0 input exposes forbidden target keys: {sorted(forbidden)}")
        intervals = row.get("video_intervals_so_far")
        chunk_index = int(row["chunk_index"])
        if not isinstance(intervals, list) or len(intervals) != chunk_index + 1:
            raise ValueError(f"S0 causal intervals do not end at {sample_id}")
        if len(row.get("steps", [])) != 4:  # type: ignore[arg-type]
            raise ValueError(f"S0 sample does not have four static steps: {sample_id}")
    if len(sessions) != 20:
        raise ValueError("S0 prepared input must contain exactly 20 sessions")
    return {"states": len(rows), "sessions": len(sessions)}


def _validate_existing(
    records: list[dict[str, object]], selected: list[dict[str, object]], view: str
) -> None:
    if len(records) > len(selected):
        raise ValueError("S0 resume records exceed selected inputs")
    for index, record in enumerate(records):
        if record.get("sample_id") != selected[index].get("sample_id"):
            raise ValueError(f"S0 resume sample identity differs at row {index}")
        if record.get("view") != view:
            raise ValueError(f"S0 resume view differs at row {index}")


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--view", required=True, choices=STATE_VIEWS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    args = parser.parse_args(raw_argv)
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    started = time.monotonic()
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    model_config = dict(config["model"])  # type: ignore[arg-type]
    prepared = dict(config["prepared_inputs"])  # type: ignore[arg-type]
    protocol = dict(config["protocol"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    inference = dict(config["inference"])  # type: ignore[arg-type]
    calibration_config = dict(config.get("calibration", {"mode": "none"}))  # type: ignore[arg-type]
    input_path = _resolve(prepared["path"])
    manifest_path = _resolve(prepared["manifest"])
    protocol_path = _resolve(protocol["path"])
    starter_dir = _resolve(starter_config["path"])
    video_folder = _resolve(config["video_folder"])
    model_path = _resolve(model_config["local_path"])
    output_dir = _resolve(args.output_dir)
    records_path = output_dir / "state_predictions.jsonl"
    if records_path.exists() and not args.resume:
        raise FileExistsError(f"S0 records require --resume: {records_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(output_dir, append=args.resume)

    frozen_hashes = {
        "inputs": _check_hash(input_path, prepared["sha256"]),
        "prepared_manifest": _check_hash(
            manifest_path, prepared["manifest_sha256"]
        ),
        "protocol": _check_hash(protocol_path, protocol["sha256"]),
    }
    manifest = _load_json(manifest_path)
    if manifest.get("prediction_runner_reads_targets") is not False:
        raise ValueError("S0 prepared manifest does not guarantee target isolation")
    if manifest.get("inputs_contain_dynamic_state_targets") is not False:
        raise ValueError("S0 prepared inputs may contain state targets")
    starter_hashes = _validate_static_files(config, input_path, starter_dir)
    model_audit = verify_model_snapshot(model_path, model_config)
    rows = load_jsonl(input_path)
    validation = _validate_inputs(rows)
    selected = rows[: args.max_samples] if args.max_samples else rows
    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "input_path": str(input_path),
        "target_path_available_to_runner": False,
        "view": args.view,
        "output_dir": str(output_dir),
        "device": args.device,
        "max_samples": args.max_samples,
        "audit_only": args.audit_only,
        "require_exclusive_gpu": args.require_exclusive_gpu,
        "calibration_mode": calibration_config["mode"],
    }
    config_output = output_dir / "config.json"
    if args.resume and config_output.exists():
        if _load_json(config_output) != effective:
            raise ValueError("S0 effective config changed on resume")
    else:
        write_json(config_output, effective)
        _write_command(output_dir / "command.sh", raw_argv)
        write_json(output_dir / "environment.txt", environment_snapshot())
        write_json(
            output_dir / "code_state.txt",
            code_snapshot(
                PROJECT_ROOT,
                [
                    *sorted((PROJECT_ROOT / "src/proactive_state_s0").glob("*.py")),
                    *sorted((PROJECT_ROOT / "src/proactive_state_s0/tests").glob("*.py")),
                    config_path,
                    protocol_path,
                    PROJECT_ROOT / "CURRENT_ROUTE.md",
                    PROJECT_ROOT / "Agent.md",
                ],
            ),
        )
    write_json(
        output_dir / "data_manifest.json",
        {
            "classification": config["validation_policy"],
            "frozen_hashes": frozen_hashes,
            "starter_kit_sha256": starter_hashes,
            "model": {"path": str(model_path), **model_audit},
            "prepared_validation": validation,
            "selected_states": len(selected),
            "target_path_stored": False,
            "target_file_read": False,
            "ratings_read": False,
            "answers_read": False,
        },
    )
    if args.audit_only:
        write_json(
            output_dir / "runtime.json",
            {
                "status": "complete target-isolated audit only",
                "wall_time_seconds": round(time.monotonic() - started, 3),
                "gpu_used": False,
                "states": len(selected),
            },
        )
        return

    records = load_jsonl(records_path) if records_path.exists() else []
    _validate_existing(records, selected, args.view)
    starter = load_starter_kit(starter_dir)
    model: InternVLStateCandidateScorer | None = None
    if len(records) < len(selected):
        model = InternVLStateCandidateScorer(
            model_path=str(model_path),
            device=args.device,
            dtype_name=str(model_config["dtype"]),
            attention_implementation=str(model_config["attention_implementation"]),
            seed=int(inference["seed"]),
            require_exclusive_gpu=args.require_exclusive_gpu,
            video_frame_size=int(inference["video_frame_size"]),
            pad_token_id=int(inference["pad_token_id"]),
        )
        if model.parameter_count != int(model_config["total_parameters"]):
            raise ValueError("S0 loaded model parameter count changed")
        by_input: dict[int, list[tuple[int, dict[str, object]]]] = defaultdict(list)
        for index, sample in enumerate(selected):
            by_input[int(sample["input_index"])].append((index, sample))
        completed_ids = {str(record["sample_id"]) for record in records}
        with records_path.open("a", encoding="utf-8") as handle:
            for input_index in sorted(by_input):
                samples = sorted(by_input[input_index], key=lambda value: int(value[1]["chunk_index"]))
                calibration_scores: dict[str, list[float]] | None = None
                calibration_seconds = 0.0
                if calibration_config["mode"] == "query_plan_no_observation":
                    calibration_sample = samples[0][1]
                    calibration_base = [
                        {"role": "system", "content": starter.system_prompt},
                        {"role": "user", "content": str(calibration_sample["query"])},
                    ]
                    calibration_questions = {
                        target: state_question_messages(
                            calibration_base, calibration_sample, target
                        )
                        for target in STATE_TARGETS
                    }
                    calibration_started = time.monotonic()
                    calibration_result = model.score_state([], calibration_questions)
                    calibration_seconds = time.monotonic() - calibration_started
                    if calibration_result["vision_forward_passes"] != 0:
                        raise RuntimeError("S0 content-free calibration used vision")
                    calibration_scores = calibration_result["scores"]  # type: ignore[assignment]
                elif calibration_config["mode"] != "none":
                    raise ValueError(
                        f"Unsupported S0 calibration mode: {calibration_config['mode']}"
                    )
                interval_source = max(samples, key=lambda value: int(value[1]["chunk_index"]))[1]
                intervals = interval_source["video_intervals_so_far"]
                selected_by_chunk = {
                    int(sample["chunk_index"]): sample for _, sample in samples
                }
                cumulative_frames: list[object] = []
                for chunk_index, interval in enumerate(intervals):  # type: ignore[assignment]
                    current_frames = starter.extract_frames(
                        str(video_folder / str(interval_source["video_path"])),
                        intervals=[(float(interval[0]), float(interval[1]))],
                        frames_per_interval=int(inference["frames_per_interval"]),
                    )
                    cumulative_frames.extend(current_frames)
                    sample = selected_by_chunk.get(chunk_index)
                    if sample is None or str(sample["sample_id"]) in completed_ids:
                        continue
                    frames = subsample_frames(cumulative_frames, int(inference["max_frames"]))
                    base_messages = messages_from_sample(
                        sample,
                        starter.system_prompt,
                        starter.normalize_dialog_turns,
                        int(inference["max_history_turns"]),
                        args.view,
                    )
                    questions = {
                        target: state_question_messages(base_messages, sample, target)
                        for target in STATE_TARGETS
                    }
                    sample_started = time.monotonic()
                    scored = model.score_state(frames, questions)
                    elapsed = time.monotonic() - sample_started
                    calibrated_scores = {
                        target: [
                            float(value)
                            - (
                                float(calibration_scores[target][index])
                                if calibration_scores is not None
                                else 0.0
                            )
                            for index, value in enumerate(scored["scores"][target])  # type: ignore[index]
                        ]
                        for target in STATE_TARGETS
                    }
                    predictions = {
                        target: prediction_from_scores(
                            target, calibrated_scores[target]
                        )
                        for target in STATE_TARGETS
                    }
                    record = {
                        "sample_id": sample["sample_id"],
                        "input_index": input_index,
                        "video_path": sample["video_path"],
                        "domain": sample["domain"],
                        "position_bin": sample["position_bin"],
                        "chunk_index": chunk_index,
                        "view": args.view,
                        "model_input_frames": len(frames),
                        "predictions": predictions,
                        "raw_log_probabilities": {
                            target: {
                                candidate: float(value)
                                for candidate, value in zip(
                                    inference["candidate_text"][target],  # type: ignore[index]
                                    scored["scores"][target],  # type: ignore[index]
                                )
                            }
                            for target in STATE_TARGETS
                        },
                        "content_free_log_probabilities": (
                            {
                                target: {
                                    candidate: float(value)
                                    for candidate, value in zip(
                                        inference["candidate_text"][target],  # type: ignore[index]
                                        calibration_scores[target],
                                    )
                                }
                                for target in STATE_TARGETS
                            }
                            if calibration_scores is not None
                            else None
                        ),
                        "calibration_mode": calibration_config["mode"],
                        "calibration_wall_time_seconds_per_session": calibration_seconds,
                        "prompt_tokens": scored["prompt_tokens"],
                        "vision_forward_passes": scored["vision_forward_passes"],
                        "language_forward_passes": scored["language_forward_passes"],
                        "candidate_token_ids": scored["candidate_token_ids"],
                        "wall_time_seconds": elapsed,
                    }
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    records.append(record)
                    completed_ids.add(str(sample["sample_id"]))
                    LOGGER.info(
                        "State %d/%d complete: %s view=%s elapsed=%.2fs",
                        len(records),
                        len(selected),
                        sample["sample_id"],
                        args.view,
                        elapsed,
                    )
    _validate_existing(records, selected, args.view)
    if len(records) != len(selected):
        raise RuntimeError("S0 inference did not complete every selected state")
    runtime = {
        "status": "complete target-isolated S0 state inference",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "states": len(records),
        "sessions": len({int(record["input_index"]) for record in records}),
        "view": args.view,
        "peak_gpu_memory_bytes": model.peak_memory_bytes() if model else None,
        "preexisting_gpu_processes": model.preexisting_gpu_processes if model else [],
        "target_file_read": False,
        "ratings_read": False,
        "calibration_mode": calibration_config["mode"],
        "state_predictions_sha256": sha256_file(records_path),
    }
    write_json(output_dir / "runtime.json", runtime)
    print(json.dumps(runtime, sort_keys=True))


if __name__ == "__main__":
    main()
