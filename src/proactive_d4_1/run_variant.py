"""Run one resumable D4.1 variant shard on one GPU."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Sequence

from proactive_d1.core import load_decision_head, strip_answers
from proactive_d1.internvl_features import InternVLDecisionFeatureExtractor
from proactive_d4.deploy import process_session_with_dialog_stage_head
from proactive_r0.artifacts import environment_snapshot, sha256_file, write_json
from proactive_r0.core import (
    CausalInferenceConfig,
    load_jsonl,
    load_starter_kit,
    validate_prediction_rows,
    validate_source_rows,
    write_jsonl,
)

from .core import (
    InferenceParameters,
    atomic_append_jsonl,
    canonical_json,
    load_task_records,
    object_sha256,
    partition_session_indices,
    prepare_task_directory,
    task_hash,
    task_should_run,
    validate_shard_records,
)

LOGGER = logging.getLogger("proactive_d4_1.run_variant")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resolve(project_root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _configure_logging(path: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def sanitize_inference_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    sanitized = strip_answers(rows)
    if any("answers" in row for row in sanitized):
        raise RuntimeError("D4.1 target stripping failed")
    return sanitized


class TimedDecisionModel:
    """Measure synchronized model calls while delegating the frozen implementation."""

    def __init__(self, model: InternVLDecisionFeatureExtractor) -> None:
        self.model = model
        self.generation_seconds: list[float] = []
        self.decision_feature_seconds: list[float] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self.model, name)

    def _synchronize(self) -> None:
        try:
            import torch

            if torch.cuda.is_available() and str(self.model.device).startswith("cuda"):
                torch.cuda.synchronize(self.model.device)
        except (ImportError, RuntimeError):
            pass

    def _timed(self, target: object, *args: object, **kwargs: object) -> tuple[object, float]:
        self._synchronize()
        started = time.perf_counter()
        result = target(*args, **kwargs)  # type: ignore[operator]
        self._synchronize()
        return result, time.perf_counter() - started

    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> str:
        result, elapsed = self._timed(
            self.model.generate, frames, messages, max_new_tokens=max_new_tokens
        )
        self.generation_seconds.append(elapsed)
        return str(result)

    def extract_decision_features(
        self, frames: list[object], messages: list[dict[str, str]]
    ) -> object:
        result, elapsed = self._timed(
            self.model.extract_decision_features, frames, messages
        )
        self.decision_feature_seconds.append(elapsed)
        return result

    def reset_session_timing(self) -> None:
        self.generation_seconds.clear()
        self.decision_feature_seconds.clear()

    def attach_timing(
        self, record: dict[str, object], session_wall_seconds: float
    ) -> dict[str, object]:
        chunks = record.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError("D4.1 D4 record has no chunks")
        if len(chunks) != len(self.generation_seconds) or len(chunks) != len(
            self.decision_feature_seconds
        ):
            raise ValueError("D4.1 timing calls do not align with D4 chunks")
        for chunk, generation, decision in zip(
            chunks, self.generation_seconds, self.decision_feature_seconds
        ):
            if not isinstance(chunk, dict):
                raise ValueError("D4.1 D4 chunk record is invalid")
            chunk["generation_seconds"] = generation
            chunk["decision_feature_seconds"] = decision
            chunk["model_inference_seconds"] = generation + decision
        generation_total = sum(self.generation_seconds)
        decision_total = sum(self.decision_feature_seconds)
        record["timing"] = {
            "generation_seconds": generation_total,
            "decision_feature_seconds": decision_total,
            "model_inference_seconds": generation_total + decision_total,
            "session_wall_seconds": session_wall_seconds,
            "frame_and_host_overhead_seconds": max(
                0.0, session_wall_seconds - generation_total - decision_total
            ),
        }
        return record


def _code_state() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return completed.stdout.rstrip()

    try:
        return {
            "is_git_repository": True,
            "head": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "status_short": run("status", "--short"),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"is_git_repository": False, "error": str(error)}


def _write_command(path: Path, argv: Sequence[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d4_1.run_variant", *argv])
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(PROJECT_ROOT))}",
        "export PYTHONNOUSERSITE=1",
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}",
    ]
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        lines.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(visible_devices)}")
    lines.extend([f"exec {command}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def _variant_by_id(path: Path, variant_id: str) -> dict[str, object]:
    matches = [row for row in load_jsonl(path) if row.get("variant_id") == variant_id]
    if len(matches) != 1:
        raise ValueError(f"D4.1 expected one variant {variant_id}, got {len(matches)}")
    return matches[0]


def _task_directory(
    experiment_dir: Path, stage: str, variant_id: str, shard_id: int
) -> Path:
    return experiment_dir / "runs" / stage / variant_id / f"shard_{shard_id:03d}"


def _validate_static_audit(
    experiment_dir: Path, config_sha256: str
) -> dict[str, object]:
    audit_path = experiment_dir / "static_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError("D4.1 static audit is missing; run run_search first")
    audit = _load_object(audit_path)
    if audit.get("experiment_config_sha256") != config_sha256:
        raise ValueError("D4.1 static audit uses a different experiment config")
    if audit.get("model_verified") is not True:
        raise ValueError("D4.1 model snapshot has not passed the non-dry static audit")
    return audit


def _run(args: argparse.Namespace, raw_argv: list[str]) -> dict[str, object]:
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    config_path = experiment_dir / "config.json"
    config = _load_object(config_path)
    config_sha256 = object_sha256(config)
    identity = _load_object(experiment_dir / "experiment_identity.json")
    if identity.get("experiment_config_sha256") != config_sha256:
        raise ValueError("D4.1 experiment config hash differs from its identity")
    _validate_static_audit(experiment_dir, config_sha256)

    variant = _variant_by_id(experiment_dir / "variants.jsonl", args.variant_id)
    manifest = _load_object(experiment_dir / "sample_manifest.json")
    if args.stage not in ("search", "confirmation", "full", "smoke"):
        raise ValueError(f"Unsupported D4.1 stage: {args.stage}")
    stage_indices = list(manifest[args.stage]["indices"])  # type: ignore[index]
    shards = partition_session_indices(stage_indices, manifest, args.num_shards)
    if not 0 <= args.shard_id < len(shards):
        raise ValueError(f"D4.1 shard ID {args.shard_id} is out of range")
    session_indices = shards[args.shard_id]
    expected_task_hash = task_hash(
        experiment_config_sha256=config_sha256,
        stage=args.stage,
        variant=variant,
        shard_id=args.shard_id,
        session_indices=session_indices,
    )
    task = {
        "schema_version": 1,
        "task_hash": expected_task_hash,
        "experiment_config_sha256": config_sha256,
        "stage": args.stage,
        "variant_id": args.variant_id,
        "parameters": variant["parameters"],
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "session_indices": session_indices,
    }
    task_dir = _task_directory(experiment_dir, args.stage, args.variant_id, args.shard_id)
    prepare_task_directory(task_dir, task)
    _configure_logging(task_dir / "run.log")
    records_path = task_dir / "session_records.jsonl"

    config_data = dict(config["data"])  # type: ignore[arg-type]
    config_model = dict(config["model"])  # type: ignore[arg-type]
    config_head = dict(config["decision_head"])  # type: ignore[arg-type]
    config_starter = dict(config["starter_kit"])  # type: ignore[arg-type]
    frozen = dict(config["frozen_inference"])  # type: ignore[arg-type]
    input_path = _resolve(PROJECT_ROOT, config_data["input"])
    video_folder = _resolve(PROJECT_ROOT, config_data["video_folder"])
    model_path = _resolve(PROJECT_ROOT, config_model["default_local_path"])
    head_path = _resolve(PROJECT_ROOT, config_head["path"])
    starter_dir = _resolve(PROJECT_ROOT, config_starter["path"])

    source_rows = load_jsonl(input_path)
    inference_rows = sanitize_inference_rows(source_rows)
    validate_source_rows([inference_rows[index] for index in session_indices], video_folder)
    records = load_task_records(records_path)
    validate_shard_records(
        records, session_indices, inference_rows, require_complete=False
    )
    if not task_should_run(task_dir, expected_task_hash):
        validate_shard_records(
            records, session_indices, inference_rows, require_complete=True
        )
        LOGGER.info("Completed D4.1 task skipped: %s", expected_task_hash)
        return _load_object(task_dir / "runtime.json")

    previous_status = _load_object(task_dir / "status.json")
    attempt = int(previous_status.get("attempt", 0)) + 1
    write_json(
        task_dir / "status.json",
        {
            "status": "running",
            "task_hash": expected_task_hash,
            "attempt": attempt,
            "started_at": _now(),
            "device": args.device,
            "pid": os.getpid(),
        },
    )
    attempt_started = time.monotonic()
    if not (task_dir / "environment.json").exists():
        write_json(task_dir / "environment.json", environment_snapshot())
        write_json(task_dir / "code_state.json", _code_state())
        write_json(
            task_dir / "effective_config.json",
            {"experiment": config, "task": task},
        )
        _write_command(task_dir / "command.sh", raw_argv)

    head_payload = _load_object(head_path)
    if sha256_file(head_path) != config_head["sha256"]:
        raise ValueError("D4.1 head SHA256 differs from the frozen D4 head")
    head = load_decision_head(head_payload)
    if len(head.feature_names) + 1 != int(config_head["parameters"]):
        raise ValueError("D4.1 head parameter count changed")
    if head.threshold_logit != float(config_head["threshold_logit"]):
        raise ValueError("D4.1 decision threshold changed")

    parameters = InferenceParameters.from_mapping(
        variant["parameters"]  # type: ignore[arg-type]
    )
    inference = CausalInferenceConfig(**parameters.to_dict())
    starter = load_starter_kit(starter_dir)
    model = InternVLDecisionFeatureExtractor(
        model_path=str(model_path),
        device=args.device,
        dtype_name=str(config_model["dtype"]),
        attention_implementation=str(config_model["attention_implementation"]),
        seed=int(frozen["seed"]),
        require_exclusive_gpu=not args.allow_shared_gpu,
        video_frame_size=int(frozen["video_frame_size"]),
        pad_token_id=int(frozen["pad_token_id"]),
        decision_feature_mode=str(frozen["decision_feature_mode"]),
    )
    if model.parameter_count != int(config_model["total_parameters"]):
        raise ValueError("D4.1 loaded backbone parameter count changed")
    if model.hidden_size != int(config_head["hidden_size"]):
        raise ValueError("D4.1 loaded hidden width changed")
    timed_model = TimedDecisionModel(model)
    session_model_times = [
        float(record["timing"]["model_inference_seconds"])  # type: ignore[index]
        for record in records
    ]
    for position in range(len(records), len(session_indices)):
        input_index = session_indices[position]
        timed_model.reset_session_timing()
        session_started = time.perf_counter()
        record = process_session_with_dialog_stage_head(
            inference_rows[input_index],
            input_index,
            video_folder,
            timed_model,  # type: ignore[arg-type]
            starter,
            inference,
            head,
            record_hidden_state=False,
        )
        timed_model.attach_timing(record, time.perf_counter() - session_started)
        if "answers" in inference_rows[input_index]:
            raise RuntimeError("D4.1 answers reached the inference loop")
        atomic_append_jsonl(records_path, record)
        records.append(record)
        model_seconds = float(record["timing"]["model_inference_seconds"])  # type: ignore[index]
        session_model_times.append(model_seconds)
        LOGGER.info(
            "Session %d/%d source_index=%d chunks=%d model=%.3fs wall=%.3fs",
            position + 1,
            len(session_indices),
            input_index,
            len(record["chunks"]),  # type: ignore[arg-type]
            model_seconds,
            float(record["timing"]["session_wall_seconds"]),  # type: ignore[index]
        )
    validate_shard_records(records, session_indices, inference_rows, require_complete=True)
    predictions = [record["prediction"] for record in records]
    validate_prediction_rows(
        [inference_rows[index] for index in session_indices], predictions  # type: ignore[arg-type]
    )
    predictions_path = task_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)  # type: ignore[arg-type]
    generation_seconds = sum(
        float(record["timing"]["generation_seconds"]) for record in records  # type: ignore[index]
    )
    decision_seconds = sum(
        float(record["timing"]["decision_feature_seconds"]) for record in records  # type: ignore[index]
    )
    runtime = {
        "status": "complete",
        "task_hash": expected_task_hash,
        "attempt": attempt,
        "completed_at": _now(),
        "wall_time_seconds_this_attempt": time.monotonic() - attempt_started,
        "sessions": len(records),
        "chunks": sum(len(record["chunks"]) for record in records),  # type: ignore[arg-type]
        "generation_seconds": generation_seconds,
        "decision_feature_seconds": decision_seconds,
        "total_model_inference_seconds": generation_seconds + decision_seconds,
        "maximum_session_model_inference_seconds": max(session_model_times),
        "sessions_over_300_model_seconds": sum(value > 300 for value in session_model_times),
        "peak_gpu_memory_bytes": model.peak_memory_bytes(),
        "preexisting_gpu_processes": model.preexisting_gpu_processes,
        "predictions_sha256": sha256_file(predictions_path),
        "session_records_sha256": sha256_file(records_path),
    }
    write_json(task_dir / "runtime.json", runtime)
    write_json(
        task_dir / "status.json",
        {
            "status": "complete",
            "task_hash": expected_task_hash,
            "attempt": attempt,
            "completed_at": _now(),
        },
    )
    return runtime


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("search", "confirmation", "full", "smoke"),
    )
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--shard-id", required=True, type=int)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="Diagnostic override only; the D4.1 launcher selects idle GPUs by default.",
    )
    args = parser.parse_args(raw_argv)
    task_dir: Path | None = None
    try:
        result = _run(args, raw_argv)
        print(canonical_json(result))
    except Exception as error:
        experiment_dir = Path(args.experiment_dir).expanduser().resolve()
        task_dir = _task_directory(
            experiment_dir, args.stage, args.variant_id, args.shard_id
        )
        if task_dir.exists():
            status_path = task_dir / "status.json"
            previous: dict[str, object] = {}
            if status_path.exists():
                try:
                    previous = _load_object(status_path)
                except (ValueError, json.JSONDecodeError):
                    previous = {}
            write_json(
                status_path,
                {
                    "status": "failed",
                    "task_hash": previous.get("task_hash"),
                    "attempt": previous.get("attempt", 1),
                    "failed_at": _now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    main()
