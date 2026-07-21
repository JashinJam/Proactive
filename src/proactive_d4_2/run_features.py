"""Run one resumable D4.2 policy-matched feature shard on one GPU."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from proactive_d1.core import load_decision_head, strip_answers
from proactive_d1.internvl_features import (
    InternVLDecisionFeatureExtractor,
    NeuralDecisionFeatures,
)
from proactive_d4.deploy import process_session_with_dialog_stage_head
from proactive_d4_1.core import (
    atomic_append_jsonl,
    canonical_json,
    load_task_records,
    object_sha256,
    prepare_task_directory,
    task_should_run,
)
from proactive_d4_1.run_variant import TimedDecisionModel
from proactive_r0.artifacts import environment_snapshot, sha256_file, write_json
from proactive_r0.core import (
    CausalInferenceConfig,
    load_jsonl,
    load_starter_kit,
    validate_source_rows,
    write_jsonl,
)

from .core import (
    BASELINE,
    PolicyParameters,
    feature_task_hash,
    load_candidates,
    partition_indices,
    recover_duplicate_feature_prefix,
    stable_candidate_id,
    validate_feature_records,
)

LOGGER = logging.getLogger("proactive_d4_2.run_features")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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
        raise RuntimeError("D4.2 target stripping failed")
    return sanitized


class ReusedComponentModel:
    """Run only the policy-dependent forward and replay the identical component."""

    def __init__(
        self, model: InternVLDecisionFeatureExtractor, mode: str
    ) -> None:
        self.model = model
        self.mode = mode
        self.reference: Mapping[str, object] | None = None
        self.cursor = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self.model, name)

    def begin_session(self, reference: Mapping[str, object] | None) -> None:
        self.reference = reference
        self.cursor = 0

    def _chunk(self) -> Mapping[str, object]:
        if self.reference is None:
            raise ValueError("D4.2 reuse mode has no reference session")
        chunks = self.reference.get("chunks")
        if not isinstance(chunks, list) or self.cursor >= len(chunks):
            raise ValueError("D4.2 reuse reference chunk coverage changed")
        chunk = chunks[self.cursor]
        if not isinstance(chunk, Mapping):
            raise ValueError("D4.2 reuse reference chunk is malformed")
        return chunk

    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> str:
        if self.mode == "cached_generation_recomputed_decision_features":
            return str(self._chunk()["raw_response"])
        return self.model.generate(
            frames, messages, max_new_tokens=max_new_tokens
        )

    def extract_decision_features(
        self, frames: list[object], messages: list[dict[str, str]]
    ) -> NeuralDecisionFeatures:
        if self.mode == "generation_with_d4_2_baseline_decision_features":
            chunk = self._chunk()
            result = NeuralDecisionFeatures(
                hidden_state=np.asarray(chunk["hidden_state"], dtype=np.float32),
                silent_log_probability=float(chunk["silent_log_probability"]),
                interrupt_log_probability=float(chunk["interrupt_log_probability"]),
                tag_margin=float(chunk["tag_margin"]),
                prompt_tokens=int(chunk["prompt_tokens"]),
                hidden_max_abs_difference=float(
                    chunk.get("hidden_max_abs_difference", 0.0)
                ),
                hidden_cosine_similarity=float(
                    chunk.get("hidden_cosine_similarity", 1.0)
                ),
                extraction_mode=str(chunk["decision_feature_mode"]),
                candidate_forward_passes=int(chunk["candidate_forward_passes"]),
            )
        else:
            result = self.model.extract_decision_features(frames, messages)
        self.cursor += 1
        return result

    def finish_session(self) -> None:
        if self.reference is None:
            return
        chunks = self.reference.get("chunks")
        if not isinstance(chunks, list) or self.cursor != len(chunks):
            raise ValueError("D4.2 did not consume the complete reuse reference")


def _reference_records(
    *,
    experiment_dir: Path,
    config: Mapping[str, object],
    candidate: Mapping[str, object],
    shard_id: int,
    expected_indices: Sequence[int],
) -> tuple[str, dict[int, dict[str, object]]]:
    reuse = config.get("compute_reuse")
    if not isinstance(reuse, Mapping):
        raise ValueError("D4.2 config lacks compute-reuse policy")
    policy = reuse.get(str(candidate["name"]))
    if not isinstance(policy, Mapping):
        raise ValueError("D4.2 candidate lacks compute-reuse policy")
    mode = str(policy.get("mode"))
    if mode == "full_inference":
        return mode, {}
    if mode == "cached_generation_recomputed_decision_features":
        path = _resolve(policy["path"])
        if sha256_file(path) != str(policy["sha256"]):
            raise ValueError("D4.2 cached D4.1 generation SHA256 changed")
        records = load_jsonl(path)
    elif mode == "generation_with_d4_2_baseline_decision_features":
        baseline_id = stable_candidate_id(BASELINE)
        baseline_dir = task_directory(experiment_dir, baseline_id, shard_id)
        status_path = baseline_dir / "status.json"
        deadline = time.monotonic() + 3600.0
        while True:
            if status_path.exists() and _load_object(status_path).get("status") == "complete":
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("D4.2 tokens16 timed out waiting for baseline features")
            time.sleep(5.0)
        records = load_jsonl(baseline_dir / "session_records.jsonl")
    else:
        raise ValueError(f"Unsupported D4.2 compute-reuse mode: {mode}")
    by_index = {int(record["input_index"]): record for record in records}
    if len(by_index) != len(records):
        raise ValueError("D4.2 reuse reference contains duplicate sessions")
    missing = [index for index in expected_indices if index not in by_index]
    if missing:
        raise ValueError(f"D4.2 reuse reference lacks sessions: {missing[:8]}")
    return mode, {index: by_index[index] for index in expected_indices}


def _validate_reuse_identity(
    record: Mapping[str, object],
    reference: Mapping[str, object],
    mode: str,
) -> None:
    if (record.get("input_index"), record.get("video_path")) != (
        reference.get("input_index"),
        reference.get("video_path"),
    ):
        raise ValueError("D4.2 reuse session identity changed")
    chunks = record.get("chunks")
    reference_chunks = reference.get("chunks")
    if not isinstance(chunks, list) or not isinstance(reference_chunks, list):
        raise ValueError("D4.2 reuse session chunks are malformed")
    if len(chunks) != len(reference_chunks):
        raise ValueError("D4.2 reuse chunk coverage changed")
    for chunk, reference_chunk in zip(chunks, reference_chunks):
        if not isinstance(chunk, Mapping) or not isinstance(reference_chunk, Mapping):
            raise ValueError("D4.2 reuse chunk is malformed")
        for name in ("chunk_index", "interval", "model_input_frames", "prompt_tokens"):
            if chunk.get(name) != reference_chunk.get(name):
                raise ValueError(f"D4.2 reuse identity changed for {name}")
        if mode == "cached_generation_recomputed_decision_features":
            for name in (
                "raw_response",
                "r0_answer",
                "silent_log_probability",
                "interrupt_log_probability",
                "tag_margin",
                "decision_feature_mode",
                "candidate_forward_passes",
            ):
                if chunk.get(name) != reference_chunk.get(name):
                    raise ValueError(f"D4.2 recomputed feature differs for {name}")
        elif mode == "generation_with_d4_2_baseline_decision_features":
            for name in (
                "silent_log_probability",
                "interrupt_log_probability",
                "tag_margin",
                "hidden_state",
                "decision_feature_mode",
                "candidate_forward_passes",
            ):
                if chunk.get(name) != reference_chunk.get(name):
                    raise ValueError(f"D4.2 reused baseline feature differs for {name}")


def _apply_reuse_timing(
    record: dict[str, object], reference: Mapping[str, object] | None, mode: str
) -> None:
    timing = record.get("timing")
    chunks = record.get("chunks")
    if not isinstance(timing, dict) or not isinstance(chunks, list):
        raise ValueError("D4.2 reuse timing record is malformed")
    experimental = dict(timing)
    if mode == "full_inference":
        timing["compute_reuse_mode"] = mode
        timing["experimental_session_wall_seconds"] = timing["session_wall_seconds"]
        return
    if reference is None or not isinstance(reference.get("chunks"), list):
        raise ValueError("D4.2 reuse timing lacks reference chunks")
    reference_chunks = reference["chunks"]
    reused_seconds = 0.0
    for chunk, reference_chunk in zip(chunks, reference_chunks):
        if not isinstance(chunk, dict) or not isinstance(reference_chunk, Mapping):
            raise ValueError("D4.2 reuse timing chunk is malformed")
        if mode == "cached_generation_recomputed_decision_features":
            value = float(reference_chunk["generation_seconds"])
            chunk["generation_seconds"] = value
        else:
            value = float(reference_chunk["decision_feature_seconds"])
            chunk["decision_feature_seconds"] = value
        reused_seconds += value
        chunk["model_inference_seconds"] = float(chunk["generation_seconds"]) + float(
            chunk["decision_feature_seconds"]
        )
        chunk["compute_reuse_mode"] = mode
    generation = sum(float(chunk["generation_seconds"]) for chunk in chunks)
    decision = sum(float(chunk["decision_feature_seconds"]) for chunk in chunks)
    timing.update(
        {
            "generation_seconds": generation,
            "decision_feature_seconds": decision,
            "model_inference_seconds": generation + decision,
            "experimental_session_wall_seconds": experimental["session_wall_seconds"],
            "session_wall_seconds": float(experimental["session_wall_seconds"])
            + reused_seconds,
            "compute_reuse_mode": mode,
            "reused_component_seconds": reused_seconds,
        }
    )


def _candidate_by_id(
    config: Mapping[str, object], candidate_id: str
) -> dict[str, object]:
    matches = [
        candidate
        for candidate in load_candidates(config)
        if candidate["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"D4.2 expected one candidate {candidate_id}, got {len(matches)}"
        )
    return matches[0]


def task_directory(experiment_dir: Path, candidate_id: str, shard_id: int) -> Path:
    return experiment_dir / "features" / candidate_id / f"shard_{shard_id:03d}"


def _write_command(path: Path, argv: Sequence[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d4_2.run_features", *argv])
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(PROJECT_ROOT))}",
        "export PYTHONNOUSERSITE=1",
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}",
    ]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        lines.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(visible)}")
    lines.extend([f"exec {command}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def _next_recovery_paths(task_dir: Path) -> tuple[Path, Path]:
    sequence = 1
    while True:
        backup = task_dir / f"session_records.pre_duplicate_recovery_{sequence:03d}.jsonl"
        audit = task_dir / f"duplicate_recovery_{sequence:03d}.json"
        if not backup.exists() and not audit.exists():
            return backup, audit
        sequence += 1


def _recover_duplicate_records(
    *,
    task_dir: Path,
    records_path: Path,
    records: Sequence[Mapping[str, object]],
    expected_indices: Sequence[int],
    rows: Sequence[Mapping[str, object]],
    hidden_size: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    recovered, recovery = recover_duplicate_feature_prefix(
        records,
        expected_indices,
        rows,
        hidden_size=hidden_size,
    )
    if not recovery["recovered"]:
        return [dict(record) for record in recovered], recovery

    backup_path, audit_path = _next_recovery_paths(task_dir)
    status_path = task_dir / "status.json"
    previous_status = _load_object(status_path) if status_path.exists() else {}
    original_sha256 = sha256_file(records_path)
    shutil.copy2(records_path, backup_path)
    if sha256_file(backup_path) != original_sha256:
        raise RuntimeError("D4.2 duplicate-record backup verification failed")
    write_jsonl(records_path, [dict(record) for record in recovered])
    recovered_sha256 = sha256_file(records_path)
    audit = {
        **recovery,
        "recovered_at": _now(),
        "reason": "concurrent shard workers appended identical session records",
        "original_path": str(records_path),
        "original_sha256": original_sha256,
        "backup_path": str(backup_path),
        "backup_sha256": sha256_file(backup_path),
        "recovered_sha256": recovered_sha256,
        "previous_status": previous_status,
    }
    write_json(audit_path, audit)
    LOGGER.warning(
        "Recovered %d duplicate records; preserved original as %s",
        int(recovery["discarded_duplicates"]),
        backup_path,
    )
    return [dict(record) for record in recovered], audit


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


def _run(args: argparse.Namespace, raw_argv: list[str]) -> dict[str, object]:
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    config = _load_object(experiment_dir / "config.json")
    config_sha256 = object_sha256(config)
    identity = _load_object(experiment_dir / "experiment_identity.json")
    if identity.get("experiment_config_sha256") != config_sha256:
        raise ValueError("D4.2 effective config differs from experiment identity")
    audit = _load_object(experiment_dir / "static_audit.json")
    if audit.get("experiment_config_sha256") != config_sha256:
        raise ValueError("D4.2 static audit uses a different config")
    if audit.get("model_verified") is not True:
        raise ValueError("D4.2 model snapshot has not passed static verification")

    candidate = _candidate_by_id(config, args.candidate_id)
    manifest = _load_object(experiment_dir / "source_manifest.json")
    shards = partition_indices(manifest, args.num_shards)
    if not 0 <= args.shard_id < len(shards):
        raise ValueError(f"D4.2 shard ID {args.shard_id} is out of range")
    session_indices = shards[args.shard_id]
    expected_hash = feature_task_hash(
        experiment_config_sha256=config_sha256,
        candidate=candidate,
        shard_id=args.shard_id,
        session_indices=session_indices,
    )
    task = {
        "schema_version": 1,
        "task_hash": expected_hash,
        "experiment_config_sha256": config_sha256,
        "stage": "features",
        "candidate_id": args.candidate_id,
        "candidate_name": candidate["name"],
        "parameters": candidate["parameters"],
        "record_hidden_state": True,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "session_indices": session_indices,
    }
    task_dir = task_directory(experiment_dir, args.candidate_id, args.shard_id)
    prepare_task_directory(task_dir, task)
    _configure_logging(task_dir / "run.log")
    records_path = task_dir / "session_records.jsonl"

    data = dict(config["data"])  # type: ignore[arg-type]
    model_config = dict(config["model"])  # type: ignore[arg-type]
    head_config = dict(config["frozen_d4_head"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    frozen = dict(config["frozen_inference"])  # type: ignore[arg-type]
    input_path = _resolve(data["input"])
    video_folder = _resolve(data["video_folder"])
    model_path = _resolve(model_config["default_local_path"])
    head_path = _resolve(head_config["path"])
    starter_dir = _resolve(starter_config["path"])

    source_rows = load_jsonl(input_path)
    inference_rows = sanitize_inference_rows(source_rows)
    validate_source_rows([inference_rows[index] for index in session_indices], video_folder)
    records = load_task_records(records_path)
    records, recovery = _recover_duplicate_records(
        task_dir=task_dir,
        records_path=records_path,
        records=records,
        expected_indices=session_indices,
        rows=inference_rows,
        hidden_size=int(head_config["hidden_size"]),
    )
    validate_feature_records(
        records,
        session_indices,
        inference_rows,
        hidden_size=int(head_config["hidden_size"]),
        require_complete=False,
    )
    if not task_should_run(task_dir, expected_hash):
        validate_feature_records(
            records,
            session_indices,
            inference_rows,
            hidden_size=int(head_config["hidden_size"]),
            require_complete=True,
        )
        LOGGER.info("Completed D4.2 feature task skipped: %s", expected_hash)
        return _load_object(task_dir / "runtime.json")

    previous = _load_object(task_dir / "status.json")
    attempt = 1 if recovery["recovered"] else int(previous.get("attempt", 0)) + 1
    write_json(
        task_dir / "status.json",
        {
            "status": "running",
            "task_hash": expected_hash,
            "attempt": attempt,
            "started_at": _now(),
            "device": args.device,
            "pid": os.getpid(),
        },
    )
    attempt_started = time.monotonic()
    reuse_mode, references = _reference_records(
        experiment_dir=experiment_dir,
        config=config,
        candidate=candidate,
        shard_id=args.shard_id,
        expected_indices=session_indices,
    )
    if not (task_dir / "environment.json").exists():
        write_json(task_dir / "environment.json", environment_snapshot())
        write_json(task_dir / "code_state.json", _code_state())
        write_json(task_dir / "effective_config.json", {"experiment": config, "task": task})
        _write_command(task_dir / "command.sh", raw_argv)

    if sha256_file(head_path) != str(head_config["sha256"]):
        raise ValueError("D4.2 frozen D4 head SHA256 changed")
    head = load_decision_head(_load_object(head_path))
    if len(head.feature_names) + 1 != int(head_config["parameters"]):
        raise ValueError("D4.2 frozen D4 head parameter count changed")

    parameters = PolicyParameters.from_mapping(
        candidate["parameters"]  # type: ignore[arg-type]
    )
    inference = CausalInferenceConfig(**parameters.to_dict())
    starter = load_starter_kit(starter_dir)
    model = InternVLDecisionFeatureExtractor(
        model_path=str(model_path),
        device=args.device,
        dtype_name=str(model_config["dtype"]),
        attention_implementation=str(model_config["attention_implementation"]),
        seed=int(frozen["seed"]),
        require_exclusive_gpu=not args.allow_shared_gpu,
        video_frame_size=int(frozen["video_frame_size"]),
        pad_token_id=int(frozen["pad_token_id"]),
        decision_feature_mode=str(frozen["decision_feature_mode"]),
    )
    if model.parameter_count != int(model_config["total_parameters"]):
        raise ValueError("D4.2 loaded backbone parameter count changed")
    if model.hidden_size != int(head_config["hidden_size"]):
        raise ValueError("D4.2 loaded hidden width changed")
    reuse_model = ReusedComponentModel(model, reuse_mode)
    timed_model = TimedDecisionModel(reuse_model)  # type: ignore[arg-type]
    session_model_times = [
        float(record["timing"]["model_inference_seconds"])  # type: ignore[index]
        for record in records
    ]
    for position in range(len(records), len(session_indices)):
        input_index = session_indices[position]
        reference = references.get(input_index)
        reuse_model.begin_session(reference)
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
            record_hidden_state=True,
        )
        timed_model.attach_timing(record, time.perf_counter() - session_started)
        reuse_model.finish_session()
        if reference is not None:
            _validate_reuse_identity(record, reference, reuse_mode)
        _apply_reuse_timing(record, reference, reuse_mode)
        if "answers" in inference_rows[input_index]:
            raise RuntimeError("D4.2 answers reached feature inference")
        validate_feature_records(
            [record],
            [input_index],
            inference_rows,
            hidden_size=int(head_config["hidden_size"]),
            require_complete=True,
        )
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

    validation = validate_feature_records(
        records,
        session_indices,
        inference_rows,
        hidden_size=int(head_config["hidden_size"]),
        require_complete=True,
    )
    generation_seconds = sum(
        float(record["timing"]["generation_seconds"])  # type: ignore[index]
        for record in records
    )
    decision_seconds = sum(
        float(record["timing"]["decision_feature_seconds"])  # type: ignore[index]
        for record in records
    )
    runtime = {
        "status": "complete",
        "task_hash": expected_hash,
        "attempt": attempt,
        "completed_at": _now(),
        "wall_time_seconds_this_attempt": time.monotonic() - attempt_started,
        **validation,
        "generation_seconds": generation_seconds,
        "decision_feature_seconds": decision_seconds,
        "total_model_inference_seconds": generation_seconds + decision_seconds,
        "maximum_session_model_inference_seconds": max(session_model_times),
        "sessions_over_300_model_seconds": sum(value > 300 for value in session_model_times),
        "peak_gpu_memory_bytes": model.peak_memory_bytes(),
        "preexisting_gpu_processes": model.preexisting_gpu_processes,
        "compute_reuse_mode": reuse_mode,
        "session_records_sha256": sha256_file(records_path),
    }
    write_json(task_dir / "runtime.json", runtime)
    write_json(
        task_dir / "status.json",
        {
            "status": "complete",
            "task_hash": expected_hash,
            "attempt": attempt,
            "completed_at": _now(),
        },
    )
    return runtime


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--shard-id", required=True, type=int)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args(raw_argv)
    task_dir = task_directory(
        Path(args.experiment_dir).expanduser().resolve(),
        args.candidate_id,
        args.shard_id,
    )
    task_dir.mkdir(parents=True, exist_ok=True)
    with (task_dir / ".worker.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            result = _run(args, raw_argv)
            print(canonical_json(result))
        except Exception as error:
            status_path = task_dir / "status.json"
            if status_path.exists():
                previous: dict[str, object] = {}
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
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
