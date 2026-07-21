"""Initialize and run the resumable D4.2 policy-matched OOF experiment."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from proactive_d1.core import make_fold_manifest, strip_answers
from proactive_d4_1.core import (
    canonical_json,
    ensure_experiment_identity,
    event,
    object_sha256,
    prepare_task_directory,
    task_should_run,
)
from proactive_d4_1.run_search import discover_idle_gpus
from proactive_r0.artifacts import (
    environment_snapshot,
    sha256_file,
    verify_model_snapshot,
    write_json,
)
from proactive_r0.core import load_jsonl

from .core import (
    EXPERIMENT_ID,
    build_source_manifest,
    feature_task_hash,
    load_candidates,
    partition_indices,
)
from .evaluate import evaluate_experiment
from .run_features import task_directory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "d4_2_internvl35_1b_adapted_input_policy_oof_v1.json"
)
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "output" / "experiments" / EXPERIMENT_ID
LOGGER = logging.getLogger("proactive_d4_2.run")


def _now() -> str:
    return datetime.now().astimezone().isoformat()


@contextmanager
def experiment_lock(experiment_dir: Path) -> Iterator[None]:
    """Reject concurrent master schedulers for the same experiment directory."""
    experiment_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = experiment_dir.parent / f".{experiment_dir.name}.master.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another D4.2 master process is already running for {experiment_dir}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _configure_logging(experiment_dir: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(
        experiment_dir / "master.log", mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def select_gpus(
    num_gpus: int, explicit: str | None, *, allow_shared_gpu: bool
) -> list[str]:
    """Select GPUs, retaining D4.1 idle-only safety unless sharing is explicit."""
    if not allow_shared_gpu:
        return discover_idle_gpus(num_gpus, explicit)
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    available = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    requested = (
        [value.strip() for value in explicit.split(",") if value.strip()]
        if explicit
        else available[:num_gpus]
    )
    if len(requested) != len(set(requested)):
        raise ValueError("D4.2 GPU_IDS contains duplicates")
    if len(requested) != num_gpus:
        raise ValueError("D4.2 GPU_IDS count must equal NUM_GPUS")
    missing = [index for index in requested if index not in available]
    if missing:
        raise ValueError(f"D4.2 requested unknown GPUs: {missing}")
    return requested


def _effective_config(args: argparse.Namespace) -> dict[str, object]:
    config = copy.deepcopy(_load_object(Path(args.config).expanduser().resolve()))
    data = dict(config["data"])  # type: ignore[arg-type]
    model = dict(config["model"])  # type: ignore[arg-type]
    head = dict(config["frozen_d4_head"])  # type: ignore[arg-type]
    starter = dict(config["starter_kit"])  # type: ignore[arg-type]
    if args.input_jsonl:
        data["input"] = str(Path(args.input_jsonl).expanduser().resolve())
    if args.video_dir:
        data["video_folder"] = str(Path(args.video_dir).expanduser().resolve())
    if args.model_path:
        model["default_local_path"] = str(Path(args.model_path).expanduser().resolve())
    if args.head_path:
        head["path"] = str(Path(args.head_path).expanduser().resolve())
    if args.starter_kit_dir:
        starter["path"] = str(Path(args.starter_kit_dir).expanduser().resolve())
    config["data"] = data
    config["model"] = model
    config["frozen_d4_head"] = head
    config["starter_kit"] = starter
    return config


def _validate_static_artifacts(config: Mapping[str, object]) -> dict[str, object]:
    data = dict(config["data"])  # type: ignore[arg-type]
    head = dict(config["frozen_d4_head"])  # type: ignore[arg-type]
    starter = dict(config["starter_kit"])  # type: ignore[arg-type]
    protocol = dict(config["protocol"])  # type: ignore[arg-type]
    reference = dict(config["d4_1_reference"])  # type: ignore[arg-type]
    reuse = dict(config["compute_reuse"])  # type: ignore[arg-type]
    input_path = _resolve(data["input"])
    head_path = _resolve(head["path"])
    starter_dir = _resolve(starter["path"])
    protocol_path = _resolve(protocol["path"])
    reference_dir = _resolve(reference["experiment_dir"])
    actual = {
        "input_sha256": sha256_file(input_path),
        "head_sha256": sha256_file(head_path),
        "starter_model_py_sha256": sha256_file(starter_dir / "model.py"),
        "starter_proactive_py_sha256": sha256_file(
            starter_dir / "run_generate_proactive.py"
        ),
        "starter_scorer_py_sha256": sha256_file(starter_dir / "run_evaluation.py"),
        "protocol_sha256": sha256_file(protocol_path),
        "d4_1_comparison_sha256": sha256_file(reference_dir / "comparison.json"),
        "d4_1_best_sha256": sha256_file(reference_dir / "best_inference.json"),
        "baseline_cached_generation_sha256": sha256_file(
            _resolve(reuse["baseline"]["path"])  # type: ignore[index]
        ),
        "history8_cached_generation_sha256": sha256_file(
            _resolve(reuse["history8"]["path"])  # type: ignore[index]
        ),
    }
    expected = {
        "input_sha256": str(data["input_sha256"]),
        "head_sha256": str(head["sha256"]),
        "starter_model_py_sha256": str(starter["model_py_sha256"]),
        "starter_proactive_py_sha256": str(starter["proactive_py_sha256"]),
        "starter_scorer_py_sha256": str(starter["scorer_py_sha256"]),
        "protocol_sha256": str(protocol["sha256"]),
        "d4_1_comparison_sha256": str(reference["comparison_sha256"]),
        "d4_1_best_sha256": str(reference["best_inference_sha256"]),
        "baseline_cached_generation_sha256": str(
            reuse["baseline"]["sha256"]  # type: ignore[index]
        ),
        "history8_cached_generation_sha256": str(
            reuse["history8"]["sha256"]  # type: ignore[index]
        ),
    }
    mismatches = {
        name: {"actual": value, "expected": expected[name]}
        for name, value in actual.items()
        if value != expected[name]
    }
    if mismatches:
        raise ValueError(f"D4.2 frozen artifact hashes changed: {mismatches}")
    return {
        **actual,
        "input_path": str(input_path),
        "head_path": str(head_path),
        "starter_kit_path": str(starter_dir),
        "protocol_path": str(protocol_path),
        "d4_1_reference_dir": str(reference_dir),
    }


def _static_audit(
    experiment_dir: Path,
    config: Mapping[str, object],
    config_sha256: str,
    *,
    dry_run: bool,
) -> dict[str, object]:
    path = experiment_dir / "static_audit.json"
    if path.exists():
        existing = _load_object(path)
        if existing.get("experiment_config_sha256") != config_sha256:
            raise ValueError("D4.2 static audit config hash changed")
        if existing.get("model_verified") or dry_run:
            return existing
    small = _validate_static_artifacts(config)
    model_config = dict(config["model"])  # type: ignore[arg-type]
    model_path = _resolve(model_config["default_local_path"])
    model_audit: dict[str, object] = {
        "path": str(model_path),
        "configured_weights_sha256": model_config["weights_sha256"],
        "configured_total_parameters": model_config["total_parameters"],
    }
    if not dry_run:
        model_audit.update(verify_model_snapshot(model_path, model_config))
        model_audit["weights_mtime_ns"] = (
            model_path / "model.safetensors"
        ).stat().st_mtime_ns
    audit = {
        "status": "dry_run_small_artifacts_only" if dry_run else "complete",
        "experiment_config_sha256": config_sha256,
        "small_artifacts": small,
        "model": model_audit,
        "model_verified": not dry_run,
        "completed_at": _now(),
    }
    write_json(path, audit)
    return audit


def _write_master_command(path: Path, argv: Sequence[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d4_2.run", *argv])
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _initialize(
    experiment_dir: Path,
    config: dict[str, object],
    *,
    num_shards: int,
    dry_run: bool,
    raw_argv: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    config_sha256 = object_sha256(config)
    ensure_experiment_identity(
        experiment_dir,
        config_sha256=config_sha256,
        identity={
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "input_sha256": config["data"]["input_sha256"],  # type: ignore[index]
            "frozen_d4_head_sha256": config["frozen_d4_head"]["sha256"],  # type: ignore[index]
        },
    )
    _configure_logging(experiment_dir)
    config_path = experiment_dir / "config.json"
    if config_path.exists() and _load_object(config_path) != config:
        raise ValueError("D4.2 effective config changed on resume")
    if not config_path.exists():
        write_json(config_path, config)
    _static_audit(experiment_dir, config, config_sha256, dry_run=dry_run)

    input_path = _resolve(config["data"]["input"])  # type: ignore[index]
    source_rows = load_jsonl(input_path)
    answer_free_rows = strip_answers(source_rows)
    if any("answers" in row for row in answer_free_rows):
        raise RuntimeError("D4.2 initialization failed to strip answers")
    source_manifest = build_source_manifest(answer_free_rows)
    source_manifest["source"] = {
        "path": str(input_path),
        "sha256": config["data"]["input_sha256"],  # type: ignore[index]
        "labels_used": False,
    }
    manifest_path = experiment_dir / "source_manifest.json"
    if manifest_path.exists() and _load_object(manifest_path) != source_manifest:
        raise ValueError("D4.2 source manifest changed on resume")
    if not manifest_path.exists():
        write_json(manifest_path, source_manifest)

    fold_manifest = make_fold_manifest(
        answer_free_rows,
        folds=int(config["folds"]["count"]),  # type: ignore[index]
        seed=str(config["folds"]["seed"]),  # type: ignore[index]
    )
    fold_path = experiment_dir / "fold_manifest.json"
    if not fold_path.exists():
        write_json(fold_path, fold_manifest)
    elif _load_object(fold_path) != fold_manifest:
        raise ValueError("D4.2 fold manifest changed on resume")
    actual_fold_sha = sha256_file(fold_path)
    expected_fold_sha = str(config["folds"]["reference_file_sha256"])  # type: ignore[index]
    if actual_fold_sha != expected_fold_sha:
        raise ValueError(
            f"D4.2 fold manifest SHA256 changed: {actual_fold_sha} != {expected_fold_sha}"
        )

    candidates = load_candidates(config)
    shards = partition_indices(source_manifest, num_shards)
    plan = {
        "schema_version": 1,
        "experiment_config_sha256": config_sha256,
        "candidate_ids": [str(candidate["candidate_id"]) for candidate in candidates],
        "num_shards": len(shards),
        "shards": shards,
        "sessions": len(source_rows),
        "chunks": sum(len(row["video_intervals"]) for row in source_rows),  # type: ignore[arg-type,index]
        "record_hidden_state": True,
    }
    plan["feature_plan_sha256"] = object_sha256(plan)
    plan_path = experiment_dir / "feature_plan.json"
    if plan_path.exists() and _load_object(plan_path) != plan:
        raise ValueError("D4.2 feature plan changed on resume")
    if not plan_path.exists():
        write_json(plan_path, plan)
    for candidate in candidates:
        for shard_id, session_indices in enumerate(shards):
            identity = feature_task_hash(
                experiment_config_sha256=config_sha256,
                candidate=candidate,
                shard_id=shard_id,
                session_indices=session_indices,
            )
            prepare_task_directory(
                task_directory(
                    experiment_dir, str(candidate["candidate_id"]), shard_id
                ),
                {
                    "schema_version": 1,
                    "task_hash": identity,
                    "experiment_config_sha256": config_sha256,
                    "stage": "features",
                    "candidate_id": candidate["candidate_id"],
                    "candidate_name": candidate["name"],
                    "parameters": candidate["parameters"],
                    "record_hidden_state": True,
                    "shard_id": shard_id,
                    "num_shards": len(shards),
                    "session_indices": session_indices,
                },
            )

    if not (experiment_dir / "environment.json").exists():
        write_json(experiment_dir / "environment.json", environment_snapshot())
        write_json(
            experiment_dir / "data_manifest.json",
            {
                "data": config["data"],
                "model": config["model"],
                "frozen_d4_head": config["frozen_d4_head"],
                "starter_kit": config["starter_kit"],
                "protocol": config["protocol"],
                "d4_1_reference": config["d4_1_reference"],
                "compute_reuse": config["compute_reuse"],
                "fold_manifest_sha256": actual_fold_sha,
                "feature_inference_rows_contain_answers": False,
                "external_data_used": False,
                "supervision": config["validation_policy"]["classification"],  # type: ignore[index]
            },
        )
        (experiment_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# {config['experiment_id']}",
                    "",
                    "状态：已初始化 D4.2 policy-matched 五折 OOF 实验。",
                    "",
                    "所有候选均重新生成 answer-stripped causal features，再训练完整 D4 线性头。",
                    "本实验为 post-D4.1 val-supervised 机制诊断，不是独立泛化证据。",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        _write_master_command(experiment_dir / "command.sh", raw_argv)
    return candidates, plan


def _schedule(
    experiment_dir: Path,
    candidates: Sequence[Mapping[str, object]],
    plan: Mapping[str, object],
    gpu_ids: Sequence[str],
    *,
    max_task_attempts: int,
    allow_shared_gpu: bool,
) -> None:
    tasks: list[dict[str, object]] = []
    num_shards = int(plan["num_shards"])
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        for shard_id in range(num_shards):
            task_dir = task_directory(experiment_dir, candidate_id, shard_id)
            task = _load_object(task_dir / "task.json")
            if task_should_run(task_dir, str(task["task_hash"])):
                tasks.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_name": candidate["name"],
                        "shard_id": shard_id,
                        "task_dir": task_dir,
                    }
                )
    if not tasks:
        LOGGER.info("D4.2 feature extraction already complete")
        return
    LOGGER.info("D4.2 scheduling %d feature tasks on GPUs %s", len(tasks), gpu_ids)
    active: dict[subprocess.Popen[bytes], dict[str, object]] = {}
    available = list(gpu_ids)
    try:
        while tasks or active:
            while tasks and available:
                task = tasks.pop(0)
                gpu_id = available.pop(0)
                command = [
                    sys.executable,
                    "-m",
                    "proactive_d4_2.run_features",
                    "--experiment-dir",
                    str(experiment_dir),
                    "--candidate-id",
                    str(task["candidate_id"]),
                    "--shard-id",
                    str(task["shard_id"]),
                    "--num-shards",
                    str(num_shards),
                    "--device",
                    "cuda:0",
                ]
                if allow_shared_gpu:
                    command.append("--allow-shared-gpu")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu_id
                environment["PYTHONNOUSERSITE"] = "1"
                environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
                output = (Path(task["task_dir"]) / "launcher.log").open("ab")
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                active[process] = {**task, "gpu_id": gpu_id, "output": output}
                event(
                    experiment_dir / "events.jsonl",
                    "feature_task_started",
                    timestamp=_now(),
                    candidate_id=task["candidate_id"],
                    candidate_name=task["candidate_name"],
                    shard_id=task["shard_id"],
                    gpu_id=gpu_id,
                    pid=process.pid,
                )
            if not active:
                continue
            time.sleep(1.0)
            for process, task in list(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                task["output"].close()  # type: ignore[union-attr]
                active.pop(process)
                available.append(str(task["gpu_id"]))
                available.sort(key=int)
                status = _load_object(Path(task["task_dir"]) / "status.json")
                event(
                    experiment_dir / "events.jsonl",
                    "feature_task_finished" if return_code == 0 else "feature_task_failed",
                    timestamp=_now(),
                    candidate_id=task["candidate_id"],
                    shard_id=task["shard_id"],
                    gpu_id=task["gpu_id"],
                    return_code=return_code,
                    attempt=status.get("attempt"),
                    error=status.get("error"),
                )
                if return_code != 0:
                    if int(status.get("attempt", 1)) < max_task_attempts:
                        tasks.append(
                            {
                                "candidate_id": task["candidate_id"],
                                "candidate_name": task["candidate_name"],
                                "shard_id": task["shard_id"],
                                "task_dir": task["task_dir"],
                            }
                        )
                    else:
                        raise RuntimeError(
                            "D4.2 feature task failed after "
                            f"{status.get('attempt')} attempts: "
                            f"{task['candidate_id']}/shard_{task['shard_id']}"
                        )
    finally:
        for process, task in active.items():
            if process.poll() is None:
                process.terminate()
        for process, task in active.items():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            task["output"].close()  # type: ignore[union-attr]


def _run_locked(
    args: argparse.Namespace,
    raw_argv: Sequence[str],
    experiment_dir: Path,
) -> dict[str, object]:
    config = _effective_config(args)
    num_shards = args.num_shards or args.num_gpus
    candidates, plan = _initialize(
        experiment_dir,
        config,
        num_shards=num_shards,
        dry_run=args.dry_run,
        raw_argv=raw_argv,
    )
    if args.dry_run:
        result = {
            "status": "dry_run",
            "experiment_id": config["experiment_id"],
            "experiment_config_sha256": object_sha256(config),
            "candidates": candidates,
            "sessions_per_candidate": plan["sessions"],
            "chunks_per_candidate": plan["chunks"],
            "num_shards": plan["num_shards"],
            "feature_tasks": len(candidates) * int(plan["num_shards"]),
            "gpu_required": False,
            "model_loaded": False,
            "evaluation_run": False,
        }
        write_json(experiment_dir / "dry_run.json", result)
        LOGGER.info("D4.2 CPU dry run complete: %s", canonical_json(result))
        return result
    if args.evaluate_only:
        return evaluate_experiment(experiment_dir)

    gpu_ids = select_gpus(
        args.num_gpus, args.gpu_ids, allow_shared_gpu=args.allow_shared_gpu
    )
    LOGGER.info(
        "D4.2 selected GPUs: %s (shared=%s)", gpu_ids, args.allow_shared_gpu
    )
    event(
        experiment_dir / "events.jsonl",
        "gpu_selection",
        timestamp=_now(),
        gpu_ids=gpu_ids,
        allow_shared_gpu=args.allow_shared_gpu,
    )
    _schedule(
        experiment_dir,
        candidates,
        plan,
        gpu_ids,
        max_task_attempts=args.max_task_attempts,
        allow_shared_gpu=args.allow_shared_gpu,
    )
    LOGGER.info("D4.2 feature extraction complete; starting five-fold OOF")
    result = evaluate_experiment(experiment_dir)
    event(
        experiment_dir / "events.jsonl",
        "experiment_complete",
        timestamp=_now(),
        winner_id=result["winner"]["candidate_id"],  # type: ignore[index]
        oof_macro_f1=result["winner"]["official_metrics"]["macro_f1"],  # type: ignore[index]
        train_fit_macro_f1=result["final_refit"]["train_fit_official"]["macro_f1"],  # type: ignore[index]
    )
    LOGGER.info("D4.2 complete: %s", canonical_json(result["winner"]))
    return result


def run(args: argparse.Namespace, raw_argv: Sequence[str]) -> dict[str, object]:
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    with experiment_lock(experiment_dir):
        return _run_locked(args, raw_argv, experiment_dir)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--gpu-ids")
    parser.add_argument("--max-task-attempts", type=int, default=2)
    parser.add_argument("--model-path")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--video-dir")
    parser.add_argument("--starter-kit-dir")
    parser.add_argument("--head-path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args(raw_argv)
    if not 1 <= args.num_gpus <= 8:
        parser.error("--num-gpus must be between 1 and 8")
    if args.num_shards is not None and args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if args.dry_run and args.evaluate_only:
        parser.error("--dry-run and --evaluate-only are mutually exclusive")
    result = run(args, raw_argv)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
