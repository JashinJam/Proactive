"""Create and execute the resumable D4.1 search, confirmation, and full stages."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from proactive_d1.core import strip_answers
from proactive_r0.artifacts import (
    environment_snapshot,
    sha256_file,
    verify_model_snapshot,
    write_json,
)
from proactive_r0.core import load_jsonl, write_jsonl

from .compare import compare_experiment
from .core import (
    EXPERIMENT_ID,
    atomic_append_jsonl,
    build_sample_manifest,
    canonical_json,
    compose_joint_variant,
    default_variants,
    ensure_experiment_identity,
    event,
    object_sha256,
    partition_session_indices,
    prepare_task_directory,
    rank_summaries,
    select_search_components,
    task_hash,
    task_should_run,
    validate_variants,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "d4_1_internvl35_1b_input_policy_search_v1.json"
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "output" / "experiments" / EXPERIMENT_ID
LOGGER = logging.getLogger("proactive_d4_1.run_search")


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


def _effective_config(args: argparse.Namespace) -> dict[str, object]:
    config = copy.deepcopy(_load_object(Path(args.config).expanduser().resolve()))
    data = dict(config["data"])  # type: ignore[arg-type]
    model = dict(config["model"])  # type: ignore[arg-type]
    head = dict(config["decision_head"])  # type: ignore[arg-type]
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
    config["decision_head"] = head
    config["starter_kit"] = starter
    return config


def discover_idle_gpus(num_gpus: int, explicit: str | None = None) -> list[str]:
    if num_gpus <= 0:
        raise ValueError("D4.1 NUM_GPUS must be positive")
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    gpu_by_index: dict[str, str] = {}
    for line in gpu_query.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid = (value.strip() for value in line.split(",", 1))
        gpu_by_index[index] = uuid
    if not gpu_by_index:
        raise RuntimeError("D4.1 found no NVIDIA GPUs")
    process_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    occupied = {
        line.split(",", 1)[0].strip()
        for line in process_query.stdout.splitlines()
        if line.strip() and "," in line
    }
    requested = (
        [value.strip() for value in explicit.split(",") if value.strip()]
        if explicit
        else [
            index
            for index in sorted(gpu_by_index, key=int)
            if gpu_by_index[index] not in occupied
        ][:num_gpus]
    )
    if explicit and len(requested) != len(set(requested)):
        raise ValueError("D4.1 GPU_IDS contains duplicates")
    if explicit and len(requested) != num_gpus:
        raise ValueError("D4.1 GPU_IDS count must equal NUM_GPUS")
    missing = [index for index in requested if index not in gpu_by_index]
    busy = [index for index in requested if gpu_by_index.get(index) in occupied]
    if missing:
        raise ValueError(f"D4.1 requested unknown GPUs: {missing}")
    if busy:
        raise RuntimeError(f"D4.1 requested GPUs already have compute processes: {busy}")
    if len(requested) < num_gpus:
        raise RuntimeError(
            f"D4.1 requires {num_gpus} idle GPUs but found {len(requested)}"
        )
    return requested


def _validate_small_artifacts(config: Mapping[str, object]) -> dict[str, object]:
    data = dict(config["data"])  # type: ignore[arg-type]
    head = dict(config["decision_head"])  # type: ignore[arg-type]
    starter = dict(config["starter_kit"])  # type: ignore[arg-type]
    protocol = dict(config["protocol"])  # type: ignore[arg-type]
    input_path = _resolve(data["input"])
    head_path = _resolve(head["path"])
    starter_dir = _resolve(starter["path"])
    protocol_path = _resolve(protocol["path"])
    actual = {
        "input_sha256": sha256_file(input_path),
        "head_sha256": sha256_file(head_path),
        "starter_model_py_sha256": sha256_file(starter_dir / "model.py"),
        "starter_proactive_py_sha256": sha256_file(
            starter_dir / "run_generate_proactive.py"
        ),
        "starter_scorer_py_sha256": sha256_file(starter_dir / "run_evaluation.py"),
        "protocol_sha256": sha256_file(protocol_path),
    }
    expected = {
        "input_sha256": str(data["input_sha256"]),
        "head_sha256": str(head["sha256"]),
        "starter_model_py_sha256": str(starter["model_py_sha256"]),
        "starter_proactive_py_sha256": str(starter["proactive_py_sha256"]),
        "starter_scorer_py_sha256": str(starter["scorer_py_sha256"]),
        "protocol_sha256": str(protocol["sha256"]),
    }
    mismatches = {
        name: {"expected": expected[name], "actual": value}
        for name, value in actual.items()
        if value != expected[name]
    }
    if mismatches:
        raise ValueError(f"D4.1 frozen artifact hashes changed: {mismatches}")
    return {
        **actual,
        "input_path": str(input_path),
        "head_path": str(head_path),
        "starter_kit_path": str(starter_dir),
        "protocol_path": str(protocol_path),
    }


def _static_audit(
    experiment_dir: Path,
    config: Mapping[str, object],
    config_sha256: str,
    *,
    dry_run: bool,
) -> dict[str, object]:
    audit_path = experiment_dir / "static_audit.json"
    if audit_path.exists():
        existing = _load_object(audit_path)
        if existing.get("experiment_config_sha256") != config_sha256:
            raise ValueError("D4.1 static audit config hash changed")
        if existing.get("model_verified") or dry_run:
            return existing
    small = _validate_small_artifacts(config)
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
    write_json(audit_path, audit)
    return audit


def _initialize(
    experiment_dir: Path,
    config: dict[str, object],
    *,
    dry_run: bool,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    config_sha256 = object_sha256(config)
    ensure_experiment_identity(
        experiment_dir,
        config_sha256=config_sha256,
        identity={
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "input_sha256": config["data"]["input_sha256"],  # type: ignore[index]
            "head_sha256": config["decision_head"]["sha256"],  # type: ignore[index]
        },
    )
    _configure_logging(experiment_dir)
    config_path = experiment_dir / "config.json"
    if config_path.exists() and _load_object(config_path) != config:
        raise ValueError("D4.1 effective config changed on resume")
    if not config_path.exists():
        write_json(config_path, config)
    _static_audit(experiment_dir, config, config_sha256, dry_run=dry_run)
    source_rows = load_jsonl(_resolve(config["data"]["input"]))  # type: ignore[index]
    inference_rows = strip_answers(source_rows)
    if any("answers" in row for row in inference_rows):
        raise RuntimeError("D4.1 initialization failed to strip answers")
    manifest = build_sample_manifest(
        inference_rows,
        seed=int(config["search"]["seed"]),  # type: ignore[index]
        sessions_per_domain=int(
            config["search"]["sessions_per_domain_per_subset"]  # type: ignore[index]
        ),
    )
    manifest["source"] = {
        "path": str(_resolve(config["data"]["input"])),  # type: ignore[index]
        "sha256": config["data"]["input_sha256"],  # type: ignore[index]
        "sessions": len(source_rows),
        "labels_used_for_sampling": False,
    }
    manifest_path = experiment_dir / "sample_manifest.json"
    if manifest_path.exists() and _load_object(manifest_path) != manifest:
        raise ValueError("D4.1 frozen sample manifest changed on resume")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    variants_path = experiment_dir / "variants.jsonl"
    predefined = default_variants()
    if variants_path.exists():
        variants = load_jsonl(variants_path)
        by_id = {str(value["variant_id"]): value for value in variants}
        for expected in predefined:
            if by_id.get(str(expected["variant_id"])) != expected:
                raise ValueError("D4.1 predefined variants changed on resume")
    else:
        write_jsonl(variants_path, predefined)
        variants = list(predefined)
    validate_variants(variants)
    if not (experiment_dir / "environment.json").exists():
        write_json(experiment_dir / "environment.json", environment_snapshot())
        write_json(
            experiment_dir / "data_manifest.json",
            {
                "data": config["data"],
                "model": config["model"],
                "decision_head": config["decision_head"],
                "starter_kit": config["starter_kit"],
                "generation_rows_contain_answers": False,
                "selection_reads_public_validation_labels_after_inference": True,
            },
        )
        (experiment_dir / "README.md").write_text(
            "\n".join(
                [
                    f"# {config['experiment_id']}",
                    "",
                    "Status: initialized; no D4.1 GPU result is implied by this file.",
                    "",
                    f"Hypothesis: {config['hypothesis']}",
                    "",
                    "Classification: val-supervised public-validation input-policy audit.",
                    "",
                    "Run: `bash scripts/run_d4_1_input_policy_search.sh`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    event(
        experiment_dir / "events.jsonl",
        "initialized",
        timestamp=_now(),
        dry_run=dry_run,
        experiment_config_sha256=config_sha256,
    )
    return manifest, variants, source_rows


def _stage_plan(
    experiment_dir: Path,
    *,
    config_sha256: str,
    stage: str,
    variant_ids: Sequence[str],
    indices: Sequence[int],
    manifest: Mapping[str, object],
    num_shards: int,
    variants: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    shards = partition_session_indices(indices, manifest, num_shards)
    plan = {
        "schema_version": 1,
        "stage": stage,
        "experiment_config_sha256": config_sha256,
        "variant_ids": list(variant_ids),
        "indices": list(indices),
        "num_shards": len(shards),
        "shards": shards,
        "variant_parameters": {
            variant_id: variants[variant_id]["parameters"]
            for variant_id in variant_ids
        },
    }
    plan["stage_plan_sha256"] = object_sha256(plan)
    path = experiment_dir / "stage_plans" / f"{stage}.json"
    if path.exists() and _load_object(path) != plan:
        raise ValueError(f"D4.1 {stage} plan changed on resume")
    if not path.exists():
        write_json(path, plan)
    for variant_id in variant_ids:
        variant = variants[variant_id]
        for shard_id, session_indices in enumerate(shards):
            identity = task_hash(
                experiment_config_sha256=config_sha256,
                stage=stage,
                variant=variant,
                shard_id=shard_id,
                session_indices=session_indices,
            )
            prepare_task_directory(
                experiment_dir
                / "runs"
                / stage
                / variant_id
                / f"shard_{shard_id:03d}",
                {
                    "schema_version": 1,
                    "task_hash": identity,
                    "experiment_config_sha256": config_sha256,
                    "stage": stage,
                    "variant_id": variant_id,
                    "parameters": variant["parameters"],
                    "shard_id": shard_id,
                    "num_shards": len(shards),
                    "session_indices": session_indices,
                },
            )
    return plan


def _schedule_stage(
    experiment_dir: Path,
    stage_plan: Mapping[str, object],
    gpu_ids: Sequence[str],
    *,
    max_task_attempts: int,
    allow_shared_gpu: bool,
) -> None:
    tasks: list[dict[str, object]] = []
    stage = str(stage_plan["stage"])
    num_shards = int(stage_plan["num_shards"])
    for variant_id in stage_plan["variant_ids"]:  # type: ignore[index]
        for shard_id in range(num_shards):
            task_dir = (
                experiment_dir
                / "runs"
                / stage
                / str(variant_id)
                / f"shard_{shard_id:03d}"
            )
            task = _load_object(task_dir / "task.json")
            if task_should_run(task_dir, str(task["task_hash"])):
                tasks.append(
                    {
                        "variant_id": str(variant_id),
                        "shard_id": shard_id,
                        "task_dir": task_dir,
                    }
                )
    if not tasks:
        LOGGER.info("D4.1 %s stage already complete", stage)
        return
    LOGGER.info("D4.1 %s scheduling %d tasks on GPUs %s", stage, len(tasks), gpu_ids)
    active: dict[subprocess.Popen[bytes], dict[str, object]] = {}
    available = list(gpu_ids)
    while tasks or active:
        while tasks and available:
            task = tasks.pop(0)
            gpu_id = available.pop(0)
            command = [
                sys.executable,
                "-m",
                "proactive_d4_1.run_variant",
                "--experiment-dir",
                str(experiment_dir),
                "--stage",
                stage,
                "--variant-id",
                str(task["variant_id"]),
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
                "task_started",
                timestamp=_now(),
                stage=stage,
                variant_id=task["variant_id"],
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
                "task_finished" if return_code == 0 else "task_failed",
                timestamp=_now(),
                stage=stage,
                variant_id=task["variant_id"],
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
                            "variant_id": task["variant_id"],
                            "shard_id": task["shard_id"],
                            "task_dir": task["task_dir"],
                        }
                    )
                else:
                    for other, running_task in active.items():
                        other.terminate()
                    for other, running_task in active.items():
                        try:
                            other.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            other.kill()
                            other.wait()
                        running_task["output"].close()  # type: ignore[union-attr]
                    raise RuntimeError(
                        f"D4.1 task failed after {status.get('attempt')} attempts: "
                        f"{stage}/{task['variant_id']}/shard_{task['shard_id']}"
                    )


def _completed_stage_summaries(
    comparison: Mapping[str, object], stage: str
) -> list[dict[str, object]]:
    value = comparison["stages"].get(stage)  # type: ignore[union-attr]
    if not isinstance(value, Mapping) or value.get("status") != "complete":
        raise ValueError(f"D4.1 {stage} comparison is not complete")
    return list(value["ranking"])  # type: ignore[arg-type]


def run_search(args: argparse.Namespace) -> dict[str, object]:
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    config = _effective_config(args)
    manifest, variants, source_rows = _initialize(
        experiment_dir, config, dry_run=args.dry_run
    )
    config_sha256 = object_sha256(config)
    by_id = {str(variant["variant_id"]): variant for variant in variants}
    predefined = default_variants()
    baseline_id = next(
        str(variant["variant_id"]) for variant in predefined if variant["is_baseline"]
    )
    search_plan = _stage_plan(
        experiment_dir,
        config_sha256=config_sha256,
        stage="search",
        variant_ids=[str(variant["variant_id"]) for variant in predefined],
        indices=manifest["search"]["indices"],  # type: ignore[index]
        manifest=manifest,
        num_shards=args.num_shards or args.num_gpus,
        variants=by_id,
    )
    if args.dry_run:
        projected = {
            "status": "dry_run",
            "experiment_id": config["experiment_id"],
            "experiment_config_sha256": config_sha256,
            "search": {
                "variants": len(predefined),
                "sessions_per_variant": len(search_plan["indices"]),  # type: ignore[arg-type]
                "tasks": len(predefined) * int(search_plan["num_shards"]),
            },
            "confirmation": {
                "variants": "16 or 17 after joint composition",
                "sessions_per_variant": 80,
            },
            "full": {"variants": 3, "sessions_per_variant": len(source_rows)},
            "gpu_required": False,
            "model_loaded": False,
        }
        write_json(experiment_dir / "dry_run.json", projected)
        LOGGER.info("D4.1 CPU dry run complete: %s", canonical_json(projected))
        return projected

    if args.smoke_only:
        smoke_gpu_ids = discover_idle_gpus(1, args.gpu_ids)
        non_baseline_id = next(
            str(variant["variant_id"])
            for variant in predefined
            if not variant["is_baseline"]
        )
        smoke_ids = [baseline_id, non_baseline_id]
        smoke_plan = _stage_plan(
            experiment_dir,
            config_sha256=config_sha256,
            stage="smoke",
            variant_ids=smoke_ids,
            indices=manifest["smoke"]["indices"],  # type: ignore[index]
            manifest=manifest,
            num_shards=1,
            variants=by_id,
        )
        _schedule_stage(
            experiment_dir,
            smoke_plan,
            smoke_gpu_ids,
            max_task_attempts=args.max_task_attempts,
            allow_shared_gpu=False,
        )
        variants_result: list[dict[str, object]] = []
        for variant_id in smoke_ids:
            task_dir = (
                experiment_dir
                / "runs"
                / "smoke"
                / variant_id
                / "shard_000"
            )
            runtime = _load_object(task_dir / "runtime.json")
            records = load_jsonl(task_dir / "session_records.jsonl")
            if len(records) != 1 or runtime.get("preexisting_gpu_processes"):
                raise ValueError("D4.1 GPU smoke did not use one exclusive short session")
            variants_result.append(
                {
                    "variant_id": variant_id,
                    "parameters": by_id[variant_id]["parameters"],
                    "input_index": records[0]["input_index"],
                    "chunks": len(records[0]["chunks"]),  # type: ignore[arg-type]
                    "timing": records[0]["timing"],
                    "peak_gpu_memory_bytes": runtime["peak_gpu_memory_bytes"],
                    "predictions_sha256": runtime["predictions_sha256"],
                }
            )
        smoke_result = {
            "status": "pass",
            "classification": "D4.1 baseline/non-baseline same-session GPU smoke",
            "gpu_id": smoke_gpu_ids[0],
            "exclusive_gpu": True,
            "variants": variants_result,
        }
        write_json(experiment_dir / "gpu_smoke.json", smoke_result)
        event(
            experiment_dir / "events.jsonl",
            "gpu_smoke_complete",
            timestamp=_now(),
            gpu_id=smoke_gpu_ids[0],
            variant_ids=smoke_ids,
        )
        return smoke_result

    gpu_ids = discover_idle_gpus(args.num_gpus, args.gpu_ids)
    LOGGER.info("D4.1 selected idle GPUs: %s", gpu_ids)
    event(
        experiment_dir / "events.jsonl",
        "gpu_selection",
        timestamp=_now(),
        gpu_ids=gpu_ids,
        existing_compute_processes_allowed=args.allow_shared_gpu,
    )
    _schedule_stage(
        experiment_dir,
        search_plan,
        gpu_ids,
        max_task_attempts=args.max_task_attempts,
        allow_shared_gpu=args.allow_shared_gpu,
    )
    comparison = compare_experiment(experiment_dir)
    search_summaries = _completed_stage_summaries(comparison, "search")
    summary_by_id = {
        str(summary["variant_id"]): summary for summary in search_summaries
    }
    components = select_search_components(predefined, summary_by_id)
    joint = compose_joint_variant(
        components["visual"], components["history"], components["generation"]
    )
    joint_id = str(joint["variant_id"])
    alias_of = joint_id if joint_id in by_id else None
    if alias_of is None:
        atomic_append_jsonl(experiment_dir / "variants.jsonl", joint)
        variants.append(joint)
        by_id[joint_id] = joint
    joint_selection = {
        "schema_version": 1,
        "selected_components": {
            name: str(value["variant_id"]) for name, value in components.items()
        },
        "joint_variant_id": joint_id,
        "alias_of": alias_of,
        "inference_reused": alias_of is not None,
        "parameters": joint["parameters"],
    }
    joint_path = experiment_dir / "joint_selection.json"
    if joint_path.exists() and _load_object(joint_path) != joint_selection:
        raise ValueError("D4.1 joint selection changed on resume")
    if not joint_path.exists():
        write_json(joint_path, joint_selection)

    confirmation_ids = [str(variant["variant_id"]) for variant in variants]
    confirmation_plan = _stage_plan(
        experiment_dir,
        config_sha256=config_sha256,
        stage="confirmation",
        variant_ids=confirmation_ids,
        indices=manifest["confirmation"]["indices"],  # type: ignore[index]
        manifest=manifest,
        num_shards=args.num_shards or args.num_gpus,
        variants=by_id,
    )
    _schedule_stage(
        experiment_dir,
        confirmation_plan,
        gpu_ids,
        max_task_attempts=args.max_task_attempts,
        allow_shared_gpu=args.allow_shared_gpu,
    )
    comparison = compare_experiment(experiment_dir)
    confirmation = _completed_stage_summaries(comparison, "confirmation")
    non_baseline = [
        summary for summary in rank_summaries(confirmation) if not summary["is_baseline"]
    ]
    candidate_count = int(config["evaluation"]["full_candidates"])  # type: ignore[index]
    if len(non_baseline) < candidate_count:
        raise ValueError("D4.1 confirmation has too few non-baseline candidates")
    selected_candidates = non_baseline[:candidate_count]
    confirmation_selection = {
        "schema_version": 1,
        "selected_variant_ids": [
            str(summary["variant_id"]) for summary in selected_candidates
        ],
        "selection_stage": "confirmation",
        "ranking_rule": config["evaluation"]["tie_break"],  # type: ignore[index]
    }
    selection_path = experiment_dir / "confirmation_selection.json"
    if selection_path.exists() and _load_object(selection_path) != confirmation_selection:
        raise ValueError("D4.1 confirmation selection changed on resume")
    if not selection_path.exists():
        write_json(selection_path, confirmation_selection)

    full_ids = [
        baseline_id,
        *[str(summary["variant_id"]) for summary in selected_candidates],
    ]
    full_plan = _stage_plan(
        experiment_dir,
        config_sha256=config_sha256,
        stage="full",
        variant_ids=full_ids,
        indices=manifest["full"]["indices"],  # type: ignore[index]
        manifest=manifest,
        num_shards=args.num_shards or args.num_gpus,
        variants=by_id,
    )
    _schedule_stage(
        experiment_dir,
        full_plan,
        gpu_ids,
        max_task_attempts=args.max_task_attempts,
        allow_shared_gpu=args.allow_shared_gpu,
    )
    comparison = compare_experiment(experiment_dir)
    write_json(
        experiment_dir / "task_state.json",
        {"status": "complete", "completed_at": _now(), "gpu_ids": gpu_ids},
    )
    event(
        experiment_dir / "events.jsonl",
        "experiment_complete",
        timestamp=_now(),
        best_inference_path=str(experiment_dir / "best_inference.json"),
    )
    return comparison


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--model-path")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--video-dir")
    parser.add_argument("--starter-kit-dir")
    parser.add_argument("--head-path")
    parser.add_argument("--gpu-ids")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--max-task-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="Diagnostic override only; default D4.1 execution requires idle GPUs.",
    )
    args = parser.parse_args(argv)
    if args.num_gpus <= 0:
        parser.error("--num-gpus must be positive")
    if args.num_shards is not None and args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if args.max_task_attempts <= 0:
        parser.error("--max-task-attempts must be positive")
    if args.dry_run and args.smoke_only:
        parser.error("--dry-run and --smoke-only are mutually exclusive")
    result = run_search(args)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
