"""Run and audit the frozen C1 Small no-plan baseline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from .artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    verify_model_snapshot,
    write_json,
)
from .core import (
    CausalInferenceConfig,
    INTERRUPT_TAG,
    load_jsonl,
    load_starter_kit,
    process_session,
    validate_prediction_rows,
    validate_source_rows,
    write_jsonl,
)
from .internvl import InternVLProactiveModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "r0_internvl35_1b_no_plan.json"
LOGGER = logging.getLogger("proactive_r0")


def contiguous_shard_bounds(total: int, num_shards: int, shard_index: int) -> tuple[int, int]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    base, remainder = divmod(total, num_shards)
    start = shard_index * base + min(shard_index, remainder)
    stop = start + base + (1 if shard_index < remainder else 0)
    return start, stop


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def _load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "experiment_id",
        "hypothesis",
        "model",
        "data",
        "starter_kit",
        "inference",
        "evaluation",
        "validation_policy",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Config is missing keys: {sorted(missing)}")
    return config


def _tracked_code_paths(config_path: Path) -> list[Path]:
    return [
        *sorted((PROJECT_ROOT / "src" / "proactive_r0").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_r0" / "tests").glob("*.py")),
        config_path,
        PROJECT_ROOT / "models" / "internvl35_1b_hf.json",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
        PROJECT_ROOT / "C1_SPEC.md",
        PROJECT_ROOT / "Agent.md",
    ]


def _write_command(path: Path, argv: list[str]) -> None:
    pythonpath = str(PROJECT_ROOT / "src")
    command = shlex.join([sys.executable, "-m", "proactive_r0.run", *argv])
    text = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(pythonpath)}\n"
        f"exec {command}\n"
    )
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _write_text_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _validate_static_files(
    config: dict[str, object],
    input_path: Path,
    starter_dir: Path,
) -> dict[str, str]:
    data_config: dict[str, object] = config["data"]  # type: ignore[assignment]
    starter_config: dict[str, object] = config["starter_kit"]  # type: ignore[assignment]
    actual = {
        "input_sha256": sha256_file(input_path),
        "model_py_sha256": sha256_file(starter_dir / "model.py"),
        "proactive_py_sha256": sha256_file(
            starter_dir / "run_generate_proactive.py"
        ),
        "scorer_py_sha256": sha256_file(starter_dir / "run_evaluation.py"),
    }
    expected = {
        "input_sha256": str(data_config["input_sha256"]),
        "model_py_sha256": str(starter_config["model_py_sha256"]),
        "proactive_py_sha256": str(starter_config["proactive_py_sha256"]),
        "scorer_py_sha256": str(starter_config["scorer_py_sha256"]),
    }
    if actual != expected:
        raise ValueError(f"Pinned file fingerprint mismatch: {actual} != {expected}")
    return actual


def _effective_config(
    config: dict[str, object],
    config_path: Path,
    model_path: Path,
    input_path: Path,
    video_folder: Path,
    starter_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "model_path": str(model_path),
        "input_path": str(input_path),
        "video_folder": str(video_folder),
        "starter_kit_path": str(starter_dir),
        "output_dir": str(output_dir),
        "device": args.device,
        "max_sessions": args.max_sessions,
        "audit_only": args.audit_only,
        "skip_evaluation": args.skip_eval,
        "require_exclusive_gpu": args.require_exclusive_gpu,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    return effective


def _load_records(path: Path) -> list[dict[str, object]]:
    return load_jsonl(path) if path.exists() else []


def _validate_record_prefix(
    records: list[dict[str, object]],
    rows: list[dict[str, object]],
    input_index_offset: int = 0,
) -> None:
    if len(records) > len(rows):
        raise ValueError("Resume records are longer than the selected input")
    predictions: list[dict[str, object]] = []
    for expected_index, record in enumerate(records):
        if record.get("input_index") != input_index_offset + expected_index:
            raise ValueError(f"Resume record {expected_index} has wrong input_index")
        prediction = record.get("prediction")
        if not isinstance(prediction, dict):
            raise ValueError(f"Resume record {expected_index} has no prediction")
        predictions.append(prediction)
    validate_prediction_rows(rows[: len(records)], predictions)


def _run_official_scorer(
    starter_dir: Path,
    golden_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    scorer_log_path: Path,
) -> None:
    command = [
        sys.executable,
        str(starter_dir / "run_evaluation.py"),
        "--task",
        "proactive",
        "--eval-only",
        "--golden",
        str(golden_path),
        "--predictions",
        str(predictions_path),
        "--output",
        str(metrics_path),
    ]
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    scorer_log_path.write_text(
        f"$ {shlex.join(command)}\n{completed.stdout}", encoding="utf-8"
    )
    LOGGER.info("Official scorer completed:\n%s", completed.stdout.rstrip())


def _diagnostics(
    records: list[dict[str, object]],
    prediction_validation: dict[str, int],
    predictions_path: Path,
) -> dict[str, object]:
    reasons: Counter[str] = Counter()
    for record in records:
        for chunk in record.get("chunks", []):  # type: ignore[union-attr]
            reason = chunk.get("normalization")
            if reason:
                reasons[str(reason)] += 1
    chunks = prediction_validation["chunks"]
    interrupts = prediction_validation["interrupts"]
    return {
        **prediction_validation,
        "predicted_interrupt_rate": interrupts / chunks if chunks else 0.0,
        "normalization_counts": dict(sorted(reasons.items())),
        "predictions_sha256": sha256_file(predictions_path),
    }


def _readme_text(
    effective: dict[str, object],
    status: str,
    diagnostics: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
    runtime: dict[str, object] | None = None,
) -> str:
    model = effective["model"]  # type: ignore[assignment]
    inference = effective["inference"]  # type: ignore[assignment]
    validation = effective["validation_policy"]  # type: ignore[assignment]
    lines = [
        f"# {effective['experiment_id']}",
        "",
        f"Status: **{status}**",
        "",
        "## Hypothesis",
        "",
        str(effective["hypothesis"]),
        "",
        "## Frozen System",
        "",
        f"- Model: `{model['repo_id']}` at `{model['revision']}`",
        f"- Total parameters: `{model['total_parameters']}` (all inference-time learned parameters)",
        f"- License: `{model['license']}`",
        f"- Plan state: `{inference['plan_state']}`",
        f"- Frame policy: `{inference['frames_per_interval']}` per interval, `{inference['max_frames']}` cumulative cap",
        f"- Dialog history cap: `{inference['max_history_turns']}` turns",
        f"- Decoding: greedy, max `{inference['max_new_tokens']}` new tokens",
        "",
        "## Validation Policy",
        "",
        f"- Gold `answers` accessed by generation logic: `{validation['labels_read_during_generation']}`",
        "- The public source container includes labels, but only video path, intervals, query, and dialog enter generation",
        f"- Labels used for training or tuning: `{validation['labels_used_for_prompt_or_threshold_tuning']}`",
        f"- Classification: {validation['classification']}",
    ]
    if diagnostics:
        lines.extend(
            [
                "",
                "## Output Validation",
                "",
                f"- Sessions: `{diagnostics['sessions']}`",
                f"- Chunks: `{diagnostics['chunks']}`",
                f"- Predicted interrupt rate: `{diagnostics['predicted_interrupt_rate']:.6f}`",
                f"- Predictions SHA256: `{diagnostics['predictions_sha256']}`",
            ]
        )
    if metrics:
        overall = metrics["overall"]  # type: ignore[index]
        lines.extend(
            [
                "",
                "## Official Metrics",
                "",
                f"- Macro F1: `{overall['macro_f1']}`",
                f"- G-mean F1: `{overall['gmean_f1']}`",
                f"- Interrupt P/R/F1: `{overall['interrupt_precision']}` / `{overall['interrupt_recall']}` / `{overall['interrupt_f1']}`",
                f"- Silent P/R/F1: `{overall['silent_precision']}` / `{overall['silent_recall']}` / `{overall['silent_f1']}`",
                f"- TP/FP/TN/FN: `{overall['tp']}` / `{overall['fp']}` / `{overall['tn']}` / `{overall['fn']}`",
            ]
        )
    if runtime:
        lines.extend(
            [
                "",
                "## Runtime",
                "",
                f"- Wall time seconds: `{runtime['wall_time_seconds']}`",
                f"- Peak allocated GPU bytes: `{runtime['peak_gpu_memory_bytes']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "This run is the no-plan reference point. Plan-state claims require a controlled comparison against this exact backbone, context, and decoding policy.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_completed_report(
    effective: dict[str, object],
    output_dir: Path,
    diagnostics: dict[str, object],
    metrics: dict[str, object],
    runtime: dict[str, object],
) -> None:
    report_path = PROJECT_ROOT / "reports" / f"{effective['experiment_id']}.md"
    artifact_rel = output_dir.relative_to(PROJECT_ROOT)
    report = _readme_text(
        effective, "complete full public-validation R0", diagnostics, metrics, runtime
    )
    report += f"\nArtifact directory: [`{artifact_rel}`](../{artifact_rel})\n"
    report_path.write_text(report, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--video-folder", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_sessions is not None and args.max_sessions <= 0:
        parser.error("--max-sessions must be positive")
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= index < num_shards")
    if args.num_shards > 1 and args.max_sessions is not None:
        parser.error("--max-sessions cannot be combined with --num-shards > 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    config_path = _project_path(args.config).resolve()
    config = _load_config(config_path)
    model_config: dict[str, object] = config["model"]  # type: ignore[assignment]
    data_config: dict[str, object] = config["data"]  # type: ignore[assignment]
    starter_config: dict[str, object] = config["starter_kit"]  # type: ignore[assignment]
    inference_config: dict[str, object] = config["inference"]  # type: ignore[assignment]

    model_path = Path(
        args.model_path or str(model_config["default_local_path"])
    ).expanduser().resolve()
    input_path = _project_path(args.input or str(data_config["input"])).resolve()
    video_folder = _project_path(
        args.video_folder or str(data_config["video_folder"])
    ).resolve()
    starter_dir = _project_path(str(starter_config["path"])).resolve()
    output_dir = _project_path(
        args.output_dir
        or f"output/experiments/{config['experiment_id']}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "session_records.jsonl"
    if records_path.exists() and not args.resume:
        raise FileExistsError(
            f"Existing session records require --resume: {records_path}"
        )
    _configure_logging(output_dir, append=args.resume)
    started_at = time.monotonic()
    effective = _effective_config(
        config,
        config_path,
        model_path,
        input_path,
        video_folder,
        starter_dir,
        output_dir,
        args,
    )

    existing_config = output_dir / "config.json"
    if args.resume and existing_config.exists():
        if json.loads(existing_config.read_text(encoding="utf-8")) != effective:
            raise ValueError("Effective config differs from the run being resumed")
    else:
        write_json(existing_config, effective)
        _write_command(output_dir / "command.sh", raw_argv)
        _write_text_json(output_dir / "environment.txt", environment_snapshot())
        _write_text_json(
            output_dir / "code_state.txt",
            code_snapshot(PROJECT_ROOT, _tracked_code_paths(config_path)),
        )

    fingerprints = _validate_static_files(config, input_path, starter_dir)
    model_audit = verify_model_snapshot(model_path, model_config)
    all_rows = load_jsonl(input_path)
    selection_start, selection_stop = contiguous_shard_bounds(
        len(all_rows), args.num_shards, args.shard_index
    )
    selected_rows = all_rows[selection_start:selection_stop]
    if args.max_sessions is not None:
        selected_rows = selected_rows[: args.max_sessions]
        selection_stop = selection_start + len(selected_rows)
    source_validation = validate_source_rows(selected_rows, video_folder)
    full_public_run = (
        args.max_sessions is None
        and args.num_shards == 1
        and source_validation["sessions"] == 700
        and source_validation["chunks"] == 9935
    )
    dataset_license_path = PROJECT_ROOT / "data" / "LICENSE"
    data_manifest = {
        "source": {
            "repo_id": data_config["dataset_repo"],
            "split": data_config["split"],
            "input_path": str(input_path),
            "input_bytes": input_path.stat().st_size,
            "input_sha256": fingerprints["input_sha256"],
            "sessions_selected": source_validation["sessions"],
            "chunks_selected": source_validation["chunks"],
            "full_public_validation": full_public_run,
            "selection_start": selection_start,
            "selection_stop": selection_stop,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "top_level_license": data_config["license"],
            "license_file_sha256": sha256_file(dataset_license_path)
            if dataset_license_path.is_file()
            else None,
        },
        "model": {
            "repo_id": model_config["repo_id"],
            "revision": model_config["revision"],
            "license": model_config["license"],
            "local_path": str(model_path),
            **model_audit,
        },
        "starter_kit_sha256": fingerprints,
        "supervision": {
            "generation_reads_gold_answers": False,
            "training_or_tuning_uses_public_labels": False,
            "official_scoring_reads_labels_after_predictions_are_frozen": not args.skip_eval,
        },
    }
    write_json(output_dir / "data_manifest.json", data_manifest)
    (output_dir / "README.md").write_text(
        _readme_text(effective, "audited; generation pending"), encoding="utf-8"
    )
    LOGGER.info(
        "Static audit passed: %d parameters, %d sessions, %d chunks",
        model_audit["stored_unique_parameters"],
        source_validation["sessions"],
        source_validation["chunks"],
    )

    if args.audit_only:
        runtime = {
            "status": "audit_only",
            "wall_time_seconds": round(time.monotonic() - started_at, 3),
            "peak_gpu_memory_bytes": None,
        }
        write_json(output_dir / "runtime.json", runtime)
        LOGGER.info("Audit-only run complete")
        return

    records = _load_records(records_path)
    _validate_record_prefix(records, selected_rows, selection_start)
    start_index = len(records)
    starter = load_starter_kit(starter_dir)
    causal_config = CausalInferenceConfig(
        frames_per_interval=int(inference_config["frames_per_interval"]),
        max_frames=int(inference_config["max_frames"]),
        max_history_turns=int(inference_config["max_history_turns"]),
        max_new_tokens=int(inference_config["max_new_tokens"]),
    )

    model: InternVLProactiveModel | None = None
    if start_index < len(selected_rows):
        model = InternVLProactiveModel(
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
            raise ValueError(
                f"Loaded model has {model.parameter_count} parameters; expected "
                f"{model_config['total_parameters']}"
            )
        LOGGER.info("Loaded model on %s", args.device)

        with records_path.open("a", encoding="utf-8") as record_file:
            for row_index in range(start_index, len(selected_rows)):
                session_started = time.monotonic()
                result = process_session(
                    row=selected_rows[row_index],
                    input_index=selection_start + row_index,
                    video_folder=video_folder,
                    model=model,
                    starter=starter,
                    config=causal_config,
                )
                record_file.write(json.dumps(result, ensure_ascii=True) + "\n")
                record_file.flush()
                os.fsync(record_file.fileno())
                records.append(result)
                LOGGER.info(
                    "Session %d/%d complete: %s chunks=%d elapsed=%.2fs",
                    row_index + 1,
                    len(selected_rows),
                    result["video_path"],
                    len(result["chunks"]),  # type: ignore[arg-type]
                    time.monotonic() - session_started,
                )

    _validate_record_prefix(records, selected_rows, selection_start)
    predictions = [record["prediction"] for record in records]
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)  # type: ignore[arg-type]
    prediction_validation = validate_prediction_rows(selected_rows, predictions)  # type: ignore[arg-type]
    diagnostics = _diagnostics(records, prediction_validation, predictions_path)
    write_json(output_dir / "diagnostics.json", diagnostics)

    metrics: dict[str, object] | None = None
    if not args.skip_eval:
        golden_path = input_path
        if not full_public_run:
            golden_path = output_dir / "evaluation_golden_subset.jsonl"
            write_jsonl(golden_path, selected_rows)
        metrics_path = output_dir / "metrics.json"
        _run_official_scorer(
            starter_dir,
            golden_path,
            predictions_path,
            metrics_path,
            output_dir / "scorer.log",
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    runtime = {
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started_at, 3),
        "peak_gpu_memory_bytes": model.peak_memory_bytes() if model else None,
        "preexisting_gpu_processes": model.preexisting_gpu_processes if model else [],
        "selection_start": selection_start,
        "selection_stop": selection_stop,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "sessions": prediction_validation["sessions"],
        "chunks": prediction_validation["chunks"],
    }
    write_json(output_dir / "runtime.json", runtime)
    status = "complete full public-validation R0" if full_public_run else "complete partial smoke run"
    (output_dir / "README.md").write_text(
        _readme_text(effective, status, diagnostics, metrics, runtime),
        encoding="utf-8",
    )
    if full_public_run and metrics is not None:
        _write_completed_report(
            effective, output_dir, diagnostics, metrics, runtime
        )
    LOGGER.info(
        "Run complete: sessions=%d chunks=%d predictions_sha256=%s",
        prediction_validation["sessions"],
        prediction_validation["chunks"],
        diagnostics["predictions_sha256"],
    )


if __name__ == "__main__":
    main()
