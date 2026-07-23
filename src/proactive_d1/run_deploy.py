"""Run a serialized D1/D3/D4 decision head in the causal online R0 loop."""

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

from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    verify_model_snapshot,
    write_json,
)
from proactive_r0.core import (
    CausalInferenceConfig,
    load_jsonl,
    load_starter_kit,
    validate_prediction_rows,
    validate_source_rows,
    write_jsonl,
)
from proactive_r0.internvl import InternVLProactiveModel
from proactive_r0.run import _run_official_scorer, _validate_static_files

from .core import load_decision_head, strip_answers
from .deploy import process_session_with_fused_head, process_session_with_scalar_head
from .internvl_features import (
    DECISION_FEATURE_MODES,
    InternVLDecisionFeatureExtractor,
)
from proactive_d3.deploy import process_session_with_dynamics_head
from proactive_d4.deploy import process_session_with_dialog_stage_head

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "d1_internvl35_1b_scalar_deploy.json"
LOGGER = logging.getLogger("proactive_d1.run_deploy")


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
    command = shlex.join([sys.executable, "-m", "proactive_d1.run_deploy", *argv])
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
    paths = [
        *sorted((PROJECT_ROOT / "src" / "proactive_r0").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d1").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d1" / "tests").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d3").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d3" / "tests").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d4").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d4" / "tests").glob("*.py")),
        PROJECT_ROOT / "configs" / "d1_internvl35_1b_scalar_final.json",
        PROJECT_ROOT / "configs" / "d1_internvl35_1b_neural_final.json",
        PROJECT_ROOT / "models" / "internvl35_1b_hf.json",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
        PROJECT_ROOT / "C1_SPEC.md",
        PROJECT_ROOT / "Agent.md",
    ]
    # Submission runtime configs live under /tmp and are fingerprinted separately.
    # code_snapshot only accepts files whose paths are relative to PROJECT_ROOT.
    if config_path.resolve().is_relative_to(PROJECT_ROOT.resolve()):
        paths.append(config_path)
    return paths


def _validate_records(
    records: list[dict[str, object]],
    rows: list[dict[str, object]],
    input_indices: list[int],
) -> None:
    if len(records) > len(rows) or len(rows) != len(input_indices):
        raise ValueError("D1 deployment records exceed selected sessions")
    for position, (record, row, input_index) in enumerate(
        zip(records, rows, input_indices)
    ):
        if record.get("input_index") != input_index or record.get(
            "video_path"
        ) != row.get("video_path"):
            raise ValueError(f"D1 deployment resume identity mismatch at {position}")
        prediction = record.get("prediction")
        if not isinstance(prediction, dict):
            raise ValueError(f"D1 deployment record {index} has no prediction")
        validate_prediction_rows([row], [prediction])


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument(
        "--session-indices",
        help="Comma-separated original zero-based session indices",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    parser.add_argument("--record-hidden-state", action="store_true")
    args = parser.parse_args(raw_argv)
    if args.max_sessions is not None and args.max_sessions <= 0:
        parser.error("--max-sessions must be positive")
    if args.max_sessions is not None and args.session_indices:
        parser.error("--max-sessions and --session-indices are mutually exclusive")
    started_at = time.monotonic()

    config_path = _resolve(args.config)
    config = _load_json(config_path)
    model_config = dict(config["model"])  # type: ignore[arg-type]
    head_config = dict(config["decision_head"])  # type: ignore[arg-type]
    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    inference_config = dict(config["inference"])  # type: ignore[arg-type]
    model_path = _resolve(args.model_path or model_config["default_local_path"])
    head_path = _resolve(head_config["path"])
    input_path = _resolve(data_config["input"])
    video_folder = _resolve(data_config["video_folder"])
    starter_dir = _resolve(starter_config["path"])
    output_dir = _resolve(args.output_dir or f"output/experiments/{config['experiment_id']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "session_records.jsonl"
    if records_path.exists() and not args.resume:
        raise FileExistsError(f"Existing D1 deployment records require --resume: {records_path}")
    _configure_logging(output_dir, append=args.resume)

    fingerprints = _validate_static_files(config, input_path, starter_dir)
    model_audit = verify_model_snapshot(model_path, model_config)
    if sha256_file(head_path) != head_config["sha256"]:
        raise ValueError("Serialized D1 deployment head fingerprint mismatch")
    head = load_decision_head(_load_json(head_path))
    if len(head.feature_names) + 1 != int(head_config["parameters"]):
        raise ValueError("Serialized D1 deployment head parameter count mismatch")
    feature_variant = str(head_config["feature_variant"])
    neural_variants = ("fused_linear", "dynamics_fused", "dialog_stage_fused")
    if feature_variant not in ("response_temporal", *neural_variants):
        raise ValueError(f"Unsupported D1 deployment feature variant: {feature_variant}")
    if feature_variant in neural_variants:
        hidden_names = [
            name
            for name in head.feature_names
            if name.startswith("hidden_")
            and len(name) == len("hidden_") + 4
            and name[len("hidden_") :].isdigit()
        ]
        if len(hidden_names) != int(head_config["hidden_size"]):
            raise ValueError("Serialized neural deployment head hidden width mismatch")
    elif any(name == "tag_margin" or name.startswith("hidden_") for name in head.feature_names):
        raise ValueError("Scalar D1 deployment head unexpectedly contains neural features")
    decision_feature_mode: str | None = None
    if feature_variant in neural_variants:
        decision_feature_mode = str(
            inference_config.get("decision_feature_mode", "sequential")
        )
        if decision_feature_mode not in DECISION_FEATURE_MODES:
            raise ValueError(
                f"Unsupported D1 decision feature mode: {decision_feature_mode}"
            )
    elif args.record_hidden_state:
        parser.error("--record-hidden-state is only valid for a neural feature head")
    frame_sampling = str(
        inference_config.get("frame_sampling", "uniform_cumulative_v1")
    )
    if (
        frame_sampling != "uniform_cumulative_v1"
        and feature_variant != "dialog_stage_fused"
    ):
        raise ValueError(
            "Non-uniform frame sampling is currently implemented only for "
            "dialog_stage_fused deployment"
        )
    all_source_rows = load_jsonl(input_path)
    if args.session_indices:
        try:
            input_indices = [
                int(value.strip())
                for value in args.session_indices.split(",")
                if value.strip()
            ]
        except ValueError as error:
            parser.error(f"Invalid --session-indices: {error}")
        if not input_indices or len(set(input_indices)) != len(input_indices):
            parser.error("--session-indices must be non-empty and unique")
        if input_indices != sorted(input_indices):
            parser.error("--session-indices must preserve ascending source order")
        if input_indices[0] < 0 or input_indices[-1] >= len(all_source_rows):
            parser.error("--session-indices contains an out-of-range index")
    else:
        selected_count = min(
            args.max_sessions or len(all_source_rows), len(all_source_rows)
        )
        input_indices = list(range(selected_count))
    source_rows = [all_source_rows[index] for index in input_indices]
    generation_rows = strip_answers(source_rows)
    source_validation = validate_source_rows(generation_rows, video_folder)
    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "model_path": str(model_path),
        "head_path": str(head_path),
        "input_path": str(input_path),
        "video_folder": str(video_folder),
        "output_dir": str(output_dir),
        "device": args.device,
        "max_sessions": args.max_sessions,
        "session_indices": input_indices if args.session_indices else None,
        "audit_only": args.audit_only,
        "skip_eval": args.skip_eval,
        "require_exclusive_gpu": args.require_exclusive_gpu,
        "decision_feature_mode": decision_feature_mode,
        "record_hidden_state": args.record_hidden_state,
    }
    existing_config = output_dir / "config.json"
    if args.resume and existing_config.exists():
        if _load_json(existing_config) != effective:
            raise ValueError("Effective D1 deployment config differs on resume")
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
                "input_indices": input_indices,
                "generation_rows_contain_answers": False,
            },
            "model": {"path": str(model_path), **model_audit},
            "decision_head": {
                "path": str(head_path),
                "sha256": sha256_file(head_path),
                "parameters": len(head.feature_names) + 1,
                "feature_variant": feature_variant,
                "decision_feature_mode": decision_feature_mode,
            },
            "starter_kit_sha256": fingerprints,
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
    _validate_records(records, generation_rows, input_indices)
    starter = load_starter_kit(starter_dir)
    causal_config = CausalInferenceConfig(
        frames_per_interval=int(inference_config["frames_per_interval"]),
        max_frames=int(inference_config["max_frames"]),
        max_history_turns=int(inference_config["max_history_turns"]),
        max_new_tokens=int(inference_config["max_new_tokens"]),
        frame_sampling=frame_sampling,
    )
    model: InternVLProactiveModel | None = None
    if len(records) < len(generation_rows):
        model_kwargs = {
            "model_path": str(model_path),
            "device": args.device,
            "dtype_name": str(model_config["dtype"]),
            "attention_implementation": str(model_config["attention_implementation"]),
            "seed": int(inference_config["seed"]),
            "require_exclusive_gpu": args.require_exclusive_gpu,
            "video_frame_size": int(inference_config["video_frame_size"]),
            "pad_token_id": int(inference_config["pad_token_id"]),
        }
        if feature_variant in neural_variants:
            model = InternVLDecisionFeatureExtractor(
                **model_kwargs,
                decision_feature_mode=decision_feature_mode,
            )
        else:
            model = InternVLProactiveModel(**model_kwargs)
        if model.parameter_count != int(model_config["total_parameters"]):
            raise ValueError("Loaded D1 deployment model parameter count mismatch")
        if feature_variant in neural_variants and (
            not isinstance(model, InternVLDecisionFeatureExtractor)
            or model.hidden_size != int(head_config["hidden_size"])
        ):
            raise ValueError("Loaded D1 deployment hidden width mismatch")
        with records_path.open("a", encoding="utf-8") as handle:
            for position in range(len(records), len(generation_rows)):
                session_started = time.monotonic()
                input_index = input_indices[position]
                if feature_variant in neural_variants:
                    if not isinstance(model, InternVLDecisionFeatureExtractor):
                        raise RuntimeError("Neural deployment model type changed")
                    if feature_variant == "dynamics_fused":
                        processor = process_session_with_dynamics_head
                    elif feature_variant == "dialog_stage_fused":
                        processor = process_session_with_dialog_stage_head
                    else:
                        processor = process_session_with_fused_head
                    record = processor(
                        generation_rows[position],
                        input_index,
                        video_folder,
                        model,
                        starter,
                        causal_config,
                        head,
                        record_hidden_state=args.record_hidden_state,
                    )
                else:
                    record = process_session_with_scalar_head(
                        generation_rows[position],
                        input_index,
                        video_folder,
                        model,
                        starter,
                        causal_config,
                        head,
                    )
                session_elapsed = time.monotonic() - session_started
                record["session_wall_time_seconds"] = session_elapsed
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                records.append(record)
                LOGGER.info(
                    "Session %d/%d complete: chunks=%d elapsed=%.2fs",
                    position + 1,
                    len(generation_rows),
                    len(record["chunks"]),  # type: ignore[arg-type]
                    session_elapsed,
                )
    _validate_records(records, generation_rows, input_indices)
    predictions = [record["prediction"] for record in records]
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)  # type: ignore[arg-type]
    validation = validate_prediction_rows(generation_rows, predictions)  # type: ignore[arg-type]
    write_json(
        output_dir / "diagnostics.json",
        {**validation, "predictions_sha256": sha256_file(predictions_path)},
    )
    metrics: dict[str, object] | None = None
    if not args.skip_eval:
        golden_path = input_path
        if len(source_rows) != 700:
            golden_path = output_dir / "evaluation_golden_subset.jsonl"
            write_jsonl(golden_path, source_rows)
        metrics_path = output_dir / "metrics.json"
        _run_official_scorer(
            starter_dir,
            golden_path,
            predictions_path,
            metrics_path,
            output_dir / "scorer.log",
        )
        metrics = _load_json(metrics_path)
    runtime = {
        "status": f"complete online {feature_variant} deployment run",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started_at, 3),
        "peak_gpu_memory_bytes": model.peak_memory_bytes() if model else None,
        "preexisting_gpu_processes": model.preexisting_gpu_processes if model else [],
        "sessions": validation["sessions"],
        "chunks": validation["chunks"],
        "total_parameters": int(model_config["total_parameters"]) + len(head.feature_names) + 1,
        "decision_feature_mode": decision_feature_mode,
        "hidden_state_recorded": args.record_hidden_state,
        "max_session_wall_time_seconds": max(
            (
                float(record["session_wall_time_seconds"])
                for record in records
                if "session_wall_time_seconds" in record
            ),
            default=None,
        ),
    }
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "Status: **complete online causal deployment run**",
                "",
                f"- Sessions: `{validation['sessions']}`",
                f"- Chunks: `{validation['chunks']}`",
                f"- Feature variant: `{feature_variant}`",
                f"- Decision feature mode: `{decision_feature_mode}`",
                f"- Predictions SHA256: `{sha256_file(predictions_path)}`",
                "- Official Macro F1: "
                f"`{metrics['overall']['macro_f1'] if metrics else 'not scored'}`",  # type: ignore[index]
                "- Result type: `val-supervised deployment verification`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"validation": validation, "metrics": metrics, "runtime": runtime},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
