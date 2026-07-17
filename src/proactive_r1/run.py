"""Run the controlled C1 Small R1 oracle compact-state pilot."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
import time
from collections import Counter
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

from .core import process_session_variants
from .state import (
    STATE_VARIANTS,
    annotation_paths_from_config,
    load_json,
    validate_and_select_manifest,
    validate_annotations,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "r1_internvl35_1b_oracle_state_pilot_v1.json"
LOGGER = logging.getLogger("proactive_r1")


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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
    config = load_json(path)
    required = {
        "experiment_id",
        "hypothesis",
        "model",
        "data",
        "starter_kit",
        "inference",
        "oracle_state",
        "r0_reference",
        "evaluation",
        "validation_policy",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Config is missing keys: {sorted(missing)}")
    return config


def _load_annotations(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("R1 annotations must be a JSON list of objects")
    return value


def _tracked_paths(config_path: Path, manifest_path: Path, annotation_path: Path) -> list[Path]:
    return [
        *sorted((PROJECT_ROOT / "src" / "proactive_r0").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_r1").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_r1" / "tests").glob("*.py")),
        config_path,
        manifest_path,
        annotation_path,
        manifest_path.parent / "PROTOCOL.md",
        PROJECT_ROOT / "models" / "internvl35_1b_hf.json",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
        PROJECT_ROOT / "C1_SPEC.md",
        PROJECT_ROOT / "Agent.md",
    ]


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_r1.run", *argv])
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


def _strip_answers(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key != "answers"}


def _records_by_index(path: Path) -> dict[int, dict[str, object]]:
    rows = load_jsonl(path)
    result: dict[int, dict[str, object]] = {}
    for row in rows:
        index = row.get("input_index")
        if not isinstance(index, int) or index in result:
            raise ValueError(f"Invalid or duplicate input_index in {path}: {index!r}")
        result[index] = row
    return result


def _validate_record_prefix(
    records: list[dict[str, object]],
    selected: list[tuple[int, dict[str, object]]],
) -> None:
    if len(records) > len(selected):
        raise ValueError("Resume records exceed selected R1 sessions")
    for position, record in enumerate(records):
        input_index, source = selected[position]
        if record.get("input_index") != input_index:
            raise ValueError(f"Resume record {position} has wrong input_index")
        if record.get("video_path") != source.get("video_path"):
            raise ValueError(f"Resume record {position} has wrong video_path")
        variants = record.get("variants")
        if not isinstance(variants, dict) or set(variants) != set(STATE_VARIANTS):
            raise ValueError(f"Resume record {position} has wrong variants")
        stripped_source = _strip_answers(source)
        for variant in STATE_VARIANTS:
            payload = variants[variant]
            if not isinstance(payload, dict):
                raise ValueError(f"Resume record {position} variant {variant} is invalid")
            prediction = payload.get("prediction")
            if not isinstance(prediction, dict):
                raise ValueError(f"Resume record {position} variant {variant} lacks prediction")
            validate_prediction_rows([stripped_source], [prediction])


def _variant_payloads(
    records: list[dict[str, object]], variant: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    predictions: list[dict[str, object]] = []
    chunks_by_session: list[dict[str, object]] = []
    for record in records:
        variants = record["variants"]
        assert isinstance(variants, dict)
        payload = variants[variant]
        assert isinstance(payload, dict)
        prediction = payload["prediction"]
        chunks = payload["chunks"]
        assert isinstance(prediction, dict) and isinstance(chunks, list)
        predictions.append(prediction)
        chunks_by_session.append(
            {
                "input_index": record["input_index"],
                "video_path": record["video_path"],
                "prediction": prediction,
                "chunks": chunks,
            }
        )
    return predictions, chunks_by_session


def _diagnostics(
    source_rows: list[dict[str, object]],
    predictions: list[dict[str, object]],
    records: list[dict[str, object]],
    predictions_path: Path,
) -> dict[str, object]:
    validation = validate_prediction_rows(source_rows, predictions)
    reasons: Counter[str] = Counter()
    for record in records:
        chunks = record.get("chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                if isinstance(chunk, dict) and chunk.get("normalization"):
                    reasons[str(chunk["normalization"])] += 1
    return {
        **validation,
        "predicted_interrupt_rate": validation["interrupts"] / validation["chunks"],
        "normalization_counts": dict(sorted(reasons.items())),
        "predictions_sha256": sha256_file(predictions_path),
    }


def _metric_summary(metrics: dict[str, object]) -> dict[str, object]:
    overall = metrics["overall"]
    assert isinstance(overall, dict)
    return {
        key: overall[key]
        for key in (
            "macro_f1",
            "gmean_f1",
            "interrupt_precision",
            "interrupt_recall",
            "interrupt_f1",
            "silent_precision",
            "silent_recall",
            "silent_f1",
            "tp",
            "fp",
            "tn",
            "fn",
            "support",
        )
    }


def _comparison(
    metrics_by_variant: dict[str, dict[str, object]],
    diagnostics_by_variant: dict[str, dict[str, object]],
) -> dict[str, object]:
    summaries = {
        variant: {
            **_metric_summary(metrics),
            "predicted_interrupt_rate": diagnostics_by_variant[variant][
                "predicted_interrupt_rate"
            ],
            "normalization_counts": diagnostics_by_variant[variant][
                "normalization_counts"
            ],
            "predictions_sha256": diagnostics_by_variant[variant][
                "predictions_sha256"
            ],
        }
        for variant, metrics in metrics_by_variant.items()
    }
    null = summaries["null"]
    r0 = summaries["r0_frozen"]
    for variant, summary in summaries.items():
        summary["delta_macro_f1_vs_null"] = round(
            float(summary["macro_f1"]) - float(null["macro_f1"]), 4
        )
        summary["delta_interrupt_recall_vs_null"] = round(
            float(summary["interrupt_recall"]) - float(null["interrupt_recall"]), 4
        )
        summary["delta_macro_f1_vs_r0_frozen"] = round(
            float(summary["macro_f1"]) - float(r0["macro_f1"]), 4
        )
    return {
        "interpretation": (
            "Protocol pilot only. Four label-independent sessions are insufficient "
            "for a population-level R1 claim. State effects are primarily compared "
            "against null; r0_frozen measures the state-wrapper confound."
        ),
        "variants": summaries,
    }


def _readme(
    config: dict[str, object],
    status: str,
    source_sessions: int,
    source_chunks: int,
    comparison: dict[str, object] | None = None,
) -> str:
    lines = [
        f"# {config['experiment_id']}",
        "",
        f"Status: **{status}**",
        "",
        str(config["hypothesis"]),
        "",
        "## Scope",
        "",
        f"- Pilot sessions/chunks: `{source_sessions}` / `{source_chunks}`",
        "- Frozen model and R0 inference policy; only the procedural-state block varies",
        "- Oracle annotations are causal, evaluation-only, non-deployable, and not a held-out result",
        "- The pilot is a protocol check, not a population estimate",
    ]
    if comparison:
        variants = comparison["variants"]
        assert isinstance(variants, dict)
        lines.extend(["", "## Official Subset Metrics", ""])
        for variant in ("r0_frozen", *STATE_VARIANTS):
            summary = variants[variant]
            assert isinstance(summary, dict)
            lines.append(
                f"- `{variant}`: Macro F1 `{summary['macro_f1']}`, "
                f"interrupt recall `{summary['interrupt_recall']}`, "
                f"silent precision `{summary['silent_precision']}`"
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "Any promising effect must be repeated on a larger pre-registered session subset before the R1 scientific gate can pass.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.max_sessions is not None and args.max_sessions <= 0:
        parser.error("--max-sessions must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    started_at = time.monotonic()
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    model_config = config["model"]
    data_config = config["data"]
    starter_config = config["starter_kit"]
    inference_config = config["inference"]
    r0_reference = config["r0_reference"]
    assert all(
        isinstance(value, dict)
        for value in (
            model_config,
            data_config,
            starter_config,
            inference_config,
            r0_reference,
        )
    )
    model_config = dict(model_config)  # type: ignore[arg-type]
    data_config = dict(data_config)  # type: ignore[arg-type]
    starter_config = dict(starter_config)  # type: ignore[arg-type]
    inference_config = dict(inference_config)  # type: ignore[arg-type]
    r0_reference = dict(r0_reference)  # type: ignore[arg-type]

    input_path = _resolve(data_config["input"])
    video_folder = _resolve(data_config["video_folder"])
    starter_dir = _resolve(starter_config["path"])
    model_path = _resolve(args.model_path or model_config["default_local_path"])
    manifest_path, annotation_path = annotation_paths_from_config(PROJECT_ROOT, config)
    output_dir = _resolve(
        args.output_dir or f"output/experiments/{config['experiment_id']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "session_records.jsonl"
    if records_path.exists() and not args.resume:
        raise FileExistsError(f"Existing R1 records require --resume: {records_path}")
    _configure_logging(output_dir, append=args.resume)

    fingerprints = _validate_static_files(config, input_path, starter_dir)
    model_audit = verify_model_snapshot(model_path, model_config)
    source_rows = load_jsonl(input_path)
    manifest = load_json(manifest_path)
    selected = validate_and_select_manifest(manifest, source_rows)
    annotations = _load_annotations(annotation_path)
    annotations_by_index = validate_annotations(annotations, selected)
    if args.max_sessions is not None:
        selected = selected[: args.max_sessions]
    generation_rows = [_strip_answers(row) for _, row in selected]
    source_validation = validate_source_rows(generation_rows, video_folder)
    if list(inference_config["state_variants"]) != list(STATE_VARIANTS):
        raise ValueError("Config state_variants must match the frozen R1 order")

    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "device": args.device,
        "max_sessions": args.max_sessions,
        "audit_only": args.audit_only,
        "require_exclusive_gpu": args.require_exclusive_gpu,
    }
    existing_config = output_dir / "config.json"
    if args.resume and existing_config.exists():
        if json.loads(existing_config.read_text(encoding="utf-8")) != effective:
            raise ValueError("Effective config differs from the R1 run being resumed")
    else:
        write_json(existing_config, effective)
        _write_command(output_dir / "command.sh", raw_argv)
        write_json(output_dir / "environment.txt", environment_snapshot())
        write_json(
            output_dir / "code_state.txt",
            code_snapshot(
                PROJECT_ROOT,
                _tracked_paths(config_path, manifest_path, annotation_path),
            ),
        )

    r0_dir = _resolve(r0_reference["experiment_dir"])
    r0_predictions_path = r0_dir / "predictions.jsonl"
    actual_r0_sha = sha256_file(r0_predictions_path)
    if actual_r0_sha != r0_reference["predictions_sha256"]:
        raise ValueError("Frozen R0 prediction SHA256 mismatch")
    protocol_path = manifest_path.parent / "PROTOCOL.md"
    data_manifest = {
        "source": {
            "repo_id": data_config["dataset_repo"],
            "split": data_config["split"],
            "input_path": str(input_path),
            "input_sha256": fingerprints["input_sha256"],
            "sessions_selected": source_validation["sessions"],
            "chunks_selected": source_validation["chunks"],
            "selection_is_label_independent": True,
            "top_level_license": data_config["license"],
        },
        "oracle_state": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "annotations_path": str(annotation_path),
            "annotations_sha256": sha256_file(annotation_path),
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "annotation_type": "evaluation_only_oracle_non_deployable",
        },
        "model": {
            "repo_id": model_config["repo_id"],
            "revision": model_config["revision"],
            "license": model_config["license"],
            "local_path": str(model_path),
            **model_audit,
        },
        "starter_kit_sha256": fingerprints,
        "r0_reference": {
            "path": str(r0_dir),
            "predictions_sha256": actual_r0_sha,
        },
        "supervision": {
            "generation_rows_contain_answers": False,
            "oracle_plan_inputs": ["task", "query"],
            "oracle_dynamic_inputs": [
                "task",
                "query",
                "dialog_at_chunk",
                "video_through_interval_end",
            ],
            "oracle_excluded_inputs": ["answers", "future_dialog", "future_video"],
            "training_or_parameter_updates": False,
            "official_scoring_reads_labels_after_predictions_are_frozen": True,
        },
    }
    write_json(output_dir / "data_manifest.json", data_manifest)
    (output_dir / "README.md").write_text(
        _readme(
            config,
            "audited; generation pending" if not args.audit_only else "audit only",
            int(source_validation["sessions"]),
            int(source_validation["chunks"]),
        ),
        encoding="utf-8",
    )
    LOGGER.info(
        "R1 audit passed: %d parameters, %d sessions, %d chunks, annotations=%s",
        model_audit["stored_unique_parameters"],
        source_validation["sessions"],
        source_validation["chunks"],
        sha256_file(annotation_path),
    )
    if args.audit_only:
        write_json(
            output_dir / "runtime.json",
            {
                "status": "audit_only",
                "wall_time_seconds": round(time.monotonic() - started_at, 3),
                "peak_gpu_memory_bytes": None,
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
        max_new_tokens=int(inference_config["max_new_tokens"]),
    )
    model: InternVLProactiveModel | None = None
    if len(records) < len(selected):
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
            raise ValueError("Loaded parameter count differs from the frozen R1 config")
        with records_path.open("a", encoding="utf-8") as handle:
            for position in range(len(records), len(selected)):
                session_started = time.monotonic()
                input_index, source = selected[position]
                result = process_session_variants(
                    row=_strip_answers(source),
                    input_index=input_index,
                    annotation=annotations_by_index[input_index],
                    video_folder=video_folder,
                    model=model,
                    starter=starter,
                    config=causal_config,
                )
                handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                records.append(result)
                LOGGER.info(
                    "Session %d/%d complete: index=%d chunks=%d variants=%d elapsed=%.2fs",
                    position + 1,
                    len(selected),
                    input_index,
                    len(source["video_intervals"]),  # type: ignore[arg-type]
                    len(STATE_VARIANTS),
                    time.monotonic() - session_started,
                )
    _validate_record_prefix(records, selected)

    # Gold rows are materialized only after every generated response is durable.
    golden_rows = [row for _, row in selected]
    golden_path = output_dir / "evaluation_golden_subset.jsonl"
    write_jsonl(golden_path, golden_rows)
    r0_records_by_index = _records_by_index(r0_dir / "session_records.jsonl")
    r0_records: list[dict[str, object]] = []
    for input_index, source in selected:
        record = r0_records_by_index[input_index]
        r0_records.append(record)
        prediction = record.get("prediction")
        if not isinstance(prediction, dict):
            raise ValueError(f"Frozen R0 record {input_index} lacks prediction")
        validate_prediction_rows([_strip_answers(source)], [prediction])

    metrics_by_variant: dict[str, dict[str, object]] = {}
    diagnostics_by_variant: dict[str, dict[str, object]] = {}
    all_variant_records: dict[str, list[dict[str, object]]] = {}
    for variant in STATE_VARIANTS:
        predictions, variant_records = _variant_payloads(records, variant)
        all_variant_records[variant] = variant_records
        variant_dir = output_dir / "variants" / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = variant_dir / "predictions.jsonl"
        write_jsonl(predictions_path, predictions)
        write_jsonl(variant_dir / "session_records.jsonl", variant_records)
        diagnostics = _diagnostics(
            generation_rows, predictions, variant_records, predictions_path
        )
        write_json(variant_dir / "diagnostics.json", diagnostics)
        metrics_path = variant_dir / "metrics.json"
        _run_official_scorer(
            starter_dir,
            golden_path,
            predictions_path,
            metrics_path,
            variant_dir / "scorer.log",
        )
        metrics = load_json(metrics_path)
        metrics_by_variant[variant] = metrics
        diagnostics_by_variant[variant] = diagnostics

    r0_predictions = [record["prediction"] for record in r0_records]
    r0_variant_dir = output_dir / "variants" / "r0_frozen"
    r0_variant_dir.mkdir(parents=True, exist_ok=True)
    r0_subset_path = r0_variant_dir / "predictions.jsonl"
    write_jsonl(r0_subset_path, r0_predictions)  # type: ignore[arg-type]
    write_jsonl(r0_variant_dir / "session_records.jsonl", r0_records)
    r0_diagnostics = _diagnostics(
        generation_rows,
        r0_predictions,  # type: ignore[arg-type]
        r0_records,
        r0_subset_path,
    )
    write_json(r0_variant_dir / "diagnostics.json", r0_diagnostics)
    r0_metrics_path = r0_variant_dir / "metrics.json"
    _run_official_scorer(
        starter_dir,
        golden_path,
        r0_subset_path,
        r0_metrics_path,
        r0_variant_dir / "scorer.log",
    )
    metrics_by_variant["r0_frozen"] = load_json(r0_metrics_path)
    diagnostics_by_variant["r0_frozen"] = r0_diagnostics

    comparison = _comparison(metrics_by_variant, diagnostics_by_variant)
    write_json(output_dir / "comparison.json", comparison)
    runtime = {
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started_at, 3),
        "peak_gpu_memory_bytes": model.peak_memory_bytes() if model else None,
        "preexisting_gpu_processes": model.preexisting_gpu_processes if model else [],
        "sessions": source_validation["sessions"],
        "chunks": source_validation["chunks"],
        "generated_variants": list(STATE_VARIANTS),
        "reference_variants": ["r0_frozen"],
    }
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "README.md").write_text(
        _readme(
            config,
            "complete R1 protocol pilot",
            int(source_validation["sessions"]),
            int(source_validation["chunks"]),
            comparison,
        ),
        encoding="utf-8",
    )
    LOGGER.info("R1 pilot complete: %s", json.dumps(comparison["variants"]))


if __name__ == "__main__":
    main()
