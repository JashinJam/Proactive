"""Run the R1 tag-grammar factorial on the frozen four-session pilot."""

from __future__ import annotations

import argparse
import json
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
from proactive_r0.run import _run_official_scorer, _validate_static_files

from .constrained import (
    ConstrainedInternVLProactiveModel,
    SequenceChoiceConstrainedInternVLProactiveModel,
)
from .format_core import FORMAT_VARIANTS, process_format_session
from .run import (
    _configure_logging,
    _diagnostics,
    _load_annotations,
    _load_config,
    _metric_summary,
    _records_by_index,
    _resolve,
    _strip_answers,
    _tracked_paths,
)
from .state import annotation_paths_from_config, load_json, validate_and_select_manifest, validate_annotations

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "r1f_internvl35_1b_tag_grammar_factorial_pilot_v1.json"


def _validate_format_config(model_path: Path, config: dict[str, object]) -> dict[str, object]:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer = processor.tokenizer
    actual = {
        "silent_text": "$silent$",
        "silent_token_ids": tokenizer.encode("$silent$", add_special_tokens=False),
        "interrupt_text": "$interrupt$",
        "interrupt_token_ids": tokenizer.encode("$interrupt$", add_special_tokens=False),
        "eos_token_id": tokenizer.eos_token_id,
    }
    expected = {key: config[key] for key in actual}
    if actual != expected:
        raise ValueError(f"Tag grammar tokenizer audit mismatch: {actual} != {expected}")
    if tokenizer.decode(actual["silent_token_ids"], skip_special_tokens=False) != "$silent$":
        raise ValueError("Silent tag tokenizer round-trip failed")
    if tokenizer.decode(actual["interrupt_token_ids"], skip_special_tokens=False) != "$interrupt$":
        raise ValueError("Interrupt tag tokenizer round-trip failed")
    return actual


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_r1.run_format", *argv])
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


def _validate_records(
    records: list[dict[str, object]], selected: list[tuple[int, dict[str, object]]]
) -> None:
    if len(records) > len(selected):
        raise ValueError("Format-factorial records exceed selected sessions")
    for position, record in enumerate(records):
        input_index, source = selected[position]
        if record.get("input_index") != input_index or record.get("video_path") != source.get("video_path"):
            raise ValueError(f"Format-factorial record {position} identity mismatch")
        variants = record.get("variants")
        if not isinstance(variants, dict) or set(variants) != set(FORMAT_VARIANTS):
            raise ValueError(f"Format-factorial record {position} variant mismatch")
        for variant in FORMAT_VARIANTS:
            payload = variants[variant]
            if not isinstance(payload, dict) or not isinstance(payload.get("prediction"), dict):
                raise ValueError(f"Record {position} variant {variant} is invalid")
            validate_prediction_rows([_strip_answers(source)], [payload["prediction"]])


def _variant_payloads(
    records: list[dict[str, object]], variant: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    predictions: list[dict[str, object]] = []
    variant_records: list[dict[str, object]] = []
    for record in records:
        variants = record["variants"]
        assert isinstance(variants, dict)
        payload = variants[variant]
        assert isinstance(payload, dict)
        prediction = payload["prediction"]
        chunks = payload["chunks"]
        assert isinstance(prediction, dict) and isinstance(chunks, list)
        predictions.append(prediction)
        variant_records.append(
            {
                "input_index": record["input_index"],
                "video_path": record["video_path"],
                "prediction": prediction,
                "chunks": chunks,
            }
        )
    return predictions, variant_records


def _comparison(
    metrics: dict[str, dict[str, object]], diagnostics: dict[str, dict[str, object]]
) -> dict[str, object]:
    summaries = {
        variant: {
            **_metric_summary(metric),
            "predicted_interrupt_rate": diagnostics[variant]["predicted_interrupt_rate"],
            "normalization_counts": diagnostics[variant]["normalization_counts"],
            "predictions_sha256": diagnostics[variant]["predictions_sha256"],
        }
        for variant, metric in metrics.items()
    }
    for summary in summaries.values():
        summary["delta_macro_f1_vs_r0_format"] = round(
            float(summary["macro_f1"]) - float(summaries["r0_format"]["macro_f1"]), 4
        )
        summary["delta_macro_f1_vs_null"] = round(
            float(summary["macro_f1"]) - float(summaries["null"]["macro_f1"]), 4
        )
        summary["delta_macro_f1_vs_r0_frozen"] = round(
            float(summary["macro_f1"]) - float(summaries["r0_frozen"]["macro_f1"]), 4
        )
    return {
        "interpretation": (
            "Four-session format-controlled protocol pilot. Grammar is identical "
            "across r0_format/null/step/cues/full and does not use labels."
        ),
        "variants": summaries,
    }


def _readme(config: dict[str, object], status: str, comparison: dict[str, object] | None = None) -> str:
    lines = [
        f"# {config['experiment_id']}",
        "",
        f"Status: **{status}**",
        "",
        str(config["hypothesis"]),
        "",
        "The required-tag FSM is label-independent and identical for all generated variants.",
        "This four-session result is a protocol diagnostic, not a population estimate or leaderboard score.",
    ]
    if comparison:
        lines.extend(["", "## Official Subset Metrics", ""])
        variants = comparison["variants"]
        assert isinstance(variants, dict)
        for variant in ("r0_frozen", *FORMAT_VARIANTS):
            value = variants[variant]
            assert isinstance(value, dict)
            lines.append(
                f"- `{variant}`: Macro F1 `{value['macro_f1']}`, "
                f"interrupt recall `{value['interrupt_recall']}`, "
                f"silent recall `{value['silent_recall']}`"
            )
    return "\n".join(lines) + "\n"


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
    model_config = dict(config["model"])  # type: ignore[arg-type]
    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    inference_config = dict(config["inference"])  # type: ignore[arg-type]
    format_config = dict(config["format_constraint"])  # type: ignore[arg-type]
    r0_reference = dict(config["r0_reference"])  # type: ignore[arg-type]
    input_path = _resolve(data_config["input"])
    video_folder = _resolve(data_config["video_folder"])
    starter_dir = _resolve(starter_config["path"])
    model_path = _resolve(args.model_path or model_config["default_local_path"])
    manifest_path, annotation_path = annotation_paths_from_config(PROJECT_ROOT, config)
    output_dir = _resolve(args.output_dir or f"output/experiments/{config['experiment_id']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "session_records.jsonl"
    if records_path.exists() and not args.resume:
        raise FileExistsError(f"Existing format records require --resume: {records_path}")
    _configure_logging(output_dir, append=args.resume)

    fingerprints = _validate_static_files(config, input_path, starter_dir)
    model_audit = verify_model_snapshot(model_path, model_config)
    tokenizer_audit = _validate_format_config(model_path, format_config)
    source_rows = load_jsonl(input_path)
    selected = validate_and_select_manifest(load_json(manifest_path), source_rows)
    annotations = _load_annotations(annotation_path)
    annotations_by_index = validate_annotations(annotations, selected)
    if args.max_sessions is not None:
        selected = selected[: args.max_sessions]
    generation_rows = [_strip_answers(row) for _, row in selected]
    source_validation = validate_source_rows(generation_rows, video_folder)
    if tuple(inference_config["format_variants"]) != FORMAT_VARIANTS:
        raise ValueError("Config format_variants do not match the frozen factorial")

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
            raise ValueError("Effective config differs from the run being resumed")
    else:
        write_json(existing_config, effective)
        _write_command(output_dir / "command.sh", raw_argv)
        write_json(output_dir / "environment.txt", environment_snapshot())
        write_json(
            output_dir / "code_state.txt",
            code_snapshot(PROJECT_ROOT, _tracked_paths(config_path, manifest_path, annotation_path)),
        )

    r0_dir = _resolve(r0_reference["experiment_dir"])
    r0_predictions_path = r0_dir / "predictions.jsonl"
    if sha256_file(r0_predictions_path) != r0_reference["predictions_sha256"]:
        raise ValueError("Frozen R0 prediction SHA256 mismatch")
    data_manifest = {
        "source": {
            "input_path": str(input_path),
            "input_sha256": fingerprints["input_sha256"],
            "sessions_selected": source_validation["sessions"],
            "chunks_selected": source_validation["chunks"],
            "selection_is_label_independent": True,
        },
        "model": {
            "repo_id": model_config["repo_id"],
            "revision": model_config["revision"],
            "license": model_config["license"],
            **model_audit,
        },
        "format_constraint": {
            **format_config,
            "tokenizer_audit": tokenizer_audit,
            "tokenizer_revision": model_config["revision"],
            "learned_parameters_added": 0,
        },
        "oracle_state": {
            "manifest_sha256": sha256_file(manifest_path),
            "annotations_sha256": sha256_file(annotation_path),
        },
        "starter_kit_sha256": fingerprints,
        "supervision": {
            "generation_rows_contain_answers": False,
            "grammar_uses_labels_or_thresholds": False,
            "training_or_parameter_updates": False,
            "official_scoring_reads_labels_after_predictions_are_frozen": True,
        },
    }
    write_json(output_dir / "data_manifest.json", data_manifest)
    (output_dir / "README.md").write_text(_readme(config, "audited; generation pending"), encoding="utf-8")
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
    _validate_records(records, selected)
    starter = load_starter_kit(starter_dir)
    causal_config = CausalInferenceConfig(
        frames_per_interval=int(inference_config["frames_per_interval"]),
        max_frames=int(inference_config["max_frames"]),
        max_history_turns=int(inference_config["max_history_turns"]),
        max_new_tokens=int(inference_config["max_new_tokens"]),
    )
    model: ConstrainedInternVLProactiveModel | None = None
    if len(records) < len(selected):
        constraint_type = str(format_config["type"])
        if constraint_type == "required_tag_prefix_fsm_v1":
            model_class = ConstrainedInternVLProactiveModel
        elif constraint_type == "required_tag_sequence_beam_v2":
            model_class = SequenceChoiceConstrainedInternVLProactiveModel
        else:
            raise ValueError(f"Unsupported format constraint: {constraint_type}")
        model = model_class(
            model_path=str(model_path),
            device=args.device,
            dtype_name=str(model_config["dtype"]),
            attention_implementation=str(model_config["attention_implementation"]),
            seed=int(inference_config["seed"]),
            require_exclusive_gpu=args.require_exclusive_gpu,
            video_frame_size=int(inference_config["video_frame_size"]),
            pad_token_id=int(inference_config["pad_token_id"]),
        )
        if model.tag_token_ids != {
            "silent": format_config["silent_token_ids"],
            "interrupt": format_config["interrupt_token_ids"],
            "eos": format_config["eos_token_id"],
        }:
            raise ValueError("Runtime tokenizer IDs differ from the frozen grammar config")
        with records_path.open("a", encoding="utf-8") as handle:
            for position in range(len(records), len(selected)):
                session_started = time.monotonic()
                input_index, source = selected[position]
                result = process_format_session(
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
                print(
                    f"session {position + 1}/{len(selected)} index={input_index} "
                    f"elapsed={time.monotonic() - session_started:.2f}s",
                    flush=True,
                )
    _validate_records(records, selected)

    golden_rows = [row for _, row in selected]
    golden_path = output_dir / "evaluation_golden_subset.jsonl"
    write_jsonl(golden_path, golden_rows)
    metrics_by_variant: dict[str, dict[str, object]] = {}
    diagnostics_by_variant: dict[str, dict[str, object]] = {}
    for variant in FORMAT_VARIANTS:
        predictions, variant_records = _variant_payloads(records, variant)
        variant_dir = output_dir / "variants" / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = variant_dir / "predictions.jsonl"
        write_jsonl(predictions_path, predictions)
        write_jsonl(variant_dir / "session_records.jsonl", variant_records)
        diagnostics = _diagnostics(generation_rows, predictions, variant_records, predictions_path)
        if "malformed_response_scored_as_silent" in diagnostics["normalization_counts"]:
            raise RuntimeError(f"Tag grammar failed for variant {variant}")
        write_json(variant_dir / "diagnostics.json", diagnostics)
        metrics_path = variant_dir / "metrics.json"
        _run_official_scorer(
            starter_dir, golden_path, predictions_path, metrics_path, variant_dir / "scorer.log"
        )
        metrics_by_variant[variant] = load_json(metrics_path)
        diagnostics_by_variant[variant] = diagnostics

    frozen_records_by_index = _records_by_index(r0_dir / "session_records.jsonl")
    frozen_records = [frozen_records_by_index[input_index] for input_index, _ in selected]
    frozen_predictions = [record["prediction"] for record in frozen_records]
    frozen_dir = output_dir / "variants" / "r0_frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = frozen_dir / "predictions.jsonl"
    write_jsonl(frozen_path, frozen_predictions)  # type: ignore[arg-type]
    write_jsonl(frozen_dir / "session_records.jsonl", frozen_records)
    frozen_diagnostics = _diagnostics(
        generation_rows,
        frozen_predictions,  # type: ignore[arg-type]
        frozen_records,
        frozen_path,
    )
    write_json(frozen_dir / "diagnostics.json", frozen_diagnostics)
    frozen_metrics_path = frozen_dir / "metrics.json"
    _run_official_scorer(
        starter_dir,
        golden_path,
        frozen_path,
        frozen_metrics_path,
        frozen_dir / "scorer.log",
    )
    metrics_by_variant["r0_frozen"] = load_json(frozen_metrics_path)
    diagnostics_by_variant["r0_frozen"] = frozen_diagnostics

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
        "generated_variants": list(FORMAT_VARIANTS),
        "reference_variants": ["r0_frozen"],
    }
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "README.md").write_text(
        _readme(config, "complete format-controlled pilot", comparison), encoding="utf-8"
    )
    print(json.dumps(comparison["variants"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
