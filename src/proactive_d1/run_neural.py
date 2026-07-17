"""Run D1 neural and fused heads on the frozen session-level OOF protocol."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

from proactive_r0.artifacts import code_snapshot, environment_snapshot, sha256_file, write_json
from proactive_r0.core import INTERRUPT_TAG, load_jsonl, validate_prediction_rows, write_jsonl
from proactive_r0.run import _run_official_scorer, _validate_static_files

from .core import (
    attach_gold_labels,
    build_label_free_chunks,
    decisions_from_feature,
    feature_names,
    metrics_for_subset,
    paired_session_bootstrap,
    prediction_rows,
    strip_answers,
    validate_fold_manifest,
)
from .neural_core import (
    NEURAL_VARIANTS,
    cross_validate_neural_matrix,
    load_aligned_neural_cache,
    neural_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "d1_internvl35_1b_neural_oof.json"


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
        raise ValueError(f"Frozen D1 artifact mismatch for {path}: {actual} != {expected}")
    return actual


def _verify_generation_reference(reference: dict[str, object]) -> tuple[Path, dict[str, str]]:
    directory = _resolve(reference["experiment_dir"])
    hashes = {
        "predictions_sha256": _check_hash(
            directory / "predictions.jsonl", reference["predictions_sha256"]
        ),
        "session_records_sha256": _check_hash(
            directory / "session_records.jsonl", reference["session_records_sha256"]
        ),
        "metrics_sha256": _check_hash(directory / "metrics.json", reference["metrics_sha256"]),
    }
    return directory, hashes


def _prediction_decisions(
    predictions: list[dict[str, object]],
    examples: object,
) -> dict[tuple[int, int], int]:
    decisions: dict[tuple[int, int], int] = {}
    for input_index, row in enumerate(predictions):
        answers = row.get("answers")
        if not isinstance(answers, list):
            raise ValueError(f"D1 scalar baseline row {input_index} has no answers")
        for chunk_index, answer in enumerate(answers):
            decisions[(input_index, chunk_index)] = int(
                str(answer).startswith(INTERRUPT_TAG)
            )
    expected = {example.key for example in examples}  # type: ignore[union-attr]
    if set(decisions) != expected:
        raise ValueError("D1 scalar baseline predictions do not cover neural examples")
    return decisions


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d1.run_neural", *argv])
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
        *sorted((PROJECT_ROOT / "src" / "proactive_r0f").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d1").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d1" / "tests").glob("*.py")),
        config_path,
        PROJECT_ROOT / "configs" / "d1_internvl35_1b_scalar_oof.json",
        PROJECT_ROOT / "configs" / "d1_internvl35_1b_neural_features.json",
        PROJECT_ROOT / "models" / "internvl35_1b_hf.json",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
        PROJECT_ROOT / "C1_SPEC.md",
        PROJECT_ROOT / "Agent.md",
    ]


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(raw_argv)
    started_at = time.monotonic()

    config_path = _resolve(args.config)
    config = _load_json(config_path)
    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    model_config = dict(config["model"])  # type: ignore[arg-type]
    r0_reference = dict(config["r0_reference"])  # type: ignore[arg-type]
    r0f_reference = dict(config["r0f_reference"])  # type: ignore[arg-type]
    scalar_reference = dict(config["scalar_oof_reference"])  # type: ignore[arg-type]
    cache_config = dict(config["neural_cache"])  # type: ignore[arg-type]
    feature_config = dict(config["features"])  # type: ignore[arg-type]
    split_config = dict(config["split"])  # type: ignore[arg-type]
    training_config = dict(config["training"])  # type: ignore[arg-type]
    evaluation_config = dict(config["evaluation"])  # type: ignore[arg-type]

    input_path = _resolve(data_config["input"])
    starter_dir = _resolve(starter_config["path"])
    cache_dir = _resolve(cache_config["path"])
    cache_path = cache_dir / "features.npz"
    scalar_dir = _resolve(scalar_reference["experiment_dir"])
    output_dir = _resolve(args.output_dir or f"output/experiments/{config['experiment_id']}")
    if output_dir.exists():
        raise FileExistsError(f"D1 neural output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    fingerprints = _validate_static_files(config, input_path, starter_dir)
    r0_dir, r0_hashes = _verify_generation_reference(r0_reference)
    r0f_dir, r0f_hashes = _verify_generation_reference(r0f_reference)
    scalar_hashes = {
        "split_manifest_sha256": _check_hash(
            scalar_dir / "split_manifest.json", scalar_reference["split_manifest_sha256"]
        ),
        "comparison_sha256": _check_hash(
            scalar_dir / "comparison.json", scalar_reference["comparison_sha256"]
        ),
        "predictions_sha256": _check_hash(
            scalar_dir
            / "variants"
            / str(scalar_reference["feature_variant"])
            / "predictions.jsonl",
            scalar_reference["predictions_sha256"],
        ),
        "metrics_sha256": _check_hash(
            scalar_dir / "variants" / str(scalar_reference["feature_variant"]) / "metrics.json",
            scalar_reference["metrics_sha256"],
        ),
    }
    expected_cache_hash = str(cache_config["features_sha256"])
    if len(expected_cache_hash) != 64 or expected_cache_hash == "PENDING_MERGE_DO_NOT_RUN":
        raise ValueError("Pin the completed merged neural feature SHA256 before running D1 OOF")
    cache_hash = _check_hash(cache_path, expected_cache_hash)
    cache_summary = _load_json(cache_dir / "summary.json")
    if cache_summary.get("labels_read_or_stored") is not False:
        raise ValueError("D1 neural OOF requires a label-free merged feature cache")
    if int(cache_summary["sessions"]) != 700 or int(cache_summary["chunks"]) != 9935:
        raise ValueError("D1 neural cache coverage differs from 700/9935")

    source_rows = load_jsonl(input_path)
    r0_records = load_jsonl(r0_dir / "session_records.jsonl")
    manifest = _load_json(scalar_dir / "split_manifest.json")
    fold_by_index = validate_fold_manifest(manifest, strip_answers(source_rows))
    label_free = build_label_free_chunks(
        strip_answers(source_rows),
        r0_records,
        fold_by_index,
        max_history_turns=int(feature_config["max_history_turns"]),
        max_model_frames=int(feature_config["max_model_frames"]),
    )
    examples = attach_gold_labels(label_free, source_rows)
    neural_cache = load_aligned_neural_cache(
        cache_path, examples, hidden_size=int(cache_config["hidden_size"])
    )
    domains = sorted({example.feature.domain for example in examples})
    scalar_names = feature_names(  # type: ignore[arg-type]
        str(feature_config["scalar_variant"]), domains
    )
    variants = tuple(str(value) for value in feature_config["variants"])
    if any(variant not in NEURAL_VARIANTS for variant in variants):
        raise ValueError(f"Unsupported D1 neural variants: {variants}")

    scalar_predictions = load_jsonl(
        scalar_dir
        / "variants"
        / str(scalar_reference["feature_variant"])
        / "predictions.jsonl"
    )
    scalar_decisions = _prediction_decisions(scalar_predictions, examples)
    r0f_decisions = decisions_from_feature(examples, "r0f_decision_interrupt")
    scalar_non_first = metrics_for_subset(examples, scalar_decisions, include_first=False)
    r0f_non_first = metrics_for_subset(examples, r0f_decisions, include_first=False)

    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "input_path": str(input_path),
        "neural_cache_path": str(cache_path),
        "output_dir": str(output_dir),
        "audit_only": args.audit_only,
        "gpu_used_for_head_training": False,
    }
    write_json(output_dir / "config.json", effective)
    _write_command(output_dir / "command.sh", raw_argv)
    write_json(output_dir / "environment.txt", environment_snapshot())
    write_json(
        output_dir / "code_state.txt",
        code_snapshot(PROJECT_ROOT, _tracked_paths(config_path)),
    )
    write_json(
        output_dir / "data_manifest.json",
        {
            "source": {"path": str(input_path), "sha256": fingerprints["input_sha256"]},
            "r0_reference": {"path": str(r0_dir), **r0_hashes},
            "r0f_reference": {"path": str(r0f_dir), **r0f_hashes},
            "scalar_oof_reference": {"path": str(scalar_dir), **scalar_hashes},
            "neural_cache": {
                "path": str(cache_path),
                "sha256": cache_hash,
                "labels_read_or_stored": False,
            },
            "starter_kit_sha256": fingerprints,
            "supervision": config["validation_policy"],
        },
    )
    audit = {
        "sessions": 700,
        "chunks": 9935,
        "domains": domains,
        "folds": split_config["folds"],
        "neural_hidden_shape": list(neural_cache.hidden_state.shape),
        "scalar_feature_names": scalar_names,
        "variants": variants,
        "cache_contains_labels": False,
        "feature_rows_contain_gold": False,
    }
    write_json(output_dir / "feature_audit.json", audit)
    if args.audit_only:
        write_json(
            output_dir / "runtime.json",
            {
                "status": "audit_only",
                "wall_time_seconds": round(time.monotonic() - started_at, 3),
                "gpu_used": False,
            },
        )
        print(json.dumps(audit, sort_keys=True))
        return

    comparison_variants: dict[str, object] = {}
    max_head_parameters = 0
    for variant in variants:
        values, names = neural_matrix(
            examples, neural_cache, scalar_names, variant  # type: ignore[arg-type]
        )
        max_head_parameters = max(max_head_parameters, len(names) + 1)
        dimensional = (
            dict(training_config["hidden_dimensional"])  # type: ignore[arg-type]
            if variant in ("hidden_linear", "fused_linear")
            else dict(training_config["low_dimensional"])  # type: ignore[arg-type]
        )
        decisions, fold_details = cross_validate_neural_matrix(
            examples,
            values,
            names,
            folds=int(split_config["folds"]),
            calibration_fold_offset=int(split_config["calibration_fold_offset"]),
            seed=int(training_config["seed"]),
            max_iterations=int(training_config["max_iterations"]),
            l2_weights=[float(value) for value in dimensional["l2_weights"]],
            l2_reduction=str(dimensional["l2_reduction"]),  # type: ignore[arg-type]
        )
        variant_dir = output_dir / "variants" / variant
        variant_dir.mkdir(parents=True)
        predictions = prediction_rows(examples, decisions)
        validation = validate_prediction_rows(source_rows, predictions)
        predictions_path = variant_dir / "predictions.jsonl"
        write_jsonl(predictions_path, predictions)
        internal = metrics_for_subset(examples, decisions, include_first=True)
        non_first = metrics_for_subset(examples, decisions, include_first=False)
        bootstrap_scalar = paired_session_bootstrap(
            examples,
            decisions,
            scalar_decisions,
            repetitions=int(evaluation_config["bootstrap_repetitions"]),
            seed=int(evaluation_config["bootstrap_seed"]),
        )
        bootstrap_r0f = paired_session_bootstrap(
            examples,
            decisions,
            r0f_decisions,
            repetitions=int(evaluation_config["bootstrap_repetitions"]),
            seed=int(evaluation_config["bootstrap_seed"]),
        )
        write_jsonl(
            variant_dir / "oof_records.jsonl",
            [
                {
                    "input_index": example.feature.input_index,
                    "video_path": example.feature.video_path,
                    "fold": example.feature.fold,
                    "chunk_index": example.feature.chunk_index,
                    "gold_interrupt": example.gold_interrupt,
                    "predicted_interrupt": decisions[example.key],
                    "scalar_interrupt": scalar_decisions[example.key],
                    "r0f_interrupt": r0f_decisions[example.key],
                    "tag_margin": float(neural_cache.tag_margin[row_index]),
                    "prompt_tokens": int(neural_cache.prompt_tokens[row_index]),
                }
                for row_index, example in enumerate(examples)
            ],
        )
        metrics_path = variant_dir / "metrics.json"
        _run_official_scorer(
            starter_dir,
            input_path,
            predictions_path,
            metrics_path,
            variant_dir / "scorer.log",
        )
        metrics = _load_json(metrics_path)
        overall = metrics["overall"]
        assert isinstance(overall, dict)
        delta_scalar = float(overall["macro_f1"]) - float(scalar_reference["macro_f1"])
        delta_r0f = float(overall["macro_f1"]) - float(r0f_reference["macro_f1"])
        promotion = (
            delta_scalar >= float(evaluation_config["promotion_min_delta_vs_scalar_oof"])
            and float(bootstrap_scalar["delta_macro_f1_p2_5"]) > 0
            and float(overall["interrupt_f1"]) > 0
            and float(overall["silent_f1"]) > 0
            and float(non_first["macro_f1"]) > float(scalar_non_first["macro_f1"])
        )
        diagnostics = {
            **validation,
            "feature_variant": variant,
            "feature_count": len(names),
            "head_parameters_per_fold": len(names) + 1,
            "fold_details": fold_details,
            "internal_oof_metrics": internal,
            "non_first_chunk_metrics": non_first,
            "scalar_non_first_chunk_metrics": scalar_non_first,
            "r0f_non_first_chunk_metrics": r0f_non_first,
            "paired_session_bootstrap_vs_scalar": bootstrap_scalar,
            "paired_session_bootstrap_vs_r0f": bootstrap_r0f,
            "predictions_sha256": sha256_file(predictions_path),
        }
        write_json(variant_dir / "diagnostics.json", diagnostics)
        comparison_variants[variant] = {
            "macro_f1": overall["macro_f1"],
            "interrupt_f1": overall["interrupt_f1"],
            "silent_f1": overall["silent_f1"],
            "predicted_interrupt_rate": internal["predicted_interrupt_rate"],
            "delta_macro_f1_vs_scalar_oof": round(delta_scalar, 6),
            "delta_macro_f1_vs_r0f": round(delta_r0f, 6),
            "non_first_chunk_macro_f1": non_first["macro_f1"],
            "scalar_non_first_chunk_macro_f1": scalar_non_first["macro_f1"],
            "bootstrap_vs_scalar": bootstrap_scalar,
            "promotion_gate_passed": promotion,
        }
        print(json.dumps({"variant": variant, **comparison_variants[variant]}, sort_keys=True))

    comparison = {
        "classification": config["validation_policy"],
        "r0f_reference_macro_f1": r0f_reference["macro_f1"],
        "scalar_oof_reference_macro_f1": scalar_reference["macro_f1"],
        "variants": comparison_variants,
        "promotion_rule": {
            "min_delta_macro_f1_vs_scalar_oof": evaluation_config[
                "promotion_min_delta_vs_scalar_oof"
            ],
            "positive_session_bootstrap_lower_bound": True,
            "non_first_chunk_gain": True,
            "both_class_f1_nonzero": True,
        },
    }
    write_json(output_dir / "comparison.json", comparison)
    total_parameters = int(model_config["total_parameters"]) + max_head_parameters
    if total_parameters > 2_000_000_000:
        raise ValueError("D1 neural head exceeds the Small parameter limit")
    runtime = {
        "status": "complete neural OOF controls",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started_at, 3),
        "gpu_used_for_head_training": False,
        "model_inference_rerun": False,
        "sessions": 700,
        "chunks": 9935,
        "max_head_parameters": max_head_parameters,
        "max_total_parameters": total_parameters,
    }
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "Status: **complete neural OOF controls**",
                "",
                "This is session-level val-supervised OOF development, not hidden-test evidence.",
                f"The promoted scalar reference is Macro F1 `{scalar_reference['macro_f1']}`.",
                "",
                *[
                    f"- `{name}`: Macro F1 `{payload['macro_f1']}`, "
                    f"delta vs scalar `{payload['delta_macro_f1_vs_scalar_oof']}`"
                    for name, payload in comparison_variants.items()  # type: ignore[union-attr]
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
