"""Run the preregistered CPU-only D3 causal-dynamics OOF experiment."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from proactive_d1.core import (
    LabeledChunk,
    attach_gold_labels,
    binary_metrics,
    build_label_free_chunks,
    feature_names,
    metrics_for_subset,
    paired_session_bootstrap,
    prediction_rows,
    strip_answers,
    validate_fold_manifest,
)
from proactive_d1.neural_core import (
    cross_validate_neural_matrix,
    load_aligned_neural_cache,
)
from proactive_d3.core import (
    D3_VARIANTS,
    DYNAMIC_SCALAR_NAMES,
    PRIMARY_VARIANT,
    build_causal_dynamics,
    d3_matrix,
)
from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    write_json,
)
from proactive_r0.core import (
    INTERRUPT_TAG,
    load_jsonl,
    validate_prediction_rows,
    write_jsonl,
)
from proactive_r0.run import _run_official_scorer, _validate_static_files


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/d3_internvl35_1b_causal_dynamics_oof.json"


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
        raise ValueError(f"Frozen D3 artifact mismatch for {path}: {actual} != {expected}")
    return actual


def _prediction_decisions(
    predictions: Sequence[dict[str, object]], examples: Sequence[LabeledChunk]
) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for input_index, row in enumerate(predictions):
        answers = row.get("answers")
        if not isinstance(answers, list):
            raise ValueError(f"D3 reference prediction is malformed at {input_index}")
        for chunk_index, answer in enumerate(answers):
            result[(input_index, chunk_index)] = int(
                str(answer).startswith(INTERRUPT_TAG)
            )
    if set(result) != {example.key for example in examples}:
        raise ValueError("D3 reference decisions do not cover all examples")
    return result


def _subset_metrics(
    examples: Sequence[LabeledChunk],
    decisions: dict[tuple[int, int], int],
    include: Callable[[LabeledChunk], bool],
) -> dict[str, float | int]:
    selected = [example for example in examples if include(example)]
    if not selected:
        raise ValueError("D3 diagnostic subset is empty")
    return binary_metrics(
        [example.gold_interrupt for example in selected],
        [decisions[example.key] for example in selected],
    )


def _group_comparison(
    examples: Sequence[LabeledChunk],
    candidate: dict[tuple[int, int], int],
    reference: dict[tuple[int, int], int],
    key: Callable[[LabeledChunk], str],
) -> dict[str, dict[str, object]]:
    groups: dict[str, list[LabeledChunk]] = defaultdict(list)
    for example in examples:
        groups[key(example)].append(example)
    result: dict[str, dict[str, object]] = {}
    for name, rows in sorted(groups.items()):
        gold = [row.gold_interrupt for row in rows]
        candidate_metrics = binary_metrics(gold, [candidate[row.key] for row in rows])
        reference_metrics = binary_metrics(gold, [reference[row.key] for row in rows])
        result[name] = {
            "chunks": len(rows),
            "candidate": candidate_metrics,
            "reference": reference_metrics,
            "delta_macro_f1": float(candidate_metrics["macro_f1"])
            - float(reference_metrics["macro_f1"]),
        }
    return result


def _position(example: LabeledChunk) -> str:
    chunk = example.feature.chunk_index
    if chunk == 0:
        return "first"
    if chunk == 1:
        return "second"
    if chunk <= 4:
        return "2-4"
    if chunk <= 9:
        return "5-9"
    return "10+"


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d3.run", *argv])
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _tracked_paths(config_path: Path) -> list[Path]:
    return [
        *sorted((PROJECT_ROOT / "src/proactive_r0").glob("*.py")),
        *sorted((PROJECT_ROOT / "src/proactive_r0f").glob("*.py")),
        *sorted((PROJECT_ROOT / "src/proactive_d1").glob("*.py")),
        *sorted((PROJECT_ROOT / "src/proactive_d3").glob("*.py")),
        *sorted((PROJECT_ROOT / "src/proactive_d3/tests").glob("*.py")),
        config_path,
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
    started = time.monotonic()

    config_path = _resolve(args.config)
    config = _load_json(config_path)
    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    model_config = dict(config["model"])  # type: ignore[arg-type]
    r0_reference = dict(config["r0_reference"])  # type: ignore[arg-type]
    split_reference = dict(config["split_reference"])  # type: ignore[arg-type]
    cache_config = dict(config["neural_cache"])  # type: ignore[arg-type]
    d1_reference = dict(config["d1_reference"])  # type: ignore[arg-type]
    feature_config = dict(config["features"])  # type: ignore[arg-type]
    training_config = dict(config["training"])  # type: ignore[arg-type]
    evaluation_config = dict(config["evaluation"])  # type: ignore[arg-type]
    promotion_config = dict(evaluation_config["promotion"])  # type: ignore[arg-type]

    input_path = _resolve(data_config["input"])
    starter_dir = _resolve(starter_config["path"])
    r0_dir = _resolve(r0_reference["experiment_dir"])
    split_dir = _resolve(split_reference["experiment_dir"])
    split_path = split_dir / str(split_reference["manifest"])
    cache_dir = _resolve(cache_config["path"])
    cache_path = cache_dir / "features.npz"
    cache_summary_path = cache_dir / "summary.json"
    d1_dir = _resolve(d1_reference["experiment_dir"])
    d1_variant_dir = d1_dir / "variants" / str(d1_reference["variant"])
    output_dir = _resolve(args.output_dir or f"output/experiments/{config['experiment_id']}")
    if output_dir.exists():
        raise FileExistsError(f"D3 output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    fingerprints = _validate_static_files(config, input_path, starter_dir)
    source_hashes = {
        "r0_session_records": _check_hash(
            r0_dir / "session_records.jsonl", r0_reference["session_records_sha256"]
        ),
        "split_manifest": _check_hash(
            split_path, split_reference["manifest_sha256"]
        ),
        "neural_features": _check_hash(
            cache_path, cache_config["features_sha256"]
        ),
        "neural_summary": _check_hash(
            cache_summary_path, cache_config["summary_sha256"]
        ),
        "d1_predictions": _check_hash(
            d1_variant_dir / "predictions.jsonl", d1_reference["predictions_sha256"]
        ),
        "d1_metrics": _check_hash(
            d1_variant_dir / "metrics.json", d1_reference["metrics_sha256"]
        ),
        "d1_oof_records": _check_hash(
            d1_variant_dir / "oof_records.jsonl", d1_reference["oof_records_sha256"]
        ),
    }
    cache_summary = _load_json(cache_summary_path)
    if cache_summary.get("labels_read_or_stored") is not False:
        raise ValueError("D3 requires the frozen label-free D1 cache")
    if int(cache_summary["sessions"]) != 700 or int(cache_summary["chunks"]) != 9935:
        raise ValueError("D3 cache coverage differs from 700/9935")

    source_rows = load_jsonl(input_path)
    r0_records = load_jsonl(r0_dir / "session_records.jsonl")
    split_manifest = _load_json(split_path)
    fold_by_index = validate_fold_manifest(split_manifest, strip_answers(source_rows))
    label_free = build_label_free_chunks(
        strip_answers(source_rows),
        r0_records,
        fold_by_index,
        max_history_turns=int(feature_config["max_history_turns"]),
        max_model_frames=int(feature_config["max_model_frames"]),
    )
    examples = attach_gold_labels(label_free, source_rows)
    cache = load_aligned_neural_cache(
        cache_path, examples, hidden_size=int(cache_config["hidden_size"])
    )
    dynamics = build_causal_dynamics(cache)
    domains = sorted({example.feature.domain for example in examples})
    scalar_names = feature_names(
        str(feature_config["scalar_variant"]), domains  # type: ignore[arg-type]
    )
    variants = tuple(str(value) for value in feature_config["variants"])
    if variants != D3_VARIANTS:
        raise ValueError(f"D3 variants differ from preregistration: {variants}")
    if str(feature_config["primary_variant"]) != PRIMARY_VARIANT:
        raise ValueError("D3 primary variant differs from preregistration")
    if tuple(feature_config["dynamic_scalars"]) != DYNAMIC_SCALAR_NAMES:  # type: ignore[arg-type]
        raise ValueError("D3 dynamic scalar names differ from preregistration")

    d1_predictions = load_jsonl(d1_variant_dir / "predictions.jsonl")
    d1_decisions = _prediction_decisions(d1_predictions, examples)
    d1_metrics = _load_json(d1_variant_dir / "metrics.json")
    d1_overall = dict(d1_metrics["overall"])  # type: ignore[arg-type]
    if float(d1_overall["macro_f1"]) != float(d1_reference["macro_f1"]):
        raise ValueError("D3 D1 reference metric differs from preregistration")
    d1_non_first = metrics_for_subset(examples, d1_decisions, include_first=False)
    if abs(
        float(d1_non_first["macro_f1"])
        - float(d1_reference["non_first_macro_f1"])
    ) > 1e-12:
        raise ValueError("D3 D1 non-first reference differs from preregistration")

    matrix_shapes: dict[str, list[int]] = {}
    feature_counts: dict[str, int] = {}
    for variant in variants:
        values, names = d3_matrix(
            examples, cache, scalar_names, dynamics, variant  # type: ignore[arg-type]
        )
        matrix_shapes[variant] = list(values.shape)
        feature_counts[variant] = len(names)
    audit = {
        "sessions": len(set(cache.input_index.tolist())),
        "chunks": len(examples),
        "domains": domains,
        "folds": split_reference["folds"],
        "scalar_feature_count": len(scalar_names),
        "dynamic_scalar_names": list(DYNAMIC_SCALAR_NAMES),
        "dynamic_scalar_shape": list(dynamics.scalar.shape),
        "hidden_delta_shape": list(dynamics.hidden_delta.shape),
        "first_chunk_rows": int(np.sum(cache.chunk_index == 0)),
        "first_chunk_dynamic_scalar_abs_max": float(
            np.abs(dynamics.scalar[cache.chunk_index == 0]).max()
        ),
        "first_chunk_hidden_delta_abs_max": float(
            np.abs(dynamics.hidden_delta[cache.chunk_index == 0]).max()
        ),
        "dynamic_scalar_summary": {
            name: {
                "min": float(dynamics.scalar[:, index].min()),
                "max": float(dynamics.scalar[:, index].max()),
                "mean": float(dynamics.scalar[:, index].mean()),
                "std": float(dynamics.scalar[:, index].std()),
            }
            for index, name in enumerate(DYNAMIC_SCALAR_NAMES)
        },
        "matrix_shapes": matrix_shapes,
        "feature_counts": feature_counts,
        "primary_variant": PRIMARY_VARIANT,
        "dynamic_features_read_labels": False,
        "dynamic_features_read_future_rows": False,
    }

    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "input_path": str(input_path),
        "cache_path": str(cache_path),
        "output_dir": str(output_dir),
        "audit_only": args.audit_only,
        "gpu_used": False,
    }
    write_json(output_dir / "config.json", effective)
    _write_command(output_dir / "command.sh", raw_argv)
    write_json(output_dir / "environment.txt", environment_snapshot())
    write_json(output_dir / "code_state.txt", code_snapshot(PROJECT_ROOT, _tracked_paths(config_path)))
    write_json(output_dir / "feature_audit.json", audit)
    write_json(
        output_dir / "data_manifest.json",
        {
            "source": {"path": str(input_path), "sha256": fingerprints["input_sha256"]},
            "source_hashes": source_hashes,
            "starter_kit_sha256": fingerprints,
            "dynamic_feature_supervision": {
                "labels_read": False,
                "future_rows_read": False,
                "source": "frozen D1 neural cache only",
            },
            "head_supervision": config["validation_policy"],
            "external_data_used": False,
        },
    )
    if args.audit_only:
        write_json(
            output_dir / "runtime.json",
            {
                "status": "audit_only",
                "wall_time_seconds": round(time.monotonic() - started, 3),
                "gpu_used": False,
            },
        )
        print(json.dumps(audit, sort_keys=True))
        return

    comparison_variants: dict[str, object] = {}
    max_head_parameters = 0
    baseline_reproduced = False
    for variant in variants:
        values, names = d3_matrix(
            examples, cache, scalar_names, dynamics, variant  # type: ignore[arg-type]
        )
        max_head_parameters = max(max_head_parameters, len(names) + 1)
        decisions, fold_details = cross_validate_neural_matrix(
            examples,
            values,
            names,
            folds=int(split_reference["folds"]),
            calibration_fold_offset=int(split_reference["calibration_fold_offset"]),
            seed=int(training_config["seed"]),
            max_iterations=int(training_config["max_iterations"]),
            l2_weights=[float(value) for value in training_config["l2_weights"]],  # type: ignore[index]
            l2_reduction=str(training_config["l2_reduction"]),  # type: ignore[arg-type]
        )
        variant_dir = output_dir / "variants" / variant
        variant_dir.mkdir(parents=True)
        predictions = prediction_rows(examples, decisions)
        validation = validate_prediction_rows(source_rows, predictions)
        predictions_path = variant_dir / "predictions.jsonl"
        write_jsonl(predictions_path, predictions)
        prediction_hash = sha256_file(predictions_path)

        if variant == "d1_fused_replay":
            if decisions != d1_decisions:
                changed = sum(
                    decisions[key] != d1_decisions[key] for key in d1_decisions
                )
                raise ValueError(f"D3 failed to reproduce D1 decisions: {changed} changed")
            if prediction_hash != str(d1_reference["predictions_sha256"]):
                raise ValueError("D3 failed to reproduce the D1 prediction SHA256")
            baseline_reproduced = True
        elif not baseline_reproduced:
            raise ValueError("D3 dynamic variant ran before exact D1 reproduction")

        internal = metrics_for_subset(examples, decisions, include_first=True)
        non_first = metrics_for_subset(examples, decisions, include_first=False)
        bootstrap = paired_session_bootstrap(
            examples,
            decisions,
            d1_decisions,
            repetitions=int(evaluation_config["bootstrap_repetitions"]),
            seed=int(evaluation_config["bootstrap_seed"]),
        )
        fold_comparison = _group_comparison(
            examples, decisions, d1_decisions, lambda example: str(example.feature.fold)
        )
        domain_comparison = _group_comparison(
            examples, decisions, d1_decisions, lambda example: example.feature.domain
        )
        position_comparison = _group_comparison(
            examples, decisions, d1_decisions, _position
        )
        positive_folds = sum(
            float(value["delta_macro_f1"]) > 0 for value in fold_comparison.values()
        )
        positive_domains = sum(
            float(value["delta_macro_f1"]) > 0 for value in domain_comparison.values()
        )
        write_jsonl(
            variant_dir / "oof_records.jsonl",
            [
                {
                    "input_index": example.feature.input_index,
                    "video_path": example.feature.video_path,
                    "domain": example.feature.domain,
                    "fold": example.feature.fold,
                    "chunk_index": example.feature.chunk_index,
                    "gold_interrupt": example.gold_interrupt,
                    "predicted_interrupt": decisions[example.key],
                    "d1_interrupt": d1_decisions[example.key],
                    "tag_margin": float(cache.tag_margin[row_index]),
                    **{
                        name: float(dynamics.scalar[row_index, scalar_index])
                        for scalar_index, name in enumerate(DYNAMIC_SCALAR_NAMES)
                    },
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
        overall = dict(metrics["overall"])  # type: ignore[arg-type]
        if variant == "d1_fused_replay" and sha256_file(metrics_path) != str(
            d1_reference["metrics_sha256"]
        ):
            raise ValueError("D3 failed to reproduce the D1 official metrics SHA256")
        delta = float(overall["macro_f1"]) - float(d1_reference["macro_f1"])
        checks = {
            "primary_variant": variant == PRIMARY_VARIANT,
            "minimum_macro_f1_delta": delta
            >= float(promotion_config["minimum_macro_f1_delta_vs_d1"]),
            "positive_session_bootstrap_lower_bound": float(
                bootstrap["delta_macro_f1_p2_5"]
            )
            > 0,
            "minimum_positive_folds": positive_folds
            >= int(promotion_config["minimum_positive_folds"]),
            "minimum_positive_domains": positive_domains
            >= int(promotion_config["minimum_positive_domains"]),
            "non_first_chunk_gain": float(non_first["macro_f1"])
            > float(d1_non_first["macro_f1"]),
            "minimum_interrupt_f1": float(overall["interrupt_f1"])
            >= float(promotion_config["minimum_interrupt_f1"]),
            "minimum_silent_f1": float(overall["silent_f1"])
            >= float(promotion_config["minimum_silent_f1"]),
        }
        promotion_passed = all(checks.values())
        diagnostics = {
            **validation,
            "feature_variant": variant,
            "promotion_eligible": variant == PRIMARY_VARIANT,
            "feature_count": len(names),
            "head_parameters_per_fold": len(names) + 1,
            "fold_details": fold_details,
            "internal_oof_metrics": internal,
            "official_metrics": overall,
            "non_first_chunk_metrics": non_first,
            "d1_non_first_chunk_metrics": d1_non_first,
            "paired_session_bootstrap_vs_d1": bootstrap,
            "fold_comparison": fold_comparison,
            "domain_comparison": domain_comparison,
            "position_comparison": position_comparison,
            "positive_folds": positive_folds,
            "positive_domains": positive_domains,
            "promotion_gate": {"checks": checks, "passed": promotion_passed},
            "predictions_sha256": prediction_hash,
            "metrics_sha256": sha256_file(metrics_path),
        }
        write_json(variant_dir / "diagnostics.json", diagnostics)
        comparison_variants[variant] = {
            "macro_f1": overall["macro_f1"],
            "interrupt_f1": overall["interrupt_f1"],
            "silent_f1": overall["silent_f1"],
            "predicted_interrupt_rate": internal["predicted_interrupt_rate"],
            "delta_macro_f1_vs_d1": delta,
            "non_first_chunk_macro_f1": non_first["macro_f1"],
            "paired_session_bootstrap_vs_d1": bootstrap,
            "positive_folds": positive_folds,
            "positive_domains": positive_domains,
            "promotion_eligible": variant == PRIMARY_VARIANT,
            "promotion_gate": {"checks": checks, "passed": promotion_passed},
            "predictions_sha256": prediction_hash,
        }
        print(json.dumps({"variant": variant, **comparison_variants[variant]}, sort_keys=True))

    comparison = {
        "classification": config["validation_policy"],
        "d1_reference": {
            "macro_f1": d1_reference["macro_f1"],
            "non_first_macro_f1": d1_reference["non_first_macro_f1"],
            "predictions_sha256": d1_reference["predictions_sha256"],
            "metrics_sha256": d1_reference["metrics_sha256"],
        },
        "baseline_reproduced_exactly": baseline_reproduced,
        "primary_variant": PRIMARY_VARIANT,
        "variants": comparison_variants,
        "promotion_rule": promotion_config,
        "diagnostic_variants_cannot_be_promoted": True,
    }
    write_json(output_dir / "comparison.json", comparison)
    max_total_parameters = int(model_config["total_parameters"]) + max_head_parameters
    if max_total_parameters > 2_000_000_000:
        raise ValueError("D3 exceeds the Small parameter limit")
    runtime = {
        "status": "complete D3 causal-dynamics OOF",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "gpu_used": False,
        "model_inference_rerun": False,
        "sessions": 700,
        "chunks": 9935,
        "max_head_parameters": max_head_parameters,
        "max_total_parameters": max_total_parameters,
    }
    write_json(output_dir / "runtime.json", runtime)
    primary = comparison_variants[PRIMARY_VARIANT]
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "Status: complete CPU-only D3 causal-dynamics OOF.",
                "",
                "This is public-validation-supervised OOF development, not hidden-test evidence.",
                f"D1 replay exact: {baseline_reproduced}.",
                f"Primary `{PRIMARY_VARIANT}` Macro F1: {primary['macro_f1']}.",
                f"Promotion passed: {primary['promotion_gate']['passed']}.",  # type: ignore[index]
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
