"""Run the preregistered CPU-only D5 decision-fusion OOF experiment."""

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
    make_fold_manifest,
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
from proactive_d3.core import DYNAMIC_SCALAR_NAMES, build_causal_dynamics
from proactive_d3.dialog_control_core import (
    DIALOG_POLICY_NAMES,
    build_dialog_policy_features,
)
from proactive_d5.core import (
    ACTION_HISTORY_NAMES,
    D5_VARIANTS,
    PRIMARY_VARIANT,
    build_action_history_features,
    d5_matrix,
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
        raise ValueError(f"Frozen D5 artifact mismatch for {path}: {actual} != {expected}")
    return actual


def _decisions(
    predictions: Sequence[dict[str, object]], examples: Sequence[LabeledChunk]
) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for input_index, row in enumerate(predictions):
        answers = row.get("answers")
        if not isinstance(answers, list):
            raise ValueError(f"D5 prediction is malformed at {input_index}")
        for chunk_index, answer in enumerate(answers):
            result[(input_index, chunk_index)] = int(
                str(answer).startswith(INTERRUPT_TAG)
            )
    if set(result) != {example.key for example in examples}:
        raise ValueError("D5 reference decisions do not cover every example")
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
        labels = [row.gold_interrupt for row in rows]
        candidate_metrics = binary_metrics(labels, [candidate[row.key] for row in rows])
        reference_metrics = binary_metrics(labels, [reference[row.key] for row in rows])
        result[name] = {
            "chunks": len(rows),
            "candidate": candidate_metrics,
            "reference": reference_metrics,
            "delta_macro_f1": float(candidate_metrics["macro_f1"])
            - float(reference_metrics["macro_f1"]),
        }
    return result


def _write_command(path: Path, argv: Sequence[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d5.run", *argv])
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


def _score(
    *,
    source_rows: Sequence[dict[str, object]],
    examples: Sequence[LabeledChunk],
    decisions: dict[tuple[int, int], int],
    input_path: Path,
    starter_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, object], str, str]:
    output_dir.mkdir(parents=True)
    predictions = prediction_rows(examples, decisions)
    validate_prediction_rows(source_rows, predictions)
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    metrics_path = output_dir / "metrics.json"
    _run_official_scorer(
        starter_dir,
        input_path,
        predictions_path,
        metrics_path,
        output_dir / "scorer.log",
    )
    metrics = _load_json(metrics_path)
    return metrics, sha256_file(predictions_path), sha256_file(metrics_path)


def _validate_feature_contract(config: dict[str, object]) -> None:
    feature_config = dict(config["features"])  # type: ignore[arg-type]
    if tuple(feature_config["dialog_policy_names"]) != DIALOG_POLICY_NAMES:
        raise ValueError("D5 dialog feature names differ from preregistration")
    if tuple(feature_config["dynamic_scalar_names"]) != DYNAMIC_SCALAR_NAMES:
        raise ValueError("D5 dynamic feature names differ from preregistration")
    if tuple(feature_config["omitted_duplicate_dynamic_scalars"]) != (
        "has_previous_chunk",
    ):
        raise ValueError("D5 duplicate-column policy differs from preregistration")
    if tuple(feature_config["action_history_names"]) != ACTION_HISTORY_NAMES:
        raise ValueError("D5 action-history names differ from preregistration")
    if tuple(feature_config["variants"]) != D5_VARIANTS:
        raise ValueError("D5 variants differ from preregistration")
    if str(feature_config["primary_variant"]) != PRIMARY_VARIANT:
        raise ValueError("D5 primary differs from preregistration")


def _fit_variant(
    *,
    examples: Sequence[LabeledChunk],
    values: np.ndarray,
    names: Sequence[str],
    training: dict[str, object],
) -> tuple[dict[tuple[int, int], int], list[dict[str, object]]]:
    return cross_validate_neural_matrix(
        examples,
        values,
        names,
        folds=int(training["folds"]),
        calibration_fold_offset=int(training["calibration_fold_offset"]),
        seed=int(training["seed"]),
        max_iterations=int(training["max_iterations"]),
        l2_weights=[float(value) for value in training["l2_weights"]],  # type: ignore[union-attr]
        l2_reduction=str(training["l2_reduction"]),  # type: ignore[arg-type]
    )


def _stability_runs(
    *,
    config: dict[str, object],
    source_rows: Sequence[dict[str, object]],
    label_free_rows: Sequence[dict[str, object]],
    r0_records: Sequence[dict[str, object]],
    cache: object,
    dialog_values: np.ndarray,
    dynamics: object,
    action_history: np.ndarray,
    input_path: Path,
    starter_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    feature_config = dict(config["features"])  # type: ignore[arg-type]
    training = dict(config["training"])  # type: ignore[arg-type]
    stability = dict(config["stability"])  # type: ignore[arg-type]
    evaluation = dict(config["evaluation"])  # type: ignore[arg-type]
    if stability.get("enabled") is not True:
        raise ValueError("D5 preregistration requires stability runs")
    if tuple(stability["variants"]) != ("d4_replay", PRIMARY_VARIANT):
        raise ValueError("D5 stability variants differ from preregistration")
    seeds = [str(value) for value in stability["split_seeds"]]  # type: ignore[union-attr]
    if seeds != [
        "d5-stability-20260721-a",
        "d5-stability-20260721-b",
        "d5-stability-20260721-c",
    ]:
        raise ValueError("D5 stability seeds differ from preregistration")

    results: dict[str, object] = {}
    for split_index, seed in enumerate(seeds):
        split_dir = output_dir / seed
        split_dir.mkdir(parents=True)
        manifest = make_fold_manifest(
            label_free_rows,
            folds=int(training["folds"]),
            seed=seed,
        )
        manifest_path = split_dir / "split_manifest.json"
        write_json(manifest_path, manifest)
        fold_by_index = validate_fold_manifest(manifest, label_free_rows)
        chunks = build_label_free_chunks(
            label_free_rows,
            r0_records,
            fold_by_index,
            max_history_turns=int(feature_config["max_history_turns"]),
            max_model_frames=int(feature_config["max_model_frames"]),
        )
        examples = attach_gold_labels(chunks, source_rows)
        domains = sorted({example.feature.domain for example in examples})
        scalar_names = feature_names(str(feature_config["d1_scalar_variant"]), domains)
        split_decisions: dict[str, dict[tuple[int, int], int]] = {}
        split_metrics: dict[str, dict[str, object]] = {}
        split_fold_details: dict[str, object] = {}
        for variant in ("d4_replay", PRIMARY_VARIANT):
            values, names = d5_matrix(
                examples,
                cache,  # type: ignore[arg-type]
                scalar_names,
                dialog_values,
                dynamics,  # type: ignore[arg-type]
                action_history,
                variant,
            )
            decisions, fold_details = _fit_variant(
                examples=examples,
                values=values,
                names=names,
                training=training,
            )
            metrics, prediction_hash, metrics_hash = _score(
                source_rows=source_rows,
                examples=examples,
                decisions=decisions,
                input_path=input_path,
                starter_dir=starter_dir,
                output_dir=split_dir / variant,
            )
            split_decisions[variant] = decisions
            split_metrics[variant] = {
                **dict(metrics["overall"]),  # type: ignore[arg-type]
                "feature_count": len(names),
                "head_parameters_per_fold": len(names) + 1,
                "predictions_sha256": prediction_hash,
                "metrics_sha256": metrics_hash,
            }
            split_fold_details[variant] = fold_details
        reference = split_decisions["d4_replay"]
        candidate = split_decisions[PRIMARY_VARIANT]
        candidate_macro = float(split_metrics[PRIMARY_VARIANT]["macro_f1"])
        reference_macro = float(split_metrics["d4_replay"]["macro_f1"])
        bootstrap = paired_session_bootstrap(
            examples,
            candidate,
            reference,
            repetitions=int(evaluation["bootstrap_repetitions"]),
            seed=int(evaluation["bootstrap_seed"]) + 100 + split_index,
        )
        result = {
            "seed": seed,
            "split_manifest_sha256": sha256_file(manifest_path),
            "variants": split_metrics,
            "delta_macro_f1": candidate_macro - reference_macro,
            "paired_session_bootstrap": bootstrap,
            "fold_details": split_fold_details,
        }
        write_json(split_dir / "comparison.json", result)
        results[seed] = result
        print(json.dumps({"stability": seed, **result}, sort_keys=True))
    return {
        "classification": "same-public-data label-independent split robustness; not independent evidence",
        "splits": results,
        "all_primary_deltas_positive": all(
            float(value["delta_macro_f1"]) > 0  # type: ignore[index]
            for value in results.values()  # type: ignore[union-attr]
        ),
    }


def run(config_path: Path, output_dir: Path, raw_argv: Sequence[str]) -> dict[str, object]:
    started = time.monotonic()
    config = _load_json(config_path)
    if output_dir.exists():
        raise FileExistsError(f"D5 output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    protocol = dict(config["protocol"])  # type: ignore[arg-type]
    base_configs = dict(config["base_configs"])  # type: ignore[arg-type]
    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    source_config = dict(config["sources"])  # type: ignore[arg-type]
    feature_config = dict(config["features"])  # type: ignore[arg-type]
    training = dict(config["training"])  # type: ignore[arg-type]
    references = dict(config["references"])  # type: ignore[arg-type]
    evaluation = dict(config["evaluation"])  # type: ignore[arg-type]
    promotion = dict(evaluation["promotion"])  # type: ignore[arg-type]

    protocol_path = _resolve(protocol["path"])
    _check_hash(protocol_path, protocol["sha256"])
    for value in base_configs.values():
        entry = dict(value)  # type: ignore[arg-type]
        _check_hash(_resolve(entry["path"]), entry["sha256"])
    input_path = _resolve(data_config["input"])
    starter_dir = _resolve(starter_config["path"])
    fingerprints = _validate_static_files(config, input_path, starter_dir)
    source_paths = {
        name: _resolve(dict(value)["path"])  # type: ignore[arg-type]
        for name, value in source_config.items()
    }
    source_hashes = {
        name: _check_hash(source_paths[name], dict(value)["sha256"])  # type: ignore[arg-type]
        for name, value in source_config.items()
    }
    _validate_feature_contract(config)

    source_rows = load_jsonl(input_path)
    label_free_rows = strip_answers(source_rows)
    r0_records = load_jsonl(source_paths["r0_session_records"])
    split_manifest = _load_json(source_paths["split_manifest"])
    fold_by_index = validate_fold_manifest(split_manifest, label_free_rows)
    label_free_chunks = build_label_free_chunks(
        label_free_rows,
        r0_records,
        fold_by_index,
        max_history_turns=int(feature_config["max_history_turns"]),
        max_model_frames=int(feature_config["max_model_frames"]),
    )
    dialog_values, dialog_audit = build_dialog_policy_features(
        label_free_rows, label_free_chunks
    )
    action_history, previous_actions, action_audit = build_action_history_features(
        label_free_rows, label_free_chunks
    )
    if not np.array_equal(previous_actions, dialog_values[:, 1].astype(np.int8)):
        raise ValueError("D5 action lag-1 differs from frozen D4 assistant addition")

    examples = attach_gold_labels(label_free_chunks, source_rows)
    cache = load_aligned_neural_cache(
        source_paths["neural_features"],
        examples,
        hidden_size=int(dict(source_config["neural_features"])["hidden_size"]),  # type: ignore[arg-type]
    )
    dynamics = build_causal_dynamics(cache)
    if not np.array_equal(dynamics.scalar[:, 0], dialog_values[:, 0]):
        raise ValueError("D5 duplicate has_previous_chunk columns differ")
    domains = sorted({example.feature.domain for example in examples})
    scalar_names = feature_names(str(feature_config["d1_scalar_variant"]), domains)

    d3_predictions = load_jsonl(source_paths["d3_predictions"])
    d4_predictions = load_jsonl(source_paths["d4_predictions"])
    d3_decisions = _decisions(d3_predictions, examples)
    d4_decisions = _decisions(d4_predictions, examples)
    d3_metrics = dict(_load_json(source_paths["d3_metrics"])["overall"])  # type: ignore[arg-type]
    d4_metrics = dict(_load_json(source_paths["d4_metrics"])["overall"])  # type: ignore[arg-type]
    if float(d3_metrics["macro_f1"]) != float(references["d3_macro_f1"]):
        raise ValueError("D5 frozen D3 metric differs from preregistration")
    if float(d4_metrics["macro_f1"]) != float(references["d4_macro_f1"]):
        raise ValueError("D5 frozen D4 metric differs from preregistration")

    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "input_path": str(input_path),
        "gpu_used": False,
    }
    write_json(output_dir / "config.json", effective)
    _write_command(output_dir / "command.sh", raw_argv)
    write_json(output_dir / "environment.txt", environment_snapshot())
    write_json(
        output_dir / "code_state.txt",
        code_snapshot(
            PROJECT_ROOT,
            [
                config_path,
                protocol_path,
                PROJECT_ROOT / "src/proactive_d3/core.py",
                PROJECT_ROOT / "src/proactive_d3/dialog_control_core.py",
                PROJECT_ROOT / "src/proactive_d5/core.py",
                PROJECT_ROOT / "src/proactive_d5/run.py",
                PROJECT_ROOT / "src/proactive_d5/tests/test_core.py",
            ],
        ),
    )
    write_json(
        output_dir / "feature_audit.json",
        {
            "dialog": dialog_audit,
            "action_history": action_audit,
            "lag1_matches_dialog_addition": True,
            "duplicate_has_previous_matches": True,
            "dynamic_scalar_shape": list(dynamics.scalar.shape),
            "hidden_delta_shape": list(dynamics.hidden_delta.shape),
        },
    )
    write_json(
        output_dir / "data_manifest.json",
        {
            "input": {"path": str(input_path), "sha256": fingerprints["input_sha256"]},
            "source_hashes": source_hashes,
            "starter_kit_sha256": fingerprints,
            "protocol_sha256": protocol["sha256"],
            "feature_supervision": {
                "labels_read": False,
                "predictions_read": False,
                "future_chunks_read": False,
                "source": "answer-stripped organizer-visible causal prefix",
            },
            "head_supervision": config["validation_policy"],
            "external_data_used": False,
            "human_evaluation_used": False,
        },
    )

    variant_results: dict[str, dict[str, object]] = {}
    variant_diagnostics: dict[str, dict[str, object]] = {}
    fixed_decisions: dict[str, dict[tuple[int, int], int]] = {}
    baseline_reproduced = False
    max_head_parameters = 0
    for variant in D5_VARIANTS:
        values, names = d5_matrix(
            examples,
            cache,
            scalar_names,
            dialog_values,
            dynamics,
            action_history,
            variant,
        )
        max_head_parameters = max(max_head_parameters, len(names) + 1)
        decisions, fold_details = _fit_variant(
            examples=examples,
            values=values,
            names=names,
            training=training,
        )
        variant_dir = output_dir / "variants" / variant
        metrics, prediction_hash, metrics_hash = _score(
            source_rows=source_rows,
            examples=examples,
            decisions=decisions,
            input_path=input_path,
            starter_dir=starter_dir,
            output_dir=variant_dir,
        )
        if variant == "d4_replay":
            if decisions != d4_decisions:
                changed = sum(decisions[key] != d4_decisions[key] for key in decisions)
                raise ValueError(f"D5 failed to reproduce D4 decisions: {changed}")
            if prediction_hash != str(dict(source_config["d4_predictions"])["sha256"]):  # type: ignore[arg-type]
                raise ValueError("D5 D4 replay prediction hash differs")
            if metrics_hash != str(dict(source_config["d4_metrics"])["sha256"]):  # type: ignore[arg-type]
                raise ValueError("D5 D4 replay metric hash differs")
            baseline_reproduced = True
        elif not baseline_reproduced:
            raise ValueError("D5 variants ran before exact D4 reproduction")

        overall = dict(metrics["overall"])  # type: ignore[arg-type]
        candidate_macro = float(overall["macro_f1"])
        delta_d4 = candidate_macro - float(d4_metrics["macro_f1"])
        bootstrap = paired_session_bootstrap(
            examples,
            decisions,
            d4_decisions,
            repetitions=int(evaluation["bootstrap_repetitions"]),
            seed=int(evaluation["bootstrap_seed"]),
        )
        non_first = metrics_for_subset(examples, decisions, include_first=False)
        fold_comparison = _group_comparison(
            examples, decisions, d4_decisions, lambda example: str(example.feature.fold)
        )
        domain_comparison = _group_comparison(
            examples, decisions, d4_decisions, lambda example: example.feature.domain
        )
        position_comparison = _group_comparison(
            examples, decisions, d4_decisions, _position
        )
        positive_folds = sum(
            float(value["delta_macro_f1"]) > 0 for value in fold_comparison.values()
        )
        positive_domains = sum(
            float(value["delta_macro_f1"]) > 0 for value in domain_comparison.values()
        )
        agreement_d3 = sum(
            decisions[key] == d3_decisions[key] for key in decisions
        ) / len(decisions)
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
                    "d4_interrupt": d4_decisions[example.key],
                    "d3_interrupt": d3_decisions[example.key],
                    "previous_action_interrupt": int(previous_actions[row_index]),
                }
                for row_index, example in enumerate(examples)
            ],
        )
        diagnostics = {
            "variant": variant,
            "primary_variant": variant == PRIMARY_VARIANT,
            "feature_count": len(names),
            "head_parameters_per_fold": len(names) + 1,
            "fold_details": fold_details,
            "official_metrics": overall,
            "non_first_chunk_metrics": non_first,
            "delta_macro_f1_vs_d4": delta_d4,
            "delta_macro_f1_vs_d3": candidate_macro - float(d3_metrics["macro_f1"]),
            "paired_session_bootstrap_vs_d4": bootstrap,
            "fold_comparison_vs_d4": fold_comparison,
            "domain_comparison_vs_d4": domain_comparison,
            "position_comparison_vs_d4": position_comparison,
            "positive_folds": positive_folds,
            "positive_domains": positive_domains,
            "decision_agreement_with_d3": agreement_d3,
            "predictions_sha256": prediction_hash,
            "metrics_sha256": metrics_hash,
        }
        write_json(variant_dir / "diagnostics.json", diagnostics)
        result = {
            "macro_f1": overall["macro_f1"],
            "interrupt_f1": overall["interrupt_f1"],
            "silent_f1": overall["silent_f1"],
            "delta_macro_f1_vs_d4": delta_d4,
            "delta_macro_f1_vs_d3": diagnostics["delta_macro_f1_vs_d3"],
            "non_first_macro_f1": non_first["macro_f1"],
            "paired_session_bootstrap_vs_d4": bootstrap,
            "positive_folds": positive_folds,
            "positive_domains": positive_domains,
            "feature_count": len(names),
            "head_parameters_per_fold": len(names) + 1,
            "predictions_sha256": prediction_hash,
        }
        variant_results[variant] = result
        variant_diagnostics[variant] = diagnostics
        fixed_decisions[variant] = decisions
        print(json.dumps({"variant": variant, **result}, sort_keys=True))

    if not baseline_reproduced:
        raise ValueError("D5 did not complete exact D4 replay")

    by_key = {example.key: example for example in examples}
    gold_matches = 0
    non_first_count = 0
    for row_index, example in enumerate(examples):
        if example.feature.chunk_index == 0:
            continue
        previous = by_key[(example.feature.input_index, example.feature.chunk_index - 1)]
        gold_matches += int(bool(previous_actions[row_index]) == bool(previous.gold_interrupt))
        non_first_count += 1
    previous_action_cross_check = {
        "runs_after_all_fixed_oof_predictions": True,
        "visible_previous_action_equals_previous_gold_interrupt": gold_matches,
        "non_first_chunks": non_first_count,
        "agreement_rate": gold_matches / non_first_count,
    }

    stability_results = _stability_runs(
        config=config,
        source_rows=source_rows,
        label_free_rows=label_free_rows,
        r0_records=r0_records,
        cache=cache,
        dialog_values=dialog_values,
        dynamics=dynamics,
        action_history=action_history,
        input_path=input_path,
        starter_dir=starter_dir,
        output_dir=output_dir / "stability",
    )

    primary = variant_results[PRIMARY_VARIANT]
    primary_diagnostics = variant_diagnostics[PRIMARY_VARIANT]
    promotion_checks = {
        "minimum_macro_f1_delta": float(primary["delta_macro_f1_vs_d4"])
        >= float(promotion["minimum_macro_f1_delta_vs_d4"]),
        "positive_session_bootstrap_lower_bound": float(
            dict(primary["paired_session_bootstrap_vs_d4"])["delta_macro_f1_p2_5"]  # type: ignore[arg-type]
        )
        > 0,
        "minimum_positive_folds": int(primary["positive_folds"])
        >= int(promotion["minimum_positive_folds"]),
        "minimum_positive_domains": int(primary["positive_domains"])
        >= int(promotion["minimum_positive_domains"]),
        "non_first_chunk_gain": float(primary["non_first_macro_f1"])
        > float(references["d4_non_first_macro_f1"]),
        "minimum_interrupt_f1": float(primary["interrupt_f1"])
        >= float(promotion["minimum_interrupt_f1"]),
        "minimum_silent_f1": float(primary["silent_f1"])
        >= float(promotion["minimum_silent_f1"]),
        "all_stability_splits_positive": bool(
            stability_results["all_primary_deltas_positive"]
        ),
    }
    promotion_passed = all(promotion_checks.values())
    primary_diagnostics["promotion"] = {
        "eligible": True,
        "checks": promotion_checks,
        "passed": promotion_passed,
    }
    write_json(
        output_dir / "variants" / PRIMARY_VARIANT / "diagnostics.json",
        primary_diagnostics,
    )
    comparison = {
        "status": "complete preregistered D5 decision-fusion OOF and stability study",
        "classification": config["classification"],
        "baseline_reproduced_exactly": True,
        "d3_reference": d3_metrics,
        "d4_reference": d4_metrics,
        "variants": variant_results,
        "primary_variant": PRIMARY_VARIANT,
        "primary_promotion": {
            "checks": promotion_checks,
            "passed": promotion_passed,
            "positive_but_below_minimum": 0
            < float(primary["delta_macro_f1_vs_d4"])
            < float(promotion["minimum_macro_f1_delta_vs_d4"]),
        },
        "stability": stability_results,
        "previous_action_cross_check": previous_action_cross_check,
        "human_evaluation_used": False,
        "gpu_used": False,
        "external_actions_performed": False,
    }
    write_json(output_dir / "comparison.json", comparison)
    runtime = {
        "status": comparison["status"],
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "gpu_used": False,
        "model_inference_rerun": False,
        "sessions": len(source_rows),
        "chunks": len(examples),
        "variants": len(D5_VARIANTS),
        "stability_splits": len(dict(stability_results["splits"])),  # type: ignore[arg-type]
        "max_head_parameters": max_head_parameters,
    }
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "状态：D5 冻结五折 OOF 与三组稳定性 split 已完成。",
                "",
                f"D4 精确复现：{baseline_reproduced}。",
                f"D4 Macro F1：{d4_metrics['macro_f1']}。",
                f"D5 primary Macro F1：{primary['macro_f1']}。",
                f"D5 primary delta：{primary['delta_macro_f1_vs_d4']}。",
                f"晋级门槛通过：{promotion_passed}。",
                "",
                "所有结果均为 public-validation-supervised OOF，不是隐藏测试证据。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(comparison, sort_keys=True))
    return comparison


def main() -> None:
    raw_argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/d5_internvl35_1b_decision_fusion_oof_v1.json"
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args(raw_argv)
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    output_dir = _resolve(
        args.output_dir or f"output/experiments/{config['experiment_id']}"
    )
    run(config_path, output_dir, raw_argv)


if __name__ == "__main__":
    main()
