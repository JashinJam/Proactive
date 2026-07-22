"""Run the preregistered CPU-only D6 structured-calibration OOF study."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

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
from proactive_d1.neural_core import load_aligned_neural_cache
from proactive_d3.dialog_control_core import (
    DIALOG_POLICY_NAMES,
    build_dialog_policy_features,
    dialog_control_matrix,
)
from proactive_d6.core import (
    D6_VARIANTS,
    LAST2_GROUPS,
    LAST_ACTION_GROUPS,
    POSITION_GROUPS,
    PRIMARY_VARIANT,
    build_structured_stages,
    cross_validate_structured_calibration,
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
        raise ValueError(f"Frozen D6 artifact mismatch for {path}: {actual} != {expected}")
    return actual


def _decisions(
    predictions: Sequence[dict[str, object]], examples: Sequence[LabeledChunk]
) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for input_index, row in enumerate(predictions):
        answers = row.get("answers")
        if not isinstance(answers, list):
            raise ValueError(f"D6 prediction is malformed at {input_index}")
        for chunk_index, answer in enumerate(answers):
            result[(input_index, chunk_index)] = int(
                str(answer).startswith(INTERRUPT_TAG)
            )
    if set(result) != {example.key for example in examples}:
        raise ValueError("D6 reference decisions do not cover every example")
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
    candidate: Mapping[tuple[int, int], int],
    reference: Mapping[tuple[int, int], int],
    key: Callable[[LabeledChunk], str],
) -> dict[str, dict[str, object]]:
    groups: dict[str, list[LabeledChunk]] = defaultdict(list)
    for example in examples:
        groups[key(example)].append(example)
    result: dict[str, dict[str, object]] = {}
    for name, rows in sorted(groups.items()):
        labels = [row.gold_interrupt for row in rows]
        candidate_metrics = binary_metrics(
            labels, [candidate[row.key] for row in rows]
        )
        reference_metrics = binary_metrics(
            labels, [reference[row.key] for row in rows]
        )
        result[name] = {
            "chunks": len(rows),
            "candidate": candidate_metrics,
            "reference": reference_metrics,
            "delta_macro_f1": float(candidate_metrics["macro_f1"])
            - float(reference_metrics["macro_f1"]),
        }
    return result


def _write_command(path: Path, argv: Sequence[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d6.run", *argv])
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
    decisions: Mapping[tuple[int, int], int],
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


def _validate_contract(config: dict[str, object]) -> None:
    feature_config = dict(config["features"])  # type: ignore[arg-type]
    calibration = dict(config["calibration"])  # type: ignore[arg-type]
    training = dict(config["training"])  # type: ignore[arg-type]
    stability = dict(config["stability"])  # type: ignore[arg-type]
    if tuple(feature_config["dialog_policy_names"]) != DIALOG_POLICY_NAMES:
        raise ValueError("D6 dialog feature names differ from preregistration")
    if tuple(feature_config["variants"]) != D6_VARIANTS:
        raise ValueError("D6 variants differ from preregistration")
    if str(feature_config["primary_variant"]) != PRIMARY_VARIANT:
        raise ValueError("D6 primary differs from preregistration")
    if tuple(feature_config["position_groups"]) != POSITION_GROUPS:
        raise ValueError("D6 position groups differ from preregistration")
    if tuple(feature_config["last_action_groups"]) != LAST_ACTION_GROUPS:
        raise ValueError("D6 last-action groups differ from preregistration")
    if tuple(feature_config["last2_groups"]) != LAST2_GROUPS:
        raise ValueError("D6 last-two groups differ from preregistration")
    expected_calibration = {
        "first_group_uses_global_threshold": True,
        "minimum_group_rows": 64,
        "require_both_classes": True,
        "shrinkage_effective_n": "2_times_min_class_count",
        "shrinkage_pseudocount": 256.0,
        "local_threshold_selection": "exact_group_macro_f1",
        "variants_reuse_d4_selected_model_and_l2": True,
    }
    if calibration != expected_calibration:
        raise ValueError("D6 calibration rule differs from preregistration")
    if training.get("d4_model_selection") != "exact_global_calibration_fold_macro_f1":
        raise ValueError("D6 D4 model-selection rule differs from preregistration")
    expected_seeds = [
        "d6-stability-20260721-a",
        "d6-stability-20260721-b",
        "d6-stability-20260721-c",
    ]
    if stability.get("enabled") is not True:
        raise ValueError("D6 stability runs must be enabled")
    if list(stability["split_seeds"]) != expected_seeds:  # type: ignore[arg-type]
        raise ValueError("D6 stability seeds differ from preregistration")
    if tuple(stability["variants"]) != ("d4_global_replay", PRIMARY_VARIANT):
        raise ValueError("D6 stability variants differ from preregistration")


def _fit_policies(
    *,
    examples: Sequence[LabeledChunk],
    values: np.ndarray,
    names: Sequence[str],
    stage_families: Mapping[str, Sequence[str]],
    training: dict[str, object],
    calibration: dict[str, object],
) -> tuple[
    dict[str, dict[tuple[int, int], int]],
    dict[str, list[dict[str, object]]],
    dict[tuple[int, int], float],
    dict[str, dict[tuple[int, int], float]],
]:
    return cross_validate_structured_calibration(
        examples,
        values,
        names,
        stage_families,
        folds=int(training["folds"]),
        calibration_fold_offset=int(training["calibration_fold_offset"]),
        seed=int(training["seed"]),
        max_iterations=int(training["max_iterations"]),
        l2_weights=[float(value) for value in training["l2_weights"]],  # type: ignore[union-attr]
        l2_reduction=str(training["l2_reduction"]),  # type: ignore[arg-type]
        minimum_group_rows=int(calibration["minimum_group_rows"]),
        shrinkage_pseudocount=float(calibration["shrinkage_pseudocount"]),
        pin_first=bool(calibration["first_group_uses_global_threshold"]),
    )


def _stability_runs(
    *,
    config: dict[str, object],
    source_rows: Sequence[dict[str, object]],
    label_free_rows: Sequence[dict[str, object]],
    r0_records: Sequence[dict[str, object]],
    cache: object,
    input_path: Path,
    starter_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    feature_config = dict(config["features"])  # type: ignore[arg-type]
    calibration = dict(config["calibration"])  # type: ignore[arg-type]
    training = dict(config["training"])  # type: ignore[arg-type]
    stability = dict(config["stability"])  # type: ignore[arg-type]
    evaluation = dict(config["evaluation"])  # type: ignore[arg-type]
    seeds = [str(value) for value in stability["split_seeds"]]  # type: ignore[union-attr]
    results: dict[str, object] = {}
    for split_index, split_seed in enumerate(seeds):
        split_dir = output_dir / split_seed
        split_dir.mkdir(parents=True)
        manifest = make_fold_manifest(
            label_free_rows,
            folds=int(training["folds"]),
            seed=split_seed,
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
        stages, previous_actions, stage_audit = build_structured_stages(
            label_free_rows, chunks
        )
        dialog_values, dialog_audit = build_dialog_policy_features(
            label_free_rows, chunks
        )
        if not np.array_equal(
            previous_actions, dialog_values[:, 1].astype(np.int8)
        ):
            raise ValueError("D6 stability stage action differs from D4 dialog action")
        examples = attach_gold_labels(chunks, source_rows)
        domains = sorted({example.feature.domain for example in examples})
        scalar_names = feature_names(str(feature_config["d1_scalar_variant"]), domains)
        values, names = dialog_control_matrix(
            examples,
            cache,  # type: ignore[arg-type]
            scalar_names,
            dialog_values,
            "d1_fused_plus_dialog_stage",
        )
        decisions, fold_details, _, _ = _fit_policies(
            examples=examples,
            values=values,
            names=names,
            stage_families=stages,
            training=training,
            calibration=calibration,
        )
        variant_metrics: dict[str, dict[str, object]] = {}
        for variant in ("d4_global_replay", PRIMARY_VARIANT):
            metrics, prediction_hash, metrics_hash = _score(
                source_rows=source_rows,
                examples=examples,
                decisions=decisions[variant],
                input_path=input_path,
                starter_dir=starter_dir,
                output_dir=split_dir / variant,
            )
            variant_metrics[variant] = {
                **dict(metrics["overall"]),  # type: ignore[arg-type]
                "predictions_sha256": prediction_hash,
                "metrics_sha256": metrics_hash,
            }
        reference = decisions["d4_global_replay"]
        primary = decisions[PRIMARY_VARIANT]
        delta = float(variant_metrics[PRIMARY_VARIANT]["macro_f1"]) - float(
            variant_metrics["d4_global_replay"]["macro_f1"]
        )
        bootstrap = paired_session_bootstrap(
            examples,
            primary,
            reference,
            repetitions=int(evaluation["bootstrap_repetitions"]),
            seed=int(evaluation["bootstrap_seed"]) + 200 + split_index,
        )
        result = {
            "seed": split_seed,
            "split_manifest_sha256": sha256_file(manifest_path),
            "variants": variant_metrics,
            "delta_macro_f1": delta,
            "paired_session_bootstrap": bootstrap,
            "fold_details": {
                "d4_global_replay": fold_details["d4_global_replay"],
                PRIMARY_VARIANT: fold_details[PRIMARY_VARIANT],
            },
            "stage_audit": stage_audit,
            "dialog_audit": dialog_audit,
        }
        write_json(split_dir / "comparison.json", result)
        results[split_seed] = result
        print(json.dumps({"stability": split_seed, "delta_macro_f1": delta}))
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
        raise FileExistsError(f"D6 output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    protocol = dict(config["protocol"])  # type: ignore[arg-type]
    prior_evidence = dict(config["prior_evidence"])  # type: ignore[arg-type]
    base_configs = dict(config["base_configs"])  # type: ignore[arg-type]
    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    source_config = dict(config["sources"])  # type: ignore[arg-type]
    feature_config = dict(config["features"])  # type: ignore[arg-type]
    calibration = dict(config["calibration"])  # type: ignore[arg-type]
    training = dict(config["training"])  # type: ignore[arg-type]
    references = dict(config["references"])  # type: ignore[arg-type]
    evaluation = dict(config["evaluation"])  # type: ignore[arg-type]
    promotion = dict(evaluation["promotion"])  # type: ignore[arg-type]

    protocol_path = _resolve(protocol["path"])
    _check_hash(protocol_path, protocol["sha256"])
    for group in (prior_evidence, base_configs):
        for value in group.values():
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
    _validate_contract(config)

    source_rows = load_jsonl(input_path)
    label_free_rows = strip_answers(source_rows)
    r0_records = load_jsonl(source_paths["r0_session_records"])
    split_manifest = _load_json(source_paths["split_manifest"])
    fold_by_index = validate_fold_manifest(split_manifest, label_free_rows)
    chunks = build_label_free_chunks(
        label_free_rows,
        r0_records,
        fold_by_index,
        max_history_turns=int(feature_config["max_history_turns"]),
        max_model_frames=int(feature_config["max_model_frames"]),
    )
    stage_families, previous_actions, stage_audit = build_structured_stages(
        label_free_rows, chunks
    )
    dialog_values, dialog_audit = build_dialog_policy_features(label_free_rows, chunks)
    if not np.array_equal(previous_actions, dialog_values[:, 1].astype(np.int8)):
        raise ValueError("D6 previous action differs from frozen D4 assistant addition")
    examples = attach_gold_labels(chunks, source_rows)
    cache = load_aligned_neural_cache(
        source_paths["neural_features"],
        examples,
        hidden_size=int(dict(source_config["neural_features"])["hidden_size"]),  # type: ignore[arg-type]
    )
    domains = sorted({example.feature.domain for example in examples})
    scalar_names = feature_names(str(feature_config["d1_scalar_variant"]), domains)
    d4_values, d4_names = dialog_control_matrix(
        examples,
        cache,
        scalar_names,
        dialog_values,
        "d1_fused_plus_dialog_stage",
    )
    if len(d4_names) != 1051:
        raise ValueError("D6 exact D4 matrix width changed")

    d4_predictions = load_jsonl(source_paths["d4_predictions"])
    frozen_d4_decisions = _decisions(d4_predictions, examples)
    d4_metrics = dict(_load_json(source_paths["d4_metrics"])["overall"])  # type: ignore[arg-type]
    if float(d4_metrics["macro_f1"]) != float(references["d4_macro_f1"]):
        raise ValueError("D6 frozen D4 metric differs from preregistration")

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
                PROJECT_ROOT / "src/proactive_d3/dialog_control_core.py",
                PROJECT_ROOT / "src/proactive_d6/core.py",
                PROJECT_ROOT / "src/proactive_d6/run.py",
                PROJECT_ROOT / "src/proactive_d6/tests/test_core.py",
            ],
        ),
    )
    write_json(
        output_dir / "stage_audit.json",
        {
            "structured_stages": stage_audit,
            "dialog": dialog_audit,
            "previous_action_matches_d4_dialog_addition": True,
            "d4_matrix_shape": list(d4_values.shape),
            "d4_feature_count": len(d4_names),
            "candidate_numeric_inputs_per_row": 1,
            "candidate_additional_feature_columns": 0,
            "candidate_duplicate_columns": 0,
        },
    )
    write_json(
        output_dir / "data_manifest.json",
        {
            "input": {"path": str(input_path), "sha256": fingerprints["input_sha256"]},
            "source_hashes": source_hashes,
            "starter_kit_sha256": fingerprints,
            "protocol_sha256": protocol["sha256"],
            "config_sha256": sha256_file(config_path),
            "stage_supervision": {
                "labels_read": False,
                "predictions_read": False,
                "future_chunks_read": False,
                "source": "answer-stripped organizer-visible causal prefix",
            },
            "head_and_calibration_supervision": config["validation_policy"],
            "external_data_used": False,
            "human_evaluation_used": False,
        },
    )

    decisions, fold_details, oof_logits, thresholds = _fit_policies(
        examples=examples,
        values=d4_values,
        names=d4_names,
        stage_families=stage_families,
        training=training,
        calibration=calibration,
    )
    variant_results: dict[str, dict[str, object]] = {}
    variant_diagnostics: dict[str, dict[str, object]] = {}
    baseline_reproduced = False
    for variant in D6_VARIANTS:
        variant_dir = output_dir / "variants" / variant
        metrics, prediction_hash, metrics_hash = _score(
            source_rows=source_rows,
            examples=examples,
            decisions=decisions[variant],
            input_path=input_path,
            starter_dir=starter_dir,
            output_dir=variant_dir,
        )
        if variant == "d4_global_replay":
            if decisions[variant] != frozen_d4_decisions:
                changed = sum(
                    decisions[variant][key] != frozen_d4_decisions[key]
                    for key in frozen_d4_decisions
                )
                raise ValueError(f"D6 failed to reproduce D4 decisions: {changed}")
            if prediction_hash != str(dict(source_config["d4_predictions"])["sha256"]):  # type: ignore[arg-type]
                raise ValueError("D6 D4 replay prediction hash differs")
            if metrics_hash != str(dict(source_config["d4_metrics"])["sha256"]):  # type: ignore[arg-type]
                raise ValueError("D6 D4 replay metric hash differs")
            baseline_reproduced = True
        elif not baseline_reproduced:
            raise ValueError("D6 candidate ran before exact D4 reproduction")

        overall = dict(metrics["overall"])  # type: ignore[arg-type]
        reference = decisions["d4_global_replay"]
        candidate = decisions[variant]
        macro = float(overall["macro_f1"])
        delta = macro - float(d4_metrics["macro_f1"])
        bootstrap = paired_session_bootstrap(
            examples,
            candidate,
            reference,
            repetitions=int(evaluation["bootstrap_repetitions"]),
            seed=int(evaluation["bootstrap_seed"]),
        )
        non_first = metrics_for_subset(examples, candidate, include_first=False)
        fold_comparison = _group_comparison(
            examples, candidate, reference, lambda example: str(example.feature.fold)
        )
        domain_comparison = _group_comparison(
            examples, candidate, reference, lambda example: example.feature.domain
        )
        position_comparison = _group_comparison(
            examples, candidate, reference, _position
        )
        stage_family = (
            "position"
            if variant == "position_shrunk"
            else "last_action"
            if variant == "last_action_shrunk"
            else "last2"
        )
        stage_by_key = {
            example.key: stage_families[stage_family][row_index]
            for row_index, example in enumerate(examples)
        }
        stage_comparison = (
            {}
            if variant == "d4_global_replay"
            else _group_comparison(
                examples,
                candidate,
                reference,
                lambda example: stage_by_key[example.key],
            )
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
                    "predicted_interrupt": candidate[example.key],
                    "d4_interrupt": reference[example.key],
                    "d4_logit": oof_logits[example.key],
                    "applied_threshold": thresholds[variant][example.key],
                    "position_stage": stage_families["position"][row_index],
                    "last_action_stage": stage_families["last_action"][row_index],
                    "last2_stage": stage_families["last2"][row_index],
                    "previous_action_interrupt": int(previous_actions[row_index]),
                }
                for row_index, example in enumerate(examples)
            ],
        )
        diagnostics = {
            "variant": variant,
            "primary_variant": variant == PRIMARY_VARIANT,
            "d4_feature_count": len(d4_names),
            "additional_feature_columns": 0,
            "maximum_stage_thresholds": 1
            if variant == "d4_global_replay"
            else len(
                POSITION_GROUPS
                if variant == "position_shrunk"
                else LAST_ACTION_GROUPS
                if variant == "last_action_shrunk"
                else LAST2_GROUPS
            ),
            "fold_details": fold_details[variant],
            "official_metrics": overall,
            "non_first_chunk_metrics": non_first,
            "delta_macro_f1_vs_d4": delta,
            "paired_session_bootstrap_vs_d4": bootstrap,
            "fold_comparison_vs_d4": fold_comparison,
            "domain_comparison_vs_d4": domain_comparison,
            "position_comparison_vs_d4": position_comparison,
            "stage_comparison_vs_d4": stage_comparison,
            "positive_folds": positive_folds,
            "positive_domains": positive_domains,
            "predictions_sha256": prediction_hash,
            "metrics_sha256": metrics_hash,
        }
        write_json(variant_dir / "diagnostics.json", diagnostics)
        result = {
            "macro_f1": overall["macro_f1"],
            "interrupt_f1": overall["interrupt_f1"],
            "silent_f1": overall["silent_f1"],
            "delta_macro_f1_vs_d4": delta,
            "non_first_macro_f1": non_first["macro_f1"],
            "paired_session_bootstrap_vs_d4": bootstrap,
            "positive_folds": positive_folds,
            "positive_domains": positive_domains,
            "maximum_stage_thresholds": diagnostics["maximum_stage_thresholds"],
            "predictions_sha256": prediction_hash,
        }
        variant_results[variant] = result
        variant_diagnostics[variant] = diagnostics
        print(json.dumps({"variant": variant, **result}, sort_keys=True))

    if not baseline_reproduced:
        raise ValueError("D6 did not complete exact D4 replay")

    by_key = {example.key: example for example in examples}
    gold_matches = 0
    non_first_count = 0
    for row_index, example in enumerate(examples):
        if example.feature.chunk_index == 0:
            continue
        previous = by_key[(example.feature.input_index, example.feature.chunk_index - 1)]
        gold_matches += int(
            bool(previous_actions[row_index]) == bool(previous.gold_interrupt)
        )
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
        "status": "complete preregistered D6 structured-calibration OOF and stability study",
        "classification": config["classification"],
        "baseline_reproduced_exactly": True,
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
        "variants": len(D6_VARIANTS),
        "stability_splits": len(dict(stability_results["splits"])),  # type: ignore[arg-type]
        "d4_head_parameters_per_fold": len(d4_names) + 1,
        "maximum_stage_thresholds": len(LAST2_GROUPS),
    }
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "状态：D6 冻结五折 OOF 与三组稳定性 split 已完成。",
                "",
                f"D4 精确复现：{baseline_reproduced}。",
                f"D4 Macro F1：{d4_metrics['macro_f1']}。",
                f"D6 primary Macro F1：{primary['macro_f1']}。",
                f"D6 primary delta：{primary['delta_macro_f1_vs_d4']}。",
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
        "--config",
        default="configs/d6_internvl35_1b_structured_calibration_oof_v1.json",
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
