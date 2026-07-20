"""Run the preregistered CPU-only D3 dialog-policy mechanism control."""

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
from proactive_d3.dialog_control_core import (
    DIALOG_POLICY_NAMES,
    VARIANTS,
    build_dialog_policy_features,
    dialog_control_matrix,
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
        raise ValueError(f"Frozen D3-D artifact mismatch for {path}: {actual} != {expected}")
    return actual


def _decisions(
    predictions: Sequence[dict[str, object]], examples: Sequence[LabeledChunk]
) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for input_index, row in enumerate(predictions):
        answers = row.get("answers")
        if not isinstance(answers, list):
            raise ValueError(f"D3-D prediction is malformed at {input_index}")
        for chunk_index, answer in enumerate(answers):
            result[(input_index, chunk_index)] = int(
                str(answer).startswith(INTERRUPT_TAG)
            )
    if set(result) != {example.key for example in examples}:
        raise ValueError("D3-D reference decisions do not cover every example")
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


def _explanation_band(fraction: float, evaluation: dict[str, object]) -> str:
    if fraction >= float(evaluation["mostly_explained_fraction"]):
        return "mostly_reconstructed_by_dialog_policy"
    if fraction >= float(evaluation["partly_explained_fraction"]):
        return "partly_reconstructed_by_dialog_policy"
    return "little_reconstructed_by_dialog_policy"


def _write_command(path: Path, argv: Sequence[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d3.run_dialog_control", *argv])
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


def run(config_path: Path, output_dir: Path, raw_argv: Sequence[str]) -> dict[str, object]:
    started = time.monotonic()
    config = _load_json(config_path)
    if output_dir.exists():
        raise FileExistsError(f"D3-D output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    protocol = dict(config["protocol"])
    base_config = dict(config["base_d3_config"])
    data_config = dict(config["data"])
    starter_config = dict(config["starter_kit"])
    source_config = dict(config["sources"])
    feature_config = dict(config["features"])
    training = dict(config["training"])
    references = dict(config["references"])
    evaluation = dict(config["evaluation"])

    protocol_path = _resolve(protocol["path"])
    _check_hash(protocol_path, protocol["sha256"])
    _check_hash(_resolve(base_config["path"]), base_config["sha256"])
    input_path = _resolve(data_config["input"])
    starter_dir = _resolve(starter_config["path"])
    fingerprints = _validate_static_files(config, input_path, starter_dir)
    source_paths = {
        name: _resolve(value["path"]) for name, value in source_config.items()
    }
    source_hashes = {
        name: _check_hash(source_paths[name], value["sha256"])
        for name, value in source_config.items()
    }
    if tuple(feature_config["dialog_policy_names"]) != DIALOG_POLICY_NAMES:
        raise ValueError("D3-D feature names differ from preregistration")
    if tuple(feature_config["variants"]) != VARIANTS:
        raise ValueError("D3-D variants differ from preregistration")

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
    dialog_values, feature_audit = build_dialog_policy_features(
        label_free_rows, label_free_chunks
    )
    examples = attach_gold_labels(label_free_chunks, source_rows)
    cache = load_aligned_neural_cache(
        source_paths["neural_features"],
        examples,
        hidden_size=int(source_config["neural_features"]["hidden_size"]),
    )
    domains = sorted({example.feature.domain for example in examples})
    scalar_names = feature_names(str(feature_config["d1_scalar_variant"]), domains)

    d1_predictions = load_jsonl(source_paths["d1_predictions"])
    d3_predictions = load_jsonl(source_paths["d3_predictions"])
    d1_decisions = _decisions(d1_predictions, examples)
    d3_decisions = _decisions(d3_predictions, examples)
    d1_metrics = _load_json(source_paths["d1_metrics"])["overall"]
    d3_metrics = _load_json(source_paths["d3_metrics"])["overall"]
    if float(d1_metrics["macro_f1"]) != float(references["d1_macro_f1"]):
        raise ValueError("D3-D frozen D1 metric differs from preregistration")
    if float(d3_metrics["macro_f1"]) != float(references["d3_macro_f1"]):
        raise ValueError("D3-D frozen D3 metric differs from preregistration")
    reference_gain = float(d3_metrics["macro_f1"]) - float(d1_metrics["macro_f1"])
    if abs(reference_gain - float(references["d3_gain_vs_d1"])) > 1e-12:
        raise ValueError("D3-D frozen D3 gain differs from preregistration")

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
                PROJECT_ROOT / "src/proactive_d1/core.py",
                PROJECT_ROOT / "src/proactive_d1/neural_core.py",
                PROJECT_ROOT / "src/proactive_d3/dialog_control_core.py",
                PROJECT_ROOT / "src/proactive_d3/run_dialog_control.py",
                PROJECT_ROOT / "src/proactive_d3/tests/test_dialog_control_core.py",
            ],
        ),
    )
    write_json(output_dir / "feature_audit.json", feature_audit)
    write_json(
        output_dir / "data_manifest.json",
        {
            "input": {"path": str(input_path), "sha256": fingerprints["input_sha256"]},
            "source_hashes": source_hashes,
            "starter_kit_sha256": fingerprints,
            "protocol_sha256": protocol["sha256"],
            "dialog_feature_supervision": {
                "labels_read": False,
                "predictions_read": False,
                "future_chunks_read": False,
                "source": "answer-stripped official dialog prefix",
            },
            "head_supervision": config["validation_policy"],
            "external_data_used": False,
        },
    )

    variant_results: dict[str, object] = {}
    baseline_reproduced = False
    max_head_parameters = 0
    for variant in VARIANTS:
        values, names = dialog_control_matrix(
            examples, cache, scalar_names, dialog_values, variant
        )
        max_head_parameters = max(max_head_parameters, len(names) + 1)
        decisions, fold_details = cross_validate_neural_matrix(
            examples,
            values,
            names,
            folds=int(training["folds"]),
            calibration_fold_offset=int(training["calibration_fold_offset"]),
            seed=int(training["seed"]),
            max_iterations=int(training["max_iterations"]),
            l2_weights=[float(value) for value in training["l2_weights"]],
            l2_reduction=str(training["l2_reduction"]),
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
                changed = sum(decisions[key] != d1_decisions[key] for key in decisions)
                raise ValueError(f"D3-D failed to reproduce D1 decisions: {changed}")
            if prediction_hash != source_config["d1_predictions"]["sha256"]:
                raise ValueError("D3-D D1 replay prediction hash differs")
            baseline_reproduced = True
        elif not baseline_reproduced:
            raise ValueError("D3-D controls ran before exact D1 reproduction")

        metrics_path = variant_dir / "metrics.json"
        _run_official_scorer(
            starter_dir,
            input_path,
            predictions_path,
            metrics_path,
            variant_dir / "scorer.log",
        )
        metrics = _load_json(metrics_path)
        overall = dict(metrics["overall"])
        if variant == "d1_fused_replay" and sha256_file(metrics_path) != source_config[
            "d1_metrics"
        ]["sha256"]:
            raise ValueError("D3-D D1 replay metric hash differs")
        candidate_macro = float(overall["macro_f1"])
        delta_d1 = candidate_macro - float(d1_metrics["macro_f1"])
        captured = delta_d1 / reference_gain
        bootstrap = paired_session_bootstrap(
            examples,
            decisions,
            d1_decisions,
            repetitions=int(evaluation["bootstrap_repetitions"]),
            seed=int(evaluation["bootstrap_seed"]),
        )
        agreement_d3 = sum(
            decisions[key] == d3_decisions[key] for key in decisions
        ) / len(decisions)
        non_first = metrics_for_subset(examples, decisions, include_first=False)
        fold_comparison = _group_comparison(
            examples, decisions, d1_decisions, lambda example: str(example.feature.fold)
        )
        domain_comparison = _group_comparison(
            examples, decisions, d1_decisions, lambda example: example.feature.domain
        )
        position_comparison = _group_comparison(
            examples, decisions, d1_decisions, _position
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
                    "d3_interrupt": d3_decisions[example.key],
                    **{
                        name: float(dialog_values[row_index, index])
                        for index, name in enumerate(DIALOG_POLICY_NAMES)
                    },
                }
                for row_index, example in enumerate(examples)
            ],
        )
        diagnostics = {
            **validation,
            "variant": variant,
            "promotion_eligible": False,
            "feature_count": len(names),
            "head_parameters_per_fold": len(names) + 1,
            "fold_details": fold_details,
            "official_metrics": overall,
            "non_first_chunk_metrics": non_first,
            "delta_macro_f1_vs_d1": delta_d1,
            "delta_macro_f1_vs_d3": candidate_macro - float(d3_metrics["macro_f1"]),
            "captured_d3_gain": captured,
            "explanation_band": _explanation_band(captured, evaluation),
            "stable_positive_vs_d1": float(bootstrap["delta_macro_f1_p2_5"]) > 0,
            "paired_session_bootstrap_vs_d1": bootstrap,
            "decision_agreement_with_d3": agreement_d3,
            "fold_comparison_vs_d1": fold_comparison,
            "domain_comparison_vs_d1": domain_comparison,
            "position_comparison_vs_d1": position_comparison,
            "predictions_sha256": prediction_hash,
            "metrics_sha256": sha256_file(metrics_path),
        }
        write_json(variant_dir / "diagnostics.json", diagnostics)
        variant_results[variant] = {
            "macro_f1": overall["macro_f1"],
            "interrupt_f1": overall["interrupt_f1"],
            "silent_f1": overall["silent_f1"],
            "delta_macro_f1_vs_d1": delta_d1,
            "delta_macro_f1_vs_d3": diagnostics["delta_macro_f1_vs_d3"],
            "captured_d3_gain": captured,
            "explanation_band": diagnostics["explanation_band"],
            "stable_positive_vs_d1": diagnostics["stable_positive_vs_d1"],
            "non_first_macro_f1": non_first["macro_f1"],
            "decision_agreement_with_d3": agreement_d3,
            "paired_session_bootstrap_vs_d1": bootstrap,
            "feature_count": len(names),
            "head_parameters_per_fold": len(names) + 1,
            "promotion_eligible": False,
            "predictions_sha256": prediction_hash,
        }
        print(json.dumps({"variant": variant, **variant_results[variant]}, sort_keys=True))

    by_key = {example.key: example for example in examples}
    matches = 0
    non_first_count = 0
    for row_index, example in enumerate(examples):
        if example.feature.chunk_index == 0:
            continue
        previous = by_key[(example.feature.input_index, example.feature.chunk_index - 1)]
        visible_addition = bool(dialog_values[row_index, 1])
        matches += int(visible_addition == bool(previous.gold_interrupt))
        non_first_count += 1
    cross_check = {
        "runs_after_all_oof_predictions": True,
        "assistant_added_equals_previous_gold_interrupt": matches,
        "non_first_chunks": non_first_count,
        "agreement_rate": matches / non_first_count,
    }
    comparison = {
        "status": "complete preregistered D3 dialog-policy mechanism control",
        "classification": config["classification"],
        "baseline_reproduced_exactly": baseline_reproduced,
        "d1_reference": d1_metrics,
        "d3_reference": d3_metrics,
        "d3_gain_vs_d1": reference_gain,
        "variants": variant_results,
        "previous_gold_cross_check": cross_check,
        "interpretation_controls": {
            name: variant_results[name]
            for name in (
                "d1_fused_plus_dialog_increment",
                "d1_fused_plus_dialog_stage",
            )
        },
        "all_variants_promotion_eligible": False,
        "residual_is_not_automatically_visual_understanding": True,
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
        "max_head_parameters": max_head_parameters,
    }
    write_json(output_dir / "runtime.json", runtime)
    increment = variant_results["d1_fused_plus_dialog_increment"]
    stage = variant_results["d1_fused_plus_dialog_stage"]
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "状态：已完成预注册的 CPU-only D3 dialog-policy 机制控制。",
                "",
                "所有结果均为公共 validation 上的 val-supervised OOF 诊断，不可晋级。",
                f"D1 精确复现：{baseline_reproduced}。",
                f"D1 + dialog increment Macro F1：{increment['macro_f1']}。",
                f"D1 + dialog stage Macro F1：{stage['macro_f1']}。",
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
        "--config", default="configs/d3_internvl35_1b_dialog_policy_control_v1.json"
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
