"""Audit transport of the deployable D1 threshold across frozen OOF models."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

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

from .core import (
    LabeledChunk,
    attach_gold_labels,
    binary_metrics,
    build_label_free_chunks,
    feature_names,
    fit_linear_logistic,
    metrics_for_subset,
    paired_session_bootstrap,
    predict_logits,
    prediction_rows,
    select_threshold,
    strip_answers,
    validate_fold_manifest,
)
from .neural_core import load_aligned_neural_cache, neural_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "d1_internvl35_1b_threshold_robustness.json"


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


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d1.audit_threshold", *argv])
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
        PROJECT_ROOT / "configs" / "d1_internvl35_1b_neural_oof.json",
        PROJECT_ROOT / "configs" / "d1_internvl35_1b_neural_final.json",
        PROJECT_ROOT / "models" / "internvl35_1b_hf.json",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
        PROJECT_ROOT / "C1_SPEC.md",
        PROJECT_ROOT / "Agent.md",
    ]


def decisions_at_threshold(
    examples: Sequence[LabeledChunk],
    logits: dict[tuple[int, int], float],
    threshold: float,
) -> dict[tuple[int, int], int]:
    expected = {example.key for example in examples}
    if set(logits) != expected or not math.isfinite(threshold):
        raise ValueError("Threshold decisions require finite, complete OOF logits")
    return {key: int(value >= threshold) for key, value in logits.items()}


def _prediction_decisions(
    predictions: Sequence[dict[str, object]],
    examples: Sequence[LabeledChunk],
) -> dict[tuple[int, int], int]:
    decisions: dict[tuple[int, int], int] = {}
    for input_index, row in enumerate(predictions):
        answers = row.get("answers")
        if not isinstance(answers, list):
            raise ValueError(f"Prediction row {input_index} has no answers")
        for chunk_index, answer in enumerate(answers):
            decisions[(input_index, chunk_index)] = int(
                str(answer).lstrip().startswith(INTERRUPT_TAG)
            )
    if set(decisions) != {example.key for example in examples}:
        raise ValueError("Reference predictions do not cover all D1 examples")
    return decisions


def _group_metrics(
    examples: Sequence[LabeledChunk],
    decisions: dict[tuple[int, int], int],
    attribute: str,
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[LabeledChunk]] = {}
    for example in examples:
        value = getattr(example.feature, attribute)
        groups.setdefault(str(value), []).append(example)
    return {
        name: binary_metrics(
            [example.gold_interrupt for example in selected],
            [decisions[example.key] for example in selected],
        )
        for name, selected in sorted(groups.items())
    }


def threshold_gate(
    reference: dict[str, float | int],
    unified: dict[str, float | int],
    fold_deltas: Sequence[float],
    local_sweep_deltas: Sequence[float],
    bootstrap: dict[str, object],
    gate: dict[str, object],
) -> dict[str, object]:
    if not fold_deltas or not local_sweep_deltas:
        raise ValueError("Threshold gate requires fold and local-sweep results")
    checks = {
        "overall_macro_drop": (
            float(unified["macro_f1"]) - float(reference["macro_f1"])
            >= -float(gate["max_overall_macro_f1_drop"])
        ),
        "worst_fold_macro_drop": min(fold_deltas)
        >= -float(gate["max_worst_fold_macro_f1_drop"]),
        "local_threshold_plateau": min(local_sweep_deltas)
        >= -float(gate["max_local_offset_macro_f1_drop"]),
        "bootstrap_lower_bound": float(bootstrap["delta_macro_f1_p2_5"])
        >= float(gate["min_bootstrap_delta_p2_5"]),
        "both_class_f1": min(
            float(unified["interrupt_f1"]), float(unified["silent_f1"])
        )
        >= float(gate["min_class_f1"]),
    }
    return {"passed": all(checks.values()), "checks": checks, "criteria": gate}


def _reproduce_fold_models(
    examples: Sequence[LabeledChunk],
    values: np.ndarray,
    names: Sequence[str],
    split: dict[str, object],
    training: dict[str, object],
    frozen_fold_details: Sequence[dict[str, object]],
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], float],
    dict[int, float],
    list[dict[str, object]],
]:
    folds = int(split["folds"])
    offset = int(split["calibration_fold_offset"])
    seed = int(training["seed"])
    max_iterations = int(training["max_iterations"])
    l2_weights = [float(value) for value in training["l2_weights"]]  # type: ignore[arg-type]
    l2_reduction = str(training["l2_reduction"])
    labels = np.asarray([example.gold_interrupt for example in examples], dtype=np.int64)
    fold_values = np.asarray([example.feature.fold for example in examples], dtype=np.int64)
    decisions: dict[tuple[int, int], int] = {}
    logits: dict[tuple[int, int], float] = {}
    thresholds: dict[int, float] = {}
    reproduced: list[dict[str, object]] = []
    if len(frozen_fold_details) != folds:
        raise ValueError("Frozen D1 diagnostics do not contain every fold")

    for test_fold in range(folds):
        calibration_fold = (test_fold + offset) % folds
        fit_indices = np.flatnonzero(
            (fold_values != test_fold) & (fold_values != calibration_fold)
        )
        calibration_indices = np.flatnonzero(fold_values == calibration_fold)
        test_indices = np.flatnonzero(fold_values == test_fold)
        candidates: list[tuple[float, object, float, dict[str, float | int]]] = []
        for grid_index, l2_weight in enumerate(l2_weights):
            model = fit_linear_logistic(
                values[fit_indices],
                labels[fit_indices],
                seed=seed + test_fold * 100 + grid_index,
                max_iterations=max_iterations,
                l2_weight=l2_weight,
                l2_reduction=l2_reduction,  # type: ignore[arg-type]
            )
            calibration_logits = predict_logits(model, values[calibration_indices])
            threshold, metrics = select_threshold(
                calibration_logits, labels[calibration_indices].tolist()
            )
            candidates.append((l2_weight, model, threshold, metrics))
        selected_l2, model, threshold, calibration_metrics = max(
            candidates,
            key=lambda item: (
                float(item[3]["macro_f1"]),
                -item[0],
                -abs(item[2]),
            ),
        )
        frozen = frozen_fold_details[test_fold]
        if int(frozen["test_fold"]) != test_fold:
            raise ValueError("Frozen fold details are not in test-fold order")
        if selected_l2 != float(frozen["selected_l2_weight"]):
            raise ValueError(f"Fold {test_fold} selected L2 changed")
        if not math.isclose(
            threshold, float(frozen["threshold_logit"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"Fold {test_fold} calibration threshold changed")

        test_logits = predict_logits(model, values[test_indices])
        test_predictions = [int(logit >= threshold) for logit in test_logits]
        for index, logit, decision in zip(
            test_indices.tolist(), test_logits, test_predictions
        ):
            key = examples[index].key
            if key in logits:
                raise ValueError(f"Duplicate reproduced OOF logit for {key}")
            logits[key] = float(logit)
            decisions[key] = decision
        thresholds[test_fold] = threshold
        reproduced.append(
            {
                "test_fold": test_fold,
                "calibration_fold": calibration_fold,
                "selected_l2_weight": selected_l2,
                "threshold_logit": threshold,
                "calibration_metrics": calibration_metrics,
                "test_metrics": binary_metrics(
                    labels[test_indices].tolist(), test_predictions
                ),
            }
        )
    expected = {example.key for example in examples}
    if set(decisions) != expected or set(logits) != expected:
        raise ValueError("Reproduced OOF outputs do not cover every chunk")
    return decisions, logits, thresholds, reproduced


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    args = parser.parse_args(raw_argv)
    started_at = time.monotonic()

    config_path = _resolve(args.config)
    config = _load_json(config_path)
    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    r0_reference = dict(config["r0_reference"])  # type: ignore[arg-type]
    scalar_reference = dict(config["scalar_oof_reference"])  # type: ignore[arg-type]
    cache_config = dict(config["neural_cache"])  # type: ignore[arg-type]
    oof_reference = dict(config["neural_oof_reference"])  # type: ignore[arg-type]
    final_reference = dict(config["final_head_reference"])  # type: ignore[arg-type]
    feature_config = dict(config["features"])  # type: ignore[arg-type]
    split_config = dict(config["split"])  # type: ignore[arg-type]
    training_config = dict(config["training_reproduction"])  # type: ignore[arg-type]
    audit_config = dict(config["audit"])  # type: ignore[arg-type]

    input_path = _resolve(data_config["input"])
    starter_dir = _resolve(starter_config["path"])
    r0_dir = _resolve(r0_reference["experiment_dir"])
    scalar_dir = _resolve(scalar_reference["experiment_dir"])
    cache_dir = _resolve(cache_config["path"])
    cache_path = cache_dir / "features.npz"
    oof_dir = _resolve(oof_reference["experiment_dir"])
    variant_dir = oof_dir / "variants" / str(oof_reference["variant"])
    final_dir = _resolve(final_reference["experiment_dir"])
    output_dir = _resolve(args.output_dir or f"output/experiments/{config['experiment_id']}")
    if output_dir.exists():
        raise FileExistsError(f"D1 threshold audit output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    fingerprints = _validate_static_files(config, input_path, starter_dir)
    hashes = {
        "r0_session_records_sha256": _check_hash(
            r0_dir / "session_records.jsonl", r0_reference["session_records_sha256"]
        ),
        "split_manifest_sha256": _check_hash(
            scalar_dir / "split_manifest.json", scalar_reference["split_manifest_sha256"]
        ),
        "scalar_predictions_sha256": _check_hash(
            scalar_dir
            / "variants"
            / str(scalar_reference["feature_variant"])
            / "predictions.jsonl",
            scalar_reference["predictions_sha256"],
        ),
        "neural_cache_sha256": _check_hash(
            cache_path, cache_config["features_sha256"]
        ),
        "oof_diagnostics_sha256": _check_hash(
            variant_dir / "diagnostics.json", oof_reference["diagnostics_sha256"]
        ),
        "oof_records_sha256": _check_hash(
            variant_dir / "oof_records.jsonl", oof_reference["oof_records_sha256"]
        ),
        "oof_predictions_sha256": _check_hash(
            variant_dir / "predictions.jsonl", oof_reference["predictions_sha256"]
        ),
        "oof_metrics_sha256": _check_hash(
            variant_dir / "metrics.json", oof_reference["metrics_sha256"]
        ),
        "final_head_sha256": _check_hash(
            final_dir / "decision_head.json", final_reference["decision_head_sha256"]
        ),
    }
    cache_summary = _load_json(cache_dir / "summary.json")
    if cache_summary.get("labels_read_or_stored") is not False:
        raise ValueError("D1 threshold audit requires the frozen label-free cache")

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
    cache = load_aligned_neural_cache(
        cache_path, examples, hidden_size=int(cache_config["hidden_size"])
    )
    domains = sorted({example.feature.domain for example in examples})
    scalar_names = feature_names(
        str(feature_config["scalar_variant"]), domains  # type: ignore[arg-type]
    )
    values, names = neural_matrix(
        examples,
        cache,
        scalar_names,
        str(feature_config["variant"]),  # type: ignore[arg-type]
    )
    frozen_diagnostics = _load_json(variant_dir / "diagnostics.json")
    frozen_fold_details = frozen_diagnostics.get("fold_details")
    if not isinstance(frozen_fold_details, list):
        raise ValueError("Frozen D1 diagnostics have no fold details")

    fold_decisions, logits, thresholds, reproduced_folds = _reproduce_fold_models(
        examples,
        values,
        names,
        split_config,
        training_config,
        frozen_fold_details,  # type: ignore[arg-type]
    )
    reproduced_predictions = prediction_rows(examples, fold_decisions)
    validate_prediction_rows(source_rows, reproduced_predictions)
    reproduced_path = output_dir / "fold_calibrated_predictions.jsonl"
    write_jsonl(reproduced_path, reproduced_predictions)
    reproduced_hash = sha256_file(reproduced_path)
    if reproduced_hash != str(oof_reference["predictions_sha256"]):
        raise ValueError("Reproduced fold-calibrated predictions are not byte-identical")
    reproduced_metrics_path = output_dir / "fold_calibrated_metrics.json"
    _run_official_scorer(
        starter_dir,
        input_path,
        reproduced_path,
        reproduced_metrics_path,
        output_dir / "fold_calibrated_scorer.log",
    )

    final_head_payload = _load_json(final_dir / "decision_head.json")
    final_threshold = float(final_head_payload["threshold_logit"])
    median_threshold = float(statistics.median(thresholds.values()))
    if not math.isclose(final_threshold, median_threshold, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Final D1 threshold is not the frozen fold-threshold median")
    unified_decisions = decisions_at_threshold(examples, logits, final_threshold)
    unified_predictions = prediction_rows(examples, unified_decisions)
    validation = validate_prediction_rows(source_rows, unified_predictions)
    unified_path = output_dir / "predictions.jsonl"
    write_jsonl(unified_path, unified_predictions)
    unified_metrics_path = output_dir / "metrics.json"
    _run_official_scorer(
        starter_dir,
        input_path,
        unified_path,
        unified_metrics_path,
        output_dir / "scorer.log",
    )

    reference_internal = metrics_for_subset(examples, fold_decisions, include_first=True)
    unified_internal = metrics_for_subset(examples, unified_decisions, include_first=True)
    reference_by_fold = _group_metrics(examples, fold_decisions, "fold")
    unified_by_fold = _group_metrics(examples, unified_decisions, "fold")
    fold_deltas = {
        fold: float(unified_by_fold[fold]["macro_f1"])
        - float(reference_by_fold[fold]["macro_f1"])
        for fold in reference_by_fold
    }
    scalar_predictions = load_jsonl(
        scalar_dir
        / "variants"
        / str(scalar_reference["feature_variant"])
        / "predictions.jsonl"
    )
    scalar_decisions = _prediction_decisions(scalar_predictions, examples)
    bootstrap_reference = paired_session_bootstrap(
        examples,
        unified_decisions,
        fold_decisions,
        repetitions=int(audit_config["bootstrap_repetitions"]),
        seed=int(audit_config["bootstrap_seed"]),
    )
    bootstrap_scalar = paired_session_bootstrap(
        examples,
        unified_decisions,
        scalar_decisions,
        repetitions=int(audit_config["bootstrap_repetitions"]),
        seed=int(audit_config["bootstrap_seed"]),
    )

    sweep: list[dict[str, object]] = []
    offsets = audit_config.get("threshold_offsets")
    if not isinstance(offsets, list):
        raise ValueError("Threshold offsets must be a JSON list")
    for offset_value in offsets:
        offset = float(offset_value)
        threshold = final_threshold + offset
        decisions = decisions_at_threshold(examples, logits, threshold)
        metrics = metrics_for_subset(examples, decisions, include_first=True)
        sweep.append(
            {
                "offset": offset,
                "threshold_logit": threshold,
                "metrics": metrics,
                "delta_macro_f1_vs_fold_calibrated": float(metrics["macro_f1"])
                - float(reference_internal["macro_f1"]),
            }
        )
    gate_config = dict(audit_config["gate"])  # type: ignore[arg-type]
    local_radius = float(gate_config["local_offset_radius"])
    local_sweep_deltas = [
        float(item["delta_macro_f1_vs_fold_calibrated"])
        for item in sweep
        if abs(float(item["offset"])) <= local_radius + 1e-12
    ]
    gate_result = threshold_gate(
        reference_internal,
        unified_internal,
        list(fold_deltas.values()),
        local_sweep_deltas,
        bootstrap_reference,
        gate_config,
    )

    write_jsonl(
        output_dir / "oof_logits.jsonl",
        [
            {
                "input_index": example.feature.input_index,
                "video_path": example.feature.video_path,
                "domain": example.feature.domain,
                "fold": example.feature.fold,
                "chunk_index": example.feature.chunk_index,
                "gold_interrupt": example.gold_interrupt,
                "oof_logit": logits[example.key],
                "fold_threshold_logit": thresholds[example.feature.fold],
                "unified_threshold_logit": final_threshold,
                "fold_calibrated_interrupt": fold_decisions[example.key],
                "unified_threshold_interrupt": unified_decisions[example.key],
                "scalar_interrupt": scalar_decisions[example.key],
            }
            for example in examples
        ],
    )
    result = {
        **validation,
        "classification": config["validation_policy"],
        "protocol_disclosure": audit_config["protocol_disclosure"],
        "feature_count": len(names),
        "fold_thresholds": thresholds,
        "threshold_min": min(thresholds.values()),
        "threshold_max": max(thresholds.values()),
        "threshold_range": max(thresholds.values()) - min(thresholds.values()),
        "final_unified_threshold": final_threshold,
        "final_threshold_equals_fold_median": True,
        "fold_reproduction": reproduced_folds,
        "fold_calibrated_internal_metrics": reference_internal,
        "unified_threshold_internal_metrics": unified_internal,
        "delta_macro_f1": float(unified_internal["macro_f1"])
        - float(reference_internal["macro_f1"]),
        "fold_calibrated_by_fold": reference_by_fold,
        "unified_threshold_by_fold": unified_by_fold,
        "unified_delta_by_fold": fold_deltas,
        "unified_by_domain": _group_metrics(examples, unified_decisions, "domain"),
        "fold_calibrated_non_first": metrics_for_subset(
            examples, fold_decisions, include_first=False
        ),
        "unified_non_first": metrics_for_subset(
            examples, unified_decisions, include_first=False
        ),
        "paired_session_bootstrap_vs_fold_calibrated": bootstrap_reference,
        "paired_session_bootstrap_vs_scalar": bootstrap_scalar,
        "threshold_sweep": sweep,
        "gate": gate_result,
        "fold_calibrated_predictions_sha256": reproduced_hash,
        "unified_predictions_sha256": sha256_file(unified_path),
        "fold_calibrated_metrics_sha256": sha256_file(reproduced_metrics_path),
        "unified_metrics_sha256": sha256_file(unified_metrics_path),
    }
    write_json(output_dir / "audit.json", result)
    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "gpu_used": False,
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
            "frozen_artifacts": hashes,
            "starter_kit_sha256": fingerprints,
            "supervision": config["validation_policy"],
        },
    )
    runtime = {
        "status": "complete threshold robustness audit",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started_at, 3),
        "gpu_used": False,
        "model_inference_rerun": False,
        "sessions": len(source_rows),
        "chunks": len(examples),
        "gate_passed": gate_result["passed"],
    }
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "状态：**D1 单一部署阈值稳健性审计完成**。",
                "",
                "该实验重新拟合冻结的五折线性模型，并先验证原始 OOF 预测逐字节复现。",
                "统一阈值来自既有 final head，不使用本次全量 OOF 指标重新选阈值。",
                f"审计门槛：`{'PASS' if gate_result['passed'] else 'FAIL'}`。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"gate": gate_result, "delta_macro_f1": result["delta_macro_f1"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
