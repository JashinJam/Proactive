"""Run the preregistered D2 residual MLP on the frozen D1 OOF protocol."""

from __future__ import annotations

import argparse
import json
import math
import shlex
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

from proactive_d1.core import (
    LabeledChunk,
    LinearModel,
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
from proactive_d1.neural_core import load_aligned_neural_cache, neural_matrix

from .core import (
    ResidualDecisionHead,
    fit_residual_mlp,
    predict_residual_logits,
    residual_parameter_count,
    serialize_residual_head,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "d2_internvl35_1b_residual_mlp_oof.json"


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
        raise ValueError(f"Frozen D2 dependency mismatch for {path}: {actual} != {expected}")
    return actual


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d2.run_mlp", *argv])
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
        *sorted((PROJECT_ROOT / "src" / "proactive_d2").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d2" / "tests").glob("*.py")),
        config_path,
        PROJECT_ROOT / "configs" / "d1_internvl35_1b_neural_oof.json",
        PROJECT_ROOT / "models" / "internvl35_1b_hf.json",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
        PROJECT_ROOT / "C1_SPEC.md",
        PROJECT_ROOT / "Agent.md",
    ]


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
        raise ValueError("Reference predictions do not cover all D2 examples")
    return decisions


def _group_metrics(
    examples: Sequence[LabeledChunk],
    decisions: dict[tuple[int, int], int],
    attribute: str,
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[LabeledChunk]] = {}
    for example in examples:
        groups.setdefault(str(getattr(example.feature, attribute)), []).append(example)
    return {
        name: binary_metrics(
            [example.gold_interrupt for example in selected],
            [decisions[example.key] for example in selected],
        )
        for name, selected in sorted(groups.items())
    }


def _select_linear_base(
    fit_values: np.ndarray,
    fit_labels: np.ndarray,
    calibration_values: np.ndarray,
    calibration_labels: np.ndarray,
    *,
    test_fold: int,
    config: dict[str, object],
) -> tuple[LinearModel, float, float, dict[str, float | int], list[dict[str, object]]]:
    l2_values = config.get("l2_weights")
    if not isinstance(l2_values, list) or not l2_values:
        raise ValueError("D2 base L2 grid must be a non-empty list")
    candidates: list[
        tuple[float, LinearModel, float, dict[str, float | int]]
    ] = []
    grid: list[dict[str, object]] = []
    for grid_index, l2_value in enumerate(l2_values):
        l2_weight = float(l2_value)
        model = fit_linear_logistic(
            fit_values,
            fit_labels,
            seed=int(config["seed"]) + test_fold * 100 + grid_index,
            max_iterations=int(config["max_iterations"]),
            l2_weight=l2_weight,
            l2_reduction=str(config["l2_reduction"]),  # type: ignore[arg-type]
        )
        calibration_logits = predict_logits(model, calibration_values)
        threshold, metrics = select_threshold(
            calibration_logits, calibration_labels.tolist()
        )
        candidates.append((l2_weight, model, threshold, metrics))
        grid.append(
            {
                "l2_weight": l2_weight,
                "threshold_logit": threshold,
                "calibration_metrics": metrics,
            }
        )
    selected_l2, model, threshold, metrics = max(
        candidates,
        key=lambda item: (
            float(item[3]["macro_f1"]),
            -item[0],
            -abs(item[2]),
        ),
    )
    return model, selected_l2, threshold, metrics, grid


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
    model_config = dict(config["model"])  # type: ignore[arg-type]
    r0_reference = dict(config["r0_reference"])  # type: ignore[arg-type]
    scalar_reference = dict(config["scalar_oof_reference"])  # type: ignore[arg-type]
    cache_config = dict(config["neural_cache"])  # type: ignore[arg-type]
    d1_reference = dict(config["d1_reference"])  # type: ignore[arg-type]
    threshold_reference = dict(config["threshold_audit_reference"])  # type: ignore[arg-type]
    feature_config = dict(config["features"])  # type: ignore[arg-type]
    split_config = dict(config["split"])  # type: ignore[arg-type]
    base_config = dict(config["base_linear_reproduction"])  # type: ignore[arg-type]
    residual_config = dict(config["residual_mlp"])  # type: ignore[arg-type]
    evaluation_config = dict(config["evaluation"])  # type: ignore[arg-type]
    promotion_config = dict(evaluation_config["promotion"])  # type: ignore[arg-type]

    if residual_config.get("activation") != "gelu":
        raise ValueError("This preregistered D2 runner only supports GELU")
    if residual_config.get("output_initialization") != "zeros":
        raise ValueError("D2 residual output must start at zero")
    if residual_config.get("device") != "cpu":
        raise ValueError("This D2 experiment is preregistered as CPU-only")
    import torch

    torch.set_num_threads(int(residual_config["torch_threads"]))

    input_path = _resolve(data_config["input"])
    starter_dir = _resolve(starter_config["path"])
    r0_dir = _resolve(r0_reference["experiment_dir"])
    scalar_dir = _resolve(scalar_reference["experiment_dir"])
    cache_dir = _resolve(cache_config["path"])
    cache_path = cache_dir / "features.npz"
    d1_dir = _resolve(d1_reference["experiment_dir"])
    d1_variant_dir = d1_dir / "variants" / str(d1_reference["variant"])
    threshold_dir = _resolve(threshold_reference["experiment_dir"])
    output_dir = _resolve(args.output_dir or f"output/experiments/{config['experiment_id']}")
    if output_dir.exists():
        raise FileExistsError(f"D2 output already exists: {output_dir}")
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
        "d1_diagnostics_sha256": _check_hash(
            d1_variant_dir / "diagnostics.json", d1_reference["diagnostics_sha256"]
        ),
        "d1_predictions_sha256": _check_hash(
            d1_variant_dir / "predictions.jsonl", d1_reference["predictions_sha256"]
        ),
        "d1_metrics_sha256": _check_hash(
            d1_variant_dir / "metrics.json", d1_reference["metrics_sha256"]
        ),
        "threshold_audit_sha256": _check_hash(
            threshold_dir / "audit.json", threshold_reference["audit_sha256"]
        ),
    }
    threshold_audit = _load_json(threshold_dir / "audit.json")
    threshold_gate = threshold_audit.get("gate")
    if not isinstance(threshold_gate, dict) or threshold_gate.get("passed") is not True:
        raise ValueError("D2 requires a passing D1 deployment-threshold audit")
    cache_summary = _load_json(cache_dir / "summary.json")
    if cache_summary.get("labels_read_or_stored") is not False:
        raise ValueError("D2 requires the frozen label-free neural cache")

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
    if len(names) != int(feature_config["input_features"]):
        raise ValueError("D2 frozen input feature count changed")

    d1_predictions = load_jsonl(d1_variant_dir / "predictions.jsonl")
    d1_decisions = _prediction_decisions(d1_predictions, examples)
    scalar_predictions = load_jsonl(
        scalar_dir
        / "variants"
        / str(scalar_reference["feature_variant"])
        / "predictions.jsonl"
    )
    scalar_decisions = _prediction_decisions(scalar_predictions, examples)
    d1_diagnostics = _load_json(d1_variant_dir / "diagnostics.json")
    frozen_folds = d1_diagnostics.get("fold_details")
    if not isinstance(frozen_folds, list) or len(frozen_folds) != int(split_config["folds"]):
        raise ValueError("D1 reference fold diagnostics are incomplete")

    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "gpu_used": False,
        "torch_threads": torch.get_num_threads(),
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
            "protocol_disclosure": config["protocol_disclosure"],
        },
    )

    labels = np.asarray([example.gold_interrupt for example in examples], dtype=np.int64)
    folds = np.asarray([example.feature.fold for example in examples], dtype=np.int64)
    decisions: dict[tuple[int, int], int] = {}
    logits: dict[tuple[int, int], float] = {}
    base_reproduced: dict[tuple[int, int], int] = {}
    fold_details: list[dict[str, object]] = []
    heads_dir = output_dir / "fold_heads"
    heads_dir.mkdir()

    fold_count = int(split_config["folds"])
    calibration_offset = int(split_config["calibration_fold_offset"])
    for test_fold in range(fold_count):
        calibration_fold = (test_fold + calibration_offset) % fold_count
        fit_indices = np.flatnonzero(
            (folds != test_fold) & (folds != calibration_fold)
        )
        calibration_indices = np.flatnonzero(folds == calibration_fold)
        test_indices = np.flatnonzero(folds == test_fold)
        base, selected_l2, base_threshold, base_calibration, base_grid = (
            _select_linear_base(
                values[fit_indices],
                labels[fit_indices],
                values[calibration_indices],
                labels[calibration_indices],
                test_fold=test_fold,
                config=base_config,
            )
        )
        frozen = frozen_folds[test_fold]
        if not isinstance(frozen, dict) or int(frozen["test_fold"]) != test_fold:
            raise ValueError("D1 frozen fold ordering changed")
        if selected_l2 != float(frozen["selected_l2_weight"]):
            raise ValueError(f"D2 base selected L2 changed on fold {test_fold}")
        if not math.isclose(
            base_threshold,
            float(frozen["threshold_logit"]),
            rel_tol=0.0,
            abs_tol=float(base_config["threshold_reproduction_abs_tolerance"]),
        ):
            raise ValueError(
                f"D2 base threshold changed on fold {test_fold}: "
                f"{base_threshold!r} != {float(frozen['threshold_logit'])!r} "
                f"(delta={base_threshold - float(frozen['threshold_logit']):.3e})"
            )
        base_test_logits = predict_logits(base, values[test_indices])
        base_test_predictions = [
            int(value >= base_threshold) for value in base_test_logits
        ]
        for index, decision in zip(test_indices.tolist(), base_test_predictions):
            key = examples[index].key
            if decision != d1_decisions[key]:
                raise ValueError(f"D2 base does not reproduce D1 at {key}")
            base_reproduced[key] = decision

        residual_model, history = fit_residual_mlp(
            values[fit_indices],
            labels[fit_indices],
            values[calibration_indices],
            labels[calibration_indices],
            base,
            hidden_width=int(residual_config["hidden_width"]),
            learning_rate=float(residual_config["learning_rate"]),
            weight_decay=float(residual_config["weight_decay"]),
            batch_size=int(residual_config["batch_size"]),
            max_epochs=int(residual_config["max_epochs"]),
            patience=int(residual_config["early_stopping_patience"]),
            min_delta=float(residual_config["early_stopping_min_delta"]),
            gradient_clip_norm=float(residual_config["gradient_clip_norm"]),
            seed=int(residual_config["seed"]) + test_fold,
        )
        calibration_logits = predict_residual_logits(
            residual_model, values[calibration_indices]
        )
        threshold, calibration_metrics = select_threshold(
            calibration_logits, labels[calibration_indices].tolist()
        )
        test_logits = predict_residual_logits(residual_model, values[test_indices])
        test_predictions = [int(value >= threshold) for value in test_logits]
        for index, logit, decision in zip(
            test_indices.tolist(), test_logits, test_predictions
        ):
            key = examples[index].key
            if key in decisions:
                raise ValueError(f"Duplicate D2 OOF decision for {key}")
            decisions[key] = decision
            logits[key] = float(logit)
        d2_test_metrics = binary_metrics(
            labels[test_indices].tolist(), test_predictions
        )
        d1_test_metrics = binary_metrics(
            labels[test_indices].tolist(), base_test_predictions
        )
        head = ResidualDecisionHead(
            feature_names=tuple(names),
            model=residual_model,
            threshold_logit=threshold,
        )
        head_payload = serialize_residual_head(
            head,
            {
                "test_fold": test_fold,
                "calibration_fold": calibration_fold,
                "fit_folds": sorted(
                    set(range(fold_count)) - {test_fold, calibration_fold}
                ),
                "selected_base_l2": selected_l2,
                "seed": int(residual_config["seed"]) + test_fold,
            },
        )
        head_path = heads_dir / f"fold_{test_fold}.json"
        write_json(head_path, head_payload)
        fold_details.append(
            {
                "test_fold": test_fold,
                "calibration_fold": calibration_fold,
                "fit_folds": sorted(
                    set(range(fold_count)) - {test_fold, calibration_fold}
                ),
                "fit_chunks": int(len(fit_indices)),
                "calibration_chunks": int(len(calibration_indices)),
                "test_chunks": int(len(test_indices)),
                "selected_base_l2": selected_l2,
                "base_threshold_logit": base_threshold,
                "frozen_base_threshold_logit": float(frozen["threshold_logit"]),
                "base_threshold_reproduction_delta": base_threshold
                - float(frozen["threshold_logit"]),
                "base_calibration_metrics": base_calibration,
                "base_grid": base_grid,
                "residual_threshold_logit": threshold,
                "residual_calibration_metrics": calibration_metrics,
                "d1_test_metrics": d1_test_metrics,
                "d2_test_metrics": d2_test_metrics,
                "delta_macro_f1": float(d2_test_metrics["macro_f1"])
                - float(d1_test_metrics["macro_f1"]),
                "training": {
                    "best_epoch": residual_model.best_epoch,
                    "epochs_run": residual_model.epochs_run,
                    "fit_loss": residual_model.fit_loss,
                    "calibration_loss": residual_model.calibration_loss,
                    "history": history,
                },
                "head_sha256": sha256_file(head_path),
            }
        )

    expected = {example.key for example in examples}
    if set(base_reproduced) != expected or set(decisions) != expected or set(logits) != expected:
        raise ValueError("D2 OOF coverage is incomplete")
    base_predictions = prediction_rows(examples, base_reproduced)
    base_path = output_dir / "d1_reproduced_predictions.jsonl"
    write_jsonl(base_path, base_predictions)
    if sha256_file(base_path) != str(d1_reference["predictions_sha256"]):
        raise ValueError("D2 did not reproduce the frozen D1 prediction bytes")

    predictions = prediction_rows(examples, decisions)
    validation = validate_prediction_rows(source_rows, predictions)
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
    official = _load_json(metrics_path)
    overall = official.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("Official D2 scorer returned no overall metrics")

    d2_internal = metrics_for_subset(examples, decisions, include_first=True)
    d1_internal = metrics_for_subset(examples, d1_decisions, include_first=True)
    scalar_internal = metrics_for_subset(examples, scalar_decisions, include_first=True)
    d2_non_first = metrics_for_subset(examples, decisions, include_first=False)
    d1_non_first = metrics_for_subset(examples, d1_decisions, include_first=False)
    bootstrap_d1 = paired_session_bootstrap(
        examples,
        decisions,
        d1_decisions,
        repetitions=int(evaluation_config["bootstrap_repetitions"]),
        seed=int(evaluation_config["bootstrap_seed"]),
    )
    bootstrap_scalar = paired_session_bootstrap(
        examples,
        decisions,
        scalar_decisions,
        repetitions=int(evaluation_config["bootstrap_repetitions"]),
        seed=int(evaluation_config["bootstrap_seed"]),
    )
    d2_by_fold = _group_metrics(examples, decisions, "fold")
    d1_by_fold = _group_metrics(examples, d1_decisions, "fold")
    fold_deltas = {
        fold: float(d2_by_fold[fold]["macro_f1"])
        - float(d1_by_fold[fold]["macro_f1"])
        for fold in d1_by_fold
    }
    positive_folds = sum(delta > 0 for delta in fold_deltas.values())
    d2_by_domain = _group_metrics(examples, decisions, "domain")
    d1_by_domain = _group_metrics(examples, d1_decisions, "domain")
    domain_deltas = {
        domain: float(d2_by_domain[domain]["macro_f1"])
        - float(d1_by_domain[domain]["macro_f1"])
        for domain in d1_by_domain
    }
    delta_d1 = float(d2_internal["macro_f1"]) - float(d1_internal["macro_f1"])
    promotion_checks = {
        "minimum_delta": delta_d1
        >= float(promotion_config["min_delta_macro_f1_vs_d1"]),
        "positive_bootstrap_lower_bound": (
            float(bootstrap_d1["delta_macro_f1_p2_5"]) > 0
            if promotion_config["require_positive_session_bootstrap_lower_bound"]
            else True
        ),
        "non_first_chunk_gain": (
            float(d2_non_first["macro_f1"]) > float(d1_non_first["macro_f1"])
            if promotion_config["require_non_first_chunk_gain"]
            else True
        ),
        "positive_fold_count": positive_folds
        >= int(promotion_config["min_positive_folds"]),
        "both_class_f1_nonzero": (
            min(float(d2_internal["interrupt_f1"]), float(d2_internal["silent_f1"])) > 0
            if promotion_config["require_both_class_f1_nonzero"]
            else True
        ),
    }
    promotion = {
        "passed": all(promotion_checks.values()),
        "checks": promotion_checks,
        "criteria": promotion_config,
    }

    write_jsonl(
        output_dir / "oof_records.jsonl",
        [
            {
                "input_index": example.feature.input_index,
                "video_path": example.feature.video_path,
                "domain": example.feature.domain,
                "fold": example.feature.fold,
                "chunk_index": example.feature.chunk_index,
                "gold_interrupt": example.gold_interrupt,
                "d2_logit": logits[example.key],
                "d2_interrupt": decisions[example.key],
                "d1_interrupt": d1_decisions[example.key],
                "scalar_interrupt": scalar_decisions[example.key],
            }
            for example in examples
        ],
    )
    residual_parameters = residual_parameter_count(
        len(names), int(residual_config["hidden_width"])
    )
    head_parameters = len(names) + 1 + residual_parameters
    if residual_parameters != int(residual_config["residual_parameters"]):
        raise ValueError("D2 residual parameter count differs from preregistration")
    if head_parameters != int(residual_config["total_head_parameters"]):
        raise ValueError("D2 total head parameter count differs from preregistration")
    total_parameters = int(model_config["total_parameters"]) + head_parameters
    if total_parameters > 2_000_000_000:
        raise ValueError("D2 system exceeds the Small parameter cap")

    diagnostics = {
        **validation,
        "protocol_disclosure": config["protocol_disclosure"],
        "feature_count": len(names),
        "base_parameters": len(names) + 1,
        "residual_parameters": residual_parameters,
        "total_head_parameters": head_parameters,
        "total_system_parameters": total_parameters,
        "d1_reproduction_exact": True,
        "fold_details": fold_details,
        "d2_internal_metrics": d2_internal,
        "d1_internal_metrics": d1_internal,
        "scalar_internal_metrics": scalar_internal,
        "d2_non_first": d2_non_first,
        "d1_non_first": d1_non_first,
        "d2_by_fold": d2_by_fold,
        "d1_by_fold": d1_by_fold,
        "fold_deltas": fold_deltas,
        "positive_folds": positive_folds,
        "d2_by_domain": d2_by_domain,
        "d1_by_domain": d1_by_domain,
        "domain_deltas": domain_deltas,
        "paired_session_bootstrap_vs_d1": bootstrap_d1,
        "paired_session_bootstrap_vs_scalar": bootstrap_scalar,
        "predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
    }
    write_json(output_dir / "diagnostics.json", diagnostics)
    comparison = {
        "classification": config["validation_policy"],
        "d1_reference_macro_f1_official": d1_reference["macro_f1"],
        "d2_macro_f1_official": overall["macro_f1"],
        "d1_macro_f1_internal": d1_internal["macro_f1"],
        "d2_macro_f1_internal": d2_internal["macro_f1"],
        "delta_macro_f1_vs_d1": delta_d1,
        "fold_deltas": fold_deltas,
        "domain_deltas": domain_deltas,
        "paired_session_bootstrap_vs_d1": bootstrap_d1,
        "promotion": promotion,
    }
    write_json(output_dir / "comparison.json", comparison)
    runtime = {
        "status": "complete D2 residual MLP OOF",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started_at, 3),
        "gpu_used": False,
        "model_inference_rerun": False,
        "sessions": len(source_rows),
        "chunks": len(examples),
        "head_parameters": head_parameters,
        "total_parameters": total_parameters,
        "promotion_passed": promotion["passed"],
    }
    write_json(output_dir / "runtime.json", runtime)
    summary_line = (
        f"D2 official Macro F1={overall['macro_f1']}; "
        f"delta vs D1={delta_d1:.6f}; promotion={promotion['passed']}"
    )
    (output_dir / "run.log").write_text(summary_line + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "状态：**D2 residual MLP OOF 完成**。",
                "",
                "这是单一预注册配置的 session-level、val-supervised OOF 实验，",
                "不是 hidden-test 结果。",
                "",
                summary_line,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
