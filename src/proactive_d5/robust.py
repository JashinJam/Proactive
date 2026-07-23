"""Answer-free robustness transforms and fixed multiview linear OOF training."""

from __future__ import annotations

import copy
import math
from typing import Literal, Mapping, Sequence

import numpy as np

from proactive_d1.core import (
    LabeledChunk,
    LinearDecisionHead,
    fit_linear_logistic,
    predict_logits,
    select_threshold,
)


Method = Literal["standard", "robust"]


def drop_assistant_history(
    rows: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Delete assistant turns without accepting rows that still expose answers."""
    if any("answers" in row for row in rows):
        raise ValueError("Assistant-drop transformation requires answer-free rows")
    transformed: list[dict[str, object]] = []
    prefixes = 0
    input_turns = 0
    removed = 0
    retained = 0
    for row_index, row in enumerate(rows):
        dialog = row.get("dialog")
        intervals = row.get("video_intervals")
        if not isinstance(dialog, list) or not isinstance(intervals, list):
            raise ValueError(f"Assistant-drop row {row_index} lacks dialog/intervals")
        if len(dialog) != len(intervals):
            raise ValueError(f"Assistant-drop row {row_index} dialog is not chunk aligned")
        output_dialog: list[list[dict[str, object]]] = []
        for chunk_index, prefix in enumerate(dialog):
            if not isinstance(prefix, list):
                raise ValueError(
                    f"Assistant-drop dialog[{row_index}][{chunk_index}] is not a list"
                )
            output_prefix: list[dict[str, object]] = []
            for turn in prefix:
                if not isinstance(turn, dict):
                    raise ValueError("Assistant-drop refuses malformed dialog turns")
                input_turns += 1
                if str(turn.get("role", "user")).strip().lower() == "assistant":
                    removed += 1
                else:
                    output_prefix.append(copy.deepcopy(turn))
                    retained += 1
            if not output_prefix:
                raise ValueError("Assistant-drop removed the entire dialog prefix")
            output_dialog.append(output_prefix)
            prefixes += 1
        output_row = copy.deepcopy(row)
        output_row["dialog"] = output_dialog
        transformed.append(output_row)
    return transformed, {
        "sessions": len(transformed),
        "prefixes": prefixes,
        "input_turns": input_turns,
        "assistant_turns_removed": removed,
        "non_assistant_turns_retained": retained,
    }


def _model_summary(names: Sequence[str], head: LinearDecisionHead) -> dict[str, object]:
    weight = np.asarray(head.model.weight, dtype=np.float64)
    ranked = np.argsort(np.abs(weight))[::-1][:20]
    return {
        "threshold_logit": head.threshold_logit,
        "weight_l2_norm": float(np.linalg.norm(weight)),
        "weight_abs_max": float(np.abs(weight).max()),
        "bias": head.model.bias,
        "train_loss": head.model.train_loss,
        "top_standardized_coefficients": [
            {"name": names[index], "weight": float(weight[index])}
            for index in ranked
        ],
    }


def _fit_grid(
    *,
    fit_values: np.ndarray,
    fit_labels: np.ndarray,
    calibration_values: np.ndarray,
    calibration_labels: np.ndarray,
    names: Sequence[str],
    seed: int,
    max_iterations: int,
    l2_weights: Sequence[float],
    l2_reduction: Literal["mean", "sum"],
) -> tuple[
    LinearDecisionHead,
    list[dict[str, float]],
    float,
    dict[str, float | int],
]:
    candidates: list[tuple[float, LinearDecisionHead, dict[str, float | int]]] = []
    for grid_index, l2_weight in enumerate(l2_weights):
        model = fit_linear_logistic(
            fit_values,
            fit_labels,
            seed=seed + grid_index,
            max_iterations=max_iterations,
            l2_weight=float(l2_weight),
            l2_reduction=l2_reduction,
        )
        logits = predict_logits(model, calibration_values)
        threshold, metrics = select_threshold(logits, calibration_labels.tolist())
        candidates.append(
            (
                float(l2_weight),
                LinearDecisionHead(tuple(names), model, threshold),
                metrics,
            )
        )
    selected_l2, selected, selected_metrics = max(
        candidates,
        key=lambda item: (
            float(item[2]["macro_f1"]),
            -item[0],
            -abs(item[1].threshold_logit),
        ),
    )
    grid = [
        {
            "l2_weight": l2_weight,
            "macro_f1": float(metrics["macro_f1"]),
            "threshold_logit": head.threshold_logit,
        }
        for l2_weight, head, metrics in candidates
    ]
    return selected, grid, selected_l2, selected_metrics


def cross_validate_multiview_linear(
    examples: Sequence[LabeledChunk],
    views: Mapping[str, np.ndarray],
    names: Sequence[str],
    *,
    clean_view: str,
    training_views: Sequence[str],
    folds: int,
    calibration_fold_offset: int,
    seed: int,
    max_iterations: int,
    l2_weights: Sequence[float],
    l2_reduction: Literal["mean", "sum"],
) -> tuple[
    dict[Method, dict[str, dict[tuple[int, int], int]]],
    list[dict[str, object]],
    dict[Method, dict[int, LinearDecisionHead]],
]:
    """Fit clean-only and equal-view heads in the same session-level OOF rotations."""
    if clean_view not in views or not training_views or clean_view not in training_views:
        raise ValueError("Multiview training requires the clean view in all view sets")
    if len(set(training_views)) != len(training_views):
        raise ValueError("Multiview training views must be unique")
    if set(training_views) != set(views):
        raise ValueError("Every configured view must participate exactly once")
    if folds < 3 or calibration_fold_offset % folds == 0:
        raise ValueError("Multiview OOF requires distinct fit/calibration/test folds")
    expected_shape = (len(examples), len(names))
    for view_name, values in views.items():
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"Multiview matrix is invalid: {view_name}")
    labels = np.asarray([example.gold_interrupt for example in examples], dtype=np.int64)
    fold_values = np.asarray([example.feature.fold for example in examples], dtype=np.int64)
    decisions: dict[Method, dict[str, dict[tuple[int, int], int]]] = {
        method: {view: {} for view in views} for method in ("standard", "robust")
    }
    heads: dict[Method, dict[int, LinearDecisionHead]] = {
        "standard": {},
        "robust": {},
    }
    details: list[dict[str, object]] = []
    for test_fold in range(folds):
        calibration_fold = (test_fold + calibration_fold_offset) % folds
        fit_indices = np.flatnonzero(
            (fold_values != test_fold) & (fold_values != calibration_fold)
        )
        calibration_indices = np.flatnonzero(fold_values == calibration_fold)
        test_indices = np.flatnonzero(fold_values == test_fold)

        standard, standard_grid, standard_l2, standard_calibration = _fit_grid(
            fit_values=views[clean_view][fit_indices],
            fit_labels=labels[fit_indices],
            calibration_values=views[clean_view][calibration_indices],
            calibration_labels=labels[calibration_indices],
            names=names,
            seed=seed + test_fold * 100,
            max_iterations=max_iterations,
            l2_weights=l2_weights,
            l2_reduction=l2_reduction,
        )
        robust, robust_grid, robust_l2, robust_calibration = _fit_grid(
            fit_values=np.concatenate(
                [views[name][fit_indices] for name in training_views], axis=0
            ),
            fit_labels=np.concatenate(
                [labels[fit_indices] for _ in training_views], axis=0
            ),
            calibration_values=np.concatenate(
                [views[name][calibration_indices] for name in training_views], axis=0
            ),
            calibration_labels=np.concatenate(
                [labels[calibration_indices] for _ in training_views], axis=0
            ),
            names=names,
            seed=seed + 10000 + test_fold * 100,
            max_iterations=max_iterations,
            l2_weights=l2_weights,
            l2_reduction=l2_reduction,
        )
        heads["standard"][test_fold] = standard
        heads["robust"][test_fold] = robust
        for method, head in (("standard", standard), ("robust", robust)):
            for view_name, values in views.items():
                logits = predict_logits(head.model, values[test_indices])
                predicted = [int(logit >= head.threshold_logit) for logit in logits]
                for index, decision in zip(test_indices.tolist(), predicted):
                    key = examples[index].key
                    if key in decisions[method][view_name]:
                        raise ValueError(f"Duplicate multiview OOF decision: {key}")
                    decisions[method][view_name][key] = decision
        details.append(
            {
                "test_fold": test_fold,
                "calibration_fold": calibration_fold,
                "fit_folds": sorted(set(range(folds)) - {test_fold, calibration_fold}),
                "fit_chunks_per_view": int(len(fit_indices)),
                "calibration_chunks_per_view": int(len(calibration_indices)),
                "test_chunks_per_view": int(len(test_indices)),
                "standard": {
                    **_model_summary(names, standard),
                    "selected_l2_weight": standard_l2,
                    "calibration_metrics": standard_calibration,
                    "grid": standard_grid,
                },
                "robust": {
                    **_model_summary(names, robust),
                    "selected_l2_weight": robust_l2,
                    "calibration_metrics": robust_calibration,
                    "grid": robust_grid,
                },
            }
        )
    expected = {example.key for example in examples}
    for method in decisions:
        for view_name in decisions[method]:
            if set(decisions[method][view_name]) != expected:
                raise ValueError(f"Incomplete multiview OOF decisions: {method}/{view_name}")
    return decisions, details, heads


def static_promotion_gate(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    clean_view: str,
    perturbation_views: Sequence[str],
    maximum_clean_drop: float,
    minimum_perturbation_gain: float,
) -> dict[str, object]:
    """Apply the frozen clean-retention and per-perturbation improvement gate."""
    standard = metrics.get("standard")
    robust = metrics.get("robust")
    if not isinstance(standard, Mapping) or not isinstance(robust, Mapping):
        raise ValueError("Static gate requires standard and robust metrics")
    required_views = (clean_view, *perturbation_views)
    required = set(required_views)
    if set(standard) != required or set(robust) != required:
        raise ValueError("Static gate metric views differ from the frozen protocol")

    def macro(method: Mapping[str, Mapping[str, object]], view: str) -> float:
        overall = method[view].get("overall")
        if not isinstance(overall, Mapping):
            raise ValueError(f"Static gate view lacks overall metrics: {view}")
        return float(overall["macro_f1"])

    clean_delta = round(macro(robust, clean_view) - macro(standard, clean_view), 6)
    perturbation_delta = {
        view: round(macro(robust, view) - macro(standard, view), 6)
        for view in perturbation_views
    }
    class_collapse: dict[str, dict[str, bool]] = {}
    for method_name, method in (("standard", standard), ("robust", robust)):
        class_collapse[method_name] = {}
        for view in required_views:
            overall = method[view]["overall"]
            if not isinstance(overall, Mapping):
                raise ValueError("Static gate overall metrics are malformed")
            support = int(overall["support"])
            predicted = int(overall["tp"]) + int(overall["fp"])
            class_collapse[method_name][view] = predicted in (0, support)
    checks = {
        "clean_retention": clean_delta >= -maximum_clean_drop,
        "every_static_perturbation_gain": all(
            delta >= minimum_perturbation_gain
            for delta in perturbation_delta.values()
        ),
        "no_class_collapse": not any(
            value
            for method in class_collapse.values()
            for value in method.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "clean_delta": clean_delta,
        "perturbation_delta": perturbation_delta,
        "worst_perturbation_delta": min(perturbation_delta.values()),
        "class_collapse": class_collapse,
        "self_fed_eligible": all(checks.values()),
    }
