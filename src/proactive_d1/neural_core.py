"""Aligned neural feature matrices and rotating OOF calibration for D1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from .core import (
    LabeledChunk,
    binary_metrics,
    fit_linear_logistic,
    predict_logits,
    select_threshold,
)

NeuralVariant = Literal["tag_only", "scalar_tag", "hidden_linear", "fused_linear"]
NEURAL_VARIANTS: tuple[NeuralVariant, ...] = (
    "tag_only",
    "scalar_tag",
    "hidden_linear",
    "fused_linear",
)


@dataclass(frozen=True)
class NeuralFeatureCache:
    hidden_state: np.ndarray
    tag_margin: np.ndarray
    silent_log_probability: np.ndarray
    interrupt_log_probability: np.ndarray
    prompt_tokens: np.ndarray
    input_index: np.ndarray
    chunk_index: np.ndarray


def load_aligned_neural_cache(
    path: Path,
    examples: Sequence[LabeledChunk],
    hidden_size: int,
) -> NeuralFeatureCache:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "hidden_state",
            "tag_margin",
            "silent_log_probability",
            "interrupt_log_probability",
            "prompt_tokens",
            "input_index",
            "chunk_index",
        }
        if set(archive.files) != required:
            raise ValueError("Merged D1 neural cache has unexpected arrays")
        arrays = {name: archive[name].copy() for name in archive.files}
    rows = len(examples)
    if arrays["hidden_state"].shape != (rows, hidden_size):
        raise ValueError("Merged D1 hidden matrix shape mismatch")
    for name in (
        "tag_margin",
        "silent_log_probability",
        "interrupt_log_probability",
        "prompt_tokens",
        "input_index",
        "chunk_index",
    ):
        if arrays[name].shape != (rows,):
            raise ValueError(f"Merged D1 array {name} shape mismatch")
    expected_keys = np.asarray([example.key for example in examples], dtype=np.int32)
    actual_keys = np.stack([arrays["input_index"], arrays["chunk_index"]], axis=1)
    if not np.array_equal(actual_keys, expected_keys):
        raise ValueError("Merged D1 neural cache does not align with scalar/gold rows")
    for name in (
        "hidden_state",
        "tag_margin",
        "silent_log_probability",
        "interrupt_log_probability",
    ):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Merged D1 array {name} contains non-finite values")
    return NeuralFeatureCache(
        hidden_state=arrays["hidden_state"].astype(np.float32, copy=False),
        tag_margin=arrays["tag_margin"].astype(np.float32, copy=False),
        silent_log_probability=arrays["silent_log_probability"].astype(
            np.float32, copy=False
        ),
        interrupt_log_probability=arrays["interrupt_log_probability"].astype(
            np.float32, copy=False
        ),
        prompt_tokens=arrays["prompt_tokens"].astype(np.int32, copy=False),
        input_index=arrays["input_index"].astype(np.int32, copy=False),
        chunk_index=arrays["chunk_index"].astype(np.int32, copy=False),
    )


def neural_matrix(
    examples: Sequence[LabeledChunk],
    cache: NeuralFeatureCache,
    scalar_names: Sequence[str],
    variant: NeuralVariant,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if variant not in NEURAL_VARIANTS:
        raise ValueError(f"Unknown D1 neural variant: {variant}")
    scalar = np.asarray(
        [
            [example.feature.values[name] for name in scalar_names]
            for example in examples
        ],
        dtype=np.float32,
    )
    margin = cache.tag_margin[:, None]
    hidden_names = tuple(
        f"hidden_{index:04d}" for index in range(cache.hidden_state.shape[1])
    )
    if variant == "tag_only":
        return margin, ("tag_margin",)
    if variant == "scalar_tag":
        return np.concatenate([scalar, margin], axis=1), (
            *scalar_names,
            "tag_margin",
        )
    if variant == "hidden_linear":
        return cache.hidden_state, hidden_names
    return np.concatenate([scalar, margin, cache.hidden_state], axis=1), (
        *scalar_names,
        "tag_margin",
        *hidden_names,
    )


def _model_summary(names: Sequence[str], model: object) -> dict[str, object]:
    weight = np.asarray(model.weight, dtype=np.float64)  # type: ignore[attr-defined]
    ranked = np.argsort(np.abs(weight))[::-1][:20]
    return {
        "weight_l2_norm": float(np.linalg.norm(weight)),
        "weight_abs_max": float(np.abs(weight).max()),
        "bias": float(model.bias),  # type: ignore[attr-defined]
        "train_loss": float(model.train_loss),  # type: ignore[attr-defined]
        "top_standardized_coefficients": [
            {"name": names[index], "weight": float(weight[index])}
            for index in ranked
        ],
    }


def cross_validate_neural_matrix(
    examples: Sequence[LabeledChunk],
    values: np.ndarray,
    names: Sequence[str],
    folds: int,
    calibration_fold_offset: int,
    seed: int,
    max_iterations: int,
    l2_weights: Sequence[float],
    l2_reduction: Literal["mean", "sum"],
) -> tuple[dict[tuple[int, int], int], list[dict[str, object]]]:
    if values.shape != (len(examples), len(names)):
        raise ValueError("D1 neural matrix shape does not match examples/names")
    if not np.isfinite(values).all():
        raise ValueError("D1 neural training matrix contains non-finite values")
    if folds < 3 or calibration_fold_offset % folds == 0:
        raise ValueError("D1 neural rotation needs distinct fit/calibration/test folds")
    if not l2_weights or any(value < 0 or not math.isfinite(value) for value in l2_weights):
        raise ValueError("D1 neural L2 grid must be finite and non-negative")
    labels = np.asarray([example.gold_interrupt for example in examples], dtype=np.int64)
    fold_values = np.asarray([example.feature.fold for example in examples], dtype=np.int64)
    decisions: dict[tuple[int, int], int] = {}
    details: list[dict[str, object]] = []
    for test_fold in range(folds):
        calibration_fold = (test_fold + calibration_fold_offset) % folds
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
                l2_weight=float(l2_weight),
                l2_reduction=l2_reduction,
            )
            calibration_logits = predict_logits(model, values[calibration_indices])
            threshold, metrics = select_threshold(
                calibration_logits, labels[calibration_indices].tolist()
            )
            candidates.append((float(l2_weight), model, threshold, metrics))
        selected_l2, selected_model, threshold, calibration_metrics = max(
            candidates,
            key=lambda item: (
                float(item[3]["macro_f1"]),
                -item[0],
                -abs(item[2]),
            ),
        )
        test_logits = predict_logits(selected_model, values[test_indices])
        test_predictions = [int(logit >= threshold) for logit in test_logits]
        for index, decision in zip(test_indices.tolist(), test_predictions):
            key = examples[index].key
            if key in decisions:
                raise ValueError(f"Duplicate D1 neural OOF decision for {key}")
            decisions[key] = decision
        details.append(
            {
                "test_fold": test_fold,
                "calibration_fold": calibration_fold,
                "fit_folds": sorted(
                    set(range(folds)) - {test_fold, calibration_fold}
                ),
                "fit_chunks": int(len(fit_indices)),
                "calibration_chunks": int(len(calibration_indices)),
                "test_chunks": int(len(test_indices)),
                "selected_l2_weight": selected_l2,
                "l2_reduction": l2_reduction,
                "threshold_logit": threshold,
                "calibration_metrics": calibration_metrics,
                "calibration_grid": [
                    {
                        "l2_weight": l2_weight,
                        "macro_f1": metrics["macro_f1"],
                        "threshold_logit": candidate_threshold,
                    }
                    for l2_weight, _, candidate_threshold, metrics in candidates
                ],
                "test_metrics_internal": binary_metrics(
                    labels[test_indices].tolist(), test_predictions
                ),
                "model": _model_summary(names, selected_model),
            }
        )
    if set(decisions) != {example.key for example in examples}:
        raise ValueError("D1 neural OOF decisions do not cover every chunk")
    return decisions, details
