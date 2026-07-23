"""Label-isolated D6 feature matrix and one-rotation decision head fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from proactive_d1.core import (
    LabelFreeChunk,
    LabeledChunk,
    LinearDecisionHead,
    binary_metrics,
    build_label_free_chunks,
    feature_names,
    fit_linear_logistic,
    predict_logits,
    select_threshold,
)
from proactive_d1.neural_core import NeuralFeatureCache
from proactive_d3.dialog_control_core import (
    build_dialog_policy_features,
    dialog_control_matrix,
)
from proactive_d4_2.evaluate import feature_arrays, records_for_scalar_features


@dataclass(frozen=True)
class LabelFreeMatrix:
    chunks: tuple[LabelFreeChunk, ...]
    values: np.ndarray
    names: tuple[str, ...]
    dialog_audit: dict[str, object]


@dataclass(frozen=True)
class RotationHeadResult:
    head: LinearDecisionHead
    selected_l2_weight: float
    calibration_metrics: dict[str, float | int]
    calibration_grid: tuple[dict[str, object], ...]
    test_keys: tuple[tuple[int, int], ...]
    test_logits: tuple[float, ...]
    test_decisions: tuple[int, ...]
    fit_chunks: int
    calibration_chunks: int
    test_chunks: int


def build_label_free_matrix(
    *,
    answer_free_rows: Sequence[dict[str, object]],
    records: Sequence[Mapping[str, object]],
    fold_by_index: Mapping[int, int],
    hidden_size: int = 1024,
    max_history_turns: int = 8,
    max_frames: int = 32,
) -> LabelFreeMatrix:
    if any("answers" in row for row in answer_free_rows):
        raise ValueError("D6 matrix received answers")
    chunks = build_label_free_chunks(
        answer_free_rows,
        records_for_scalar_features(records),
        dict(fold_by_index),
        max_history_turns=max_history_turns,
        max_model_frames=max_frames,
    )
    dialog_values, dialog_audit = build_dialog_policy_features(
        answer_free_rows, chunks
    )
    arrays = feature_arrays(records, hidden_size)
    cache = NeuralFeatureCache(
        hidden_state=arrays["hidden_state"],
        tag_margin=arrays["tag_margin"],
        silent_log_probability=arrays["silent_log_probability"],
        interrupt_log_probability=arrays["interrupt_log_probability"],
        prompt_tokens=arrays["prompt_tokens"],
        input_index=arrays["input_index"],
        chunk_index=arrays["chunk_index"],
    )
    domains = sorted({chunk.domain for chunk in chunks})
    scalar_names = feature_names("response_temporal", domains)
    sentinel = [LabeledChunk(feature=chunk, gold_interrupt=-1) for chunk in chunks]
    values, names = dialog_control_matrix(
        sentinel,
        cache,
        scalar_names,
        dialog_values,
        "d1_fused_plus_dialog_stage",
    )
    if values.shape != (9935, 1051) or len(names) != 1051:
        raise ValueError(f"D6 feature schema changed: {values.shape}, {len(names)}")
    if not np.isfinite(values).all():
        raise ValueError("D6 feature matrix contains non-finite values")
    return LabelFreeMatrix(
        chunks=tuple(chunks),
        values=values,
        names=tuple(names),
        dialog_audit=dialog_audit,
    )


def fit_rotation_head(
    *,
    matrix: LabelFreeMatrix,
    labels: Mapping[tuple[int, int], int],
    test_fold: int,
    l2_weights: Sequence[float],
    seed: int,
    max_iterations: int,
) -> RotationHeadResult:
    calibration_fold = (test_fold + 1) % 5
    fit_indices = np.asarray(
        [
            index
            for index, chunk in enumerate(matrix.chunks)
            if chunk.fold not in (test_fold, calibration_fold)
        ],
        dtype=np.int64,
    )
    calibration_indices = np.asarray(
        [
            index
            for index, chunk in enumerate(matrix.chunks)
            if chunk.fold == calibration_fold
        ],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [index for index, chunk in enumerate(matrix.chunks) if chunk.fold == test_fold],
        dtype=np.int64,
    )
    test_keys = tuple(
        (matrix.chunks[index].input_index, matrix.chunks[index].chunk_index)
        for index in test_indices
    )
    forbidden = [key for key in test_keys if key in labels]
    if forbidden:
        raise ValueError("D6 test labels were unsealed before prediction freeze")

    def labels_at(indices: np.ndarray) -> list[int]:
        result: list[int] = []
        for index in indices:
            chunk = matrix.chunks[int(index)]
            key = (chunk.input_index, chunk.chunk_index)
            if key not in labels:
                raise ValueError(f"D6 fit/calibration label missing: {key}")
            result.append(int(labels[key]))
        return result

    fit_y = labels_at(fit_indices)
    calibration_y = labels_at(calibration_indices)
    candidates: list[tuple[float, object, float, dict[str, float | int]]] = []
    for grid_index, l2_weight in enumerate(l2_weights):
        model = fit_linear_logistic(
            matrix.values[fit_indices],
            fit_y,
            seed=seed + test_fold * 100 + grid_index,
            max_iterations=max_iterations,
            l2_weight=float(l2_weight),
            l2_reduction="sum",
        )
        logits = predict_logits(model, matrix.values[calibration_indices])
        threshold, metrics = select_threshold(logits, calibration_y)
        candidates.append((float(l2_weight), model, threshold, metrics))
    selected_l2, selected_model, threshold, calibration_metrics = max(
        candidates,
        key=lambda item: (
            float(item[3]["macro_f1"]),
            -item[0],
            -abs(item[2]),
        ),
    )
    test_logits = tuple(
        predict_logits(selected_model, matrix.values[test_indices])
    )
    test_decisions = tuple(int(value >= threshold) for value in test_logits)
    head = LinearDecisionHead(
        feature_names=matrix.names,
        model=selected_model,  # type: ignore[arg-type]
        threshold_logit=threshold,
    )
    return RotationHeadResult(
        head=head,
        selected_l2_weight=selected_l2,
        calibration_metrics=calibration_metrics,
        calibration_grid=tuple(
            {
                "l2_weight": l2,
                "macro_f1": metrics["macro_f1"],
                "threshold_logit": candidate_threshold,
            }
            for l2, _, candidate_threshold, metrics in candidates
        ),
        test_keys=test_keys,
        test_logits=test_logits,
        test_decisions=test_decisions,
        fit_chunks=len(fit_indices),
        calibration_chunks=len(calibration_indices),
        test_chunks=len(test_indices),
    )


def apply_head_to_test(
    *,
    matrix: LabelFreeMatrix,
    head: LinearDecisionHead,
    test_fold: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...], tuple[int, ...]]:
    indices = np.asarray(
        [index for index, chunk in enumerate(matrix.chunks) if chunk.fold == test_fold],
        dtype=np.int64,
    )
    keys = tuple(
        (matrix.chunks[index].input_index, matrix.chunks[index].chunk_index)
        for index in indices
    )
    logits = tuple(predict_logits(head.model, matrix.values[indices]))
    decisions = tuple(int(value >= head.threshold_logit) for value in logits)
    return keys, logits, decisions


def metrics_after_unseal(
    keys: Sequence[tuple[int, int]],
    decisions: Sequence[int],
    labels: Mapping[tuple[int, int], int],
) -> dict[str, float | int]:
    if len(keys) != len(decisions) or not keys:
        raise ValueError("D6 test metrics require aligned frozen decisions")
    return binary_metrics([labels[key] for key in keys], decisions)

