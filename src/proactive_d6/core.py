"""Pure stage construction and structured threshold calibration for D6."""

from __future__ import annotations

import math
from collections import Counter
from typing import Literal, Mapping, Sequence

import numpy as np

from proactive_d1.core import (
    LabelFreeChunk,
    LabeledChunk,
    binary_metrics,
    fit_linear_logistic,
    predict_logits,
    select_threshold,
)


POSITION_GROUPS = ("first", "second", "2-4", "5-9", "10+")
LAST_ACTION_GROUPS = ("first", "previous_interrupt", "previous_silent")
LAST2_GROUPS = ("first", "second", "ii", "is", "si", "ss")

D6Variant = Literal[
    "d4_global_replay",
    "position_shrunk",
    "last_action_shrunk",
    "last2_unshrunk",
    "last2_shrunk",
]
D6_VARIANTS: tuple[D6Variant, ...] = (
    "d4_global_replay",
    "position_shrunk",
    "last_action_shrunk",
    "last2_unshrunk",
    "last2_shrunk",
)
PRIMARY_VARIANT: D6Variant = "last2_shrunk"


def _assistant_count(value: object) -> tuple[int, int, int]:
    if not isinstance(value, list):
        raise ValueError("D6 requires each chunk dialog to be a list")
    count = 0
    malformed = 0
    empty = 0
    for turn in value:
        if not isinstance(turn, dict):
            malformed += 1
            continue
        text = str(turn.get("text") or "").strip()
        if not text:
            empty += 1
            continue
        role = str(turn.get("role", "user")).strip().lower()
        count += int(role == "assistant")
    return count, malformed, empty


def _position_group(chunk_index: int) -> str:
    if chunk_index == 0:
        return "first"
    if chunk_index == 1:
        return "second"
    if chunk_index <= 4:
        return "2-4"
    if chunk_index <= 9:
        return "5-9"
    return "10+"


def build_structured_stages(
    rows: Sequence[dict[str, object]],
    chunks: Sequence[LabelFreeChunk],
) -> tuple[dict[str, tuple[str, ...]], np.ndarray, dict[str, object]]:
    """Build three categorical stage families from answer-stripped dialog prefixes."""
    if any("answers" in row for row in rows):
        raise ValueError("D6 stage construction requires answer-stripped source rows")
    position: list[str] = []
    last_action: list[str] = []
    last2: list[str] = []
    previous_actions: list[int] = []
    cursor = 0
    malformed_turns = 0
    empty_turns = 0
    multi_add_chunks = 0
    additions = 0
    for input_index, row in enumerate(rows):
        dialog = row.get("dialog")
        intervals = row.get("video_intervals")
        if not isinstance(dialog, list) or not isinstance(intervals, list):
            raise ValueError(f"D6 source row {input_index} is malformed")
        if len(dialog) != len(intervals):
            raise ValueError(f"D6 source row {input_index} has unequal chunk fields")
        previous_count: int | None = None
        prior_actions: list[int] = []
        for chunk_index, current_dialog in enumerate(dialog):
            if cursor >= len(chunks):
                raise ValueError("D6 source contains extra chunks")
            chunk = chunks[cursor]
            if (chunk.input_index, chunk.chunk_index) != (input_index, chunk_index):
                raise ValueError("D6 source order differs from label-free chunks")
            current_count, malformed, empty = _assistant_count(current_dialog)
            malformed_turns += malformed
            empty_turns += empty
            previous_action = 0
            if previous_count is not None:
                added_count = current_count - previous_count
                if added_count < 0:
                    raise ValueError("D6 assistant count decreased within a session")
                previous_action = int(added_count > 0)
                prior_actions.append(previous_action)
                additions += previous_action
                multi_add_chunks += int(added_count > 1)

            position.append(_position_group(chunk_index))
            if chunk_index == 0:
                last_action.append("first")
                last2.append("first")
            else:
                last_action.append(
                    "previous_interrupt" if prior_actions[-1] else "previous_silent"
                )
                if chunk_index == 1:
                    last2.append("second")
                else:
                    last2.append(
                        {
                            (1, 1): "ii",
                            (1, 0): "is",
                            (0, 1): "si",
                            (0, 0): "ss",
                        }[(prior_actions[-2], prior_actions[-1])]
                    )
            previous_actions.append(previous_action)
            previous_count = current_count
            cursor += 1
    if cursor != len(chunks):
        raise ValueError("D6 source does not cover every label-free chunk")

    families = {
        "position": tuple(position),
        "last_action": tuple(last_action),
        "last2": tuple(last2),
    }
    expected = {
        "position": set(POSITION_GROUPS),
        "last_action": set(LAST_ACTION_GROUPS),
        "last2": set(LAST2_GROUPS),
    }
    counts: dict[str, dict[str, int]] = {}
    for family, stages in families.items():
        if len(stages) != len(chunks) or any(stage not in expected[family] for stage in stages):
            raise ValueError(f"D6 {family} stage coverage is invalid")
        observed = Counter(stages)
        counts[family] = {
            group: int(observed.get(group, 0))
            for group in (
                POSITION_GROUPS
                if family == "position"
                else LAST_ACTION_GROUPS
                if family == "last_action"
                else LAST2_GROUPS
            )
        }
    previous = np.asarray(previous_actions, dtype=np.int8)
    if previous.shape != (len(chunks),):
        raise ValueError("D6 previous-action vector has the wrong shape")
    first = np.asarray([chunk.chunk_index == 0 for chunk in chunks])
    if first.any() and int(np.abs(previous[first]).max()) != 0:
        raise ValueError("D6 first chunks cannot have a previous action")
    return families, previous, {
        "sessions": len(rows),
        "chunks": len(chunks),
        "counts": counts,
        "assistant_additions": additions,
        "multi_assistant_additions": multi_add_chunks,
        "malformed_turns_ignored": malformed_turns,
        "empty_turns_ignored": empty_turns,
        "labels_read": False,
        "predictions_read": False,
        "future_chunks_read": False,
        "numeric_calibration_inputs_per_row": 1,
        "categorical_stage_per_family_per_row": 1,
    }


def apply_stage_thresholds(
    logits: Sequence[float],
    stages: Sequence[str],
    thresholds: Mapping[str, float],
) -> list[int]:
    if len(logits) != len(stages):
        raise ValueError("D6 logits and stages must align")
    if any(stage not in thresholds for stage in stages):
        raise ValueError("D6 threshold mapping does not cover every stage")
    if any(not math.isfinite(float(value)) for value in thresholds.values()):
        raise ValueError("D6 thresholds must be finite")
    return [
        int(float(logit) >= float(thresholds[stage]))
        for logit, stage in zip(logits, stages)
    ]


def calibrate_stage_thresholds(
    logits: Sequence[float],
    labels: Sequence[int],
    stages: Sequence[str],
    groups: Sequence[str],
    *,
    global_threshold: float,
    minimum_group_rows: int,
    shrinkage_pseudocount: float,
    shrunk: bool,
    pin_first: bool,
) -> tuple[dict[str, float], dict[str, object]]:
    if len(logits) != len(labels) or len(logits) != len(stages) or not logits:
        raise ValueError("D6 calibration inputs must be non-empty and aligned")
    if set(stages) - set(groups):
        raise ValueError("D6 calibration received an unknown stage")
    if minimum_group_rows < 1 or shrinkage_pseudocount < 0:
        raise ValueError("D6 calibration constants are invalid")
    if not math.isfinite(global_threshold):
        raise ValueError("D6 global threshold must be finite")

    thresholds: dict[str, float] = {}
    group_details: dict[str, dict[str, object]] = {}
    for group in groups:
        indices = [index for index, stage in enumerate(stages) if stage == group]
        group_labels = [int(labels[index]) for index in indices]
        positives = sum(group_labels)
        negatives = len(group_labels) - positives
        reason: str | None = None
        eligible = True
        if pin_first and group == "first":
            eligible = False
            reason = "first_group_pinned_to_global"
        elif len(indices) < minimum_group_rows:
            eligible = False
            reason = "below_minimum_group_rows"
        elif positives == 0 or negatives == 0:
            eligible = False
            reason = "missing_class"

        local_threshold: float | None = None
        local_metrics: dict[str, float | int] | None = None
        effective_n = 0
        weight = 0.0
        threshold = float(global_threshold)
        if eligible:
            local_threshold, local_metrics = select_threshold(
                [float(logits[index]) for index in indices], group_labels
            )
            effective_n = 2 * min(positives, negatives)
            weight = (
                effective_n / (effective_n + shrinkage_pseudocount)
                if shrunk
                else 1.0
            )
            threshold = float(global_threshold) + weight * (
                float(local_threshold) - float(global_threshold)
            )
        thresholds[group] = threshold
        group_details[group] = {
            "rows": len(indices),
            "interrupts": positives,
            "silents": negatives,
            "eligible": eligible,
            "fallback_reason": reason,
            "local_threshold": local_threshold,
            "effective_n": effective_n,
            "shrinkage_weight": weight,
            "applied_threshold": threshold,
            "local_threshold_metrics": local_metrics,
        }
    predictions = apply_stage_thresholds(logits, stages, thresholds)
    return thresholds, {
        "global_threshold": float(global_threshold),
        "shrunk": shrunk,
        "minimum_group_rows": minimum_group_rows,
        "shrinkage_pseudocount": shrinkage_pseudocount,
        "groups": group_details,
        "calibration_metrics": binary_metrics(labels, predictions),
    }


def cross_validate_structured_calibration(
    examples: Sequence[LabeledChunk],
    values: np.ndarray,
    names: Sequence[str],
    stage_families: Mapping[str, Sequence[str]],
    *,
    folds: int,
    calibration_fold_offset: int,
    seed: int,
    max_iterations: int,
    l2_weights: Sequence[float],
    l2_reduction: Literal["mean", "sum"],
    minimum_group_rows: int,
    shrinkage_pseudocount: float,
    pin_first: bool,
) -> tuple[
    dict[str, dict[tuple[int, int], int]],
    dict[str, list[dict[str, object]]],
    dict[tuple[int, int], float],
    dict[str, dict[tuple[int, int], float]],
]:
    """Fit D4 once per fold and apply all frozen D6 threshold policies."""
    if values.shape != (len(examples), len(names)) or not np.isfinite(values).all():
        raise ValueError("D6 D4 matrix is invalid")
    if folds < 3 or calibration_fold_offset % folds == 0:
        raise ValueError("D6 requires distinct fit, calibration, and test folds")
    if not l2_weights or any(value < 0 or not math.isfinite(value) for value in l2_weights):
        raise ValueError("D6 L2 grid is invalid")
    required_families = {
        "position": POSITION_GROUPS,
        "last_action": LAST_ACTION_GROUPS,
        "last2": LAST2_GROUPS,
    }
    for family, groups in required_families.items():
        stages = stage_families.get(family)
        if stages is None or len(stages) != len(examples) or set(stages) - set(groups):
            raise ValueError(f"D6 {family} stages do not align with examples")

    labels = np.asarray([example.gold_interrupt for example in examples], dtype=np.int64)
    fold_values = np.asarray([example.feature.fold for example in examples], dtype=np.int64)
    decisions = {variant: {} for variant in D6_VARIANTS}
    fold_details = {variant: [] for variant in D6_VARIANTS}
    oof_logits: dict[tuple[int, int], float] = {}
    applied_thresholds = {variant: {} for variant in D6_VARIANTS}

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
            global_threshold, metrics = select_threshold(
                calibration_logits, labels[calibration_indices].tolist()
            )
            candidates.append((float(l2_weight), model, global_threshold, metrics))
        selected_l2, selected_model, global_threshold, global_metrics = max(
            candidates,
            key=lambda item: (
                float(item[3]["macro_f1"]),
                -item[0],
                -abs(item[2]),
            ),
        )
        calibration_logits = predict_logits(selected_model, values[calibration_indices])
        test_logits = predict_logits(selected_model, values[test_indices])

        policies: dict[str, tuple[dict[str, float], dict[str, object], Sequence[str]]] = {
            "d4_global_replay": (
                {"global": float(global_threshold)},
                {
                    "global_threshold": float(global_threshold),
                    "shrunk": False,
                    "groups": {},
                    "calibration_metrics": global_metrics,
                },
                tuple("global" for _ in examples),
            )
        }
        policy_specs = {
            "position_shrunk": ("position", POSITION_GROUPS, True),
            "last_action_shrunk": ("last_action", LAST_ACTION_GROUPS, True),
            "last2_unshrunk": ("last2", LAST2_GROUPS, False),
            "last2_shrunk": ("last2", LAST2_GROUPS, True),
        }
        for variant, (family, groups, shrunk) in policy_specs.items():
            all_stages = stage_families[family]
            calibration_stages = [all_stages[index] for index in calibration_indices]
            thresholds, detail = calibrate_stage_thresholds(
                calibration_logits,
                labels[calibration_indices].tolist(),
                calibration_stages,
                groups,
                global_threshold=float(global_threshold),
                minimum_group_rows=minimum_group_rows,
                shrinkage_pseudocount=shrinkage_pseudocount,
                shrunk=shrunk,
                pin_first=pin_first,
            )
            policies[variant] = (thresholds, detail, all_stages)

        for index, logit in zip(test_indices.tolist(), test_logits):
            key = examples[index].key
            if key in oof_logits:
                raise ValueError(f"Duplicate D6 OOF logit for {key}")
            oof_logits[key] = float(logit)

        common = {
            "test_fold": test_fold,
            "calibration_fold": calibration_fold,
            "fit_folds": sorted(set(range(folds)) - {test_fold, calibration_fold}),
            "fit_chunks": int(len(fit_indices)),
            "calibration_chunks": int(len(calibration_indices)),
            "test_chunks": int(len(test_indices)),
            "selected_l2_weight": selected_l2,
            "l2_reduction": l2_reduction,
            "global_threshold_logit": float(global_threshold),
            "global_calibration_metrics": global_metrics,
            "d4_model_reused_across_all_variants": True,
            "calibration_grid": [
                {
                    "l2_weight": l2_weight,
                    "macro_f1": metrics["macro_f1"],
                    "threshold_logit": threshold,
                }
                for l2_weight, _, threshold, metrics in candidates
            ],
        }
        for variant in D6_VARIANTS:
            thresholds, policy_detail, all_stages = policies[variant]
            test_stages = [all_stages[index] for index in test_indices]
            test_predictions = apply_stage_thresholds(
                test_logits, test_stages, thresholds
            )
            for index, decision, stage in zip(
                test_indices.tolist(), test_predictions, test_stages
            ):
                key = examples[index].key
                if key in decisions[variant]:
                    raise ValueError(f"Duplicate D6 decision for {variant} {key}")
                decisions[variant][key] = decision
                applied_thresholds[variant][key] = float(thresholds[stage])
            fold_details[variant].append(
                {
                    **common,
                    "variant": variant,
                    "threshold_policy": policy_detail,
                    "test_metrics_internal": binary_metrics(
                        labels[test_indices].tolist(), test_predictions
                    ),
                }
            )

    expected = {example.key for example in examples}
    if set(oof_logits) != expected:
        raise ValueError("D6 OOF logits do not cover every chunk")
    for variant in D6_VARIANTS:
        if set(decisions[variant]) != expected or set(applied_thresholds[variant]) != expected:
            raise ValueError(f"D6 {variant} does not cover every chunk")
    return decisions, fold_details, oof_logits, applied_thresholds
