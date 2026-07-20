"""Frozen prompts, candidate mappings, and metrics for S0 state decoding."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Callable, Sequence

import numpy as np


STATE_TARGETS = ("step", "progress", "error")
STATE_VIEWS = ("official_dialog", "no_assistant_history")
CANDIDATE_LABELS: dict[str, tuple[str, ...]] = {
    "step": ("s1", "s2", "s3", "s4"),
    "progress": (
        "not_started",
        "ongoing",
        "complete",
        "deviated",
        "recovered",
    ),
    "error": ("absent", "present"),
}
CANDIDATE_TEXT: dict[str, tuple[str, ...]] = {
    target: tuple(str(index + 1) for index in range(len(labels)))
    for target, labels in CANDIDATE_LABELS.items()
}


QUESTION_TEXT = {
    "step": (
        "Which plan step best describes the work currently being performed, "
        "just completed without moving on, or needing recovery?"
    ),
    "progress": "What is the current progress status of that step?",
    "error": (
        "Is there currently visible incomplete, incorrect, blocked, or recovery "
        "evidence for the current step?"
    ),
}


def messages_from_sample(
    sample: dict[str, object],
    system_prompt: str,
    normalize_dialog_turns: Callable[
        [list[dict[str, object]]], list[dict[str, str]]
    ],
    max_history_turns: int,
    view: str,
) -> list[dict[str, str]]:
    if view not in STATE_VIEWS:
        raise ValueError(f"Unsupported S0 state view: {view}")
    query = str(sample.get("query", ""))
    prior = sample.get("prior_dialog")
    if not isinstance(prior, list) or not prior:
        raise ValueError("S0 sample requires a non-empty prior_dialog")
    turns = [copy.deepcopy(turn) for turn in prior[1:]]
    if not all(isinstance(turn, dict) for turn in turns):
        raise ValueError("S0 prior dialog contains a malformed turn")
    if view == "no_assistant_history":
        turns = [turn for turn in turns if turn.get("role") != "assistant"]
    if max_history_turns == 0:
        turns = []
    elif max_history_turns > 0:
        turns = turns[-max_history_turns:]
    messages = [{"role": "system", "content": system_prompt}]
    if query:
        messages.append({"role": "user", "content": query})
    messages.extend(normalize_dialog_turns(turns))
    return messages


def state_question_messages(
    messages: Sequence[dict[str, str]],
    sample: dict[str, object],
    target: str,
) -> list[dict[str, str]]:
    if target not in STATE_TARGETS:
        raise ValueError(f"Unsupported S0 state target: {target}")
    result = copy.deepcopy(list(messages))
    if not result or result[0].get("role") != "system":
        raise ValueError("S0 messages must begin with a system prompt")
    steps = sample.get("steps")
    if not isinstance(steps, list) or len(steps) != 4:
        raise ValueError("S0 requires exactly four oracle-plan steps")
    lines = [
        "[Structured procedural-state decoding]",
        "The following four-step plan was written from task/query before video inspection.",
        f"Task: {sample.get('task')}",
        f"Goal: {sample.get('goal')}",
        "Plan:",
    ]
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict) or not str(step.get("text", "")).strip():
            raise ValueError("S0 oracle plan contains a malformed step")
        lines.append(f"{index}. {str(step['text']).strip()}")
    labels = CANDIDATE_LABELS[target]
    lines.extend(
        [
            "Use only the currently visible video and prior dialog. Do not infer future actions.",
            QUESTION_TEXT[target],
            "Options:",
            *[
                f"{index + 1} = {label}"
                for index, label in enumerate(labels)
            ],
            "Return only the option digit.",
        ]
    )
    result[0]["content"] = result[0]["content"].rstrip() + "\n\n" + "\n".join(lines)
    return result


def probabilities(log_probabilities: Sequence[float]) -> list[float]:
    values = np.asarray(log_probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("S0 candidate scores must be a finite non-empty vector")
    normalized = np.exp(values - float(values.max()))
    normalized /= float(normalized.sum())
    return [float(value) for value in normalized]


def prediction_from_scores(target: str, scores: Sequence[float]) -> dict[str, object]:
    labels = CANDIDATE_LABELS[target]
    candidates = CANDIDATE_TEXT[target]
    if len(scores) != len(labels):
        raise ValueError(f"S0 {target} candidate score count changed")
    posterior = probabilities(scores)
    selected = max(range(len(scores)), key=lambda index: (scores[index], -index))
    entropy = -sum(value * math.log(max(value, 1e-300)) for value in posterior)
    return {
        "candidate": candidates[selected],
        "label": labels[selected],
        "log_probabilities": {
            candidate: float(score) for candidate, score in zip(candidates, scores)
        },
        "probabilities": {
            label: value for label, value in zip(labels, posterior)
        },
        "confidence": posterior[selected],
        "entropy": entropy,
    }


def multiclass_metrics(
    gold: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> dict[str, object]:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("S0 metrics require aligned non-empty values")
    if any(value not in labels for value in (*gold, *predicted)):
        raise ValueError("S0 metric value is outside the frozen label set")
    confusion = {
        true: {candidate: 0 for candidate in labels} for true in labels
    }
    for true, candidate in zip(gold, predicted):
        confusion[true][candidate] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        f1_values.append(f1)
        per_class[label] = {
            "support": sum(confusion[label].values()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "support": len(gold),
        "accuracy": sum(left == right for left, right in zip(gold, predicted))
        / len(gold),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion": confusion,
    }


def grouped_composite(
    rows: Sequence[dict[str, object]], field: str
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(float(row["composite_correctness"]))
    return {
        key: {"states": len(values), "mean_composite_correctness": sum(values) / len(values)}
        for key, values in sorted(groups.items())
    }


def paired_session_bootstrap(
    left: Sequence[dict[str, object]],
    right: Sequence[dict[str, object]],
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    if repetitions <= 0:
        raise ValueError("S0 bootstrap repetitions must be positive")
    left_by_id = {str(row["sample_id"]): row for row in left}
    right_by_id = {str(row["sample_id"]): row for row in right}
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("S0 paired views contain different samples")
    sessions = sorted({int(row["input_index"]) for row in left})
    by_session: dict[int, list[str]] = defaultdict(list)
    for sample_id, row in left_by_id.items():
        by_session[int(row["input_index"])].append(sample_id)
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        left_values: list[float] = []
        right_values: list[float] = []
        for session in sampled:
            for sample_id in by_session[int(session)]:
                left_values.append(
                    float(left_by_id[sample_id]["composite_correctness"])
                )
                right_values.append(
                    float(right_by_id[sample_id]["composite_correctness"])
                )
        deltas[repetition] = np.mean(left_values) - np.mean(right_values)
    return {
        "unit": "session",
        "repetitions": repetitions,
        "seed": seed,
        "delta_mean": float(np.mean(deltas)),
        "delta_p2_5": float(np.percentile(deltas, 2.5)),
        "delta_median": float(np.median(deltas)),
        "delta_p97_5": float(np.percentile(deltas, 97.5)),
        "positive_fraction": float(np.mean(deltas > 0)),
    }

