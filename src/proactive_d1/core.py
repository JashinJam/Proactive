"""Label-free controls and session-held-out linear calibration for D1."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from proactive_r0.core import EMPTY_INTERRUPT_UTTERANCE, INTERRUPT_TAG, SILENT_TAG
from proactive_r0f.core import repair_response_intent

FeatureVariant = Literal["temporal", "temporal_domain", "response_temporal"]
FEATURE_VARIANTS: tuple[FeatureVariant, ...] = (
    "temporal",
    "temporal_domain",
    "response_temporal",
)

TEMPORAL_FEATURES = (
    "is_first_chunk",
    "log1p_chunk_number",
    "log1p_observed_end_sec",
    "log1p_interval_duration",
    "log1p_gap_from_previous",
    "history_turn_fraction",
    "model_input_frame_fraction",
)
RESPONSE_FEATURES = (
    "r0_decision_interrupt",
    "r0f_decision_interrupt",
    "raw_explicit_interrupt",
    "raw_explicit_silent",
    "raw_malformed_nonempty",
    "raw_empty",
    "log1p_raw_length",
)


@dataclass(frozen=True)
class LabelFreeChunk:
    input_index: int
    video_path: str
    domain: str
    fold: int
    chunk_index: int
    total_chunks: int
    interval: tuple[float, float]
    raw_response: str
    values: dict[str, float]


@dataclass(frozen=True)
class LabeledChunk:
    feature: LabelFreeChunk
    gold_interrupt: int

    @property
    def key(self) -> tuple[int, int]:
        return self.feature.input_index, self.feature.chunk_index


@dataclass(frozen=True)
class LinearModel:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weight: tuple[float, ...]
    bias: float
    train_loss: float


@dataclass(frozen=True)
class LinearDecisionHead:
    feature_names: tuple[str, ...]
    model: LinearModel
    threshold_logit: float


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def strip_answers(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [{key: value for key, value in row.items() if key != "answers"} for row in rows]


def make_fold_manifest(
    rows: Sequence[dict[str, object]],
    folds: int,
    seed: str,
) -> dict[str, object]:
    """Assign complete sessions by domain without accessing target labels."""
    if folds < 3:
        raise ValueError("D1 requires at least three folds")
    if not seed:
        raise ValueError("Fold seed must be non-empty")
    by_domain: dict[str, list[tuple[str, int, dict[str, object]]]] = {}
    for input_index, row in enumerate(rows):
        domain = row.get("domain")
        video_path = row.get("video_path")
        intervals = row.get("video_intervals")
        query = row.get("query")
        if not isinstance(domain, str) or not isinstance(video_path, str):
            raise ValueError(f"Source row {input_index} lacks domain/video_path")
        if not isinstance(intervals, list) or not isinstance(query, str):
            raise ValueError(f"Source row {input_index} lacks intervals/query")
        key = _sha256_text(f"{seed}\0{domain}\0{video_path}")
        by_domain.setdefault(domain, []).append((key, input_index, row))

    assignments: dict[int, int] = {}
    for domain in sorted(by_domain):
        ranked = sorted(by_domain[domain], key=lambda item: item[0])
        for rank, (_, input_index, _) in enumerate(ranked):
            assignments[input_index] = rank % folds

    sessions = []
    for input_index, row in enumerate(rows):
        sessions.append(
            {
                "input_index": input_index,
                "video_path": row["video_path"],
                "domain": row["domain"],
                "task": row.get("task"),
                "query_sha256": _sha256_text(str(row["query"])),
                "chunks": len(row["video_intervals"]),  # type: ignore[arg-type]
                "fold": assignments[input_index],
            }
        )
    return {
        "schema_version": 1,
        "algorithm": "domain_stratified_sha256_round_robin",
        "seed": seed,
        "folds": folds,
        "labels_used_for_assignment": False,
        "sessions": sessions,
    }


def validate_fold_manifest(
    manifest: dict[str, object], rows: Sequence[dict[str, object]]
) -> dict[int, int]:
    folds = manifest.get("folds")
    sessions = manifest.get("sessions")
    if manifest.get("algorithm") != "domain_stratified_sha256_round_robin":
        raise ValueError("Unsupported D1 split algorithm")
    if manifest.get("labels_used_for_assignment") is not False:
        raise ValueError("D1 split must be label-independent")
    if not isinstance(folds, int) or folds < 3 or not isinstance(sessions, list):
        raise ValueError("Invalid D1 split manifest")
    if len(sessions) != len(rows):
        raise ValueError("D1 split session count mismatch")
    result: dict[int, int] = {}
    for position, (entry, row) in enumerate(zip(sessions, rows)):
        if not isinstance(entry, dict) or entry.get("input_index") != position:
            raise ValueError(f"Invalid split entry at {position}")
        if entry.get("video_path") != row.get("video_path"):
            raise ValueError(f"Split video mismatch at {position}")
        fold = entry.get("fold")
        if not isinstance(fold, int) or not 0 <= fold < folds:
            raise ValueError(f"Invalid fold at {position}")
        result[position] = fold
    if set(result.values()) != set(range(folds)):
        raise ValueError("Every D1 fold must contain at least one session")
    return result


def _history_turns(row: dict[str, object], chunk_index: int, maximum: int) -> int:
    dialog = row.get("dialog")
    if not isinstance(dialog, list) or chunk_index >= len(dialog):
        raise ValueError("Dialog does not cover the current chunk")
    current = dialog[chunk_index]
    if not isinstance(current, list):
        raise ValueError("dialog_at_chunk must be a list")
    nonempty = 0
    for turn in current[1:]:
        if isinstance(turn, dict) and str(turn.get("text", "")).strip():
            nonempty += 1
    return min(nonempty, maximum)


def feature_names(variant: FeatureVariant, domains: Sequence[str]) -> tuple[str, ...]:
    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"Unknown D1 feature variant: {variant}")
    names = list(TEMPORAL_FEATURES)
    if variant in ("temporal_domain", "response_temporal"):
        names.extend(f"domain={domain}" for domain in domains)
    if variant == "response_temporal":
        names.extend(RESPONSE_FEATURES)
    return tuple(names)


def causal_scalar_values(
    row: dict[str, object],
    chunk_index: int,
    interval: tuple[float, float],
    previous_end: float | None,
    model_input_frames: int,
    raw_response: str,
    r0_answer: str,
    domains: Sequence[str],
    max_history_turns: int,
    max_model_frames: int,
) -> dict[str, float]:
    """Build the exact current-chunk features shared by training and deployment."""
    if chunk_index < 0 or model_input_frames < 0:
        raise ValueError("D1 causal feature indices/counts must be non-negative")
    start, end = interval
    duration = end - start
    if start < 0 or duration <= 0:
        raise ValueError("D1 causal feature interval must satisfy 0 <= start < end")
    gap = max(0.0, start - previous_end) if previous_end is not None else 0.0
    raw = str(raw_response)
    stripped = raw.lstrip()
    explicit_interrupt = stripped.startswith(INTERRUPT_TAG)
    explicit_silent = stripped.startswith(SILENT_TAG)
    raw_empty = not stripped
    repaired, _ = repair_response_intent(raw)
    domain = str(row.get("domain", ""))
    values = {
        "is_first_chunk": float(chunk_index == 0),
        "log1p_chunk_number": math.log1p(chunk_index + 1),
        "log1p_observed_end_sec": math.log1p(max(end, 0.0)),
        "log1p_interval_duration": math.log1p(duration),
        "log1p_gap_from_previous": math.log1p(gap),
        "history_turn_fraction": _history_turns(
            row, chunk_index, max_history_turns
        )
        / max(max_history_turns, 1),
        "model_input_frame_fraction": float(model_input_frames)
        / max(max_model_frames, 1),
        "r0_decision_interrupt": float(str(r0_answer).startswith(INTERRUPT_TAG)),
        "r0f_decision_interrupt": float(repaired.startswith(INTERRUPT_TAG)),
        "raw_explicit_interrupt": float(explicit_interrupt),
        "raw_explicit_silent": float(explicit_silent),
        "raw_malformed_nonempty": float(
            not raw_empty and not explicit_interrupt and not explicit_silent
        ),
        "raw_empty": float(raw_empty),
        "log1p_raw_length": math.log1p(len(stripped)),
    }
    values.update({f"domain={value}": float(domain == value) for value in domains})
    return values


def build_label_free_chunks(
    rows: Sequence[dict[str, object]],
    r0_records: Sequence[dict[str, object]],
    fold_by_index: dict[int, int],
    max_history_turns: int,
    max_model_frames: int = 32,
) -> list[LabelFreeChunk]:
    """Build causal scalar features without reading source answers."""
    if any("answers" in row for row in rows):
        raise ValueError("Label-free D1 feature rows must not contain answers")
    if len(rows) != len(r0_records) or set(fold_by_index) != set(range(len(rows))):
        raise ValueError("D1 source, R0 records, and fold manifest must align")
    domains = sorted({str(row.get("domain")) for row in rows})
    examples: list[LabelFreeChunk] = []
    for input_index, (row, record) in enumerate(zip(rows, r0_records)):
        video_path = row.get("video_path")
        domain = row.get("domain")
        intervals = row.get("video_intervals")
        chunks = record.get("chunks")
        prediction = record.get("prediction")
        if record.get("input_index") != input_index:
            raise ValueError(f"R0 input index mismatch at {input_index}")
        if record.get("video_path") != video_path:
            raise ValueError(f"R0 video mismatch at {input_index}")
        if not isinstance(domain, str) or domain not in domains:
            raise ValueError(f"Invalid domain at {input_index}")
        if not isinstance(intervals, list) or not isinstance(chunks, list):
            raise ValueError(f"Invalid intervals/chunks at {input_index}")
        if not isinstance(prediction, dict) or not isinstance(prediction.get("answers"), list):
            raise ValueError(f"Invalid R0 prediction at {input_index}")
        r0_answers = prediction["answers"]
        if len(intervals) != len(chunks) or len(chunks) != len(r0_answers):
            raise ValueError(f"Chunk alignment mismatch at {input_index}")
        total_chunks = len(chunks)
        previous_end: float | None = None
        for chunk_index, (interval_value, chunk, r0_answer) in enumerate(
            zip(intervals, chunks, r0_answers)
        ):
            if not isinstance(interval_value, list) or len(interval_value) != 2:
                raise ValueError(f"Bad interval at {input_index}:{chunk_index}")
            if not isinstance(chunk, dict) or "raw_response" not in chunk:
                raise ValueError(f"R0 raw response missing at {input_index}:{chunk_index}")
            start, end = float(interval_value[0]), float(interval_value[1])
            raw = str(chunk["raw_response"])
            values = causal_scalar_values(
                row=row,
                chunk_index=chunk_index,
                interval=(start, end),
                previous_end=previous_end,
                model_input_frames=int(chunk.get("model_input_frames", 0)),
                raw_response=raw,
                r0_answer=str(r0_answer),
                domains=domains,
                max_history_turns=max_history_turns,
                max_model_frames=max_model_frames,
            )
            previous_end = end
            examples.append(
                LabelFreeChunk(
                    input_index=input_index,
                    video_path=str(video_path),
                    domain=domain,
                    fold=fold_by_index[input_index],
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    interval=(start, end),
                    raw_response=raw,
                    values=values,
                )
            )
    return examples


def attach_gold_labels(
    features: Sequence[LabelFreeChunk], rows: Sequence[dict[str, object]]
) -> list[LabeledChunk]:
    labels: dict[tuple[int, int], int] = {}
    for input_index, row in enumerate(rows):
        answers = row.get("answers")
        if not isinstance(answers, list):
            raise ValueError(f"Gold answers missing at {input_index}")
        for chunk_index, answer in enumerate(answers):
            labels[(input_index, chunk_index)] = int(
                str(answer).lstrip().startswith(INTERRUPT_TAG)
            )
    result: list[LabeledChunk] = []
    for feature in features:
        key = (feature.input_index, feature.chunk_index)
        if key not in labels:
            raise ValueError(f"Gold label missing for {key}")
        result.append(LabeledChunk(feature=feature, gold_interrupt=labels[key]))
    if len(result) != len(labels):
        raise ValueError("D1 feature/gold coverage mismatch")
    return result


def matrix(
    examples: Sequence[LabeledChunk], names: Sequence[str]
) -> tuple[list[list[float]], list[int]]:
    return (
        [[example.feature.values[name] for name in names] for example in examples],
        [example.gold_interrupt for example in examples],
    )


def fit_linear_logistic(
    values: Sequence[Sequence[float]],
    labels: Sequence[int],
    seed: int,
    max_iterations: int,
    l2_weight: float,
    l2_reduction: Literal["mean", "sum"] = "mean",
) -> LinearModel:
    import torch
    import torch.nn.functional as functional

    if len(values) != len(labels) or len(values) == 0:
        raise ValueError("Training values and labels must be non-empty and aligned")
    x = torch.tensor(values, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    if x.ndim != 2 or len(set(labels)) != 2:
        raise ValueError("Linear training requires a 2D matrix with both classes")
    if l2_reduction not in ("mean", "sum"):
        raise ValueError(f"Unsupported L2 reduction: {l2_reduction}")
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False)
    scale = torch.where(scale < 1e-10, torch.ones_like(scale), scale)
    normalized = (x - mean) / scale
    torch.manual_seed(seed)
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    positives = y.sum()
    negatives = y.numel() - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("Both classes are required for class-balanced training")
    positive_weight = negatives / positives
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=max_iterations,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = normalized @ weight + bias
        squared = weight.square()
        penalty = squared.mean() if l2_reduction == "mean" else squared.sum()
        loss = functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=positive_weight
        ) + l2_weight * penalty
        loss.backward()
        return loss

    optimizer.step(closure)
    final_loss = float(closure().detach())
    return LinearModel(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weight=tuple(float(value) for value in weight.detach()),
        bias=float(bias.detach()),
        train_loss=final_loss,
    )


def predict_logits(
    model: LinearModel, values: Sequence[Sequence[float]]
) -> list[float]:
    if len(values) == 0:
        return []
    result: list[float] = []
    for row in values:
        if len(row) != len(model.weight):
            raise ValueError("Feature dimension differs from fitted D1 model")
        score = model.bias
        for value, mean, scale, weight in zip(
            row, model.mean, model.scale, model.weight
        ):
            score += ((float(value) - mean) / scale) * weight
        result.append(score)
    return result


def serialize_decision_head(
    head: LinearDecisionHead, metadata: dict[str, object]
) -> dict[str, object]:
    """Create a JSON-safe deployment artifact with explicit feature ordering."""
    if not head.feature_names or len(head.feature_names) != len(head.model.weight):
        raise ValueError("Decision head feature names and weights must align")
    if len(set(head.feature_names)) != len(head.feature_names):
        raise ValueError("Decision head feature names must be unique")
    return {
        "schema_version": 1,
        "head_type": "standardized_linear_logistic",
        "feature_names": list(head.feature_names),
        "mean": list(head.model.mean),
        "scale": list(head.model.scale),
        "weight": list(head.model.weight),
        "bias": head.model.bias,
        "threshold_logit": head.threshold_logit,
        "training_loss": head.model.train_loss,
        "metadata": metadata,
    }


def load_decision_head(payload: dict[str, object]) -> LinearDecisionHead:
    """Validate and load a serialized D1 linear head."""
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported D1 decision-head schema")
    if payload.get("head_type") != "standardized_linear_logistic":
        raise ValueError("Unsupported D1 decision-head type")
    names_value = payload.get("feature_names")
    mean_value = payload.get("mean")
    scale_value = payload.get("scale")
    weight_value = payload.get("weight")
    if not all(
        isinstance(value, list)
        for value in (names_value, mean_value, scale_value, weight_value)
    ):
        raise ValueError("D1 decision-head arrays must be JSON lists")
    names = tuple(str(value) for value in names_value)  # type: ignore[arg-type]
    mean = tuple(float(value) for value in mean_value)  # type: ignore[arg-type]
    scale = tuple(float(value) for value in scale_value)  # type: ignore[arg-type]
    weight = tuple(float(value) for value in weight_value)  # type: ignore[arg-type]
    lengths = {len(names), len(mean), len(scale), len(weight)}
    if lengths != {len(names)} or not names or len(set(names)) != len(names):
        raise ValueError("D1 decision-head arrays are empty, duplicated, or misaligned")
    numeric = (*mean, *scale, *weight, float(payload["bias"]), float(payload["threshold_logit"]))
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("D1 decision head contains non-finite values")
    if any(value <= 0 for value in scale):
        raise ValueError("D1 decision-head scales must be positive")
    return LinearDecisionHead(
        feature_names=names,
        model=LinearModel(
            mean=mean,
            scale=scale,
            weight=weight,
            bias=float(payload["bias"]),
            train_loss=float(payload.get("training_loss", float("nan"))),
        ),
        threshold_logit=float(payload["threshold_logit"]),
    )


def predict_feature_values(
    head: LinearDecisionHead, values: dict[str, float]
) -> tuple[int, float]:
    missing = [name for name in head.feature_names if name not in values]
    if missing:
        raise ValueError(f"D1 feature values missing: {missing}")
    logit = predict_logits(
        head.model, [[float(values[name]) for name in head.feature_names]]
    )[0]
    return int(logit >= head.threshold_logit), logit


def binary_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float | int]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Metrics require non-empty aligned labels and predictions")
    tp = sum(gold == 1 and pred == 1 for gold, pred in zip(labels, predictions))
    fp = sum(gold == 0 and pred == 1 for gold, pred in zip(labels, predictions))
    tn = sum(gold == 0 and pred == 0 for gold, pred in zip(labels, predictions))
    fn = sum(gold == 1 and pred == 0 for gold, pred in zip(labels, predictions))

    def f1(true_positive: int, false_positive: int, false_negative: int) -> float:
        denominator = 2 * true_positive + false_positive + false_negative
        return 0.0 if denominator == 0 else 2 * true_positive / denominator

    interrupt_f1 = f1(tp, fp, fn)
    silent_f1 = f1(tn, fn, fp)
    return {
        "macro_f1": (interrupt_f1 + silent_f1) / 2,
        "interrupt_f1": interrupt_f1,
        "silent_f1": silent_f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "predicted_interrupt_rate": (tp + fp) / len(labels),
        "support": len(labels),
    }


def select_threshold(logits: Sequence[float], labels: Sequence[int]) -> tuple[float, dict[str, float | int]]:
    """Select an exact label-separable threshold on one calibration fold."""
    if len(logits) != len(labels) or not logits:
        raise ValueError("Threshold calibration requires aligned logits and labels")
    ordered = sorted(zip(logits, labels), key=lambda item: item[0], reverse=True)
    candidates: list[tuple[float, dict[str, float | int]]] = []
    maximum = ordered[0][0]
    candidates.append((maximum + 1.0, binary_metrics(labels, [0] * len(labels))))
    index = 0
    while index < len(ordered):
        value = ordered[index][0]
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == value:
            stop += 1
        next_value = ordered[stop + 1][0] if stop + 1 < len(ordered) else value - 1.0
        threshold = (value + next_value) / 2
        predictions = [int(logit >= threshold) for logit in logits]
        candidates.append((threshold, binary_metrics(labels, predictions)))
        index = stop + 1
    threshold, metrics = max(
        candidates,
        key=lambda item: (
            float(item[1]["macro_f1"]),
            -abs(item[0]),
            -item[0],
        ),
    )
    return threshold, metrics


def cross_validate_linear(
    examples: Sequence[LabeledChunk],
    names: Sequence[str],
    folds: int,
    calibration_fold_offset: int,
    seed: int,
    max_iterations: int,
    l2_weight: float,
) -> tuple[dict[tuple[int, int], int], list[dict[str, object]]]:
    if folds < 3 or calibration_fold_offset % folds == 0:
        raise ValueError("D1 rotation requires distinct fit/calibration/test folds")
    decisions: dict[tuple[int, int], int] = {}
    details: list[dict[str, object]] = []
    for test_fold in range(folds):
        calibration_fold = (test_fold + calibration_fold_offset) % folds
        train = [
            example
            for example in examples
            if example.feature.fold not in (test_fold, calibration_fold)
        ]
        calibration = [
            example for example in examples if example.feature.fold == calibration_fold
        ]
        test = [example for example in examples if example.feature.fold == test_fold]
        if not train or not calibration or not test:
            raise ValueError("Every D1 rotation needs fit, calibration, and test chunks")
        train_x, train_y = matrix(train, names)
        calibration_x, calibration_y = matrix(calibration, names)
        test_x, test_y = matrix(test, names)
        model = fit_linear_logistic(
            train_x,
            train_y,
            seed=seed + test_fold,
            max_iterations=max_iterations,
            l2_weight=l2_weight,
        )
        calibration_logits = predict_logits(model, calibration_x)
        threshold, calibration_metrics = select_threshold(
            calibration_logits, calibration_y
        )
        test_logits = predict_logits(model, test_x)
        test_predictions = [int(logit >= threshold) for logit in test_logits]
        for example, decision in zip(test, test_predictions):
            if example.key in decisions:
                raise ValueError(f"Duplicate OOF decision for {example.key}")
            decisions[example.key] = decision
        details.append(
            {
                "test_fold": test_fold,
                "calibration_fold": calibration_fold,
                "fit_folds": sorted(set(range(folds)) - {test_fold, calibration_fold}),
                "fit_chunks": len(train),
                "calibration_chunks": len(calibration),
                "test_chunks": len(test),
                "fit_positive_rate": sum(train_y) / len(train_y),
                "threshold_logit": threshold,
                "train_loss": model.train_loss,
                "calibration_metrics": calibration_metrics,
                "test_metrics_internal": binary_metrics(test_y, test_predictions),
                "standardized_coefficients": {
                    name: weight for name, weight in zip(names, model.weight)
                },
                "bias": model.bias,
            }
        )
    expected = {example.key for example in examples}
    if set(decisions) != expected:
        raise ValueError("OOF decisions do not cover every D1 chunk exactly once")
    return decisions, details


def decision_answer(raw_response: str, interrupt: int) -> str:
    if not interrupt:
        return SILENT_TAG
    repaired, _ = repair_response_intent(raw_response)
    if repaired.startswith(INTERRUPT_TAG):
        return repaired
    return f"{INTERRUPT_TAG}{EMPTY_INTERRUPT_UTTERANCE}"


def prediction_rows(
    examples: Sequence[LabeledChunk],
    decisions: dict[tuple[int, int], int],
) -> list[dict[str, object]]:
    by_session: dict[int, list[LabeledChunk]] = {}
    for example in examples:
        by_session.setdefault(example.feature.input_index, []).append(example)
    rows: list[dict[str, object]] = []
    for input_index in sorted(by_session):
        session = sorted(by_session[input_index], key=lambda item: item.feature.chunk_index)
        expected_indices = list(range(session[0].feature.total_chunks))
        actual_indices = [item.feature.chunk_index for item in session]
        if actual_indices != expected_indices:
            raise ValueError(f"Non-contiguous D1 chunks for session {input_index}")
        rows.append(
            {
                "video_path": session[0].feature.video_path,
                "answers": [
                    decision_answer(item.feature.raw_response, decisions[item.key])
                    for item in session
                ],
            }
        )
    return rows


def decisions_from_feature(
    examples: Iterable[LabeledChunk], name: str
) -> dict[tuple[int, int], int]:
    return {
        example.key: int(example.feature.values[name] >= 0.5) for example in examples
    }


def metrics_for_subset(
    examples: Sequence[LabeledChunk],
    decisions: dict[tuple[int, int], int],
    include_first: bool,
) -> dict[str, float | int]:
    selected = [
        example
        for example in examples
        if include_first or example.feature.chunk_index > 0
    ]
    return binary_metrics(
        [example.gold_interrupt for example in selected],
        [decisions[example.key] for example in selected],
    )


def paired_session_bootstrap(
    examples: Sequence[LabeledChunk],
    candidate: dict[tuple[int, int], int],
    baseline: dict[tuple[int, int], int],
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    import numpy as np

    if repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be positive")
    sessions = sorted({example.feature.input_index for example in examples})
    candidate_counts = np.zeros((len(sessions), 4), dtype=np.int64)
    baseline_counts = np.zeros((len(sessions), 4), dtype=np.int64)
    position = {session: index for index, session in enumerate(sessions)}

    def add(counts: object, row: int, gold: int, pred: int) -> None:
        array = counts
        if gold == 1 and pred == 1:
            array[row, 0] += 1  # type: ignore[index]
        elif gold == 0 and pred == 1:
            array[row, 1] += 1  # type: ignore[index]
        elif gold == 0 and pred == 0:
            array[row, 2] += 1  # type: ignore[index]
        else:
            array[row, 3] += 1  # type: ignore[index]

    for example in examples:
        row = position[example.feature.input_index]
        add(candidate_counts, row, example.gold_interrupt, candidate[example.key])
        add(baseline_counts, row, example.gold_interrupt, baseline[example.key])

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(sessions), size=(repetitions, len(sessions)))
    candidate_sum = candidate_counts[sampled].sum(axis=1)
    baseline_sum = baseline_counts[sampled].sum(axis=1)

    def macro(counts: object) -> object:
        values = counts
        tp, fp, tn, fn = (values[:, index] for index in range(4))  # type: ignore[index]
        interrupt_denominator = 2 * tp + fp + fn
        silent_denominator = 2 * tn + fn + fp
        interrupt = np.divide(
            2 * tp,
            interrupt_denominator,
            out=np.zeros_like(tp, dtype=np.float64),
            where=interrupt_denominator != 0,
        )
        silent = np.divide(
            2 * tn,
            silent_denominator,
            out=np.zeros_like(tn, dtype=np.float64),
            where=silent_denominator != 0,
        )
        return (interrupt + silent) / 2

    delta = macro(candidate_sum) - macro(baseline_sum)
    lower, median, upper = np.quantile(delta, [0.025, 0.5, 0.975])
    return {
        "unit": "session",
        "repetitions": repetitions,
        "seed": seed,
        "delta_macro_f1_p2_5": float(lower),
        "delta_macro_f1_median": float(median),
        "delta_macro_f1_p97_5": float(upper),
        "positive_fraction": float((delta > 0).mean()),
    }
