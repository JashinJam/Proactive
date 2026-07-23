"""Validate and aggregate the frozen two-reviewer U0 ratings."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections import Counter, defaultdict
from typing import Callable, Iterable, Sequence

from proactive_u0.core import position_bin


REVIEWER_SLOTS = ("A", "B")
DECISION_SCORE_FIELDS = (
    "decision_confidence_1_5",
    "timeliness_1_5",
)
CONTENT_SCORE_FIELDS = (
    "correctness_1_5",
    "specificity_1_5",
    "actionability_1_5",
    "groundedness_1_5",
    "plan_consistency_1_5",
    "conciseness_1_5",
    "safety_1_5",
)
COMPOSITE_FIELDS = CONTENT_SCORE_FIELDS[:5]
FLAG_FIELDS = (
    "generic_flag",
    "hallucination_flag",
    "unsafe_flag",
)
SHOULD_INTERRUPT_VALUES = ("yes", "no", "uncertain")
ERROR_TYPES = (
    "none",
    "wrong_timing",
    "wrong_action",
    "wrong_object",
    "premature",
    "stale",
    "generic",
    "hallucination",
    "unsafe",
    "other",
)
STRATA = (
    "tp_fallback",
    "tp_nonfallback",
    "fp_fallback",
    "fp_nonfallback",
    "fn_silent",
)
BOOTSTRAP_SEED = 20260720
BOOTSTRAP_SAMPLES = 10_000


def _is_unrated(row: dict[str, str]) -> bool:
    fields = (
        "should_interrupt",
        *DECISION_SCORE_FIELDS,
        *CONTENT_SCORE_FIELDS,
        *FLAG_FIELDS,
        "primary_error_type",
    )
    return all(not str(row.get(field, "")).strip() for field in fields)


def _parse_score(value: object, field: str, key: tuple[str, str]) -> int:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(f"Invalid {field} for {key}: {value!r}") from error
    if not number.is_integer() or not 1 <= number <= 5:
        raise ValueError(f"{field} must be an integer from 1 to 5 for {key}")
    return int(number)


def _parse_flag(value: object, field: str, key: tuple[str, str]) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"0", "false", "no", "n"}:
        return 0
    if normalized in {"1", "true", "yes", "y"}:
        return 1
    raise ValueError(f"{field} must be yes/no (or binary) for {key}")


def _require_blank(
    row: dict[str, str], fields: Sequence[str], key: tuple[str, str]
) -> None:
    populated = [field for field in fields if str(row.get(field, "")).strip()]
    if populated:
        raise ValueError(f"Silent U0 row {key} populates content fields: {populated}")


def validate_ratings(
    rating_rows: Iterable[dict[str, str]],
    blind_rows: Sequence[dict[str, object]],
    key_rows: Sequence[dict[str, object]],
    expected_items: int = 200,
) -> list[dict[str, object]]:
    """Validate exact A/B coverage and U0's conditional rating schema."""
    blind_by_id = {str(row["review_id"]): row for row in blind_rows}
    key_by_id = {str(row["review_id"]): row for row in key_rows}
    if len(blind_by_id) != len(blind_rows):
        raise ValueError("U0 blind package contains duplicate review_id values")
    if len(key_by_id) != len(key_rows):
        raise ValueError("U0 answer key contains duplicate review_id values")
    if set(blind_by_id) != set(key_by_id):
        raise ValueError("U0 blind package and answer key coverage differ")
    if len(blind_by_id) != expected_items:
        raise ValueError(f"Expected {expected_items} U0 items, found {len(blind_by_id)}")

    for review_id, blind in blind_by_id.items():
        key = key_by_id[review_id]
        stratum = str(key.get("stratum"))
        if stratum not in STRATA:
            raise ValueError(f"Unexpected U0 stratum for {review_id}: {stratum}")
        model_action = str(blind.get("model_action"))
        expected_action = "silent" if stratum == "fn_silent" else "spoke"
        if model_action != expected_action:
            raise ValueError(
                f"U0 blind/key action mismatch for {review_id}: "
                f"{model_action} != {expected_action}"
            )

    expected = {
        (review_id, slot) for review_id in blind_by_id for slot in REVIEWER_SLOTS
    }
    seen: set[tuple[str, str]] = set()
    parsed: list[dict[str, object]] = []
    for raw in rating_rows:
        if _is_unrated(raw):
            continue
        review_id = str(raw.get("review_id", "")).strip()
        slot = str(raw.get("reviewer_slot", "")).strip()
        row_key = (review_id, slot)
        if row_key not in expected:
            raise ValueError(f"Unexpected U0 rating row: {row_key}")
        if row_key in seen:
            raise ValueError(f"Duplicate populated U0 rating row: {row_key}")
        seen.add(row_key)

        blind = blind_by_id[review_id]
        key = key_by_id[review_id]
        should_interrupt = str(raw.get("should_interrupt", "")).strip().lower()
        if should_interrupt not in SHOULD_INTERRUPT_VALUES:
            raise ValueError(
                f"Invalid should_interrupt for {row_key}: {should_interrupt!r}"
            )
        scores = {
            field: _parse_score(raw.get(field, ""), field, row_key)
            for field in DECISION_SCORE_FIELDS
        }
        flags: dict[str, int] = {}
        primary_error_type: str | None = None
        content_composite: float | None = None
        if str(blind["model_action"]) == "silent":
            _require_blank(
                raw,
                (*CONTENT_SCORE_FIELDS, *FLAG_FIELDS, "primary_error_type"),
                row_key,
            )
        else:
            scores.update(
                {
                    field: _parse_score(raw.get(field, ""), field, row_key)
                    for field in CONTENT_SCORE_FIELDS
                }
            )
            flags = {
                field: _parse_flag(raw.get(field, ""), field, row_key)
                for field in FLAG_FIELDS
            }
            primary_error_type = (
                str(raw.get("primary_error_type", "")).strip().lower()
            )
            if primary_error_type not in ERROR_TYPES:
                raise ValueError(
                    f"Invalid primary_error_type for {row_key}: "
                    f"{primary_error_type!r}"
                )
            content_composite = statistics.fmean(
                scores[field] for field in COMPOSITE_FIELDS
            )

        parsed.append(
            {
                "review_id": review_id,
                "reviewer_slot": slot,
                "input_index": int(key["input_index"]),
                "chunk_index": int(key["chunk_index"]),
                "domain": str(blind["domain"]),
                "task": str(blind["task"]),
                "position_bin": position_bin(int(key["chunk_index"])),
                "stratum": str(key["stratum"]),
                "confusion": str(key["confusion"]),
                "is_fallback": bool(key["is_fallback"]),
                "model_action": str(blind["model_action"]),
                "candidate_utterance": blind.get("candidate_utterance"),
                "should_interrupt": should_interrupt,
                "scores": scores,
                "content_composite": content_composite,
                "flags": flags,
                "primary_error_type": primary_error_type,
                "notes": str(raw.get("notes", "")).strip(),
                "session_id": str(raw.get("session_id", "")).strip(),
                "session_revision": str(raw.get("session_revision", "")).strip(),
                "confirmed_at": str(raw.get("confirmed_at", "")).strip(),
            }
        )

    missing = sorted(expected - seen)
    if missing:
        raise ValueError(
            f"U0 ratings are incomplete: {len(missing)} reviewer rows missing; "
            f"first={missing[:8]}"
        )
    return parsed


def _cohen_kappa(
    left: Sequence[object],
    right: Sequence[object],
    categories: Sequence[object],
    *,
    quadratic: bool,
) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa inputs must be non-empty and aligned")
    index = {value: offset for offset, value in enumerate(categories)}
    if any(value not in index for value in (*left, *right)):
        raise ValueError("Kappa input contains an unknown category")
    maximum = max(1, len(categories) - 1)

    def disagreement(a: object, b: object) -> float:
        if not quadratic:
            return float(a != b)
        distance = abs(index[a] - index[b]) / maximum
        return distance * distance

    observed = statistics.fmean(disagreement(a, b) for a, b in zip(left, right))
    left_rates = {value: left.count(value) / len(left) for value in categories}
    right_rates = {value: right.count(value) / len(right) for value in categories}
    expected = sum(
        left_rates[a] * right_rates[b] * disagreement(a, b)
        for a in categories
        for b in categories
    )
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return 1.0 - observed / expected


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("Correlation inputs must be non-empty and aligned")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for offset in ordered[start:end]:
            result[offset] = rank
        start = end
    return result


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of no values")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return seed + int(digest[:8], 16)


def _cluster_bootstrap(
    values: Sequence[tuple[int, float]],
    *,
    seed: int,
    samples: int,
    label: str,
) -> dict[str, object]:
    if not values:
        raise ValueError(f"Cannot bootstrap empty values for {label}")
    by_session: dict[int, list[float]] = defaultdict(list)
    for input_index, value in values:
        by_session[input_index].append(value)
    session_ids = sorted(by_session)
    rng = random.Random(_derived_seed(seed, label))
    distribution: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(session_ids) for _ in session_ids]
        flattened = [value for key in sampled for value in by_session[key]]
        distribution.append(statistics.fmean(flattened))
    distribution.sort()
    return {
        "unit": "session",
        "sessions": len(session_ids),
        "items": len(values),
        "resamples": samples,
        "seed": seed,
        "estimate": statistics.fmean(value for _, value in values),
        "ci95_low": _percentile(distribution, 0.025),
        "ci95_high": _percentile(distribution, 0.975),
    }


def build_item_records(
    parsed: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    by_review: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in parsed:
        by_review[str(row["review_id"])][str(row["reviewer_slot"])] = row
    items: list[dict[str, object]] = []
    for review_id in sorted(by_review):
        reviewers = by_review[review_id]
        if set(reviewers) != set(REVIEWER_SLOTS):
            raise ValueError(f"U0 item lacks both reviewers: {review_id}")
        left = reviewers["A"]
        right = reviewers["B"]
        for field in (
            "input_index",
            "chunk_index",
            "domain",
            "position_bin",
            "stratum",
            "model_action",
        ):
            if left[field] != right[field]:
                raise ValueError(f"U0 reviewer metadata differs for {review_id}: {field}")

        score_fields = list(DECISION_SCORE_FIELDS)
        if left["model_action"] == "spoke":
            score_fields.extend(CONTENT_SCORE_FIELDS)
        pair_scores = {
            field: statistics.fmean(
                [float(left["scores"][field]), float(right["scores"][field])]
            )
            for field in score_fields
        }
        score_deltas = {
            field: float(right["scores"][field]) - float(left["scores"][field])
            for field in score_fields
        }
        pair_flags: dict[str, float] = {}
        flag_deltas: dict[str, int] = {}
        if left["model_action"] == "spoke":
            pair_flags = {
                field: statistics.fmean(
                    [float(left["flags"][field]), float(right["flags"][field])]
                )
                for field in FLAG_FIELDS
            }
            flag_deltas = {
                field: int(right["flags"][field]) - int(left["flags"][field])
                for field in FLAG_FIELDS
            }
        pair_composite = None
        composite_delta = None
        if left["model_action"] == "spoke":
            pair_composite = statistics.fmean(
                [float(left["content_composite"]), float(right["content_composite"])]
            )
            composite_delta = float(right["content_composite"]) - float(
                left["content_composite"]
            )

        triggers: list[str] = []
        large_score_gaps = [
            field for field in score_fields if abs(score_deltas[field]) >= 2
        ]
        triggers.extend(f"score_gap:{field}" for field in large_score_gaps)
        if {left["should_interrupt"], right["should_interrupt"]} == {"yes", "no"}:
            triggers.append("should_interrupt_yes_no")
        triggers.extend(
            f"flag_mismatch:{field}" for field, value in flag_deltas.items() if value
        )
        if (
            left["model_action"] == "spoke"
            and left["primary_error_type"] != right["primary_error_type"]
        ):
            triggers.append("primary_error_type_mismatch")

        fallback_status = (
            "silent"
            if left["model_action"] == "silent"
            else ("fallback" if left["is_fallback"] else "nonfallback")
        )
        items.append(
            {
                "review_id": review_id,
                "input_index": left["input_index"],
                "chunk_index": left["chunk_index"],
                "domain": left["domain"],
                "task": left["task"],
                "position_bin": left["position_bin"],
                "stratum": left["stratum"],
                "confusion": left["confusion"],
                "is_fallback": left["is_fallback"],
                "fallback_status": fallback_status,
                "model_action": left["model_action"],
                "candidate_utterance": left["candidate_utterance"],
                "pair_scores": pair_scores,
                "score_deltas_b_minus_a": score_deltas,
                "pair_content_composite": pair_composite,
                "content_composite_delta_b_minus_a": composite_delta,
                "pair_flags": pair_flags,
                "flag_deltas_b_minus_a": flag_deltas,
                "reviewers": {
                    slot: {
                        "should_interrupt": reviewers[slot]["should_interrupt"],
                        "scores": reviewers[slot]["scores"],
                        "content_composite": reviewers[slot]["content_composite"],
                        "flags": reviewers[slot]["flags"],
                        "primary_error_type": reviewers[slot]["primary_error_type"],
                        "notes": reviewers[slot]["notes"],
                    }
                    for slot in REVIEWER_SLOTS
                },
                "disagreement_triggers": triggers,
                "max_score_gap": max(
                    (abs(value) for value in score_deltas.values()), default=0
                ),
            }
        )
    return items


def _contingency(
    left: Sequence[object], right: Sequence[object], categories: Sequence[object]
) -> dict[str, dict[str, int]]:
    return {
        str(a): {
            str(b): sum(x == a and y == b for x, y in zip(left, right))
            for b in categories
        }
        for a in categories
    }


def _ordinal_agreement(
    items: Sequence[dict[str, object]], field: str
) -> dict[str, object]:
    eligible = [item for item in items if field in item["pair_scores"]]
    left = [int(item["reviewers"]["A"]["scores"][field]) for item in eligible]
    right = [int(item["reviewers"]["B"]["scores"][field]) for item in eligible]
    differences = [b - a for a, b in zip(left, right)]
    return {
        "items": len(eligible),
        "quadratic_weighted_kappa": _cohen_kappa(
            left, right, tuple(range(1, 6)), quadratic=True
        ),
        "exact_agreement": statistics.fmean(a == b for a, b in zip(left, right)),
        "within_one_agreement": statistics.fmean(
            abs(a - b) <= 1 for a, b in zip(left, right)
        ),
        "mean_absolute_difference": statistics.fmean(
            abs(a - b) for a, b in zip(left, right)
        ),
        "pearson_correlation": _pearson(left, right),
        "spearman_correlation": _pearson(_ranks(left), _ranks(right)),
        "difference_histogram_b_minus_a": {
            str(value): count for value, count in sorted(Counter(differences).items())
        },
        "contingency_a_rows_b_columns": _contingency(
            left, right, tuple(range(1, 6))
        ),
    }


def _categorical_agreement(
    items: Sequence[dict[str, object]],
    categories: Sequence[object],
    getter: Callable[[dict[str, object], str], object],
) -> dict[str, object]:
    left = [getter(item, "A") for item in items]
    right = [getter(item, "B") for item in items]
    return {
        "items": len(items),
        "cohen_kappa": _cohen_kappa(left, right, categories, quadratic=False),
        "exact_agreement": statistics.fmean(a == b for a, b in zip(left, right)),
        "contingency_a_rows_b_columns": _contingency(left, right, categories),
    }


def _numeric_summary(
    items: Sequence[dict[str, object]],
    field: str,
    *,
    seed: int,
    samples: int,
    label: str,
) -> dict[str, object] | None:
    eligible = [item for item in items if field in item["pair_scores"]]
    if not eligible:
        return None
    left = [float(item["reviewers"]["A"]["scores"][field]) for item in eligible]
    right = [float(item["reviewers"]["B"]["scores"][field]) for item in eligible]
    pair_values = [
        (int(item["input_index"]), float(item["pair_scores"][field]))
        for item in eligible
    ]
    deltas = [
        (
            int(item["input_index"]),
            float(item["score_deltas_b_minus_a"][field]),
        )
        for item in eligible
    ]
    return {
        "items": len(eligible),
        "reviewer_a_mean": statistics.fmean(left),
        "reviewer_b_mean": statistics.fmean(right),
        "reviewer_b_minus_a": _cluster_bootstrap(
            deltas, seed=seed, samples=samples, label=f"{label}:delta:{field}"
        ),
        "pair_average": _cluster_bootstrap(
            pair_values, seed=seed, samples=samples, label=f"{label}:mean:{field}"
        ),
    }


def _composite_summary(
    items: Sequence[dict[str, object]],
    *,
    seed: int,
    samples: int,
    label: str,
) -> dict[str, object] | None:
    eligible = [item for item in items if item["pair_content_composite"] is not None]
    if not eligible:
        return None
    left = [float(item["reviewers"]["A"]["content_composite"]) for item in eligible]
    right = [float(item["reviewers"]["B"]["content_composite"]) for item in eligible]
    pair_values = [
        (int(item["input_index"]), float(item["pair_content_composite"]))
        for item in eligible
    ]
    deltas = [
        (
            int(item["input_index"]),
            float(item["content_composite_delta_b_minus_a"]),
        )
        for item in eligible
    ]
    return {
        "items": len(eligible),
        "reviewer_a_mean": statistics.fmean(left),
        "reviewer_b_mean": statistics.fmean(right),
        "reviewer_b_minus_a": _cluster_bootstrap(
            deltas, seed=seed, samples=samples, label=f"{label}:delta:composite"
        ),
        "pair_average": _cluster_bootstrap(
            pair_values, seed=seed, samples=samples, label=f"{label}:mean:composite"
        ),
    }


def _flag_summary(
    items: Sequence[dict[str, object]],
    field: str,
    *,
    seed: int,
    samples: int,
    label: str,
) -> dict[str, object] | None:
    eligible = [item for item in items if item["model_action"] == "spoke"]
    if not eligible:
        return None
    left = [int(item["reviewers"]["A"]["flags"][field]) for item in eligible]
    right = [int(item["reviewers"]["B"]["flags"][field]) for item in eligible]
    pair_values = [
        (int(item["input_index"]), float(item["pair_flags"][field]))
        for item in eligible
    ]
    deltas = [
        (int(item["input_index"]), float(item["flag_deltas_b_minus_a"][field]))
        for item in eligible
    ]
    return {
        "items": len(eligible),
        "reviewer_a_rate": statistics.fmean(left),
        "reviewer_b_rate": statistics.fmean(right),
        "reviewer_b_minus_a": _cluster_bootstrap(
            deltas, seed=seed, samples=samples, label=f"{label}:delta:{field}"
        ),
        "pair_average_rate": _cluster_bootstrap(
            pair_values, seed=seed, samples=samples, label=f"{label}:mean:{field}"
        ),
    }


def _should_summary(
    items: Sequence[dict[str, object]],
    *,
    seed: int,
    samples: int,
    label: str,
) -> dict[str, object]:
    counts = Counter(
        str(item["reviewers"][slot]["should_interrupt"])
        for item in items
        for slot in REVIEWER_SLOTS
    )
    rates = {value: counts[value] / (2 * len(items)) for value in SHOULD_INTERRUPT_VALUES}
    yes_values = [
        (
            int(item["input_index"]),
            statistics.fmean(
                item["reviewers"][slot]["should_interrupt"] == "yes"
                for slot in REVIEWER_SLOTS
            ),
        )
        for item in items
    ]
    return {
        "reviewer_rows": 2 * len(items),
        "counts": dict(counts),
        "rates": rates,
        "pair_average_yes_rate": _cluster_bootstrap(
            yes_values, seed=seed, samples=samples, label=f"{label}:should_yes"
        ),
    }


def _primary_error_distribution(items: Sequence[dict[str, object]]) -> dict[str, object]:
    spoken = [item for item in items if item["model_action"] == "spoke"]
    counts = Counter(
        str(item["reviewers"][slot]["primary_error_type"])
        for item in spoken
        for slot in REVIEWER_SLOTS
    )
    denominator = 2 * len(spoken)
    return {
        "reviewer_rows": denominator,
        "counts": {key: counts[key] for key in ERROR_TYPES},
        "rates": {
            key: counts[key] / denominator if denominator else None for key in ERROR_TYPES
        },
    }


def _overall_summary(
    items: Sequence[dict[str, object]],
    *,
    seed: int,
    samples: int,
    label: str,
) -> dict[str, object]:
    return {
        "items": len(items),
        "spoken_items": sum(item["model_action"] == "spoke" for item in items),
        "silent_items": sum(item["model_action"] == "silent" for item in items),
        "should_interrupt": _should_summary(
            items, seed=seed, samples=samples, label=label
        ),
        "scores": {
            field: _numeric_summary(
                items, field, seed=seed, samples=samples, label=label
            )
            for field in (*DECISION_SCORE_FIELDS, *CONTENT_SCORE_FIELDS)
        },
        "content_composite": _composite_summary(
            items, seed=seed, samples=samples, label=label
        ),
        "flags": {
            field: _flag_summary(
                items, field, seed=seed, samples=samples, label=label
            )
            for field in FLAG_FIELDS
        },
        "primary_error_type": _primary_error_distribution(items),
    }


def _group_summary(
    items: Sequence[dict[str, object]],
    *,
    seed: int,
    samples: int,
    label: str,
) -> dict[str, object]:
    return {
        "items": len(items),
        "spoken_items": sum(item["model_action"] == "spoke" for item in items),
        "should_interrupt": _should_summary(
            items, seed=seed, samples=samples, label=label
        ),
        "timeliness": _numeric_summary(
            items, "timeliness_1_5", seed=seed, samples=samples, label=label
        ),
        "content_composite": _composite_summary(
            items, seed=seed, samples=samples, label=label
        ),
        "groundedness": _numeric_summary(
            items, "groundedness_1_5", seed=seed, samples=samples, label=label
        ),
        "hallucination": _flag_summary(
            items, "hallucination_flag", seed=seed, samples=samples, label=label
        ),
        "primary_error_type": _primary_error_distribution(items),
    }


def _grouped(
    items: Sequence[dict[str, object]],
    field: str,
    *,
    seed: int,
    samples: int,
) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in items:
        groups[str(item[field])].append(item)
    return {
        name: _group_summary(
            groups[name],
            seed=seed,
            samples=samples,
            label=f"{field}:{name}",
        )
        for name in sorted(groups)
    }


def _agreement(items: Sequence[dict[str, object]]) -> dict[str, object]:
    spoken = [item for item in items if item["model_action"] == "spoke"]
    return {
        "ordinal_scores": {
            field: _ordinal_agreement(items, field)
            for field in (*DECISION_SCORE_FIELDS, *CONTENT_SCORE_FIELDS)
        },
        "should_interrupt": _categorical_agreement(
            items,
            SHOULD_INTERRUPT_VALUES,
            lambda item, slot: item["reviewers"][slot]["should_interrupt"],
        ),
        "flags": {
            field: _categorical_agreement(
                spoken,
                (0, 1),
                lambda item, slot, name=field: item["reviewers"][slot]["flags"][
                    name
                ],
            )
            for field in FLAG_FIELDS
        },
        "primary_error_type": _categorical_agreement(
            spoken,
            ERROR_TYPES,
            lambda item, slot: item["reviewers"][slot]["primary_error_type"],
        ),
    }


def _disagreement_records(
    items: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    disagreements = [item for item in items if item["disagreement_triggers"]]
    disagreements.sort(
        key=lambda item: (
            -int(
                any(
                    trigger == "flag_mismatch:unsafe_flag"
                    for trigger in item["disagreement_triggers"]
                )
            ),
            -int(
                any(
                    trigger == "flag_mismatch:hallucination_flag"
                    for trigger in item["disagreement_triggers"]
                )
            ),
            -int(item["max_score_gap"]),
            -len(item["disagreement_triggers"]),
            str(item["review_id"]),
        )
    )
    trigger_counts = Counter(
        trigger
        for item in disagreements
        for trigger in item["disagreement_triggers"]
    )
    return disagreements, {
        "items": len(disagreements),
        "item_rate": len(disagreements) / len(items),
        "trigger_counts": dict(sorted(trigger_counts.items())),
        "protocol": (
            "score gap >=2, should yes/no opposition, flag mismatch, or primary "
            "error mismatch"
        ),
    }


def analyze_ratings(
    parsed: Sequence[dict[str, object]],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Return aggregate analysis, item-level A/B records, and adjudication cases."""
    items = build_item_records(parsed)
    disagreements, disagreement_summary = _disagreement_records(items)
    analysis = {
        "status": "complete U0 two-reviewer analysis",
        "classification": (
            "public-validation hard-stratum diagnostic; not population prevalence "
            "or official metric evidence"
        ),
        "validation": {
            "items": len(items),
            "reviewer_rows": len(parsed),
            "reviewers": list(REVIEWER_SLOTS),
            "spoken_items": sum(item["model_action"] == "spoke" for item in items),
            "silent_items": sum(item["model_action"] == "silent" for item in items),
        },
        "bootstrap": {
            "unit": "session",
            "seed": bootstrap_seed,
            "resamples": bootstrap_samples,
            "method": "cluster resample sessions and retain all sampled-session items",
        },
        "overall": _overall_summary(
            items,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
            label="overall",
        ),
        "agreement": _agreement(items),
        "stratified": {
            "by_stratum": _grouped(
                items,
                "stratum",
                seed=bootstrap_seed,
                samples=bootstrap_samples,
            ),
            "by_position": _grouped(
                items,
                "position_bin",
                seed=bootstrap_seed,
                samples=bootstrap_samples,
            ),
            "by_domain": _grouped(
                items,
                "domain",
                seed=bootstrap_seed,
                samples=bootstrap_samples,
            ),
            "by_fallback_status": _grouped(
                items,
                "fallback_status",
                seed=bootstrap_seed,
                samples=bootstrap_samples,
            ),
            "by_confusion": _grouped(
                items,
                "confusion",
                seed=bootstrap_seed,
                samples=bootstrap_samples,
            ),
        },
        "disagreement": disagreement_summary,
    }
    return analysis, items, disagreements
