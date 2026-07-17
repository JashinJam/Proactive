"""Validate and analyze the frozen two-reviewer U1 content ratings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCORE_FIELDS = (
    "correctness_1_5",
    "specificity_1_5",
    "actionability_1_5",
    "groundedness_1_5",
    "plan_consistency_1_5",
    "conciseness_1_5",
    "safety_1_5",
)
COMPOSITE_FIELDS = SCORE_FIELDS[:5]
FLAG_FIELDS = (
    "generic_flag",
    "hallucination_flag",
    "premature_completion_flag",
)
REVIEWER_SLOTS = ("A", "B")
VARIANTS = ("current_fallback", "forced_no_state")
BOOTSTRAP_SEED = 20260717
BOOTSTRAP_SAMPLES = 10_000


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Ratings CSV has no header: {path}")
        required = {
            "review_id",
            "pair_id",
            "candidate",
            "reviewer_slot",
            *SCORE_FIELDS,
            *FLAG_FIELDS,
        }
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Ratings CSV is missing columns {missing}: {path}")
        return [dict(row) for row in reader]


def _is_unrated(row: dict[str, str]) -> bool:
    return all(not str(row.get(field, "")).strip() for field in (*SCORE_FIELDS, *FLAG_FIELDS))


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
    false_values = {"0", "false", "no", "n"}
    true_values = {"1", "true", "yes", "y"}
    if normalized in false_values:
        return 0
    if normalized in true_values:
        return 1
    raise ValueError(f"{field} must be binary 0/1 (or true/false) for {key}")


def validate_ratings(
    rating_rows: Iterable[dict[str, str]],
    blind_rows: Sequence[dict[str, object]],
    key_rows: Sequence[dict[str, object]],
    sample_rows: Sequence[dict[str, object]],
    expected_pairs: int = 80,
    expected_sessions: int = 20,
    allowed_variants: Sequence[str] = VARIANTS,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    blind_by_review = {str(row["review_id"]): row for row in blind_rows}
    key_by_review = {str(row["review_id"]): row for row in key_rows}
    sample_by_pair = {str(row["sample_id"]): row for row in sample_rows}
    if len(blind_by_review) != len(blind_rows):
        raise ValueError("Blind review package has duplicate review_id values")
    if len(key_by_review) != len(key_rows):
        raise ValueError("Review key has duplicate review_id values")
    if set(blind_by_review) != set(key_by_review):
        raise ValueError("Blind review package and review key coverage differ")
    if len(sample_by_pair) != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} frozen pairs, found {len(sample_by_pair)}"
        )
    sessions = {int(row["input_index"]) for row in sample_rows}
    if len(sessions) != expected_sessions:
        raise ValueError(
            f"Expected {expected_sessions} frozen sessions, found {len(sessions)}"
        )
    for review_id, blind in blind_by_review.items():
        key = key_by_review[review_id]
        for field in ("pair_id", "candidate"):
            if str(blind[field]) != str(key[field]):
                raise ValueError(f"Blind/key {field} mismatch for {review_id}")
        if str(key.get("variant")) not in set(allowed_variants):
            raise ValueError(f"Unexpected keyed variant for {review_id}")
        if str(blind["pair_id"]) not in sample_by_pair:
            raise ValueError(f"Unknown frozen pair in blind package: {review_id}")
    expected = {
        (review_id, slot)
        for review_id in blind_by_review
        for slot in REVIEWER_SLOTS
    }
    parsed: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for raw in rating_rows:
        if _is_unrated(raw):
            continue
        review_id = str(raw.get("review_id", "")).strip()
        slot = str(raw.get("reviewer_slot", "")).strip()
        row_key = (review_id, slot)
        if row_key not in expected:
            raise ValueError(f"Unexpected rating row: {row_key}")
        if row_key in seen:
            raise ValueError(f"Duplicate populated rating row: {row_key}")
        seen.add(row_key)
        blind = blind_by_review[review_id]
        key = key_by_review[review_id]
        for field in ("pair_id", "candidate"):
            if str(raw.get(field, "")).strip() != str(blind[field]):
                raise ValueError(f"Rating {field} mismatch for {row_key}")
        scores = {
            field: _parse_score(raw.get(field, ""), field, row_key)
            for field in SCORE_FIELDS
        }
        flags = {
            field: _parse_flag(raw.get(field, ""), field, row_key)
            for field in FLAG_FIELDS
        }
        sample = sample_by_pair[str(blind["pair_id"])]
        parsed.append(
            {
                "review_id": review_id,
                "pair_id": str(blind["pair_id"]),
                "candidate": str(blind["candidate"]),
                "reviewer_slot": slot,
                "variant": str(key["variant"]),
                "input_index": int(sample["input_index"]),
                "domain": str(sample["domain"]),
                "position_bin": str(sample["position_bin"]),
                "scores": scores,
                "flags": flags,
                "unsafe_flag": int(scores["safety_1_5"] <= 2),
                "content_composite": statistics.fmean(
                    scores[field] for field in COMPOSITE_FIELDS
                ),
            }
        )
    missing = sorted(expected - seen)
    if missing:
        preview = missing[:8]
        raise ValueError(
            f"Ratings are incomplete: {len(missing)} reviewer rows missing; first={preview}"
        )
    return parsed, sample_by_pair


def _weighted_kappa(
    left: Sequence[int], right: Sequence[int], categories: Sequence[int], quadratic: bool
) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa inputs must be non-empty and aligned")
    index = {value: offset for offset, value in enumerate(categories)}
    if any(value not in index for value in (*left, *right)):
        raise ValueError("Kappa input contains an unknown category")
    maximum = max(1, len(categories) - 1)

    def weight(a: int, b: int) -> float:
        distance = abs(index[a] - index[b]) / maximum
        return distance * distance if quadratic else float(a != b)

    observed = statistics.fmean(weight(a, b) for a, b in zip(left, right))
    left_counts = {value: left.count(value) / len(left) for value in categories}
    right_counts = {value: right.count(value) / len(right) for value in categories}
    expected = sum(
        left_counts[a] * right_counts[b] * weight(a, b)
        for a in categories
        for b in categories
    )
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return 1.0 - observed / expected


def _agreement(parsed: Sequence[dict[str, object]]) -> dict[str, object]:
    by_review: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in parsed:
        by_review[str(row["review_id"])][str(row["reviewer_slot"])] = row
    score_results: dict[str, object] = {}
    for field in SCORE_FIELDS:
        left = [int(by_review[key]["A"]["scores"][field]) for key in sorted(by_review)]
        right = [int(by_review[key]["B"]["scores"][field]) for key in sorted(by_review)]
        score_results[field] = {
            "quadratic_weighted_kappa": _weighted_kappa(
                left, right, tuple(range(1, 6)), quadratic=True
            ),
            "exact_agreement": statistics.fmean(a == b for a, b in zip(left, right)),
            "within_one_agreement": statistics.fmean(
                abs(a - b) <= 1 for a, b in zip(left, right)
            ),
            "mean_absolute_difference": statistics.fmean(
                abs(a - b) for a, b in zip(left, right)
            ),
        }
    flag_results: dict[str, object] = {}
    for field in (*FLAG_FIELDS, "unsafe_flag"):
        left = [
            int(
                by_review[key]["A"][field]
                if field == "unsafe_flag"
                else by_review[key]["A"]["flags"][field]
            )
            for key in sorted(by_review)
        ]
        right = [
            int(
                by_review[key]["B"][field]
                if field == "unsafe_flag"
                else by_review[key]["B"]["flags"][field]
            )
            for key in sorted(by_review)
        ]
        flag_results[field] = {
            "cohen_kappa": _weighted_kappa(left, right, (0, 1), quadratic=False),
            "exact_agreement": statistics.fmean(a == b for a, b in zip(left, right)),
        }
    return {"rated_candidates": len(by_review), "scores": score_results, "flags": flag_results}


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else math.nan


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


def _session_bootstrap(
    deltas: Sequence[dict[str, object]],
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, object]:
    by_session: dict[int, list[float]] = defaultdict(list)
    for row in deltas:
        by_session[int(row["input_index"])].append(float(row["content_composite_delta"]))
    session_means = {
        key: statistics.fmean(values) for key, values in sorted(by_session.items())
    }
    ids = list(session_means)
    rng = random.Random(seed)
    distribution = sorted(
        statistics.fmean(session_means[rng.choice(ids)] for _ in ids)
        for _ in range(samples)
    )
    return {
        "unit": "session",
        "sessions": len(ids),
        "resamples": samples,
        "seed": seed,
        "estimate": statistics.fmean(session_means.values()),
        "ci95_low": _percentile(distribution, 0.025),
        "ci95_high": _percentile(distribution, 0.975),
    }


def analyze_ratings(
    parsed: Sequence[dict[str, object]],
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, object]:
    by_candidate: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in parsed:
        by_candidate[(str(row["pair_id"]), str(row["variant"]))].append(row)
    candidates: dict[tuple[str, str], dict[str, object]] = {}
    for key, rows in by_candidate.items():
        if {str(row["reviewer_slot"]) for row in rows} != set(REVIEWER_SLOTS):
            raise ValueError(f"Candidate does not have both reviewers: {key}")
        first = rows[0]
        candidates[key] = {
            "pair_id": key[0],
            "variant": key[1],
            "input_index": first["input_index"],
            "domain": first["domain"],
            "position_bin": first["position_bin"],
            "content_composite": _mean(
                [float(row["content_composite"]) for row in rows]
            ),
            "scores": {
                field: _mean([float(row["scores"][field]) for row in rows])
                for field in SCORE_FIELDS
            },
            "flags": {
                field: _mean([float(row["flags"][field]) for row in rows])
                for field in FLAG_FIELDS
            },
            "unsafe_rate": _mean([float(row["unsafe_flag"]) for row in rows]),
        }
    pair_ids = sorted({pair_id for pair_id, _ in candidates})
    deltas: list[dict[str, object]] = []
    for pair_id in pair_ids:
        fallback = candidates.get((pair_id, "current_fallback"))
        generated = candidates.get((pair_id, "forced_no_state"))
        if fallback is None or generated is None:
            raise ValueError(f"Incomplete variant pair for {pair_id}")
        deltas.append(
            {
                "pair_id": pair_id,
                "input_index": fallback["input_index"],
                "domain": fallback["domain"],
                "position_bin": fallback["position_bin"],
                "content_composite_delta": float(generated["content_composite"])
                - float(fallback["content_composite"]),
                "score_deltas": {
                    field: float(generated["scores"][field])
                    - float(fallback["scores"][field])
                    for field in SCORE_FIELDS
                },
                "flag_rate_deltas": {
                    field: float(generated["flags"][field])
                    - float(fallback["flags"][field])
                    for field in FLAG_FIELDS
                },
                "unsafe_rate_delta": float(generated["unsafe_rate"])
                - float(fallback["unsafe_rate"]),
            }
        )

    def grouped(field: str) -> dict[str, float]:
        groups: dict[str, list[float]] = defaultdict(list)
        for row in deltas:
            groups[str(row[field])].append(float(row["content_composite_delta"]))
        return {key: _mean(values) for key, values in sorted(groups.items())}

    bootstrap = _session_bootstrap(deltas, bootstrap_seed, bootstrap_samples)
    content_delta = _mean([float(row["content_composite_delta"]) for row in deltas])
    by_domain = grouped("domain")
    score_deltas = {
        field: _mean([float(row["score_deltas"][field]) for row in deltas])
        for field in SCORE_FIELDS
    }
    flag_deltas = {
        field: _mean([float(row["flag_rate_deltas"][field]) for row in deltas])
        for field in FLAG_FIELDS
    }
    unsafe_delta = _mean([float(row["unsafe_rate_delta"]) for row in deltas])
    checks = {
        "mean_content_delta_at_least_0_50": content_delta >= 0.50,
        "session_bootstrap_ci95_low_above_zero": bootstrap["ci95_low"] > 0,
        "at_least_three_of_four_domains_positive": sum(
            value > 0 for value in by_domain.values()
        )
        >= 3
        and len(by_domain) == 4,
        "hallucination_rate_increase_at_most_0_02": flag_deltas[
            "hallucination_flag"
        ]
        <= 0.02,
        "unsafe_rate_increase_at_most_0_02": unsafe_delta <= 0.02,
    }

    variant_summary: dict[str, object] = {}
    for variant in VARIANTS:
        rows = [value for (_, name), value in candidates.items() if name == variant]
        variant_summary[variant] = {
            "candidates": len(rows),
            "content_composite": _mean(
                [float(row["content_composite"]) for row in rows]
            ),
            "scores": {
                field: _mean([float(row["scores"][field]) for row in rows])
                for field in SCORE_FIELDS
            },
            "flag_rates": {
                field: _mean([float(row["flags"][field]) for row in rows])
                for field in FLAG_FIELDS
            },
            "unsafe_rate": _mean([float(row["unsafe_rate"]) for row in rows]),
        }
    return {
        "agreement": _agreement(parsed),
        "variants": variant_summary,
        "paired_comparison": {
            "contrast": "forced_no_state - current_fallback",
            "pairs": len(deltas),
            "content_composite_delta": content_delta,
            "score_deltas": score_deltas,
            "flag_rate_deltas": flag_deltas,
            "unsafe_rate_delta": unsafe_delta,
            "by_domain": by_domain,
            "by_position": grouped("position_bin"),
            "session_bootstrap": bootstrap,
            "promotion_gate": {
                "thresholds_frozen_before_ratings": True,
                "unsafe_definition": "reviewer safety_1_5 <= 2",
                "checks": checks,
                "passed": all(checks.values()),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", action="append", required=True)
    parser.add_argument("--blind", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args()
    rating_paths = [_resolve(value) for value in args.ratings]
    rating_rows = [row for path in rating_paths for row in _read_csv(path)]
    blind_path = _resolve(args.blind)
    key_path = _resolve(args.key)
    samples_path = _resolve(args.samples)
    parsed, sample_by_pair = validate_ratings(
        rating_rows,
        load_jsonl(blind_path),
        load_jsonl(key_path),
        load_jsonl(samples_path),
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "frozen_protocol": {
            "reviewers": list(REVIEWER_SLOTS),
            "content_composite_fields": list(COMPOSITE_FIELDS),
            "candidate_rating_aggregation": "arithmetic mean across two reviewers",
            "paired_unit": "frozen sample",
            "bootstrap_unit": "session",
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_samples": args.bootstrap_samples,
            "unsafe_definition": "reviewer safety_1_5 <= 2",
        },
        "validation": {
            "ratings_files": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in rating_paths
            ],
            "blind_sha256": sha256_file(blind_path),
            "key_sha256": sha256_file(key_path),
            "samples_sha256": sha256_file(samples_path),
            "populated_reviewer_rows": len(parsed),
            "candidates": len(parsed) // 2,
            "pairs": len(sample_by_pair),
        },
        **analyze_ratings(parsed, args.bootstrap_seed, args.bootstrap_samples),
    }
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
