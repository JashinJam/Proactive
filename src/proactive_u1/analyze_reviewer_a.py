"""Reproduce provisional reviewer-A-only U0/U1 diagnostics without reading B."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Sequence

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl
from proactive_u1.ratings import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    COMPOSITE_FIELDS,
    FLAG_FIELDS,
    SCORE_FIELDS,
    _mean,
    _read_csv,
    _session_bootstrap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
U0_SCORE_FIELDS = (
    "correctness_1_5",
    "specificity_1_5",
    "actionability_1_5",
    "groundedness_1_5",
    "plan_consistency_1_5",
    "conciseness_1_5",
    "safety_1_5",
)
STOP_WORDS = {
    "a", "an", "and", "for", "in", "is", "it", "next", "now", "of",
    "on", "please", "step", "that", "the", "then", "this", "to", "with",
    "you", "your",
}
WORD_RE = re.compile(r"[a-z]+")


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _plain_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _require_reviewer_a(rows: Sequence[dict[str, str]], expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} reviewer-A rows, found {len(rows)}")
    slots = {str(row.get("reviewer_slot", "")).strip() for row in rows}
    if slots != {"A"}:
        raise ValueError(f"A-only diagnostic refuses reviewer slots {sorted(slots)}")


def _flag(value: object) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1", "y"}:
        return 1
    if normalized in {"no", "false", "0", "n"}:
        return 0
    raise ValueError(f"Expected binary flag, got {value!r}")


def _tokens(value: str) -> set[str]:
    return set(WORD_RE.findall(value.lower())) - STOP_WORDS


def _group_summary(
    rows: Sequence[dict[str, object]],
    key: str,
    value: Callable[[dict[str, object]], float],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(float(value(row)))
    return {
        name: {"n": len(values), "mean": _mean(values)}
        for name, values in sorted(grouped.items())
    }


def _u1_variant_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    return {
        "candidates": len(rows),
        "content_composite": _mean([float(row["content_composite"]) for row in rows]),
        "scores": {
            field: _mean([float(row["scores"][field]) for row in rows])
            for field in SCORE_FIELDS
        },
        "flag_rates": {
            field: _mean([float(row["flags"][field]) for row in rows])
            for field in FLAG_FIELDS
        },
        "unsafe_rate": _mean([float(row["unsafe_flag"]) for row in rows]),
        "used_fallback_rate": _mean([float(row["used_fallback"]) for row in rows]),
        "primary_error_counts": dict(Counter(str(row["primary_error_type"]) for row in rows)),
    }


def analyze_u1_a(
    ratings_path: Path,
    blind_path: Path,
    key_path: Path,
    samples_path: Path,
    history_cap: int = 4,
) -> dict[str, object]:
    rating_rows = _read_csv(ratings_path)
    _require_reviewer_a(rating_rows, 160)
    blind_rows = load_jsonl(blind_path)
    key_rows = load_jsonl(key_path)
    sample_rows = load_jsonl(samples_path)
    blind = {str(row["review_id"]): row for row in blind_rows}
    keys = {str(row["review_id"]): row for row in key_rows}
    samples = {str(row["sample_id"]): row for row in sample_rows}
    ratings = {str(row["review_id"]): row for row in rating_rows}
    if len(ratings) != len(rating_rows) or set(ratings) != set(blind) or set(blind) != set(keys):
        raise ValueError("U1 A ratings/blind/key coverage differs or contains duplicates")
    if len(samples) != 80 or len({int(row["input_index"]) for row in sample_rows}) != 20:
        raise ValueError("U1 A diagnostic requires frozen 80 samples / 20 sessions")

    parsed: list[dict[str, object]] = []
    for review_id, raw in ratings.items():
        blind_row = blind[review_id]
        key_row = keys[review_id]
        pair_id = str(blind_row["pair_id"])
        sample = samples[pair_id]
        scores = {field: int(raw[field]) for field in SCORE_FIELDS}
        flags = {field: _flag(raw[field]) for field in FLAG_FIELDS}
        assistant_history = [
            str(turn.get("text", ""))
            for turn in blind_row["prior_dialog"][1:]  # type: ignore[index]
            if isinstance(turn, dict)
            and turn.get("role") == "assistant"
            and str(turn.get("text", "")).strip()
        ][-history_cap:]
        utterance = str(blind_row.get("candidate_utterance") or "")
        utterance_tokens = _tokens(utterance)
        history_tokens = _tokens(" ".join(assistant_history))
        parsed.append(
            {
                "review_id": review_id,
                "pair_id": pair_id,
                "variant": str(key_row["variant"]),
                "used_fallback": bool(key_row["used_fallback"]),
                "input_index": int(sample["input_index"]),
                "domain": str(sample["domain"]),
                "position_bin": str(sample["position_bin"]),
                "chunk_index": int(sample["chunk_index"]),
                "effective_assistant_turns": len(assistant_history),
                "scores": scores,
                "flags": flags,
                "unsafe_flag": int(scores["safety_1_5"] <= 2),
                "content_composite": statistics.fmean(
                    scores[field] for field in COMPOSITE_FIELDS
                ),
                "primary_error_type": str(raw.get("primary_error_type", "")),
                "history_content_token_overlap": (
                    len(utterance_tokens & history_tokens) / len(utterance_tokens)
                    if utterance_tokens else 0.0
                ),
                "max_string_similarity_to_prior_turn": max(
                    [
                        difflib.SequenceMatcher(
                            None, utterance.lower(), value.lower()
                        ).ratio()
                        for value in assistant_history
                    ]
                    or [0.0]
                ),
            }
        )

    by_candidate = {
        (str(row["pair_id"]), str(row["variant"])): row for row in parsed
    }
    if len(by_candidate) != 160:
        raise ValueError("U1 A contains duplicate sample/variant candidates")
    deltas: list[dict[str, object]] = []
    for pair_id in sorted(samples):
        fallback = by_candidate[(pair_id, "current_fallback")]
        generated = by_candidate[(pair_id, "forced_no_state")]
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
                "unsafe_rate_delta": float(generated["unsafe_flag"])
                - float(fallback["unsafe_flag"]),
            }
        )

    content_delta = _mean([float(row["content_composite_delta"]) for row in deltas])
    bootstrap = _session_bootstrap(deltas, BOOTSTRAP_SEED, BOOTSTRAP_SAMPLES)
    by_domain = _group_summary(
        deltas, "domain", lambda row: float(row["content_composite_delta"])
    )
    by_position = _group_summary(
        deltas, "position_bin", lambda row: float(row["content_composite_delta"])
    )
    flag_deltas = {
        field: _mean([float(row["flag_rate_deltas"][field]) for row in deltas])
        for field in FLAG_FIELDS
    }
    unsafe_delta = _mean([float(row["unsafe_rate_delta"]) for row in deltas])
    checks = {
        "mean_content_delta_at_least_0_50": content_delta >= 0.50,
        "session_bootstrap_ci95_low_above_zero": float(bootstrap["ci95_low"]) > 0,
        "at_least_three_of_four_domains_positive": len(by_domain) == 4
        and sum(float(value["mean"]) > 0 for value in by_domain.values()) >= 3,
        "hallucination_rate_increase_at_most_0_02": flag_deltas[
            "hallucination_flag"
        ] <= 0.02,
        "unsafe_rate_increase_at_most_0_02": unsafe_delta <= 0.02,
    }

    generated_rows = [row for row in parsed if row["variant"] == "forced_no_state"]
    position_details: dict[str, object] = {}
    for position in ("1:second", "2-4", "5-9", "10+"):
        rows = [row for row in generated_rows if row["position_bin"] == position]
        position_details[position] = {
            **_u1_variant_summary(rows),
            "paired_delta": by_position[position],
        }
    history_details: dict[str, object] = {}
    for turns in sorted({int(row["effective_assistant_turns"]) for row in generated_rows}):
        rows = [
            row for row in generated_rows
            if int(row["effective_assistant_turns"]) == turns
        ]
        history_details[str(turns)] = _u1_variant_summary(rows)
    nonfallback = [row for row in generated_rows if not row["used_fallback"]]
    return {
        "status": "provisional_A_only_diagnostic; not two-reviewer promotion evidence",
        "reviewer_slot": "A",
        "actual_model_history_cap": history_cap,
        "variants": {
            variant: _u1_variant_summary(
                [row for row in parsed if row["variant"] == variant]
            )
            for variant in ("current_fallback", "forced_no_state")
        },
        "paired_comparison": {
            "contrast": "forced_no_state - current_fallback",
            "content_composite_delta": content_delta,
            "score_deltas": {
                field: _mean([float(row["score_deltas"][field]) for row in deltas])
                for field in SCORE_FIELDS
            },
            "flag_rate_deltas": flag_deltas,
            "unsafe_rate_delta": unsafe_delta,
            "by_domain": by_domain,
            "by_position": by_position,
            "session_bootstrap": bootstrap,
            "provisional_gate": {"checks": checks, "passed": all(checks.values())},
        },
        "forced_no_state_by_position": position_details,
        "forced_no_state_by_effective_assistant_turns": history_details,
        "nonfallback_history_overlap_diagnostic": {
            "candidates": len(nonfallback),
            "mean_content_token_overlap": _mean(
                [float(row["history_content_token_overlap"]) for row in nonfallback]
            ),
            "median_content_token_overlap": statistics.median(
                float(row["history_content_token_overlap"]) for row in nonfallback
            ),
            "mean_max_string_similarity_to_one_prior_turn": _mean(
                [float(row["max_string_similarity_to_prior_turn"]) for row in nonfallback]
            ),
            "warning": "Lexical overlap is not causal proof of history use.",
        },
        "coverage": {"candidates": 160, "pairs": 80, "sessions": 20},
    }


def _u0_action_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    return {
        "samples": len(rows),
        "human_should_interrupt": dict(Counter(str(row["should_interrupt"]) for row in rows)),
        "mean_decision_confidence": _mean(
            [float(row["decision_confidence"]) for row in rows]
        ),
        "mean_timeliness": _mean([float(row["timeliness"]) for row in rows]),
    }


def analyze_u0_a(ratings_path: Path, blind_path: Path, key_path: Path) -> dict[str, object]:
    ratings = _plain_csv(ratings_path)
    _require_reviewer_a(ratings, 200)
    blind = {str(row["review_id"]): row for row in load_jsonl(blind_path)}
    keys = {str(row["review_id"]): row for row in load_jsonl(key_path)}
    rating_by_id = {str(row["review_id"]): row for row in ratings}
    if len(rating_by_id) != 200 or set(rating_by_id) != set(blind) or set(blind) != set(keys):
        raise ValueError("U0 A ratings/blind/key coverage differs or contains duplicates")
    parsed: list[dict[str, object]] = []
    for review_id, raw in rating_by_id.items():
        blind_row = blind[review_id]
        key_row = keys[review_id]
        chunk = int(blind_row["chunk_index"])
        position = (
            "first" if chunk == 0 else "second" if chunk == 1 else
            "2-4" if chunk <= 4 else "5-9" if chunk <= 9 else "10+"
        )
        scores = {
            field: int(raw[field])
            for field in U0_SCORE_FIELDS if str(raw.get(field, "")).strip()
        }
        parsed.append(
            {
                "review_id": review_id,
                "model_action": str(blind_row["model_action"]),
                "should_interrupt": str(raw["should_interrupt"]),
                "decision_confidence": int(raw["decision_confidence_1_5"]),
                "timeliness": int(raw["timeliness_1_5"]),
                "stratum": str(key_row["stratum"]),
                "confusion": str(key_row["confusion"]),
                "gold_interrupt": bool(key_row["gold_interrupt"]),
                "is_fallback": bool(key_row["is_fallback"]),
                "domain": str(blind_row["domain"]),
                "position_bin": position,
                "scores": scores,
                "content_composite": (
                    statistics.fmean(scores[field] for field in COMPOSITE_FIELDS)
                    if scores else None
                ),
                "generic_flag": (
                    _flag(raw["generic_flag"]) if str(raw["generic_flag"]).strip() else None
                ),
                "hallucination_flag": (
                    _flag(raw["hallucination_flag"])
                    if str(raw["hallucination_flag"]).strip() else None
                ),
                "primary_error_type": str(raw.get("primary_error_type", "")),
            }
        )
    spoke = [row for row in parsed if row["model_action"] == "spoke"]
    silent = [row for row in parsed if row["model_action"] == "silent"]

    def content_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
        return {
            "samples": len(rows),
            "content_composite": _mean(
                [float(row["content_composite"]) for row in rows]
            ),
            "scores": {
                field: _mean([float(row["scores"][field]) for row in rows])
                for field in U0_SCORE_FIELDS
            },
            "generic_rate": _mean([float(row["generic_flag"]) for row in rows]),
            "hallucination_rate": _mean(
                [float(row["hallucination_flag"]) for row in rows]
            ),
            "primary_error_counts": dict(
                Counter(str(row["primary_error_type"]) for row in rows)
            ),
        }

    decided = [row for row in parsed if row["should_interrupt"] != "uncertain"]
    matches = sum(
        (row["model_action"] == "spoke" and row["should_interrupt"] == "yes")
        or (row["model_action"] == "silent" and row["should_interrupt"] == "no")
        for row in decided
    )
    return {
        "status": (
            "provisional_A_only_diagnostic over a deliberately balanced hard-strata "
            "sample; not representative full-validation prevalence"
        ),
        "reviewer_slot": "A",
        "coverage": {
            "samples": 200,
            "strata": dict(Counter(str(row["stratum"]) for row in parsed)),
            "true_negative_stratum_present": False,
        },
        "model_action": {
            "spoke": _u0_action_summary(spoke),
            "silent": _u0_action_summary(silent),
            "human_match_rate_excluding_uncertain": matches / len(decided),
        },
        "by_stratum": {
            stratum: _u0_action_summary(
                [row for row in parsed if row["stratum"] == stratum]
            )
            for stratum in sorted({str(row["stratum"]) for row in parsed})
        },
        "by_position": {
            position: _u0_action_summary(
                [row for row in parsed if row["position_bin"] == position]
            )
            for position in ("first", "second", "2-4", "5-9", "10+")
        },
        "human_vs_official_gold": {
            "gold_interrupt": _u0_action_summary(
                [row for row in parsed if row["gold_interrupt"]]
            ),
            "gold_silent": _u0_action_summary(
                [row for row in parsed if not row["gold_interrupt"]]
            ),
        },
        "spoke_content": {
            "all": content_summary(spoke),
            "fallback": content_summary([row for row in spoke if row["is_fallback"]]),
            "nonfallback": content_summary(
                [row for row in spoke if not row["is_fallback"]]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u0-ratings-a", required=True)
    parser.add_argument("--u0-blind", required=True)
    parser.add_argument("--u0-key", required=True)
    parser.add_argument("--u1-ratings-a", required=True)
    parser.add_argument("--u1-blind", required=True)
    parser.add_argument("--u1-key", required=True)
    parser.add_argument("--u1-samples", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {name: _resolve(getattr(args, name)) for name in (
        "u0_ratings_a", "u0_blind", "u0_key", "u1_ratings_a", "u1_blind",
        "u1_key", "u1_samples",
    )}
    result = {
        "schema_version": 1,
        "status": "complete provisional reviewer-A-only diagnostic; reviewer B unread",
        "reviewer_b_read": False,
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "u0": analyze_u0_a(paths["u0_ratings_a"], paths["u0_blind"], paths["u0_key"]),
        "u1_interface": analyze_u1_a(
            paths["u1_ratings_a"], paths["u1_blind"], paths["u1_key"],
            paths["u1_samples"], history_cap=4,
        ),
        "u1_state_package": {
            "rated_in_supplied_csv": False,
            "expected_candidates": 240,
            "reason": "The supplied U1 A CSV has 160 interface-package rows only.",
        },
    }
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite A-only diagnostic: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
