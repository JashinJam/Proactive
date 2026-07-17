"""Analyze the frozen two-reviewer, three-way U1 oracle-state ratings."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl
from proactive_u1.ratings import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    FLAG_FIELDS,
    SCORE_FIELDS,
    _agreement,
    _mean,
    _read_csv,
    _resolve,
    _session_bootstrap,
    validate_ratings,
)
from proactive_u1.state_review import STATE_VARIANTS


def _aggregate_candidates(
    parsed: Sequence[dict[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    by_candidate: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in parsed:
        by_candidate[(str(row["pair_id"]), str(row["variant"]))].append(row)
    result: dict[tuple[str, str], dict[str, object]] = {}
    for key, rows in by_candidate.items():
        if len(rows) != 2 or {str(row["reviewer_slot"]) for row in rows} != {"A", "B"}:
            raise ValueError(f"State candidate lacks two reviewers: {key}")
        first = rows[0]
        result[key] = {
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
    return result


def _contrast(
    candidates: dict[tuple[str, str], dict[str, object]],
    target: str,
    reference: str,
    bootstrap_seed: int,
    bootstrap_samples: int,
    state_gate: bool,
) -> dict[str, object]:
    pair_ids = sorted({pair_id for pair_id, _ in candidates})
    deltas: list[dict[str, object]] = []
    for pair_id in pair_ids:
        target_row = candidates.get((pair_id, target))
        reference_row = candidates.get((pair_id, reference))
        if target_row is None or reference_row is None:
            raise ValueError(f"Incomplete state contrast for {pair_id}: {target}/{reference}")
        deltas.append(
            {
                "pair_id": pair_id,
                "input_index": target_row["input_index"],
                "domain": target_row["domain"],
                "position_bin": target_row["position_bin"],
                "content_composite_delta": float(target_row["content_composite"])
                - float(reference_row["content_composite"]),
                "score_deltas": {
                    field: float(target_row["scores"][field])
                    - float(reference_row["scores"][field])
                    for field in SCORE_FIELDS
                },
                "flag_rate_deltas": {
                    field: float(target_row["flags"][field])
                    - float(reference_row["flags"][field])
                    for field in FLAG_FIELDS
                },
                "unsafe_rate_delta": float(target_row["unsafe_rate"])
                - float(reference_row["unsafe_rate"]),
            }
        )

    def grouped(field: str) -> dict[str, float]:
        values: dict[str, list[float]] = defaultdict(list)
        for row in deltas:
            values[str(row[field])].append(float(row["content_composite_delta"]))
        return {key: _mean(rows) for key, rows in sorted(values.items())}

    content_delta = _mean([float(row["content_composite_delta"]) for row in deltas])
    bootstrap = _session_bootstrap(deltas, bootstrap_seed, bootstrap_samples)
    by_domain = grouped("domain")
    flag_deltas = {
        field: _mean([float(row["flag_rate_deltas"][field]) for row in deltas])
        for field in FLAG_FIELDS
    }
    unsafe_delta = _mean([float(row["unsafe_rate_delta"]) for row in deltas])
    result: dict[str, object] = {
        "contrast": f"{target} - {reference}",
        "pairs": len(deltas),
        "content_composite_delta": content_delta,
        "score_deltas": {
            field: _mean([float(row["score_deltas"][field]) for row in deltas])
            for field in SCORE_FIELDS
        },
        "flag_rate_deltas": flag_deltas,
        "unsafe_rate_delta": unsafe_delta,
        "by_domain": by_domain,
        "by_position": grouped("position_bin"),
        "session_bootstrap": bootstrap,
    }
    if state_gate:
        checks = {
            "mean_content_delta_at_least_0_50": content_delta >= 0.50,
            "session_bootstrap_ci95_low_above_zero": bootstrap["ci95_low"] > 0,
            "at_least_three_of_four_domains_positive": len(by_domain) == 4
            and sum(value > 0 for value in by_domain.values()) >= 3,
            "hallucination_rate_increase_at_most_0_02": flag_deltas[
                "hallucination_flag"
            ]
            <= 0.02,
            "unsafe_rate_increase_at_most_0_02": unsafe_delta <= 0.02,
        }
        result["promotion_gate"] = {"checks": checks, "passed": all(checks.values())}
    return result


def analyze_state_ratings(
    parsed: Sequence[dict[str, object]],
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, object]:
    candidates = _aggregate_candidates(parsed)
    step = _contrast(
        candidates,
        "forced_oracle_step",
        "forced_no_state",
        bootstrap_seed,
        bootstrap_samples,
        state_gate=True,
    )
    full = _contrast(
        candidates,
        "forced_oracle_full",
        "forced_no_state",
        bootstrap_seed,
        bootstrap_samples,
        state_gate=True,
    )
    full_vs_step = _contrast(
        candidates,
        "forced_oracle_full",
        "forced_oracle_step",
        bootstrap_seed,
        bootstrap_samples,
        state_gate=False,
    )
    full_justified = (
        float(full_vs_step["content_composite_delta"]) >= 0.25
        and float(full_vs_step["session_bootstrap"]["ci95_low"]) > 0
    )
    step_passed = bool(step["promotion_gate"]["passed"])
    full_passed = bool(full["promotion_gate"]["passed"])
    if not step_passed and not full_passed:
        preferred = "none"
    elif full_passed and (not step_passed or full_justified):
        preferred = "forced_oracle_full"
    else:
        preferred = "forced_oracle_step"

    variants: dict[str, object] = {}
    for variant in STATE_VARIANTS:
        rows = [value for (_, name), value in candidates.items() if name == variant]
        variants[variant] = {
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
        "variants": variants,
        "contrasts": {
            "step_vs_no_state": step,
            "full_vs_no_state": full,
            "full_vs_step": full_vs_step,
        },
        "decision": {
            "state_promoted": step_passed or full_passed,
            "full_over_step_justified": full_justified,
            "full_over_step_rule": "delta >= 0.25 and session-bootstrap ci95_low > 0",
            "preferred_state_representation": preferred,
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
    blind_path = _resolve(args.blind)
    key_path = _resolve(args.key)
    sample_path = _resolve(args.samples)
    parsed, samples = validate_ratings(
        [row for path in rating_paths for row in _read_csv(path)],
        load_jsonl(blind_path),
        load_jsonl(key_path),
        load_jsonl(sample_path),
        allowed_variants=STATE_VARIANTS,
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "frozen_protocol": {
            "variants": list(STATE_VARIANTS),
            "interface_package_ratings_reused": False,
            "reviewer_aggregation": "arithmetic mean across A/B",
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
            "samples_sha256": sha256_file(sample_path),
            "populated_reviewer_rows": len(parsed),
            "candidates": len(parsed) // 2,
            "pairs": len(samples),
        },
        **analyze_state_ratings(parsed, args.bootstrap_seed, args.bootstrap_samples),
    }
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
