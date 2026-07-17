"""Build diagnostic domain, position, and decision-change summaries for D1."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Sequence

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl

from .core import binary_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _position_bin(chunk_index: int) -> str:
    if chunk_index == 0:
        return "0:first"
    if chunk_index == 1:
        return "1:second"
    if chunk_index <= 4:
        return "2-4"
    if chunk_index <= 9:
        return "5-9"
    return "10+"


def grouped_metrics(
    rows: Sequence[dict[str, object]],
    key: Callable[[dict[str, object]], str],
) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    result: dict[str, object] = {}
    for group, values in sorted(groups.items()):
        labels = [int(value["gold_interrupt"]) for value in values]
        candidate = [int(value["predicted_interrupt"]) for value in values]
        baseline = [int(value["r0f_interrupt"]) for value in values]
        candidate_metrics = binary_metrics(labels, candidate)
        baseline_metrics = binary_metrics(labels, baseline)
        result[group] = {
            "candidate": candidate_metrics,
            "r0f": baseline_metrics,
            "delta_macro_f1": float(candidate_metrics["macro_f1"])
            - float(baseline_metrics["macro_f1"]),
        }
    return result


def decision_changes(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        gold = int(row["gold_interrupt"])
        candidate = int(row["predicted_interrupt"])
        baseline = int(row["r0f_interrupt"])
        if candidate == baseline:
            counts["unchanged_correct" if candidate == gold else "unchanged_wrong"] += 1
        elif candidate == gold:
            counts["corrected_r0f_error"] += 1
            counts["corrected_fn" if gold == 1 else "corrected_fp"] += 1
        else:
            counts["introduced_error"] += 1
            counts["introduced_fp" if candidate == 1 else "introduced_fn"] += 1
    return dict(sorted(counts.items()))


def coefficient_summary(diagnostics: dict[str, object]) -> dict[str, object]:
    folds = diagnostics.get("fold_details")
    if not isinstance(folds, list) or not folds:
        raise ValueError("D1 diagnostics lack fold details")
    by_name: dict[str, list[float]] = defaultdict(list)
    thresholds: list[float] = []
    for fold in folds:
        if not isinstance(fold, dict):
            raise ValueError("Invalid D1 fold diagnostics")
        coefficients = fold.get("standardized_coefficients")
        if not isinstance(coefficients, dict):
            raise ValueError("D1 fold lacks coefficients")
        for name, value in coefficients.items():
            by_name[str(name)].append(float(value))
        thresholds.append(float(fold["threshold_logit"]))
    return {
        "standardized_coefficients": {
            name: {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "population_stdev": statistics.pstdev(values),
            }
            for name, values in sorted(by_name.items())
        },
        "threshold_logit": {
            "median": statistics.median(thresholds),
            "min": min(thresholds),
            "max": max(thresholds),
        },
    }


def analyze(experiment_dir: Path, source_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    variants_dir = experiment_dir / "variants"
    variants = sorted(path.name for path in variants_dir.iterdir() if path.is_dir())
    result: dict[str, object] = {
        "status": "diagnostic analysis of frozen OOF predictions",
        "experiment_dir": str(experiment_dir),
        "variants": {},
    }
    variant_result = result["variants"]
    assert isinstance(variant_result, dict)
    for variant in variants:
        directory = variants_dir / variant
        rows = load_jsonl(directory / "oof_records.jsonl")
        diagnostics = json.loads(
            (directory / "diagnostics.json").read_text(encoding="utf-8")
        )
        if len(rows) != sum(len(row["answers"]) for row in source_rows):  # type: ignore[arg-type]
            raise ValueError(f"OOF record coverage mismatch for {variant}")
        for row in rows:
            input_index = int(row["input_index"])
            source = source_rows[input_index]
            row["domain"] = source["domain"]
            row["position_bin"] = _position_bin(int(row["chunk_index"]))
        variant_result[variant] = {
            "predictions_sha256": sha256_file(directory / "predictions.jsonl"),
            "overall_official": json.loads(
                (directory / "metrics.json").read_text(encoding="utf-8")
            )["overall"],
            "by_domain": grouped_metrics(rows, lambda row: str(row["domain"])),
            "by_fold": grouped_metrics(rows, lambda row: str(row["fold"])),
            "by_position": grouped_metrics(rows, lambda row: str(row["position_bin"])),
            "decision_changes_vs_r0f": decision_changes(rows),
            "coefficient_summary": coefficient_summary(diagnostics),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--input",
        default="data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = _resolve(args.experiment_dir)
    source_rows = load_jsonl(_resolve(args.input))
    result = analyze(experiment_dir, source_rows)
    write_json(experiment_dir / "analysis.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

