"""Analyze frozen D1 neural OOF predictions against the promoted scalar control."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

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


def grouped_neural_metrics(
    rows: Sequence[dict[str, object]],
    key: Callable[[dict[str, object]], str],
) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    result: dict[str, object] = {}
    for group, values in sorted(groups.items()):
        labels = [int(value["gold_interrupt"]) for value in values]
        candidate = binary_metrics(
            labels, [int(value["predicted_interrupt"]) for value in values]
        )
        scalar = binary_metrics(
            labels, [int(value["scalar_interrupt"]) for value in values]
        )
        r0f = binary_metrics(labels, [int(value["r0f_interrupt"]) for value in values])
        result[group] = {
            "candidate": candidate,
            "scalar_oof": scalar,
            "r0f": r0f,
            "delta_macro_f1_vs_scalar": float(candidate["macro_f1"])
            - float(scalar["macro_f1"]),
            "delta_macro_f1_vs_r0f": float(candidate["macro_f1"])
            - float(r0f["macro_f1"]),
        }
    return result


def changes_vs_scalar(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        gold = int(row["gold_interrupt"])
        candidate = int(row["predicted_interrupt"])
        baseline = int(row["scalar_interrupt"])
        if candidate == baseline:
            counts["unchanged_correct" if candidate == gold else "unchanged_wrong"] += 1
        elif candidate == gold:
            counts["corrected_scalar_error"] += 1
            counts["corrected_fn" if gold == 1 else "corrected_fp"] += 1
        else:
            counts["introduced_error"] += 1
            counts["introduced_fp" if candidate == 1 else "introduced_fn"] += 1
    return dict(sorted(counts.items()))


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start
        while stop + 1 < len(scores) and scores[order[stop + 1]] == scores[order[start]]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop + 2) / 2
        start = stop + 1
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def tag_margin_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    labels = np.asarray([int(row["gold_interrupt"]) for row in rows], dtype=np.int64)
    margins = np.asarray([float(row["tag_margin"]) for row in rows], dtype=np.float64)
    result: dict[str, object] = {
        "roc_auc": _rank_auc(labels, margins),
        "zero_threshold_metrics": binary_metrics(
            labels.tolist(), (margins >= 0).astype(int).tolist()
        ),
    }
    for label, name in ((0, "silent_gold"), (1, "interrupt_gold")):
        selected = margins[labels == label]
        quantiles = np.quantile(selected, [0.05, 0.25, 0.5, 0.75, 0.95])
        result[name] = {
            "count": int(len(selected)),
            "mean": float(selected.mean()),
            "stdev": float(selected.std()),
            "p05": float(quantiles[0]),
            "p25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "p75": float(quantiles[3]),
            "p95": float(quantiles[4]),
        }
    return result


def fold_model_summary(diagnostics: dict[str, object]) -> dict[str, object]:
    folds = diagnostics.get("fold_details")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("D1 neural diagnostics require five fold details")
    selected_l2 = Counter(str(fold["selected_l2_weight"]) for fold in folds)
    thresholds = [float(fold["threshold_logit"]) for fold in folds]
    norms = [float(fold["model"]["weight_l2_norm"]) for fold in folds]
    top_names: Counter[str] = Counter()
    for fold in folds:
        for entry in fold["model"]["top_standardized_coefficients"]:
            top_names[str(entry["name"])] += 1
    return {
        "selected_l2_counts": dict(sorted(selected_l2.items())),
        "threshold_logit": {
            "median": statistics.median(thresholds),
            "min": min(thresholds),
            "max": max(thresholds),
        },
        "weight_l2_norm": {
            "mean": statistics.fmean(norms),
            "min": min(norms),
            "max": max(norms),
        },
        "top_coefficient_fold_frequency": dict(top_names.most_common(30)),
    }


def analyze(experiment_dir: Path, source_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    variants_dir = experiment_dir / "variants"
    variants = sorted(path.name for path in variants_dir.iterdir() if path.is_dir())
    result: dict[str, object] = {
        "status": "diagnostic analysis of frozen neural OOF predictions",
        "experiment_dir": str(experiment_dir),
        "variants": {},
    }
    variant_result = result["variants"]
    assert isinstance(variant_result, dict)
    expected_chunks = sum(len(row["answers"]) for row in source_rows)  # type: ignore[arg-type]
    for variant in variants:
        directory = variants_dir / variant
        rows = load_jsonl(directory / "oof_records.jsonl")
        if len(rows) != expected_chunks:
            raise ValueError(f"Neural OOF record coverage mismatch for {variant}")
        for row in rows:
            source = source_rows[int(row["input_index"])]
            row["domain"] = source["domain"]
            row["position_bin"] = _position_bin(int(row["chunk_index"]))
        diagnostics = json.loads(
            (directory / "diagnostics.json").read_text(encoding="utf-8")
        )
        variant_result[variant] = {
            "predictions_sha256": sha256_file(directory / "predictions.jsonl"),
            "overall_official": json.loads(
                (directory / "metrics.json").read_text(encoding="utf-8")
            )["overall"],
            "by_domain": grouped_neural_metrics(rows, lambda row: str(row["domain"])),
            "by_fold": grouped_neural_metrics(rows, lambda row: str(row["fold"])),
            "by_position": grouped_neural_metrics(
                rows, lambda row: str(row["position_bin"])
            ),
            "decision_changes_vs_scalar": changes_vs_scalar(rows),
            "tag_margin": tag_margin_summary(rows),
            "fold_model": fold_model_summary(diagnostics),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--input",
        default="data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl",
    )
    args = parser.parse_args()
    experiment_dir = _resolve(args.experiment_dir)
    source_rows = load_jsonl(_resolve(args.input))
    result = analyze(experiment_dir, source_rows)
    write_json(experiment_dir / "analysis.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
