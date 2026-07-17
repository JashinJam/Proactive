"""Generate reproducible diagnostics for a completed R1 oracle-state pilot."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("r0_frozen", "null", "step", "cues", "full")


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _tag(answer: object) -> str:
    return "interrupt" if str(answer).lstrip().startswith("$interrupt$") else "silent"


def _metrics(gold: list[str], predicted: list[str]) -> dict[str, object]:
    tp = sum(g == "interrupt" and p == "interrupt" for g, p in zip(gold, predicted))
    fp = sum(g == "silent" and p == "interrupt" for g, p in zip(gold, predicted))
    tn = sum(g == "silent" and p == "silent" for g, p in zip(gold, predicted))
    fn = sum(g == "interrupt" and p == "silent" for g, p in zip(gold, predicted))

    def divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    ip = divide(tp, tp + fp)
    ir = divide(tp, tp + fn)
    i_f1 = divide(2 * ip * ir, ip + ir)
    sp = divide(tn, tn + fn)
    sr = divide(tn, tn + fp)
    s_f1 = divide(2 * sp * sr, sp + sr)
    return {
        "macro_f1": round((i_f1 + s_f1) / 2, 4),
        "gmean_f1": round(math.sqrt(i_f1 * s_f1), 4),
        "interrupt_precision": round(ip, 4),
        "interrupt_recall": round(ir, 4),
        "interrupt_f1": round(i_f1, 4),
        "silent_precision": round(sp, 4),
        "silent_recall": round(sr, 4),
        "silent_f1": round(s_f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": len(gold),
        "predicted_interrupt_rate": round((tp + fp) / len(gold), 4),
    }


def _variant_records(experiment_dir: Path, variant: str) -> list[dict[str, object]]:
    return load_jsonl(experiment_dir / "variants" / variant / "session_records.jsonl")


def analyze(experiment_dir: Path, annotation_path: Path) -> dict[str, object]:
    gold_rows = load_jsonl(experiment_dir / "evaluation_golden_subset.jsonl")
    records_by_variant = {
        variant: _variant_records(experiment_dir, variant) for variant in VARIANTS
    }
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation_by_index = {int(item["input_index"]): item for item in annotations}
    comparison = json.loads((experiment_dir / "comparison.json").read_text(encoding="utf-8"))
    actual_metrics = comparison["variants"]

    session_diagnostics: list[dict[str, object]] = []
    progress_counts: dict[str, Counter[str]] = defaultdict(Counter)
    switch_counts: Counter[str] = Counter()
    switch_rows: list[dict[str, object]] = []
    first_chunks: list[dict[str, object]] = []
    flattened_gold: list[str] = []
    flattened_predicted: dict[str, list[str]] = {variant: [] for variant in VARIANTS}
    flattened_records: dict[str, list[dict[str, object]]] = {
        variant: [] for variant in VARIANTS
    }

    for position, gold_row in enumerate(gold_rows):
        input_index = int(records_by_variant["null"][position]["input_index"])
        if any(
            int(records_by_variant[variant][position]["input_index"]) != input_index
            for variant in VARIANTS
        ):
            raise ValueError("Variant session order mismatch")
        annotation = annotation_by_index[input_index]
        gold_tags = [_tag(answer) for answer in gold_row["answers"]]
        predictions: dict[str, list[str]] = {}
        chunks: dict[str, list[dict[str, object]]] = {}
        for variant in VARIANTS:
            record = records_by_variant[variant][position]
            prediction = record["prediction"]
            if not isinstance(prediction, dict):
                raise ValueError(f"Variant {variant} record lacks prediction")
            predictions[variant] = [_tag(answer) for answer in prediction["answers"]]
            chunks[variant] = record["chunks"]  # type: ignore[assignment]
            flattened_predicted[variant].extend(predictions[variant])
            flattened_records[variant].extend(chunks[variant])
        flattened_gold.extend(gold_tags)

        per_variant = {
            variant: _metrics(gold_tags, predictions[variant]) for variant in VARIANTS
        }
        session_diagnostics.append(
            {
                "input_index": input_index,
                "video_path": gold_row["video_path"],
                "domain": gold_row["domain"],
                "task": gold_row["task"],
                "chunks": len(gold_tags),
                "gold_interrupts": gold_tags.count("interrupt"),
                "metrics": per_variant,
            }
        )

        first_chunks.append(
            {
                "input_index": input_index,
                "domain": gold_row["domain"],
                "gold": gold_tags[0],
                "variants": {
                    variant: {
                        "prediction": predictions[variant][0],
                        "normalization": chunks[variant][0].get("normalization"),
                        "raw_response": chunks[variant][0].get("raw_response"),
                    }
                    for variant in VARIANTS
                },
            }
        )

        states = annotation["chunk_states"]
        for chunk_index, (gold_tag, null_tag, full_tag) in enumerate(
            zip(gold_tags, predictions["null"], predictions["full"])
        ):
            progress = str(states[chunk_index]["progress"])
            counter = progress_counts[progress]
            counter["support"] += 1
            counter[f"gold_{gold_tag}"] += 1
            counter[f"null_{null_tag}"] += 1
            counter[f"full_{full_tag}"] += 1
            if null_tag != full_tag:
                key = f"{null_tag}_to_{full_tag}|gold_{gold_tag}|{progress}"
                switch_counts[key] += 1
                switch_rows.append(
                    {
                        "input_index": input_index,
                        "chunk_index": chunk_index,
                        "progress": progress,
                        "gold": gold_tag,
                        "null": null_tag,
                        "full": full_tag,
                    }
                )

    first_gold = [str(row["gold"]) for row in first_chunks]
    first_metrics = {
        variant: _metrics(
            first_gold,
            [str(row["variants"][variant]["prediction"]) for row in first_chunks],  # type: ignore[index]
        )
        for variant in VARIANTS
    }

    posthoc: dict[str, object] = {}
    for variant in VARIANTS:
        counterfactual = list(flattened_predicted[variant])
        malformed_indices: list[int] = []
        for index, chunk in enumerate(flattened_records[variant]):
            if chunk.get("normalization") == "malformed_response_scored_as_silent":
                malformed_indices.append(index)
                counterfactual[index] = "interrupt"
        posthoc[variant] = {
            "rule": "Reclassify every malformed raw generation as interrupt.",
            "malformed_indices": malformed_indices,
            "metrics": _metrics(flattened_gold, counterfactual),
        }

    return {
        "status": "diagnostic analysis of frozen predictions",
        "official_metric_source": "Each variant metrics.json produced by the official scorer",
        "experiment_dir": str(experiment_dir),
        "experiment_predictions": {
            variant: {
                "sha256": sha256_file(experiment_dir / "variants" / variant / "predictions.jsonl"),
                "official_metrics": actual_metrics[variant],
            }
            for variant in VARIANTS
        },
        "shape": {
            "sessions": len(gold_rows),
            "chunks": len(flattened_gold),
            "gold_interrupts": flattened_gold.count("interrupt"),
            "gold_silent": flattened_gold.count("silent"),
        },
        "session_diagnostics": session_diagnostics,
        "first_chunk": {
            "rows": first_chunks,
            "metrics": first_metrics,
        },
        "full_vs_null_switches": {
            "count": len(switch_rows),
            "summary": dict(sorted(switch_counts.items())),
            "rows": switch_rows,
        },
        "progress_breakdown": {
            progress: dict(sorted(counts.items()))
            for progress, counts in sorted(progress_counts.items())
        },
        "posthoc_format_counterfactual_not_an_official_result": posthoc,
        "interpretation_boundary": (
            "The four-session pilot is too small for a population claim. The format "
            "counterfactual is posthoc and is not an R1 score or leaderboard claim."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--annotations",
        default="annotations/r1_oracle_pilot_v1/states.json",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    experiment_dir = _resolve(args.experiment_dir)
    annotation_path = _resolve(args.annotations)
    output_path = _resolve(args.output) if args.output else experiment_dir / "analysis.json"
    result = analyze(experiment_dir, annotation_path)
    write_json(output_path, result)
    print(json.dumps(result["shape"], sort_keys=True))


if __name__ == "__main__":
    main()
