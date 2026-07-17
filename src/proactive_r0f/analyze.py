"""Generate domain, first-chunk, and stability diagnostics for R0-F."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _tag(value: object) -> str:
    return "interrupt" if str(value).lstrip().startswith("$interrupt$") else "silent"


def _metrics(gold: list[str], predicted: list[str]) -> dict[str, object]:
    tp = sum(g == "interrupt" and p == "interrupt" for g, p in zip(gold, predicted))
    fp = sum(g == "silent" and p == "interrupt" for g, p in zip(gold, predicted))
    tn = sum(g == "silent" and p == "silent" for g, p in zip(gold, predicted))
    fn = sum(g == "interrupt" and p == "silent" for g, p in zip(gold, predicted))

    def div(a: int, b: int) -> float:
        return a / b if b else 0.0

    ip, ir = div(tp, tp + fp), div(tp, tp + fn)
    sp, sr = div(tn, tn + fn), div(tn, tn + fp)
    i_f1, s_f1 = div(2 * ip * ir, ip + ir), div(2 * sp * sr, sp + sr)
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


def analyze(experiment_dir: Path, r0_dir: Path, input_path: Path) -> dict[str, object]:
    source_rows = load_jsonl(input_path)
    repaired_records = load_jsonl(experiment_dir / "session_records.jsonl")
    r0_records = load_jsonl(r0_dir / "session_records.jsonl")
    if not (len(source_rows) == len(repaired_records) == len(r0_records) == 700):
        raise ValueError("R0-F analysis requires 700 aligned sessions")

    domain_values: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"gold": [], "r0": [], "r0f": []}
    )
    changed_gold: Counter[str] = Counter()
    changed_positions: Counter[int] = Counter()
    changed_raw_lengths: list[int] = []
    session_deltas: list[float] = []
    session_rows: list[dict[str, object]] = []
    first_gold: list[str] = []
    first_r0: list[str] = []
    first_r0f: list[str] = []
    first_changed = 0

    for index, (source, repaired, r0) in enumerate(
        zip(source_rows, repaired_records, r0_records)
    ):
        if repaired.get("input_index") != index or r0.get("input_index") != index:
            raise ValueError(f"Session order mismatch at {index}")
        gold = [_tag(answer) for answer in source["answers"]]
        r0_pred = [_tag(answer) for answer in r0["prediction"]["answers"]]  # type: ignore[index]
        repaired_pred = [
            _tag(answer) for answer in repaired["prediction"]["answers"]  # type: ignore[index]
        ]
        domain = str(source["domain"])
        domain_values[domain]["gold"].extend(gold)
        domain_values[domain]["r0"].extend(r0_pred)
        domain_values[domain]["r0f"].extend(repaired_pred)
        r0_metrics = _metrics(gold, r0_pred)
        r0f_metrics = _metrics(gold, repaired_pred)
        delta = round(
            float(r0f_metrics["macro_f1"]) - float(r0_metrics["macro_f1"]), 4
        )
        session_deltas.append(delta)
        session_rows.append(
            {
                "input_index": index,
                "domain": domain,
                "task": source["task"],
                "chunks": len(gold),
                "r0_macro_f1": r0_metrics["macro_f1"],
                "r0f_macro_f1": r0f_metrics["macro_f1"],
                "delta_macro_f1": delta,
            }
        )
        first_gold.append(gold[0])
        first_r0.append(r0_pred[0])
        first_r0f.append(repaired_pred[0])
        if r0_pred[0] != repaired_pred[0]:
            first_changed += 1
        chunks = repaired["chunks"]
        assert isinstance(chunks, list)
        for chunk_index, (gold_tag, chunk) in enumerate(zip(gold, chunks)):
            if not isinstance(chunk, dict):
                raise ValueError("Repaired chunk must be an object")
            if chunk.get("repair_reason") == "malformed_nonempty_repaired_as_interrupt":
                changed_gold[gold_tag] += 1
                changed_positions[chunk_index] += 1
                changed_raw_lengths.append(len(str(chunk["raw_response"])))

    per_domain = {
        domain: {
            "r0": _metrics(values["gold"], values["r0"]),
            "r0f": _metrics(values["gold"], values["r0f"]),
        }
        for domain, values in sorted(domain_values.items())
    }
    for value in per_domain.values():
        value["delta_macro_f1"] = round(
            float(value["r0f"]["macro_f1"]) - float(value["r0"]["macro_f1"]), 4  # type: ignore[index]
        )
    improved = sum(delta > 0 for delta in session_deltas)
    worse = sum(delta < 0 for delta in session_deltas)
    tied = len(session_deltas) - improved - worse
    return {
        "status": "diagnostic analysis of frozen R0-F predictions",
        "experiment_dir": str(experiment_dir),
        "predictions_sha256": sha256_file(experiment_dir / "predictions.jsonl"),
        "shape": {"sessions": 700, "chunks": 9935},
        "repaired_chunks": {
            "total": sum(changed_gold.values()),
            "gold_tag_counts": dict(sorted(changed_gold.items())),
            "chunk_position_counts": {
                str(position): count for position, count in sorted(changed_positions.items())
            },
            "raw_character_length": {
                "min": min(changed_raw_lengths),
                "median": statistics.median(changed_raw_lengths),
                "mean": round(statistics.mean(changed_raw_lengths), 3),
                "max": max(changed_raw_lengths),
            },
        },
        "first_chunk": {
            "changed": first_changed,
            "r0": _metrics(first_gold, first_r0),
            "r0f": _metrics(first_gold, first_r0f),
        },
        "per_domain": per_domain,
        "session_stability": {
            "improved_sessions": improved,
            "worse_sessions": worse,
            "tied_sessions": tied,
            "mean_macro_f1_delta": round(statistics.mean(session_deltas), 4),
            "median_macro_f1_delta": round(statistics.median(session_deltas), 4),
            "rows": session_rows,
        },
        "interpretation_boundary": (
            "The deterministic repair does not read per-response labels, but its rule "
            "family was chosen after public-validation error analysis. All results are "
            "val-supervised and require hidden-test confirmation."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--r0-dir",
        default="output/experiments/20260713_internvl35_1b_no_plan_r0",
    )
    parser.add_argument(
        "--input",
        default="data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    experiment_dir = _resolve(args.experiment_dir)
    output_path = _resolve(args.output) if args.output else experiment_dir / "analysis.json"
    result = analyze(experiment_dir, _resolve(args.r0_dir), _resolve(args.input))
    write_json(output_path, result)
    print(json.dumps({"repaired": result["repaired_chunks"], "first": result["first_chunk"]}, sort_keys=True))


if __name__ == "__main__":
    main()
