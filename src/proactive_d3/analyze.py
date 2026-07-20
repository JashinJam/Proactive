"""Post-hoc, read-only interpretation audit for the frozen D3 OOF result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from proactive_d1.core import binary_metrics
from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import INTERRUPT_TAG, load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _assistant_turns(dialog: object) -> int:
    if not isinstance(dialog, list):
        raise ValueError("D3 dialog audit requires a list of turns")
    return sum(
        isinstance(turn, dict) and turn.get("role") == "assistant" for turn in dialog
    )


def _subset_metrics(
    rows: Sequence[dict[str, object]], indices: Sequence[int], prediction: str
) -> dict[str, float | int]:
    labels = [int(rows[index]["gold_interrupt"]) for index in indices]
    predictions = [int(rows[index][prediction]) for index in indices]
    return binary_metrics(labels, predictions)


def _group_summary(
    rows: Sequence[dict[str, object]], indices: Sequence[int]
) -> dict[str, object]:
    d1 = _subset_metrics(rows, indices, "d1_interrupt")
    d3 = _subset_metrics(rows, indices, "predicted_interrupt")
    gold_rate = sum(int(rows[index]["gold_interrupt"]) for index in indices) / len(indices)
    return {
        "chunks": len(indices),
        "gold_interrupt_rate": gold_rate,
        "d1": d1,
        "d3": d3,
        "delta_macro_f1": float(d3["macro_f1"]) - float(d1["macro_f1"]),
        "dynamic_scalar_means": {
            name: float(np.mean([float(rows[index][name]) for index in indices]))
            for name in (
                "tag_margin_abs_delta_previous",
                "hidden_cosine_previous",
                "hidden_delta_rms_previous",
                "hidden_cosine_history_mean",
                "hidden_delta_rms_history_mean",
            )
        },
    }


def analyze(oof_dir: Path) -> dict[str, object]:
    config_path = oof_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path = Path(config["runtime"]["input_path"])
    d1_path = oof_dir / "variants/d1_fused_replay/oof_records.jsonl"
    d3_path = oof_dir / "variants/dynamics_fused/oof_records.jsonl"
    source = load_jsonl(source_path)
    d1_rows = load_jsonl(d1_path)
    d3_rows = load_jsonl(d3_path)
    if len(d1_rows) != len(d3_rows):
        raise ValueError("D1 and D3 OOF records have different row counts")
    rows: list[dict[str, object]] = []
    cursor = 0
    previous_dialog_matches = 0
    non_first = 0
    groups: dict[bool, list[int]] = {False: [], True: []}
    previous_gold_groups: dict[bool, list[int]] = {False: [], True: []}
    for input_index, session in enumerate(source):
        dialogs = session.get("dialog")
        answers = session.get("answers")
        if not isinstance(dialogs, list) or not isinstance(answers, list):
            raise ValueError(f"Session {input_index} lacks dialog/answers")
        if len(dialogs) != len(answers):
            raise ValueError(f"Session {input_index} dialog/answer length differs")
        for chunk_index in range(len(answers)):
            d1 = d1_rows[cursor]
            d3 = d3_rows[cursor]
            key = (input_index, chunk_index)
            if (int(d1["input_index"]), int(d1["chunk_index"])) != key:
                raise ValueError(f"D1 record order differs at {key}")
            if (int(d3["input_index"]), int(d3["chunk_index"])) != key:
                raise ValueError(f"D3 record order differs at {key}")
            if int(d1["predicted_interrupt"]) != int(d3["d1_interrupt"]):
                raise ValueError(f"D3 embedded D1 decision differs at {key}")
            row = dict(d3)
            row["d1_interrupt"] = int(d1["predicted_interrupt"])
            rows.append(row)
            if chunk_index:
                assistant_added = _assistant_turns(dialogs[chunk_index]) > _assistant_turns(
                    dialogs[chunk_index - 1]
                )
                previous_gold_interrupt = str(answers[chunk_index - 1]).startswith(
                    INTERRUPT_TAG
                )
                previous_dialog_matches += int(
                    assistant_added == previous_gold_interrupt
                )
                non_first += 1
                groups[assistant_added].append(cursor)
                previous_gold_groups[previous_gold_interrupt].append(cursor)
            cursor += 1
    if cursor != len(d3_rows):
        raise ValueError("Source chunks do not consume all D3 records")

    changes = {
        "corrected_false_negative": 0,
        "corrected_false_positive": 0,
        "introduced_false_negative": 0,
        "introduced_false_positive": 0,
        "first_chunk_changes": 0,
        "non_first_chunk_changes": 0,
    }
    for row in rows:
        gold = int(row["gold_interrupt"])
        d1 = int(row["d1_interrupt"])
        d3 = int(row["predicted_interrupt"])
        if d1 == d3:
            continue
        if int(row["chunk_index"]) == 0:
            changes["first_chunk_changes"] += 1
        else:
            changes["non_first_chunk_changes"] += 1
        if d1 != gold and d3 == gold:
            changes["corrected_false_negative" if gold else "corrected_false_positive"] += 1
        elif d1 == gold and d3 != gold:
            changes["introduced_false_negative" if gold else "introduced_false_positive"] += 1
        else:
            raise RuntimeError("Binary prediction change has an impossible state")
    changes["total_changes"] = changes["first_chunk_changes"] + changes[
        "non_first_chunk_changes"
    ]
    changes["net_corrections"] = (
        changes["corrected_false_negative"]
        + changes["corrected_false_positive"]
        - changes["introduced_false_negative"]
        - changes["introduced_false_positive"]
    )

    return {
        "status": "complete post-hoc interpretation audit",
        "classification": (
            "read-only analysis after frozen OOF predictions; not a promotion gate, "
            "not hidden-test evidence, and not used for tuning"
        ),
        "sessions": len(source),
        "chunks": len(rows),
        "non_first_chunks": non_first,
        "dialog_policy_audit": {
            "definition": (
                "assistant_added compares the number of assistant turns in dialog[i] "
                "with dialog[i-1]; both are official current-prefix inputs"
            ),
            "assistant_added_equals_previous_gold_interrupt": previous_dialog_matches,
            "total_non_first_chunks": non_first,
            "agreement_rate": previous_dialog_matches / non_first,
            "causal_benchmark_input": True,
            "deployment_caveat": (
                "the feature is legal under the benchmark prefix, but its distribution "
                "depends on the supplied dialog policy and can shift in self-fed use"
            ),
        },
        "grouped_by_official_dialog_assistant_added": {
            "false": _group_summary(rows, groups[False]),
            "true": _group_summary(rows, groups[True]),
        },
        "grouped_by_previous_gold_interrupt_cross_check": {
            "false": _group_summary(rows, previous_gold_groups[False]),
            "true": _group_summary(rows, previous_gold_groups[True]),
        },
        "decision_changes_vs_d1": changes,
        "interpretation": (
            "D3 is a valid causal improvement on the official validation protocol, "
            "but the gain is not evidence of purely visual temporal understanding"
        ),
        "input_artifacts": {
            "source_sha256": sha256_file(source_path),
            "d1_oof_records_sha256": sha256_file(d1_path),
            "d3_oof_records_sha256": sha256_file(d3_path),
            "config_sha256": sha256_file(config_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    oof_dir = _resolve(args.oof_dir)
    output_path = _resolve(args.output) if args.output else oof_dir / "analysis.json"
    result = analyze(oof_dir)
    write_json(output_path, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
