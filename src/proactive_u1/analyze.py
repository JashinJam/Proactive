"""Analyze U1 content records and build a paired blind-review package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl, write_jsonl
from proactive_u0.core import normalize_utterance


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_COMPLETION_RE = re.compile(
    r"\b(?:you(?:'re| are) done|finished|complete|perfect|great job|enjoy)\b",
    re.IGNORECASE,
)
_ASSISTANT_ACTION_RE = re.compile(
    r"\b(?:let me|i(?:'ll| will)|we(?:'ll| will))\b", re.IGNORECASE
)


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _stable_hash(seed: str, *parts: object) -> str:
    text = "|".join([seed, *(str(part) for part in parts)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    texts = [str(row.get("content", "")) for row in rows]
    nonempty = [text for text in texts if text]
    word_counts = [len(_WORD_RE.findall(text)) for text in nonempty]
    return {
        "samples": len(rows),
        "nonempty": len(nonempty),
        "nonempty_rate": len(nonempty) / len(rows) if rows else None,
        "empty": len(rows) - len(nonempty),
        "completion_claim_heuristic": sum(bool(_COMPLETION_RE.search(text)) for text in nonempty),
        "assistant_action_heuristic": sum(
            bool(_ASSISTANT_ACTION_RE.search(text)) for text in nonempty
        ),
        "question_form": sum(text.rstrip().endswith("?") for text in nonempty),
        "word_count_mean_nonempty": statistics.fmean(word_counts) if word_counts else None,
        "word_count_median_nonempty": statistics.median(word_counts) if word_counts else None,
    }


def analyze_content(
    sample_rows: Sequence[dict[str, object]],
    content_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    sample_by_id = {str(row["sample_id"]): row for row in sample_rows}
    selected = [
        dict(row) for row in content_rows if row["variant"] == "forced_no_state"
    ]
    if len(selected) != len(sample_rows):
        raise ValueError("U1 no-state analysis coverage differs from frozen sample")
    for row in selected:
        sample = sample_by_id.get(str(row["sample_id"]))
        if sample is None:
            raise ValueError(f"Unknown U1 content sample: {row['sample_id']}")
        for key in ("input_index", "chunk_index", "domain", "position_bin"):
            if row[key] != sample[key]:
                raise ValueError(f"U1 sample/content mismatch for {row['sample_id']}")

    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_position: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_session: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        by_domain[str(row["domain"])].append(row)
        by_position[str(row["position_bin"])].append(row)
        by_session[int(row["input_index"])].append(row)
    exact_repeat_after_first = 0
    sessions_with_repeat = 0
    for rows in by_session.values():
        seen: Counter[str] = Counter()
        repeated = False
        for row in sorted(rows, key=lambda item: int(item["chunk_index"])):
            content = str(row.get("content", ""))
            if not content:
                continue
            normalized = normalize_utterance(content)
            seen[normalized] += 1
            if seen[normalized] > 1:
                exact_repeat_after_first += 1
                repeated = True
        sessions_with_repeat += int(repeated)
    return {
        "schema_version": 1,
        "status": "automatic no-state diagnostics complete; human review pending",
        "overall": {
            **_summary(selected),
            "exact_repeat_after_first": exact_repeat_after_first,
            "sessions_with_exact_repeat": sessions_with_repeat,
        },
        "by_domain": {
            key: _summary(value) for key, value in sorted(by_domain.items())
        },
        "by_position": {
            key: _summary(value) for key, value in sorted(by_position.items())
        },
        "warning": (
            "Completion/assistant-action flags are lexical diagnostics. They do not "
            "establish correctness, grounding, or premature completion without video review."
        ),
    }


def build_blind_pairs(
    sample_rows: Sequence[dict[str, object]],
    content_rows: Sequence[dict[str, object]],
    seed: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_sample: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in content_rows:
        variant = str(row["variant"])
        if variant in {"current_fallback", "forced_no_state"}:
            by_sample[str(row["sample_id"])][variant] = row
    blind: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    for sample in sample_rows:
        sample_id = str(sample["sample_id"])
        variants = by_sample.get(sample_id, {})
        if set(variants) != {"current_fallback", "forced_no_state"}:
            raise ValueError(f"U1 pair coverage mismatch for {sample_id}")
        ordered = sorted(
            variants.items(),
            key=lambda item: _stable_hash(seed, sample_id, item[0]),
        )
        for candidate_index, (variant, row) in enumerate(ordered):
            candidate_label = chr(ord("A") + candidate_index)
            review_id = f"{sample_id}-{candidate_label}"
            blind.append(
                {
                    "review_id": review_id,
                    "pair_id": sample_id,
                    "candidate": candidate_label,
                    "video_path": sample["video_path"],
                    "video_file": f"data/egoproactive/val/{sample['video_path']}",
                    "interval": sample["interval"],
                    "observed_through_sec": sample["observed_through_sec"],
                    "video_intervals_so_far": sample["video_intervals_so_far"],
                    "query": sample["query"],
                    "task": sample["task"],
                    "domain": sample["domain"],
                    "chunk_index": sample["chunk_index"],
                    "prior_dialog": sample["prior_dialog"],
                    "candidate_utterance": row["content"],
                }
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "pair_id": sample_id,
                    "candidate": candidate_label,
                    "variant": variant,
                    "used_fallback": row["used_fallback"],
                }
            )
    blind.sort(key=lambda row: _stable_hash(seed, "blind", row["review_id"]))
    return blind, key_rows


def build_blind_multivariant(
    sample_rows: Sequence[dict[str, object]],
    content_rows: Sequence[dict[str, object]],
    variants: Sequence[str],
    seed: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    expected_variants = tuple(variants)
    if len(expected_variants) < 2 or len(set(expected_variants)) != len(
        expected_variants
    ):
        raise ValueError("Blind multivariant package requires unique variants")
    by_sample: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in content_rows:
        variant = str(row["variant"])
        if variant not in expected_variants:
            continue
        sample_id = str(row["sample_id"])
        if variant in by_sample[sample_id]:
            raise ValueError(f"Duplicate {variant} content for {sample_id}")
        by_sample[sample_id][variant] = row
    blind: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    for sample in sample_rows:
        sample_id = str(sample["sample_id"])
        selected = by_sample.get(sample_id, {})
        if set(selected) != set(expected_variants):
            raise ValueError(f"U1 multivariant coverage mismatch for {sample_id}")
        ordered = sorted(
            selected.items(),
            key=lambda item: _stable_hash(seed, sample_id, item[0]),
        )
        for candidate_index, (variant, row) in enumerate(ordered):
            candidate_label = chr(ord("A") + candidate_index)
            review_id = f"{sample_id}-{candidate_label}"
            blind.append(
                {
                    "review_id": review_id,
                    "pair_id": sample_id,
                    "candidate": candidate_label,
                    "video_path": sample["video_path"],
                    "video_file": f"data/egoproactive/val/{sample['video_path']}",
                    "interval": sample["interval"],
                    "observed_through_sec": sample["observed_through_sec"],
                    "video_intervals_so_far": sample["video_intervals_so_far"],
                    "query": sample["query"],
                    "task": sample["task"],
                    "domain": sample["domain"],
                    "chunk_index": sample["chunk_index"],
                    "prior_dialog": sample["prior_dialog"],
                    "candidate_utterance": row["content"],
                }
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "pair_id": sample_id,
                    "candidate": candidate_label,
                    "variant": variant,
                    "used_fallback": row["used_fallback"],
                }
            )
    blind.sort(key=lambda row: _stable_hash(seed, "blind", row["review_id"]))
    return blind, key_rows


def _write_ratings(path: Path, blind_rows: Sequence[dict[str, object]]) -> None:
    fields = [
        "review_id",
        "pair_id",
        "candidate",
        "reviewer_slot",
        "correctness_1_5",
        "specificity_1_5",
        "actionability_1_5",
        "groundedness_1_5",
        "plan_consistency_1_5",
        "conciseness_1_5",
        "safety_1_5",
        "generic_flag",
        "hallucination_flag",
        "premature_completion_flag",
        "primary_error_type",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in blind_rows:
            for reviewer_slot in ("A", "B"):
                value = {field: "" for field in fields}
                for key in ("review_id", "pair_id", "candidate"):
                    value[key] = row[key]
                value["reviewer_slot"] = reviewer_slot
                writer.writerow(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-items", required=True)
    parser.add_argument("--content-records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", default="20260716-u1-paired-review-v1")
    args = parser.parse_args()
    sample_path = _resolve(args.sample_items)
    content_path = _resolve(args.content_records)
    output_dir = _resolve(args.output_dir)
    sample_rows = load_jsonl(sample_path)
    content_rows = load_jsonl(content_path)
    analysis = analyze_content(sample_rows, content_rows)
    blind, key_rows = build_blind_pairs(sample_rows, content_rows, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "content_analysis.json", analysis)
    write_jsonl(output_dir / "paired_review_blind.jsonl", blind)
    write_jsonl(output_dir / "paired_review_key.jsonl", key_rows)
    _write_ratings(output_dir / "paired_ratings_template.csv", blind)
    result = {
        "analysis": analysis,
        "paired_review_items": len(blind),
        "paired_review_blind_sha256": sha256_file(
            output_dir / "paired_review_blind.jsonl"
        ),
        "paired_review_key_sha256": sha256_file(
            output_dir / "paired_review_key.jsonl"
        ),
        "source_sample_sha256": sha256_file(sample_path),
        "source_content_sha256": sha256_file(content_path),
        "seed": args.seed,
    }
    write_json(output_dir / "analysis_manifest.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
