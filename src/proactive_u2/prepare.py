"""Prepare the frozen review-informed sample for the U2 early-grounding audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import INTERRUPT_TAG, load_jsonl, write_jsonl
from proactive_u1.prepare import strip_current_answers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EARLY_POSITIONS = ("1:second", "2-4")


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _check_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"U2 preparation SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def _stable_hash(seed: str, *parts: object) -> str:
    value = "|".join([seed, *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_interrupt(value: object) -> bool:
    return str(value).lstrip().startswith(INTERRUPT_TAG)


def prepare_sample(
    item_records: Sequence[dict[str, object]],
    oof_predictions: Sequence[dict[str, object]],
    final_predictions: Sequence[dict[str, object]],
    label_free_sources: Sequence[dict[str, object]],
    *,
    seed: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Select the complete frozen Boolean intersection without score ranking."""
    if not (
        len(oof_predictions) == len(final_predictions) == len(label_free_sources) == 700
    ):
        raise ValueError("U2 preparation requires 700 aligned source sessions")
    if any("answers" in row for row in label_free_sources):
        raise ValueError("U2 preparation must receive answer-stripped source rows")

    candidates: list[tuple[dict[str, object], bool, bool]] = []
    seen_keys: set[tuple[int, int]] = set()
    for item in item_records:
        input_index = int(item["input_index"])
        chunk_index = int(item["chunk_index"])
        key = (input_index, chunk_index)
        if key in seen_keys:
            raise ValueError(f"Duplicate U0 item key in U2 preparation: {key}")
        seen_keys.add(key)
        if str(item["position_bin"]) not in EARLY_POSITIONS:
            continue
        reviewers = item.get("reviewers")
        if not isinstance(reviewers, dict):
            raise ValueError(f"U2 item lacks reviewer records: {key}")
        if any(
            str(reviewers[slot]["should_interrupt"]) != "yes" for slot in ("A", "B")
        ):
            continue

        source = label_free_sources[input_index]
        oof = oof_predictions[input_index]
        final = final_predictions[input_index]
        video_path = str(source["video_path"])
        if str(oof.get("video_path")) != video_path or str(final.get("video_path")) != video_path:
            raise ValueError(f"U2 prediction/source order mismatch at session {input_index}")
        intervals = source.get("video_intervals")
        dialog = source.get("dialog")
        oof_answers = oof.get("answers")
        final_answers = final.get("answers")
        if not all(
            isinstance(value, list)
            for value in (intervals, dialog, oof_answers, final_answers)
        ):
            raise ValueError(f"Malformed U2 aligned arrays at session {input_index}")
        assert isinstance(intervals, list)
        assert isinstance(dialog, list)
        assert isinstance(oof_answers, list)
        assert isinstance(final_answers, list)
        if not (
            len(intervals) == len(dialog) == len(oof_answers) == len(final_answers)
        ):
            raise ValueError(f"U2 chunk coverage mismatch at session {input_index}")
        if not 0 <= chunk_index < len(intervals):
            raise ValueError(f"U2 chunk index is out of range: {key}")
        oof_interrupt = _is_interrupt(oof_answers[chunk_index])
        final_interrupt = _is_interrupt(final_answers[chunk_index])
        if oof_interrupt and final_interrupt:
            candidates.append((item, oof_interrupt, final_interrupt))

    candidates.sort(
        key=lambda value: _stable_hash(
            seed,
            "sample-order",
            value[0]["input_index"],
            value[0]["chunk_index"],
        )
    )
    samples: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    for rank, (item, oof_interrupt, final_interrupt) in enumerate(candidates, start=1):
        sample_id = f"U2-{rank:04d}"
        input_index = int(item["input_index"])
        chunk_index = int(item["chunk_index"])
        source = label_free_sources[input_index]
        interval = source["video_intervals"][chunk_index]  # type: ignore[index]
        sample = {
            "sample_id": sample_id,
            "u0_review_id": item["review_id"],
            "input_index": input_index,
            "chunk_index": chunk_index,
            "video_path": source["video_path"],
            "interval": [float(interval[0]), float(interval[1])],
            "observed_through_sec": float(interval[1]),
            "video_intervals_so_far": source["video_intervals"][: chunk_index + 1],  # type: ignore[index]
            "query": source["query"],
            "task": source["task"],
            "domain": source["domain"],
            "position_bin": item["position_bin"],
            "prior_dialog": source["dialog"][chunk_index],  # type: ignore[index]
            "fixed_d4_decision": "$interrupt$",
        }
        if "answers" in sample:
            raise ValueError("U2 sanitized sample unexpectedly contains answers")
        samples.append(sample)
        key_rows.append(
            {
                "sample_id": sample_id,
                "u0_review_id": item["review_id"],
                "input_index": input_index,
                "chunk_index": chunk_index,
                "selection": {
                    "early_position": True,
                    "reviewer_a_should_interrupt": True,
                    "reviewer_b_should_interrupt": True,
                    "d4_oof_interrupt": oof_interrupt,
                    "d4_final_interrupt": final_interrupt,
                },
                "u0_fallback_status": item["fallback_status"],
                "u0_pair_content_composite": item["pair_content_composite"],
                "u0_disagreement_triggers": item["disagreement_triggers"],
                "selection_hash": _stable_hash(
                    seed, item["review_id"], input_index, chunk_index
                ),
            }
        )

    summary = {
        "seed": seed,
        "selection_policy": (
            "Complete intersection: early U0 position, A/B should_interrupt=yes, "
            "D4 OOF interrupt, and D4 final interrupt; no score ranking."
        ),
        "samples": len(samples),
        "sessions": len({int(row["input_index"]) for row in samples}),
        "by_position": dict(Counter(str(row["position_bin"]) for row in samples)),
        "by_domain": dict(Counter(str(row["domain"]) for row in samples)),
        "by_u0_fallback_status": dict(
            Counter(str(row["u0_fallback_status"]) for row in key_rows)
        ),
        "generation_sample_contains_answers": False,
        "gold_answer_used_for_selection": False,
        "human_ratings_used_only_as_boolean_consensus": True,
    }
    return samples, key_rows, summary


def run(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol_path = _resolve(config["protocol"]["path"])
    _check_hash(protocol_path, str(config["protocol"]["sha256"]))
    paths = {name: _resolve(value["path"]) for name, value in config["sources"].items()}
    hashes = {
        name: _check_hash(paths[name], str(value["sha256"]))
        for name, value in config["sources"].items()
    }
    label_free_sources = strip_current_answers(load_jsonl(paths["gold_container"]))
    samples, key_rows, summary = prepare_sample(
        load_jsonl(paths["u0_item_records"]),
        load_jsonl(paths["d4_oof_predictions"]),
        load_jsonl(paths["d4_final_predictions"]),
        label_free_sources,
        seed=str(config["selection"]["seed"]),
    )
    expected = config["selection"]["expected"]
    actual = {
        "samples": summary["samples"],
        "sessions": summary["sessions"],
        "by_position": summary["by_position"],
        "by_domain": summary["by_domain"],
        "by_u0_fallback_status": summary["by_u0_fallback_status"],
    }
    if actual != expected:
        raise ValueError(f"U2 selected sample differs from frozen expectation: {actual}")

    outputs = {name: _resolve(value) for name, value in config["outputs"].items()}
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(outputs["items"], samples)
    write_jsonl(outputs["key"], key_rows)
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "protocol_sha256": config["protocol"]["sha256"],
        "source_hashes": hashes,
        "summary": summary,
        "items_sha256": sha256_file(outputs["items"]),
        "key_sha256": sha256_file(outputs["key"]),
    }
    write_json(outputs["manifest"], manifest)
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(_resolve(args.config))


if __name__ == "__main__":
    main()
