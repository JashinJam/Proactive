"""Pure, deterministic helpers for the U0 utterance audit."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

from proactive_d1.core import decision_answer
from proactive_r0.core import INTERRUPT_TAG, SILENT_TAG


FALLBACK_ANSWER = "$interrupt$Please continue with the next step."
FALLBACK_UTTERANCE = "Please continue with the next step."

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SPACE_RE = re.compile(r"\s+")
_GENERIC_ONLY = {
    "continue",
    "continue with the next step",
    "continue to the next step",
    "keep going",
    "move on to the next step",
    "please continue",
    "please continue with the next step",
    "proceed to the next step",
}
_ACTION_VERBS = {
    "add",
    "align",
    "apply",
    "attach",
    "bend",
    "brush",
    "check",
    "clean",
    "close",
    "connect",
    "continue",
    "cover",
    "cut",
    "detach",
    "dip",
    "drill",
    "dry",
    "fill",
    "fold",
    "glue",
    "grab",
    "hold",
    "insert",
    "install",
    "lift",
    "loosen",
    "mark",
    "measure",
    "mix",
    "open",
    "place",
    "plug",
    "pour",
    "press",
    "pull",
    "push",
    "remove",
    "rinse",
    "rotate",
    "sand",
    "secure",
    "set",
    "slide",
    "stir",
    "switch",
    "take",
    "test",
    "tighten",
    "trim",
    "turn",
    "twist",
    "unplug",
    "wash",
    "wipe",
    "wrap",
}
_CONTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "gently",
    "go",
    "going",
    "good",
    "great",
    "it",
    "its",
    "just",
    "keep",
    "make",
    "next",
    "now",
    "of",
    "on",
    "once",
    "please",
    "step",
    "that",
    "the",
    "then",
    "this",
    "to",
    "up",
    "with",
    "you",
    "your",
}


def position_bin(chunk_index: int) -> str:
    if chunk_index == 0:
        return "0:first"
    if chunk_index == 1:
        return "1:second"
    if chunk_index <= 4:
        return "2-4"
    if chunk_index <= 9:
        return "5-9"
    return "10+"


def normalize_utterance(text: str) -> str:
    normalized = _SPACE_RE.sub(" ", str(text).strip().lower())
    return normalized.rstrip(".!?")


def response_class(raw_response: object) -> str:
    stripped = str(raw_response).lstrip()
    if not stripped:
        return "empty"
    if stripped.startswith(INTERRUPT_TAG):
        utterance = stripped[len(INTERRUPT_TAG) :].strip()
        return "explicit_interrupt" if utterance else "empty_interrupt"
    if stripped.startswith(SILENT_TAG):
        return "explicit_silent"
    return "malformed_nonempty"


def _utterance(answer: str) -> str | None:
    if not answer.startswith(INTERRUPT_TAG):
        return None
    return answer[len(INTERRUPT_TAG) :].strip()


def _confusion(gold_interrupt: bool, predicted_interrupt: bool) -> str:
    if gold_interrupt and predicted_interrupt:
        return "tp"
    if predicted_interrupt:
        return "fp"
    if gold_interrupt:
        return "fn"
    return "tn"


def _stable_hash(seed: str, *parts: object) -> str:
    value = "|".join([seed, *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_lengths(
    source_rows: Sequence[dict[str, object]],
    prediction_rows: Sequence[dict[str, object]],
    r0_rows: Sequence[dict[str, object]],
    expected_sessions: int,
) -> None:
    if not (len(source_rows) == len(prediction_rows) == len(r0_rows)):
        raise ValueError("U0 source, prediction, and R0 session counts differ")
    if len(source_rows) != expected_sessions:
        raise ValueError(
            f"U0 requires exactly {expected_sessions} sessions, got {len(source_rows)}"
        )


def build_chunk_records(
    source_rows: Sequence[dict[str, object]],
    prediction_rows: Sequence[dict[str, object]],
    r0_rows: Sequence[dict[str, object]],
    oof_rows: Sequence[dict[str, object]],
    expected_sessions: int = 700,
    expected_chunks: int = 9_935,
) -> list[dict[str, object]]:
    """Align all frozen sources and derive one auditable row per chunk."""
    _validate_lengths(
        source_rows, prediction_rows, r0_rows, expected_sessions=expected_sessions
    )
    oof_by_key: dict[tuple[int, int], dict[str, object]] = {}
    for row in oof_rows:
        key = (int(row["input_index"]), int(row["chunk_index"]))
        if key in oof_by_key:
            raise ValueError(f"Duplicate D1 OOF row: {key}")
        oof_by_key[key] = row

    records: list[dict[str, object]] = []
    for input_index, (source, prediction, r0) in enumerate(
        zip(source_rows, prediction_rows, r0_rows)
    ):
        video_path = str(source["video_path"])
        if str(prediction.get("video_path")) != video_path:
            raise ValueError(f"Prediction video mismatch at session {input_index}")
        if int(r0.get("input_index", -1)) != input_index:
            raise ValueError(f"R0 input index mismatch at session {input_index}")
        if str(r0.get("video_path")) != video_path:
            raise ValueError(f"R0 video mismatch at session {input_index}")

        gold_answers = source.get("answers")
        predicted_answers = prediction.get("answers")
        intervals = source.get("video_intervals")
        chunks = r0.get("chunks")
        dialog = source.get("dialog")
        if not all(
            isinstance(value, list)
            for value in (gold_answers, predicted_answers, intervals, chunks, dialog)
        ):
            raise ValueError(f"Malformed aligned arrays at session {input_index}")
        assert isinstance(gold_answers, list)
        assert isinstance(predicted_answers, list)
        assert isinstance(intervals, list)
        assert isinstance(chunks, list)
        assert isinstance(dialog, list)
        if not (
            len(gold_answers)
            == len(predicted_answers)
            == len(intervals)
            == len(chunks)
            == len(dialog)
        ):
            raise ValueError(f"Chunk coverage mismatch at session {input_index}")

        for chunk_index, (gold, predicted, interval, r0_chunk) in enumerate(
            zip(gold_answers, predicted_answers, intervals, chunks)
        ):
            if not isinstance(r0_chunk, dict):
                raise ValueError(f"Malformed R0 chunk at {(input_index, chunk_index)}")
            if int(r0_chunk.get("chunk_index", -1)) != chunk_index:
                raise ValueError(f"R0 chunk order mismatch at {(input_index, chunk_index)}")
            raw_response = str(r0_chunk["raw_response"])
            predicted_text = str(predicted)
            gold_text = str(gold)
            gold_interrupt = gold_text.startswith(INTERRUPT_TAG)
            predicted_interrupt = predicted_text.startswith(INTERRUPT_TAG)
            key = (input_index, chunk_index)
            oof = oof_by_key.get(key)
            if oof is None:
                raise ValueError(f"Missing D1 OOF row: {key}")
            if bool(int(oof["gold_interrupt"])) != gold_interrupt:
                raise ValueError(f"D1 OOF gold mismatch at {key}")
            if bool(int(oof["predicted_interrupt"])) != predicted_interrupt:
                raise ValueError(f"D1 OOF prediction mismatch at {key}")
            expected_answer = decision_answer(raw_response, int(predicted_interrupt))
            if expected_answer != predicted_text:
                raise ValueError(f"D1 answer assembly mismatch at {key}")

            interval_values = [float(value) for value in interval]  # type: ignore[union-attr]
            if len(interval_values) != 2 or interval_values[1] <= interval_values[0]:
                raise ValueError(f"Invalid interval at {key}")
            d1_utterance = _utterance(predicted_text)
            tokens = _WORD_RE.findall(d1_utterance or "")
            lowered_tokens = [token.lower() for token in tokens]
            content_tokens = [
                token
                for token in lowered_tokens
                if token not in _CONTENT_STOPWORDS and token not in _ACTION_VERBS
            ]
            records.append(
                {
                    "input_index": input_index,
                    "video_path": video_path,
                    "domain": str(source.get("domain", "")),
                    "task": str(source.get("task", "")),
                    "query": str(source.get("query", "")),
                    "chunk_index": chunk_index,
                    "position_bin": position_bin(chunk_index),
                    "interval": interval_values,
                    "observed_through_sec": interval_values[1],
                    "fold": int(oof["fold"]),
                    "gold_interrupt": gold_interrupt,
                    "predicted_interrupt": predicted_interrupt,
                    "confusion": _confusion(gold_interrupt, predicted_interrupt),
                    "gold_answer": gold_text,
                    "gold_utterance": _utterance(gold_text),
                    "d1_answer": predicted_text,
                    "d1_utterance": d1_utterance,
                    "is_fallback": predicted_text == FALLBACK_ANSWER,
                    "raw_response": raw_response,
                    "raw_response_class": response_class(raw_response),
                    "r0_answer": str(r0_chunk.get("answer", "")),
                    "r0_normalization": r0_chunk.get("normalization"),
                    "tag_margin": float(oof["tag_margin"]),
                    "utterance_word_count": len(tokens),
                    "generic_only_heuristic": bool(
                        d1_utterance
                        and normalize_utterance(d1_utterance) in _GENERIC_ONLY
                    ),
                    "action_verb_heuristic": bool(
                        set(lowered_tokens).intersection(_ACTION_VERBS)
                    ),
                    "nonstop_content_token_heuristic": bool(content_tokens),
                }
            )

    if len(records) != len(oof_rows) or len(records) != expected_chunks:
        raise ValueError(
            f"U0 requires exactly {expected_chunks} aligned chunks, got {len(records)}"
        )
    if set(oof_by_key) != {
        (int(row["input_index"]), int(row["chunk_index"])) for row in records
    }:
        raise ValueError("D1 OOF key coverage differs from aligned chunks")

    by_session: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_session[int(record["input_index"])].append(record)
    for session in by_session.values():
        counts = Counter(
            normalize_utterance(str(record["d1_utterance"]))
            for record in session
            if record["predicted_interrupt"] and record["d1_utterance"]
        )
        seen: Counter[str] = Counter()
        for record in sorted(session, key=lambda item: int(item["chunk_index"])):
            utterance = record["d1_utterance"]
            normalized = normalize_utterance(str(utterance)) if utterance else ""
            seen[normalized] += int(bool(normalized))
            record["utterance_count_in_session"] = counts.get(normalized, 0)
            record["utterance_occurrence_in_session"] = seen.get(normalized, 0)
            record["is_exact_repeat_after_first"] = bool(
                normalized and seen[normalized] > 1
            )
    return records


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_records(records: Sequence[dict[str, object]]) -> dict[str, object]:
    confusion = Counter(str(record["confusion"]) for record in records)
    predicted = [record for record in records if record["predicted_interrupt"]]
    fallback = [record for record in predicted if record["is_fallback"]]
    nonfallback = [record for record in predicted if not record["is_fallback"]]
    word_counts = [int(record["utterance_word_count"]) for record in predicted]
    fallback_tp = sum(record["confusion"] == "tp" for record in fallback)
    nonfallback_tp = sum(record["confusion"] == "tp" for record in nonfallback)
    repeat_excess = sum(bool(record["is_exact_repeat_after_first"]) for record in predicted)
    return {
        "chunks": len(records),
        "gold_interrupts": sum(bool(record["gold_interrupt"]) for record in records),
        "predicted_interrupts": len(predicted),
        "predicted_interrupt_rate": _safe_rate(len(predicted), len(records)),
        "tp": confusion["tp"],
        "fp": confusion["fp"],
        "tn": confusion["tn"],
        "fn": confusion["fn"],
        "fallback_count": len(fallback),
        "fallback_rate_among_interrupts": _safe_rate(len(fallback), len(predicted)),
        "fallback_tp": fallback_tp,
        "fallback_fp": len(fallback) - fallback_tp,
        "fallback_binary_precision": _safe_rate(fallback_tp, len(fallback)),
        "nonfallback_count": len(nonfallback),
        "nonfallback_tp": nonfallback_tp,
        "nonfallback_fp": len(nonfallback) - nonfallback_tp,
        "nonfallback_binary_precision": _safe_rate(nonfallback_tp, len(nonfallback)),
        "generic_only_heuristic_count": sum(
            bool(record["generic_only_heuristic"]) for record in predicted
        ),
        "action_verb_heuristic_count": sum(
            bool(record["action_verb_heuristic"]) for record in predicted
        ),
        "nonstop_content_token_heuristic_count": sum(
            bool(record["nonstop_content_token_heuristic"]) for record in predicted
        ),
        "utterance_word_count_mean": (
            sum(word_counts) / len(word_counts) if word_counts else None
        ),
        "utterance_word_count_median": (
            float(sorted(word_counts)[len(word_counts) // 2])
            if len(word_counts) % 2 == 1
            else (
                sum(sorted(word_counts)[len(word_counts) // 2 - 1 : len(word_counts) // 2 + 1])
                / 2
                if word_counts
                else None
            )
        ),
        "exact_repeat_after_first_count": repeat_excess,
        "exact_repeat_after_first_rate": _safe_rate(repeat_excess, len(predicted)),
    }


def grouped_summary(
    records: Sequence[dict[str, object]], key: str
) -> dict[str, dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {
        group: summarize_records(values) for group, values in sorted(groups.items())
    }


def session_summary(records: Sequence[dict[str, object]]) -> dict[str, object]:
    sessions: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        sessions[int(record["input_index"])].append(record)
    fallback_counts: Counter[int] = Counter()
    sessions_with_repeat = 0
    sessions_with_nonfallback_repeat = 0
    for values in sessions.values():
        fallback_counts[sum(bool(value["is_fallback"]) for value in values)] += 1
        sessions_with_repeat += int(
            any(bool(value["is_exact_repeat_after_first"]) for value in values)
        )
        sessions_with_nonfallback_repeat += int(
            any(
                bool(value["is_exact_repeat_after_first"])
                and not bool(value["is_fallback"])
                for value in values
            )
        )
    return {
        "sessions": len(sessions),
        "sessions_with_fallback": sum(
            count for fallback_number, count in fallback_counts.items() if fallback_number > 0
        ),
        "fallback_count_per_session_histogram": {
            str(key): value for key, value in sorted(fallback_counts.items())
        },
        "sessions_with_any_exact_utterance_repeat": sessions_with_repeat,
        "sessions_with_nonfallback_exact_repeat": sessions_with_nonfallback_repeat,
    }


def _review_stratum(record: dict[str, object]) -> str | None:
    confusion = str(record["confusion"])
    if confusion == "fn":
        return "fn_silent"
    if confusion not in {"tp", "fp"}:
        return None
    suffix = "fallback" if record["is_fallback"] else "nonfallback"
    return f"{confusion}_{suffix}"


def _balanced_select(
    pool: Sequence[dict[str, object]], count: int, seed: str, stratum: str
) -> list[dict[str, object]]:
    if len(pool) < count:
        raise ValueError(f"Review stratum {stratum} has only {len(pool)} rows")
    remaining = list(pool)
    selected: list[dict[str, object]] = []
    domain_counts: Counter[str] = Counter()
    position_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    while len(selected) < count:
        chosen = min(
            remaining,
            key=lambda record: (
                domain_counts[str(record["domain"])],
                position_counts[str(record["position_bin"])],
                task_counts[str(record["task"])],
                _stable_hash(
                    seed,
                    stratum,
                    record["input_index"],
                    record["chunk_index"],
                ),
            ),
        )
        remaining.remove(chosen)
        selected.append(chosen)
        domain_counts[str(chosen["domain"])] += 1
        position_counts[str(chosen["position_bin"])] += 1
        task_counts[str(chosen["task"])] += 1
    return selected


def build_review_package(
    records: Sequence[dict[str, object]],
    source_rows: Sequence[dict[str, object]],
    strata: dict[str, int],
    seed: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Create a blind review file and a separate outcome-bearing answer key."""
    pools: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        stratum = _review_stratum(record)
        if stratum is not None:
            pools[stratum].append(record)
    unknown = set(strata) - set(pools)
    if unknown:
        raise ValueError(f"Unknown or empty review strata: {sorted(unknown)}")

    selected: list[tuple[str, dict[str, object]]] = []
    for stratum, count in strata.items():
        selected.extend(
            (stratum, record)
            for record in _balanced_select(pools[stratum], count, seed, stratum)
        )
    selected.sort(
        key=lambda item: _stable_hash(
            seed,
            "blind-order",
            item[1]["input_index"],
            item[1]["chunk_index"],
        )
    )

    blind: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    for index, (stratum, record) in enumerate(selected, start=1):
        review_id = f"U0-{index:04d}"
        input_index = int(record["input_index"])
        chunk_index = int(record["chunk_index"])
        source = source_rows[input_index]
        blind_row = {
            "review_id": review_id,
            "video_path": record["video_path"],
            "video_file": f"data/egoproactive/val/{record['video_path']}",
            "interval": record["interval"],
            "observed_through_sec": record["observed_through_sec"],
            "video_intervals_so_far": source["video_intervals"][: chunk_index + 1],  # type: ignore[index]
            "query": record["query"],
            "task": record["task"],
            "domain": record["domain"],
            "chunk_index": chunk_index,
            "prior_dialog": source["dialog"][chunk_index],  # type: ignore[index]
            "model_action": (
                "spoke" if record["predicted_interrupt"] else "silent"
            ),
            "candidate_utterance": record["d1_utterance"],
        }
        blind.append(blind_row)
        key_rows.append(
            {
                "review_id": review_id,
                "input_index": input_index,
                "chunk_index": chunk_index,
                "stratum": stratum,
                "confusion": record["confusion"],
                "gold_interrupt": record["gold_interrupt"],
                "gold_utterance": record["gold_utterance"],
                "is_fallback": record["is_fallback"],
                "raw_response": record["raw_response"],
                "raw_response_class": record["raw_response_class"],
                "fold": record["fold"],
                "tag_margin": record["tag_margin"],
                "sample_hash": _stable_hash(
                    seed, stratum, input_index, chunk_index
                ),
            }
        )

    forbidden = {
        "gold_interrupt",
        "gold_utterance",
        "is_fallback",
        "raw_response",
        "raw_response_class",
        "confusion",
        "stratum",
        "tag_margin",
        "fold",
    }
    for row in blind:
        leaked = forbidden.intersection(row)
        if leaked:
            raise ValueError(f"Blind review row leaks protected keys: {sorted(leaked)}")

    sample_summary = {
        "seed": seed,
        "total": len(blind),
        "requested_strata": strata,
        "realized_strata": dict(Counter(str(row["stratum"]) for row in key_rows)),
        "by_domain": dict(Counter(str(row["domain"]) for row in blind)),
        "by_position": dict(
            Counter(position_bin(int(row["chunk_index"])) for row in blind)
        ),
        "unique_tasks": len({str(row["task"]) for row in blind}),
        "source_pool_sizes": {name: len(pools[name]) for name in sorted(pools)},
        "selection_policy": (
            "Within each outcome/content stratum, greedily minimize domain, "
            "position-bin, and task reuse counts; break ties with SHA256(seed,key)."
        ),
        "gold_used_for_sampling_only": True,
        "gold_exposed_in_blind_file": False,
    }
    return blind, key_rows, sample_summary


def ratings_rows(review_items: Iterable[dict[str, object]]) -> list[dict[str, str]]:
    fields = [
        "review_id",
        "reviewer_slot",
        "should_interrupt",
        "decision_confidence_1_5",
        "timeliness_1_5",
        "correctness_1_5",
        "specificity_1_5",
        "actionability_1_5",
        "groundedness_1_5",
        "plan_consistency_1_5",
        "conciseness_1_5",
        "safety_1_5",
        "generic_flag",
        "hallucination_flag",
        "unsafe_flag",
        "primary_error_type",
        "notes",
    ]
    rows: list[dict[str, str]] = []
    for item in review_items:
        for reviewer_slot in ("A", "B"):
            row = {field: "" for field in fields}
            row["review_id"] = str(item["review_id"])
            row["reviewer_slot"] = reviewer_slot
            rows.append(row)
    return rows


def finite_json(value: object) -> None:
    """Reject NaN/Infinity before serializing audit artifacts."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite audit value: {value}")
    if isinstance(value, dict):
        for nested in value.values():
            finite_json(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            finite_json(nested)
