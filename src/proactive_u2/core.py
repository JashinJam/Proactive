"""Pure controls and diagnostics for U2 early-chunk utterance grounding."""

from __future__ import annotations

import copy
import hashlib
import re
import statistics
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Sequence

from proactive_r0.core import INTERRUPT_TAG, SILENT_TAG


VIEWS = (
    "full_history",
    "no_current_video",
    "query_only_full_video",
    "query_only_current_video",
    "facts_full_history",
    "facts_query_current",
)
FACT_VIEWS = ("facts_full_history", "facts_query_current")
FACT_SYSTEM_PROMPT = """
You extract short visual evidence for a proactive procedural assistant. Inspect
only the supplied frames from the current candidate interval. Return one short
line with at most three concrete objects, actions, or states that are directly
visible and relevant to the user's task. Do not give advice. Do not use or infer
prior steps. Do not claim completion unless completion is directly visible. If
the evidence is insufficient, return exactly: unclear
""".strip()
FACT_BLOCK_HEADER = "[Predicted current-interval visual facts]"
COMPLETION_CLAIM = re.compile(
    r"(?:\b(?:you\s+are|you're)\s+(?:done|finished|all\s+set)\b"
    r"|\b(?:you\s+have|you've)\s+(?:finished|completed)\b"
    r"|\b(?:it|this|that)\s+(?:is|looks)\s+"
    r"(?:done|finished|complete|completed|ready|good)\b"
    r"|\ball\s+set\b|\b(?:great|good)\s+job\b)",
    flags=re.IGNORECASE,
)
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "you",
    "your",
}


def remove_assistant_history(
    messages: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    result = [copy.deepcopy(row) for row in messages if row.get("role") != "assistant"]
    if not result or result[0].get("role") != "system":
        raise ValueError("U2 messages must begin with a system prompt")
    return result


def fact_extraction_messages(query: object) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FACT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task: {str(query).strip()}\n"
                "Describe only directly visible evidence in the current interval."
            ),
        },
    ]


def normalize_visual_facts(value: object) -> dict[str, object]:
    raw = str(value).strip()
    text = raw
    removed_decision_tag = False
    for tag in (INTERRUPT_TAG, SILENT_TAG):
        if text.startswith(tag):
            text = text[len(tag) :].strip()
            removed_decision_tag = True
            break
    text = " ".join(text.split())
    used_unclear = not text or text.lower() in {"unclear", "visible facts: unclear"}
    if used_unclear:
        text = "unclear"
    return {
        "raw_visual_facts": raw,
        "visual_facts": text,
        "facts_used_unclear": used_unclear,
        "facts_removed_decision_tag": removed_decision_tag,
    }


def visual_fact_block(facts: object) -> str:
    return "\n".join(
        [
            FACT_BLOCK_HEADER,
            str(facts).strip() or "unclear",
            (
                "Use these only as potentially noisy visual evidence. Do not repeat "
                "unsupported details, and do not claim completion from unclear evidence."
            ),
        ]
    )


def _normalized_text(value: object) -> str:
    return " ".join(str(value).lower().split())


def _tokens(value: object) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN.findall(str(value))
        if token.lower() not in _STOPWORDS
    }


def fact_overlap(content: object, facts: object) -> dict[str, float | None]:
    content_tokens = _tokens(content)
    fact_tokens = _tokens(facts)
    overlap = content_tokens.intersection(fact_tokens)
    union = content_tokens.union(fact_tokens)
    return {
        "content_token_precision": (
            len(overlap) / len(content_tokens) if content_tokens else None
        ),
        "fact_token_recall": len(overlap) / len(fact_tokens) if fact_tokens else None,
        "jaccard": len(overlap) / len(union) if union else None,
    }


def _word_count(value: object) -> int:
    return len(_TOKEN.findall(str(value)))


def _mean_present(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("U2 cannot summarize an empty group")
    fallback = sum(bool(row["used_fallback"]) for row in rows)
    overlaps = [fact_overlap(row["content"], row["visual_facts"]) for row in rows]
    completion = sum(bool(COMPLETION_CLAIM.search(str(row["content"]))) for row in rows)
    return {
        "samples": len(rows),
        "fallback": fallback,
        "fallback_rate": fallback / len(rows),
        "nonempty": len(rows) - fallback,
        "nonempty_rate": (len(rows) - fallback) / len(rows),
        "mean_word_count": statistics.fmean(_word_count(row["content"]) for row in rows),
        "completion_claims": completion,
        "completion_claim_rate": completion / len(rows),
        "mean_predicted_fact_overlap": {
            field: _mean_present([value[field] for value in overlaps])
            for field in (
                "content_token_precision",
                "fact_token_recall",
                "jaccard",
            )
        },
    }


def _paired_comparison(
    reference_rows: Sequence[dict[str, object]],
    candidate_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    reference = {
        (int(row["input_index"]), int(row["chunk_index"])): row
        for row in reference_rows
    }
    candidate = {
        (int(row["input_index"]), int(row["chunk_index"])): row
        for row in candidate_rows
    }
    if set(reference) != set(candidate) or not reference:
        raise ValueError("U2 paired comparison keys differ or are empty")
    similarities: list[float] = []
    answer_exact = 0
    content_exact = 0
    reference_nonfallback_to_fallback = 0
    reference_fallback_to_nonfallback = 0
    for key in sorted(reference):
        left = reference[key]
        right = candidate[key]
        answer_exact += int(str(left["answer"]) == str(right["answer"]))
        left_content = _normalized_text(left["content"])
        right_content = _normalized_text(right["content"])
        content_exact += int(left_content == right_content)
        similarities.append(SequenceMatcher(None, left_content, right_content).ratio())
        left_fallback = bool(left["used_fallback"])
        right_fallback = bool(right["used_fallback"])
        reference_nonfallback_to_fallback += int(not left_fallback and right_fallback)
        reference_fallback_to_nonfallback += int(left_fallback and not right_fallback)
    left_summary = _summary(list(reference.values()))
    right_summary = _summary(list(candidate.values()))
    return {
        "samples": len(reference),
        "answer_exact": answer_exact,
        "answer_exact_rate": answer_exact / len(reference),
        "content_exact": content_exact,
        "content_exact_rate": content_exact / len(reference),
        "mean_text_similarity": statistics.fmean(similarities),
        "fallback_rate_delta_candidate_minus_reference": (
            float(right_summary["fallback_rate"]) - float(left_summary["fallback_rate"])
        ),
        "nonempty_rate_delta_candidate_minus_reference": (
            float(right_summary["nonempty_rate"]) - float(left_summary["nonempty_rate"])
        ),
        "reference_nonfallback_to_candidate_fallback": reference_nonfallback_to_fallback,
        "reference_fallback_to_candidate_nonfallback": reference_fallback_to_nonfallback,
    }


def _grouped(rows: Sequence[dict[str, object]], field: str) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {name: _summary(groups[name]) for name in sorted(groups)}


def analyze_records(
    records: Sequence[dict[str, object]],
    fact_records: Sequence[dict[str, object]],
    thresholds: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    by_view: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for row in records:
        view = str(row.get("view"))
        if view not in VIEWS:
            raise ValueError(f"Unsupported U2 view: {view}")
        key = (view, int(row["input_index"]), int(row["chunk_index"]))
        if key in seen:
            raise ValueError(f"Duplicate U2 record: {key}")
        seen.add(key)
        by_view[view].append(row)
    counts = {view: len(by_view[view]) for view in VIEWS}
    if len(set(counts.values())) != 1 or not counts[VIEWS[0]]:
        raise ValueError(f"U2 view coverage differs: {counts}")
    expected_keys = {
        (int(row["input_index"]), int(row["chunk_index"]))
        for row in by_view[VIEWS[0]]
    }
    for view in VIEWS[1:]:
        keys = {
            (int(row["input_index"]), int(row["chunk_index"]))
            for row in by_view[view]
        }
        if keys != expected_keys:
            raise ValueError(f"U2 paired keys differ for {view}")
    facts_by_key = {
        (int(row["input_index"]), int(row["chunk_index"])): row
        for row in fact_records
    }
    if set(facts_by_key) != expected_keys or len(facts_by_key) != len(fact_records):
        raise ValueError("U2 fact records do not exactly cover the paired sample")
    for view in VIEWS:
        for row in by_view[view]:
            key = (int(row["input_index"]), int(row["chunk_index"]))
            if str(row["visual_facts"]) != str(facts_by_key[key]["visual_facts"]):
                raise ValueError(f"U2 views do not reuse the same fact text at {key}")

    views = {
        view: {
            "overall": _summary(by_view[view]),
            "by_position": _grouped(by_view[view], "position_bin"),
            "by_domain": _grouped(by_view[view], "domain"),
        }
        for view in VIEWS
    }
    paired_vs_full = {
        view: _paired_comparison(by_view["full_history"], by_view[view])
        for view in VIEWS[1:]
    }
    contrasts = {
        "current_video_ablation": {
            "reference": "full_history",
            "candidate": "no_current_video",
            **_paired_comparison(
                by_view["full_history"], by_view["no_current_video"]
            ),
        },
        "assistant_history_ablation": {
            "reference": "full_history",
            "candidate": "query_only_full_video",
            **_paired_comparison(
                by_view["full_history"], by_view["query_only_full_video"]
            ),
        },
        "fact_query_cold_start": {
            "reference": "query_only_current_video",
            "candidate": "facts_query_current",
            **_paired_comparison(
                by_view["query_only_current_video"], by_view["facts_query_current"]
            ),
        },
        "fact_with_history": {
            "reference": "full_history",
            "candidate": "facts_full_history",
            **_paired_comparison(
                by_view["full_history"], by_view["facts_full_history"]
            ),
        },
    }
    fact_usable = sum(not bool(row["facts_used_unclear"]) for row in fact_records)
    fact_usable_rate = fact_usable / len(fact_records)
    fact_rescue = contrasts["fact_query_cold_start"]
    review_priority = {
        "facts_non_unclear_rate_at_least_threshold": fact_usable_rate
        >= float(thresholds["facts_non_unclear_rate_at_least"]),
        "fact_query_nonempty_gain_at_least_threshold": float(
            fact_rescue["nonempty_rate_delta_candidate_minus_reference"]
        )
        >= float(thresholds["fact_query_nonempty_gain_at_least"]),
    }

    discordant: list[dict[str, object]] = []
    full_by_key = {
        (int(row["input_index"]), int(row["chunk_index"])): row
        for row in by_view["full_history"]
    }
    for view in VIEWS[1:]:
        for row in by_view[view]:
            key = (int(row["input_index"]), int(row["chunk_index"]))
            full = full_by_key[key]
            if str(row["answer"]) == str(full["answer"]):
                continue
            discordant.append(
                {
                    "view": view,
                    "sample_id": row["sample_id"],
                    "input_index": row["input_index"],
                    "chunk_index": row["chunk_index"],
                    "domain": row["domain"],
                    "position_bin": row["position_bin"],
                    "full_used_fallback": full["used_fallback"],
                    "candidate_used_fallback": row["used_fallback"],
                    "similarity": SequenceMatcher(
                        None,
                        _normalized_text(full["content"]),
                        _normalized_text(row["content"]),
                    ).ratio(),
                    "visual_facts": row["visual_facts"],
                    "full_content": full["content"],
                    "candidate_content": row["content"],
                }
            )
    discordant.sort(key=lambda row: (str(row["view"]), float(row["similarity"])))
    analysis = {
        "status": "complete U2 automatic mechanism analysis",
        "classification": (
            "review-informed public-validation diagnostic; fallback, text change, "
            "and predicted-fact overlap are not grounding-quality evidence"
        ),
        "samples": counts[VIEWS[0]],
        "views": views,
        "paired_vs_full_history": paired_vs_full,
        "paired_contrasts": contrasts,
        "facts": {
            "samples": len(fact_records),
            "non_unclear": fact_usable,
            "non_unclear_rate": fact_usable_rate,
            "mean_word_count": statistics.fmean(
                _word_count(row["visual_facts"]) for row in fact_records
            ),
            "decision_tag_removed": sum(
                bool(row["facts_removed_decision_tag"]) for row in fact_records
            ),
        },
        "preregistered_review_priority_thresholds": thresholds,
        "review_priority_checks": review_priority,
        "human_quality_review_required": True,
        "official_decisions_changed": 0,
        "official_scorer_invoked": False,
        "discordant_cases": len(discordant),
    }
    return analysis, discordant


def _stable_hash(seed: str, *parts: object) -> str:
    value = "|".join([seed, *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_review_packages(
    records: Sequence[dict[str, object]],
    fact_records: Sequence[dict[str, object]],
    samples: Sequence[dict[str, object]],
    *,
    seed: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    """Blind six utterance views and keep the predicted-fact audit separate."""
    sample_by_id = {str(row["sample_id"]): row for row in samples}
    records_by_sample: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in records:
        sample_id = str(row["sample_id"])
        view = str(row["view"])
        if view in records_by_sample[sample_id]:
            raise ValueError(f"Duplicate U2 review candidate: {(sample_id, view)}")
        records_by_sample[sample_id][view] = row
    if set(records_by_sample) != set(sample_by_id):
        raise ValueError("U2 review candidates do not cover the frozen sample")
    facts_by_id = {str(row["sample_id"]): row for row in fact_records}
    if set(facts_by_id) != set(sample_by_id):
        raise ValueError("U2 fact review rows do not cover the frozen sample")

    blind: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    for sample_id in sorted(sample_by_id):
        sample = sample_by_id[sample_id]
        views = sorted(VIEWS, key=lambda view: _stable_hash(seed, sample_id, view))
        for offset, view in enumerate(views):
            candidate = chr(ord("A") + offset)
            review_id = f"{sample_id}-{candidate}"
            record = records_by_sample[sample_id].get(view)
            if record is None:
                raise ValueError(f"Missing U2 view for review: {(sample_id, view)}")
            blind.append(
                {
                    "review_id": review_id,
                    "sample_id": sample_id,
                    "candidate": candidate,
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
                    "model_action": "spoke",
                    "candidate_utterance": record["content"]
                    if not record["used_fallback"]
                    else "Please continue with the next step.",
                }
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "sample_id": sample_id,
                    "candidate": candidate,
                    "view": view,
                    "used_fallback": record["used_fallback"],
                    "raw_continuation": record["raw_continuation"],
                    "visual_facts": record["visual_facts"],
                    "model_input_frames": record["model_input_frames"],
                    "assistant_history_turns": record["assistant_history_turns"],
                }
            )
    blind.sort(key=lambda row: _stable_hash(seed, "blind", row["review_id"]))
    fact_blind = [
        {
            "fact_review_id": f"{sample_id}-FACT",
            "sample_id": sample_id,
            "video_path": sample_by_id[sample_id]["video_path"],
            "video_file": f"data/egoproactive/val/{sample_by_id[sample_id]['video_path']}",
            "interval": sample_by_id[sample_id]["interval"],
            "observed_through_sec": sample_by_id[sample_id]["observed_through_sec"],
            "query": sample_by_id[sample_id]["query"],
            "task": sample_by_id[sample_id]["task"],
            "domain": sample_by_id[sample_id]["domain"],
            "predicted_visual_facts": facts_by_id[sample_id]["visual_facts"],
        }
        for sample_id in sorted(sample_by_id)
    ]
    forbidden = {"view", "visual_facts", "used_fallback", "raw_continuation"}
    if any(forbidden.intersection(row) for row in blind):
        raise ValueError("U2 blind utterance package leaks variant metadata")
    summary = {
        "seed": seed,
        "samples": len(sample_by_id),
        "utterance_candidates": len(blind),
        "fact_candidates": len(fact_blind),
        "views_per_sample": len(VIEWS),
        "variant_exposed_in_blind_package": False,
        "predicted_facts_exposed_in_utterance_package": False,
    }
    return blind, key_rows, fact_blind, summary
