from __future__ import annotations

import copy
import unittest

from proactive_u0.core import (
    FALLBACK_ANSWER,
    build_chunk_records,
    build_review_package,
    ratings_rows,
    summarize_records,
)


def fixtures():
    gold_pairs = [
        ["$interrupt$Gold A", "$silent$"],
        ["$interrupt$Gold B", "$silent$"],
        ["$silent$", "$interrupt$Gold C"],
    ]
    raw_pairs = [
        ["Do action A", "$silent$"],
        ["$silent$", "$silent$"],
        ["Do wrong action", "$silent$"],
    ]
    decisions = [[1, 1], [1, 0], [1, 0]]
    sources = []
    predictions = []
    r0_rows = []
    oof_rows = []
    for input_index in range(3):
        video_path = f"video-{input_index}.mp4"
        source = {
            "video_path": video_path,
            "video_intervals": [[0.0, 2.0], [2.0, 10.0]],
            "query": f"Help with task {input_index}",
            "domain": "Chef" if input_index % 2 == 0 else "Tutorial",
            "task": f"Task {input_index}",
            "answers": gold_pairs[input_index],
            "dialog": [
                [{"role": "user", "text": f"Help with task {input_index}"}],
                [
                    {"role": "user", "text": f"Help with task {input_index}"},
                    {"role": "assistant", "text": "Prior guidance"},
                ],
            ],
        }
        answers = []
        chunks = []
        for chunk_index, raw_response in enumerate(raw_pairs[input_index]):
            decision = decisions[input_index][chunk_index]
            if decision:
                answer = (
                    FALLBACK_ANSWER
                    if raw_response == "$silent$"
                    else f"$interrupt${raw_response}"
                )
            else:
                answer = "$silent$"
            answers.append(answer)
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "interval": source["video_intervals"][chunk_index],
                    "raw_response": raw_response,
                    "answer": "$silent$",
                    "normalization": None,
                }
            )
            oof_rows.append(
                {
                    "input_index": input_index,
                    "video_path": video_path,
                    "fold": input_index,
                    "chunk_index": chunk_index,
                    "gold_interrupt": int(
                        gold_pairs[input_index][chunk_index].startswith("$interrupt$")
                    ),
                    "predicted_interrupt": decision,
                    "scalar_interrupt": decision,
                    "r0f_interrupt": int(raw_response != "$silent$"),
                    "tag_margin": float(chunk_index),
                }
            )
        sources.append(source)
        predictions.append({"video_path": video_path, "answers": answers})
        r0_rows.append(
            {
                "input_index": input_index,
                "video_path": video_path,
                "prediction": {"video_path": video_path, "answers": ["$silent$"] * 2},
                "chunks": chunks,
            }
        )
    return sources, predictions, r0_rows, oof_rows


class AlignmentTest(unittest.TestCase):
    def test_builds_all_outcomes_and_fallback_split(self) -> None:
        rows = build_chunk_records(
            *fixtures(), expected_sessions=3, expected_chunks=6
        )
        summary = summarize_records(rows)
        self.assertEqual(
            {name: summary[name] for name in ("tp", "fp", "tn", "fn")},
            {"tp": 2, "fp": 2, "tn": 1, "fn": 1},
        )
        self.assertEqual(summary["fallback_count"], 2)
        self.assertEqual(summary["nonfallback_count"], 2)

    def test_rejects_tampered_prediction_text(self) -> None:
        sources, predictions, r0_rows, oof_rows = fixtures()
        changed = copy.deepcopy(predictions)
        changed[0]["answers"][0] = "$interrupt$Different text"
        with self.assertRaisesRegex(ValueError, "answer assembly mismatch"):
            build_chunk_records(
                sources,
                changed,
                r0_rows,
                oof_rows,
                expected_sessions=3,
                expected_chunks=6,
            )


class BlindReviewTest(unittest.TestCase):
    def test_blind_rows_exclude_gold_and_internal_fields(self) -> None:
        sources, predictions, r0_rows, oof_rows = fixtures()
        records = build_chunk_records(
            sources,
            predictions,
            r0_rows,
            oof_rows,
            expected_sessions=3,
            expected_chunks=6,
        )
        strata = {
            "tp_fallback": 1,
            "tp_nonfallback": 1,
            "fp_fallback": 1,
            "fp_nonfallback": 1,
            "fn_silent": 1,
        }
        blind, key, summary = build_review_package(
            records, sources, strata, "test-seed"
        )
        repeated, repeated_key, repeated_summary = build_review_package(
            records, sources, strata, "test-seed"
        )
        self.assertEqual(blind, repeated)
        self.assertEqual(key, repeated_key)
        self.assertEqual(summary, repeated_summary)
        self.assertEqual(len(blind), 5)
        forbidden = {
            "gold_interrupt",
            "gold_utterance",
            "is_fallback",
            "raw_response",
            "confusion",
            "stratum",
            "tag_margin",
            "fold",
        }
        self.assertFalse(any(forbidden.intersection(row) for row in blind))
        self.assertEqual({row["stratum"] for row in key}, set(strata))
        self.assertEqual(len(ratings_rows(blind)), 10)


if __name__ == "__main__":
    unittest.main()
