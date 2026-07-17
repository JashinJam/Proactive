from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from proactive_review.core import RatingStore, SCORE_FIELDS, Study, validate_session_submission


PROJECT_ROOT = Path(__file__).resolve().parents[3]
U0_BLIND = (
    PROJECT_ROOT
    / "output/experiments/20260716_internvl35_1b_d1_utterance_u0_v1/review_items_blind.jsonl"
)
U1_BLIND = (
    PROJECT_ROOT
    / "output/experiments/20260716_internvl35_1b_fixed_gate_forced_generation_u1_v1_nostate_full"
    / "analysis/paired_review_blind.jsonl"
)


def complete_u1(item: dict[str, object]) -> dict[str, object]:
    value: dict[str, object] = {
        "review_id": item["review_id"],
        "generic_flag": "no",
        "hallucination_flag": "no",
        "premature_completion_flag": "no",
        "unsafe_flag": "no",
        "primary_error_type": "none",
        "notes": "",
    }
    value.update({field: 4 for field in SCORE_FIELDS})
    return value


class FrozenInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.u0 = Study("u0", U0_BLIND)
        cls.u1 = Study("u1", U1_BLIND)

    def test_expected_frozen_counts(self) -> None:
        self.assertEqual(len(self.u0.review_ids), 200)
        self.assertEqual(len(self.u0.sessions), 182)
        self.assertEqual(len(self.u1.review_ids), 160)
        self.assertEqual(len(self.u1.sessions), 20)
        self.assertEqual({len(session.items) for session in self.u1.sessions}, {8})

    def test_u1_candidates_are_paired_per_chunk(self) -> None:
        for session in self.u1.sessions:
            by_pair: dict[str, set[str]] = {}
            for item in session.items:
                by_pair.setdefault(str(item["pair_id"]), set()).add(str(item["candidate"]))
            self.assertEqual(len(by_pair), 4)
            self.assertTrue(all(candidates == {"A", "B"} for candidates in by_pair.values()))

    def test_u0_silent_rejects_missing_decision_fields(self) -> None:
        session = next(
            session for session in self.u0.sessions if session.items[0]["model_action"] == "silent"
        )
        with self.assertRaisesRegex(ValueError, "should_interrupt"):
            validate_session_submission(
                self.u0,
                session.session_id,
                "A",
                [{"review_id": item["review_id"]} for item in session.items],
            )

    def test_u0_silent_normalizes_content_fields_to_blank(self) -> None:
        session = next(
            session
            for session in self.u0.sessions
            if len(session.items) == 1 and session.items[0]["model_action"] == "silent"
        )
        item = session.items[0]
        value = {
            "review_id": item["review_id"],
            "should_interrupt": "yes",
            "decision_confidence_1_5": 4,
            "timeliness_1_5": 2,
            "correctness_1_5": 5,
            "notes": "Missed visible correction.",
        }
        result = validate_session_submission(self.u0, session.session_id, "A", [value])
        self.assertIsNone(result[0]["correctness_1_5"])
        self.assertEqual(result[0]["notes"], "Missed visible correction.")

    def test_u1_requires_new_unsafe_flag(self) -> None:
        session = self.u1.sessions[0]
        ratings = [complete_u1(item) for item in session.items]
        ratings[0].pop("unsafe_flag")
        with self.assertRaisesRegex(ValueError, "unsafe_flag"):
            validate_session_submission(self.u1, session.session_id, "B", ratings)


class RatingStoreTest(unittest.TestCase):
    def test_atomic_store_exports_csv_and_locks_session(self) -> None:
        study = Study("u1", U1_BLIND)
        session = study.sessions[0]
        ratings = [complete_u1(item) for item in session.items]
        with tempfile.TemporaryDirectory() as directory:
            store = RatingStore(Path(directory), {"u1": study})
            saved = store.save_session("u1", "A", session.session_id, ratings)
            self.assertEqual(saved["revision"], 1)
            document = store.load("u1", "A")
            self.assertIn(session.session_id, document["sessions"])
            with store.csv_path("u1", "A").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertIn("unsafe_flag", rows[0])
            with self.assertRaisesRegex(ValueError, "already confirmed"):
                store.save_session("u1", "A", session.session_id, ratings)

    def test_blind_hash_change_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blind = root / "blind.jsonl"
            row = {
                "review_id": "r-A",
                "pair_id": "r",
                "candidate": "A",
                "video_path": "v.mp4",
                "interval": [0.0, 2.0],
                "observed_through_sec": 2.0,
                "query": "q",
                "task": "t",
                "domain": "d",
                "chunk_index": 0,
                "prior_dialog": [],
                "candidate_utterance": "Do it.",
            }
            blind.write_text(json.dumps(row) + "\n", encoding="utf-8")
            study = Study("u1", blind)
            output = root / "ratings"
            store = RatingStore(output, {"u1": study})
            rating = complete_u1(row)
            store.save_session("u1", "A", study.sessions[0].session_id, [rating])
            row["candidate_utterance"] = "Changed."
            blind.write_text(json.dumps(row) + "\n", encoding="utf-8")
            changed = Study("u1", blind)
            changed_store = RatingStore(output, {"u1": changed})
            with self.assertRaisesRegex(ValueError, "Blind source changed"):
                changed_store.load("u1", "A")


if __name__ == "__main__":
    unittest.main()

