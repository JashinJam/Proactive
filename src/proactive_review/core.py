"""Blind review loading, validation, and durable rating storage."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import tempfile
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCORE_FIELDS = (
    "correctness_1_5",
    "specificity_1_5",
    "actionability_1_5",
    "groundedness_1_5",
    "plan_consistency_1_5",
    "conciseness_1_5",
    "safety_1_5",
)
CONTENT_COMPOSITE_FIELDS = SCORE_FIELDS[:5]
YES_NO = {"yes", "no"}
SHOULD_INTERRUPT = {"yes", "no", "uncertain"}
PRIMARY_ERRORS = {
    "none",
    "wrong_timing",
    "wrong_action",
    "wrong_object",
    "premature",
    "stale",
    "generic",
    "hallucination",
    "unsafe",
    "other",
}
FORBIDDEN_BLIND_FIELDS = {
    "answers",
    "variant",
    "source_variant",
    "gold",
    "gold_answer",
    "gold_decision",
    "confusion",
    "fallback",
    "fallback_flag",
    "used_fallback",
    "raw_response",
    "r0_raw_response",
    "fold",
    "tag_margin",
    "d1_margin",
    "error_category",
}

U0_CSV_FIELDS = (
    "review_id",
    "reviewer_slot",
    "should_interrupt",
    "decision_confidence_1_5",
    "timeliness_1_5",
    *SCORE_FIELDS,
    "generic_flag",
    "hallucination_flag",
    "unsafe_flag",
    "primary_error_type",
    "notes",
    "session_id",
    "session_revision",
    "confirmed_at",
)
U1_CSV_FIELDS = (
    "review_id",
    "pair_id",
    "candidate",
    "reviewer_slot",
    *SCORE_FIELDS,
    "generic_flag",
    "hallucination_flag",
    "premature_completion_flag",
    "unsafe_flag",
    "primary_error_type",
    "notes",
    "session_id",
    "session_revision",
    "confirmed_at",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            leaked = FORBIDDEN_BLIND_FIELDS.intersection(value)
            if leaked:
                raise ValueError(f"Blind row exposes forbidden fields: {sorted(leaked)}")
            rows.append(value)
    if not rows:
        raise ValueError(f"Blind review input is empty: {path}")
    return rows


def _session_id(video_path: str) -> str:
    stem = Path(video_path).stem
    digest = hashlib.sha256(video_path.encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}"


@dataclass(frozen=True)
class ReviewSession:
    session_id: str
    video_path: str
    task: str
    query: str
    domain: str
    items: tuple[dict[str, Any], ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "video_path": self.video_path,
            "task": self.task,
            "query": self.query,
            "domain": self.domain,
            "item_count": len(self.items),
            "review_points": len({(row["chunk_index"], row["observed_through_sec"]) for row in self.items}),
        }


class Study:
    def __init__(self, name: str, blind_path: Path) -> None:
        if name not in {"u0", "u1"}:
            raise ValueError(f"Unsupported study: {name}")
        self.name = name
        self.blind_path = blind_path
        rows = load_jsonl(blind_path)
        self.source_sha256 = hashlib.sha256(blind_path.read_bytes()).hexdigest()
        grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        for row in rows:
            self._validate_blind_row(row)
            grouped.setdefault(str(row["video_path"]), []).append(row)

        sessions: list[ReviewSession] = []
        seen_ids: set[str] = set()
        for video_path, items in grouped.items():
            items.sort(key=self._sort_key)
            first = items[0]
            session_id = _session_id(video_path)
            if session_id in seen_ids:
                raise ValueError(f"Duplicate session ID: {session_id}")
            seen_ids.add(session_id)
            sessions.append(
                ReviewSession(
                    session_id=session_id,
                    video_path=video_path,
                    task=str(first["task"]),
                    query=str(first["query"]),
                    domain=str(first["domain"]),
                    items=tuple(items),
                )
            )
        self.sessions = tuple(sessions)
        self.by_id = {session.session_id: session for session in sessions}
        self.review_ids = {str(row["review_id"]) for row in rows}

    def _validate_blind_row(self, row: dict[str, Any]) -> None:
        required = {
            "review_id",
            "video_path",
            "interval",
            "observed_through_sec",
            "query",
            "task",
            "domain",
            "chunk_index",
            "prior_dialog",
            "candidate_utterance",
        }
        if self.name == "u0":
            required.add("model_action")
        else:
            required.update({"pair_id", "candidate"})
        missing = required.difference(row)
        if missing:
            raise ValueError(f"{self.name} blind row missing fields: {sorted(missing)}")
        video_path = str(row["video_path"])
        if Path(video_path).name != video_path or not video_path.endswith(".mp4"):
            raise ValueError(f"Unsafe video path in blind row: {video_path!r}")
        interval = row["interval"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"Invalid interval for {row['review_id']}")
        if float(row["observed_through_sec"]) != float(interval[1]):
            raise ValueError(f"Observed cutoff differs from interval end: {row['review_id']}")
        if self.name == "u0" and row["model_action"] not in {"spoke", "silent"}:
            raise ValueError(f"Invalid U0 model action: {row['model_action']}")
        if self.name == "u1" and row["candidate"] not in {"A", "B"}:
            raise ValueError(f"Invalid U1 candidate: {row['candidate']}")

    def _sort_key(self, row: dict[str, Any]) -> tuple[Any, ...]:
        if self.name == "u1":
            return (int(row["chunk_index"]), str(row["candidate"]), str(row["review_id"]))
        return (int(row["chunk_index"]), str(row["review_id"]))

    def session(self, session_id: str) -> ReviewSession:
        try:
            return self.by_id[session_id]
        except KeyError as exc:
            raise ValueError(f"Unknown {self.name} session: {session_id}") from exc


def _required_score(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in range(1, 6):
        raise ValueError(f"{field} must be an integer from 1 to 5")
    return value


def _required_choice(value: Any, field: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}")
    return value


def _notes(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("notes must be text")
    value = value.strip()
    if len(value) > 2000:
        raise ValueError("notes must contain at most 2000 characters")
    return value


def validate_session_submission(
    study: Study,
    session_id: str,
    reviewer_slot: str,
    submitted: Any,
) -> list[dict[str, Any]]:
    if reviewer_slot not in {"A", "B"}:
        raise ValueError("reviewer_slot must be A or B")
    if not isinstance(submitted, list):
        raise ValueError("ratings must be a list")
    session = study.session(session_id)
    expected = {str(row["review_id"]): row for row in session.items}
    actual: dict[str, dict[str, Any]] = {}
    for value in submitted:
        if not isinstance(value, dict) or not isinstance(value.get("review_id"), str):
            raise ValueError("Each rating must contain a review_id")
        review_id = value["review_id"]
        if review_id in actual:
            raise ValueError(f"Duplicate submitted review ID: {review_id}")
        actual[review_id] = value
    if set(actual) != set(expected):
        raise ValueError("Submitted review IDs do not exactly match the session")

    result: list[dict[str, Any]] = []
    for item in session.items:
        review_id = str(item["review_id"])
        value = actual[review_id]
        normalized: dict[str, Any] = {
            "review_id": review_id,
            "reviewer_slot": reviewer_slot,
        }
        if study.name == "u0":
            normalized["should_interrupt"] = _required_choice(
                value.get("should_interrupt"), "should_interrupt", SHOULD_INTERRUPT
            )
            normalized["decision_confidence_1_5"] = _required_score(
                value.get("decision_confidence_1_5"), "decision_confidence_1_5"
            )
            normalized["timeliness_1_5"] = _required_score(
                value.get("timeliness_1_5"), "timeliness_1_5"
            )
            if item["model_action"] == "spoke":
                for field in SCORE_FIELDS:
                    normalized[field] = _required_score(value.get(field), field)
                for field in ("generic_flag", "hallucination_flag", "unsafe_flag"):
                    normalized[field] = _required_choice(value.get(field), field, YES_NO)
                normalized["primary_error_type"] = _required_choice(
                    value.get("primary_error_type"), "primary_error_type", PRIMARY_ERRORS
                )
            else:
                for field in SCORE_FIELDS:
                    normalized[field] = None
                for field in ("generic_flag", "hallucination_flag", "unsafe_flag"):
                    normalized[field] = None
                normalized["primary_error_type"] = None
        else:
            normalized["pair_id"] = str(item["pair_id"])
            normalized["candidate"] = str(item["candidate"])
            for field in SCORE_FIELDS:
                normalized[field] = _required_score(value.get(field), field)
            for field in (
                "generic_flag",
                "hallucination_flag",
                "premature_completion_flag",
                "unsafe_flag",
            ):
                normalized[field] = _required_choice(value.get(field), field, YES_NO)
            normalized["primary_error_type"] = _required_choice(
                value.get("primary_error_type"), "primary_error_type", PRIMARY_ERRORS
            )
        normalized["notes"] = _notes(value.get("notes"))
        result.append(normalized)
    return result


class RatingStore:
    """Store one independent file set per study and reviewer."""

    def __init__(self, root: Path, studies: dict[str, Study]) -> None:
        self.root = root
        self.studies = studies
        self._thread_lock = threading.RLock()

    def _directory(self, study: str, reviewer: str) -> Path:
        if study not in self.studies or reviewer not in {"A", "B"}:
            raise ValueError("Invalid study or reviewer")
        return self.root / study / f"reviewer_{reviewer}"

    @contextmanager
    def _locked(self, directory: Path) -> Iterator[None]:
        directory.mkdir(parents=True, exist_ok=True)
        lock_path = directory / ".ratings.lock"
        with self._thread_lock, lock_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def load(self, study: str, reviewer: str) -> dict[str, Any]:
        directory = self._directory(study, reviewer)
        path = directory / "ratings.json"
        if not path.is_file():
            return {
                "schema_version": 1,
                "study": study,
                "reviewer_slot": reviewer,
                "source_blind_sha256": self.studies[study].source_sha256,
                "updated_at": None,
                "sessions": {},
            }
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("study") != study or value.get("reviewer_slot") != reviewer:
            raise ValueError(f"Rating identity mismatch in {path}")
        if value.get("source_blind_sha256") != self.studies[study].source_sha256:
            raise ValueError(f"Blind source changed after ratings were created: {path}")
        return value

    def save_session(
        self,
        study_name: str,
        reviewer: str,
        session_id: str,
        submitted: Any,
    ) -> dict[str, Any]:
        study = self.studies[study_name]
        ratings = validate_session_submission(study, session_id, reviewer, submitted)
        directory = self._directory(study_name, reviewer)
        with self._locked(directory):
            document = self.load(study_name, reviewer)
            previous = document["sessions"].get(session_id)
            if previous:
                raise ValueError(
                    "This session is already confirmed and locked; ratings cannot be overwritten"
                )
            revision = 1
            confirmed_at = utc_now()
            document["sessions"][session_id] = {
                "revision": revision,
                "confirmed_at": confirmed_at,
                "ratings": ratings,
            }
            document["updated_at"] = confirmed_at
            self._write_json_atomic(directory / "ratings.json", document)
            self._write_csv_atomic(directory / "ratings.csv", study, document)
        return {
            "session_id": session_id,
            "revision": revision,
            "confirmed_at": confirmed_at,
            "progress": self.progress(study_name, reviewer),
        }

    def progress(self, study_name: str, reviewer: str) -> dict[str, Any]:
        study = self.studies[study_name]
        document = self.load(study_name, reviewer)
        completed = set(document["sessions"])
        completed_items = sum(
            len(study.by_id[session_id].items)
            for session_id in completed
            if session_id in study.by_id
        )
        domain_total: dict[str, int] = {}
        domain_completed: dict[str, int] = {}
        for session in study.sessions:
            domain_total[session.domain] = domain_total.get(session.domain, 0) + 1
            if session.session_id in completed:
                domain_completed[session.domain] = domain_completed.get(session.domain, 0) + 1
        return {
            "study": study_name,
            "reviewer_slot": reviewer,
            "sessions_total": len(study.sessions),
            "sessions_completed": len(completed),
            "items_total": len(study.review_ids),
            "items_completed": completed_items,
            "completed_session_ids": sorted(completed),
            "domains": {
                name: {"completed": domain_completed.get(name, 0), "total": total}
                for name, total in domain_total.items()
            },
            "updated_at": document.get("updated_at"),
        }

    def session_ratings(
        self, study: str, reviewer: str, session_id: str
    ) -> dict[str, Any] | None:
        return self.load(study, reviewer)["sessions"].get(session_id)

    def csv_path(self, study: str, reviewer: str) -> Path:
        return self._directory(study, reviewer) / "ratings.csv"

    def _write_json_atomic(self, path: Path, value: dict[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        self._write_text_atomic(path, payload)

    def _write_csv_atomic(self, path: Path, study: Study, document: dict[str, Any]) -> None:
        fields = U0_CSV_FIELDS if study.name == "u0" else U1_CSV_FIELDS
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for session in study.sessions:
                    saved = document["sessions"].get(session.session_id)
                    if not saved:
                        continue
                    for rating in saved["ratings"]:
                        row = dict(rating)
                        row.update(
                            {
                                "session_id": session.session_id,
                                "session_revision": saved["revision"],
                                "confirmed_at": saved["confirmed_at"],
                            }
                        )
                        writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _write_text_atomic(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
