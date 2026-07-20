"""Validation and target derivation for S1 procedural-state annotations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


STEP_IDS = ("s1", "s2", "s3", "s4")
PROGRESS_VALUES = (
    "not_started",
    "ongoing",
    "complete",
    "deviated",
    "recovered",
)
FORBIDDEN_MARKERS = (
    "$interrupt$",
    "$silent$",
    "gold label",
    "gold decision",
    "should interrupt",
    "should stay silent",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def load_optional_jsonl(path: Path) -> list[dict[str, object]]:
    return load_jsonl(path) if path.exists() else []


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    normalized = value.strip()
    lowered = normalized.lower()
    for marker in FORBIDDEN_MARKERS:
        if marker in lowered:
            raise ValueError(f"{field} contains forbidden marker {marker!r}")
    return normalized


def _text_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return [_text(item, f"{field}[{index}]") for index, item in enumerate(value)]


def validate_plan(annotation: dict[str, object], session: dict[str, object]) -> None:
    """Validate the static query/task-only part without reading dynamic state."""
    for field in (
        "input_index",
        "video_path",
        "query",
        "task",
        "domain",
        "length_band",
        "state_split",
    ):
        if annotation.get(field) != session.get(field):
            raise ValueError(f"row {session.get('input_index')}: {field} mismatch")
    provenance = annotation.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"row {session.get('input_index')}: missing provenance")
    if provenance.get("plan_inputs") != ["task", "query"]:
        raise ValueError(f"row {session.get('input_index')}: plan provenance mismatch")
    _text(annotation.get("goal"), f"row {session.get('input_index')}.goal")
    steps = annotation.get("steps")
    if not isinstance(steps, list) or len(steps) != 4:
        raise ValueError(f"row {session.get('input_index')}: exactly four steps required")
    for position, (step, expected_id) in enumerate(zip(steps, STEP_IDS)):
        if not isinstance(step, dict) or step.get("id") != expected_id:
            raise ValueError(
                f"row {session.get('input_index')}.steps[{position}]: expected {expected_id}"
            )
        _text(step.get("text"), f"row {session.get('input_index')}.steps[{position}].text")
        completion = _text_list(
            step.get("completion_cues"),
            f"row {session.get('input_index')}.steps[{position}].completion_cues",
        )
        incompletion = _text_list(
            step.get("incompletion_cues"),
            f"row {session.get('input_index')}.steps[{position}].incompletion_cues",
        )
        if not completion or not incompletion:
            raise ValueError(
                f"row {session.get('input_index')}.steps[{position}]: cues cannot be empty"
            )


def validate_annotation(
    annotation: dict[str, object], session: dict[str, object]
) -> dict[str, int]:
    """Validate one completed causal S1 annotation and return target counts."""
    validate_plan(annotation, session)
    input_index = session.get("input_index")
    if annotation.get("schema_version") != 1 or annotation.get("status") != "complete":
        raise ValueError(f"row {input_index}: annotation must be schema 1 and complete")
    provenance = annotation["provenance"]
    assert isinstance(provenance, dict)
    expected_chunk_inputs = [
        "task",
        "query",
        "dialog_at_chunk",
        "video_intervals_so_far",
    ]
    expected_excluded = [
        "answers",
        "future_dialog",
        "future_video",
        "model_outputs",
        "R0/D1/D3 errors",
        "ratings",
        "existing_oracle_states",
    ]
    if provenance.get("chunk_inputs") != expected_chunk_inputs:
        raise ValueError(f"row {input_index}: chunk provenance mismatch")
    if provenance.get("excluded_inputs") != expected_excluded:
        raise ValueError(f"row {input_index}: excluded-input provenance mismatch")
    if provenance.get("annotation_type") != "s1_training_or_heldout_causal_state_supervision":
        raise ValueError(f"row {input_index}: annotation type mismatch")

    chunks = session.get("chunks")
    states = annotation.get("chunk_states")
    if not isinstance(chunks, list) or not isinstance(states, list):
        raise ValueError(f"row {input_index}: chunks/states must be lists")
    if len(states) != len(chunks):
        raise ValueError(f"row {input_index}: state count does not match chunks")

    counts = {"states": 0, "error_present": 0}
    previous_progress: str | None = None
    for chunk_index, (chunk, state) in enumerate(zip(chunks, states)):
        if not isinstance(chunk, dict) or not isinstance(state, dict):
            raise ValueError(f"row {input_index} chunk {chunk_index}: malformed state")
        if state.get("chunk_index") != chunk_index:
            raise ValueError(f"row {input_index}: chunk states must be contiguous")
        timestamp = state.get("observed_through_sec")
        if not isinstance(timestamp, (int, float)) or abs(
            float(timestamp) - float(chunk["observed_through_sec"])
        ) > 1e-6:
            raise ValueError(f"row {input_index} chunk {chunk_index}: timestamp mismatch")
        current = state.get("current_step_id")
        if current not in STEP_IDS:
            raise ValueError(f"row {input_index} chunk {chunk_index}: bad current step")
        expected_next = STEP_IDS[STEP_IDS.index(str(current)) + 1] if current != "s4" else None
        if state.get("next_step_id") != expected_next:
            raise ValueError(f"row {input_index} chunk {chunk_index}: bad next step")
        progress = state.get("progress")
        if progress not in PROGRESS_VALUES:
            raise ValueError(f"row {input_index} chunk {chunk_index}: bad progress")
        completion = _text_list(
            state.get("completion_evidence"),
            f"row {input_index}.chunk_states[{chunk_index}].completion_evidence",
        )
        incomplete = _text_list(
            state.get("incompletion_or_error_evidence"),
            f"row {input_index}.chunk_states[{chunk_index}].incompletion_or_error_evidence",
        )
        if not completion and not incomplete:
            raise ValueError(f"row {input_index} chunk {chunk_index}: evidence is empty")
        if progress == "complete" and incomplete:
            raise ValueError(f"row {input_index} chunk {chunk_index}: complete has error evidence")
        if progress in {"not_started", "ongoing", "deviated"} and not incomplete:
            raise ValueError(f"row {input_index} chunk {chunk_index}: incomplete state lacks evidence")
        if progress == "recovered" and previous_progress != "deviated":
            raise ValueError(
                f"row {input_index} chunk {chunk_index}: recovered must follow deviated"
            )
        _text(state.get("recovery_action"), f"row {input_index}.chunk_states[{chunk_index}].recovery_action")
        confidence = state.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError(f"row {input_index} chunk {chunk_index}: bad confidence")
        counts["states"] += 1
        counts["error_present"] += int(bool(incomplete))
        previous_progress = str(progress)
    return counts


def validate_collection(
    annotations: Iterable[dict[str, object]],
    sessions: Iterable[dict[str, object]],
    expected_split: str | None = None,
) -> dict[str, int]:
    session_rows = list(sessions)
    annotation_rows = list(annotations)
    if expected_split is not None:
        session_rows = [row for row in session_rows if row.get("state_split") == expected_split]
    by_index: dict[int, dict[str, object]] = {}
    for row in annotation_rows:
        input_index = row.get("input_index")
        if not isinstance(input_index, int) or input_index in by_index:
            raise ValueError(f"invalid or duplicate annotation input_index {input_index!r}")
        by_index[input_index] = row
    expected = {int(row["input_index"]) for row in session_rows}
    if set(by_index) != expected:
        raise ValueError(
            f"annotation coverage mismatch: missing={sorted(expected - set(by_index))}, "
            f"extra={sorted(set(by_index) - expected)}"
        )
    summary = {"sessions": 0, "states": 0, "error_present": 0}
    for session in session_rows:
        result = validate_annotation(by_index[int(session["input_index"])], session)
        summary["sessions"] += 1
        summary["states"] += result["states"]
        summary["error_present"] += result["error_present"]
    return summary
