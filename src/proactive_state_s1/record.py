"""Append one immutable S1 state record or materialize a completed split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from proactive_r0.artifacts import write_json
from proactive_state_s1.core import (
    FORBIDDEN_MARKERS,
    PROGRESS_VALUES,
    STEP_IDS,
    load_json,
    load_jsonl,
    load_optional_jsonl,
    validate_collection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _clean_text(value: str, field: str) -> str:
    result = value.strip()
    if not result:
        raise ValueError(f"{field} must be non-empty")
    for marker in FORBIDDEN_MARKERS:
        if marker in result.lower():
            raise ValueError(f"{field} contains forbidden marker {marker!r}")
    return result


def _session(sessions_path: Path, input_index: int) -> dict[str, object]:
    selected = [
        row for row in load_jsonl(sessions_path)
        if int(row["input_index"]) == input_index
    ]
    if len(selected) != 1:
        raise ValueError(f"Unknown or duplicate S1 input_index {input_index}")
    return selected[0]


def _session_records(records_path: Path, input_index: int) -> list[dict[str, object]]:
    return [
        row for row in load_optional_jsonl(records_path)
        if int(row.get("input_index", -1)) == input_index
    ]


def append_record(
    sessions_path: Path,
    records_path: Path,
    input_index: int,
    chunk_index: int,
    step: str,
    progress: str,
    completion: list[str],
    incomplete: list[str],
    recovery: str,
    confidence: float,
) -> dict[str, object]:
    session = _session(sessions_path, input_index)
    chunks = session["chunks"]
    assert isinstance(chunks, list)
    records = _session_records(records_path, input_index)
    if chunk_index != len(records):
        raise ValueError(
            f"input {input_index}: next record must be chunk {len(records)}, got {chunk_index}"
        )
    if not 0 <= chunk_index < len(chunks):
        raise ValueError(f"input {input_index}: all chunks are already recorded")
    if step not in STEP_IDS or progress not in PROGRESS_VALUES:
        raise ValueError("Invalid S1 step or progress value")
    completion = [_clean_text(value, "completion evidence") for value in completion]
    incomplete = [_clean_text(value, "incompletion/error evidence") for value in incomplete]
    if not completion and not incomplete:
        raise ValueError("At least one evidence item is required")
    if progress == "complete" and incomplete:
        raise ValueError("A complete state cannot have incompletion/error evidence")
    if progress in {"not_started", "ongoing", "deviated"} and not incomplete:
        raise ValueError(f"{progress} requires incompletion/error evidence")
    previous = records[-1] if records else None
    if progress == "recovered" and (
        previous is None or previous.get("progress") != "deviated"
    ):
        raise ValueError("A recovered state must immediately follow deviated")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0, 1]")
    chunk = chunks[chunk_index]
    assert isinstance(chunk, dict)
    next_step = STEP_IDS[STEP_IDS.index(step) + 1] if step != "s4" else None
    row = {
        "input_index": input_index,
        "chunk_index": chunk_index,
        "observed_through_sec": chunk["observed_through_sec"],
        "current_step_id": step,
        "progress": progress,
        "completion_evidence": completion,
        "incompletion_or_error_evidence": incomplete,
        "next_step_id": next_step,
        "recovery_action": _clean_text(recovery, "recovery action"),
        "confidence": confidence,
    }
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return row


def materialize_complete(
    sessions_path: Path,
    annotations_path: Path,
    records_path: Path,
    split: str,
    output_path: Path,
) -> dict[str, int]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite S1 annotations: {output_path}")
    sessions = load_jsonl(sessions_path)
    annotations = load_json(annotations_path)
    if not isinstance(annotations, list):
        raise ValueError("S1 annotation work file must be a JSON list")
    records = load_optional_jsonl(records_path)
    by_key = {
        (int(row["input_index"]), int(row["chunk_index"])): row for row in records
    }
    if len(by_key) != len(records):
        raise ValueError("S1 records contain duplicate input/chunk keys")
    completed: list[dict[str, object]] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("Malformed S1 annotation work row")
        value = dict(annotation)
        input_index = int(value["input_index"])
        states = value["chunk_states"]
        assert isinstance(states, list)
        value["chunk_states"] = [
            {key: item for key, item in by_key[(input_index, chunk_index)].items() if key != "input_index"}
            for chunk_index in range(len(states))
        ]
        value["status"] = "complete"
        completed.append(value)
    summary = validate_collection(completed, sessions, expected_split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, completed)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("append")
    record.add_argument("--sessions", required=True)
    record.add_argument("--records", required=True)
    record.add_argument("--input-index", type=int, required=True)
    record.add_argument("--chunk-index", type=int, required=True)
    record.add_argument("--step", choices=STEP_IDS, required=True)
    record.add_argument("--progress", choices=PROGRESS_VALUES, required=True)
    record.add_argument("--completion", action="append", default=[])
    record.add_argument("--incomplete", action="append", default=[])
    record.add_argument("--recovery", required=True)
    record.add_argument("--confidence", type=float, required=True)
    complete = subparsers.add_parser("materialize")
    complete.add_argument("--sessions", required=True)
    complete.add_argument("--annotations", required=True)
    complete.add_argument("--records", required=True)
    complete.add_argument("--split", choices=("train", "heldout"), required=True)
    complete.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "append":
        result = append_record(
            _resolve(args.sessions), _resolve(args.records), args.input_index,
            args.chunk_index, args.step, args.progress, args.completion,
            args.incomplete, args.recovery, args.confidence,
        )
    else:
        result = materialize_complete(
            _resolve(args.sessions), _resolve(args.annotations), _resolve(args.records),
            args.split, _resolve(args.output),
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
