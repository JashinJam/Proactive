"""Frozen D4.2 candidates, sharding, task identity, and ranking."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from proactive_d4_1.core import object_sha256

EXPERIMENT_ID = "20260721_internvl35_1b_d4_2_adapted_input_policy_oof_v1"
BASELINE_PARAMETERS = {
    "max_frames": 32,
    "frames_per_interval": 16,
    "max_history_turns": 4,
    "max_new_tokens": 64,
}


@dataclass(frozen=True, order=True)
class PolicyParameters:
    max_frames: int
    frames_per_interval: int
    max_history_turns: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        if self.max_frames <= 0 or self.frames_per_interval <= 0:
            raise ValueError("D4.2 frame counts must be positive")
        if self.max_history_turns < 0 or self.max_new_tokens <= 0:
            raise ValueError("D4.2 history/token limits are invalid")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PolicyParameters":
        return cls(**{name: int(value[name]) for name in BASELINE_PARAMETERS})


BASELINE = PolicyParameters.from_mapping(BASELINE_PARAMETERS)


def stable_candidate_id(parameters: PolicyParameters | Mapping[str, object]) -> str:
    policy = (
        parameters
        if isinstance(parameters, PolicyParameters)
        else PolicyParameters.from_mapping(parameters)
    )
    return f"d42_{object_sha256(policy.to_dict())[:12]}"


def load_candidates(config: Mapping[str, object]) -> list[dict[str, object]]:
    configured = config.get("candidates")
    if not isinstance(configured, list):
        raise ValueError("D4.2 config has no candidate list")
    candidates: list[dict[str, object]] = []
    for item in configured:
        if not isinstance(item, Mapping):
            raise ValueError("D4.2 candidate must be an object")
        parameters = PolicyParameters.from_mapping(
            item["parameters"]  # type: ignore[arg-type]
        )
        candidates.append(
            {
                "schema_version": 1,
                "candidate_id": stable_candidate_id(parameters),
                "name": str(item["name"]),
                "mechanism": str(item["mechanism"]),
                "parameters": parameters.to_dict(),
                "is_baseline": parameters == BASELINE,
            }
        )
    validate_candidates(candidates)
    return candidates


def validate_candidates(
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    if len(candidates) != 4:
        raise ValueError(f"D4.2 requires four frozen policies, got {len(candidates)}")
    by_id: dict[str, Mapping[str, object]] = {}
    names: set[str] = set()
    parameter_keys: set[str] = set()
    baseline_count = 0
    for candidate in candidates:
        parameters = PolicyParameters.from_mapping(
            candidate["parameters"]  # type: ignore[arg-type]
        )
        candidate_id = str(candidate["candidate_id"])
        if candidate_id != stable_candidate_id(parameters):
            raise ValueError(f"D4.2 unstable candidate ID: {candidate_id}")
        if candidate_id in by_id:
            raise ValueError(f"D4.2 duplicate candidate ID: {candidate_id}")
        name = str(candidate["name"])
        if name in names:
            raise ValueError(f"D4.2 duplicate candidate name: {name}")
        parameter_key = json.dumps(parameters.to_dict(), sort_keys=True)
        if parameter_key in parameter_keys:
            raise ValueError("D4.2 duplicate policy parameters")
        baseline_count += int(parameters == BASELINE)
        names.add(name)
        parameter_keys.add(parameter_key)
        by_id[candidate_id] = candidate
    if baseline_count != 1:
        raise ValueError("D4.2 requires exactly one baseline")
    if names != {"baseline", "history8", "frames16", "tokens16"}:
        raise ValueError(f"D4.2 candidate names changed: {sorted(names)}")
    return by_id


def build_source_manifest(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if any("answers" in row for row in rows):
        raise ValueError("D4.2 source-manifest rows must have answers stripped")
    sessions: list[dict[str, object]] = []
    for input_index, row in enumerate(rows):
        intervals = row.get("video_intervals")
        if not isinstance(intervals, list) or not intervals:
            raise ValueError(f"D4.2 source row {input_index} has no intervals")
        sessions.append(
            {
                "input_index": input_index,
                "video_path": row.get("video_path"),
                "domain": row.get("domain"),
                "task": row.get("task"),
                "chunks": len(intervals),
            }
        )
    if len(sessions) != 700 or sum(int(row["chunks"]) for row in sessions) != 9935:
        raise ValueError("D4.2 requires the complete 700-session/9,935-chunk source")
    return {
        "schema_version": 1,
        "labels_used": False,
        "sessions": sessions,
        "indices": list(range(len(sessions))),
    }


def partition_indices(
    manifest: Mapping[str, object], num_shards: int
) -> list[list[int]]:
    if num_shards <= 0:
        raise ValueError("D4.2 shard count must be positive")
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("D4.2 source manifest has no sessions")
    shards: list[list[int]] = [[] for _ in range(min(num_shards, len(sessions)))]
    loads = [0] * len(shards)
    for session in sorted(
        sessions,
        key=lambda item: (-int(item["chunks"]), int(item["input_index"])),
    ):
        shard = min(range(len(shards)), key=lambda index: (loads[index], index))
        input_index = int(session["input_index"])
        shards[shard].append(input_index)
        loads[shard] += int(session["chunks"])
    for shard in shards:
        shard.sort()
    flattened = [index for shard in shards for index in shard]
    if sorted(flattened) != list(range(len(sessions))):
        raise ValueError("D4.2 shards do not exactly cover source sessions")
    return shards


def feature_task_hash(
    *,
    experiment_config_sha256: str,
    candidate: Mapping[str, object],
    shard_id: int,
    session_indices: Sequence[int],
) -> str:
    return object_sha256(
        {
            "experiment_config_sha256": experiment_config_sha256,
            "stage": "features",
            "candidate_id": candidate["candidate_id"],
            "parameters": candidate["parameters"],
            "record_hidden_state": True,
            "shard_id": shard_id,
            "session_indices": list(session_indices),
        }
    )


def validate_feature_records(
    records: Sequence[Mapping[str, object]],
    expected_indices: Sequence[int],
    rows: Sequence[Mapping[str, object]],
    *,
    hidden_size: int,
    require_complete: bool,
) -> dict[str, int]:
    if len(records) > len(expected_indices):
        raise ValueError("D4.2 shard contains extra records")
    if require_complete and len(records) != len(expected_indices):
        raise ValueError("D4.2 shard is incomplete")
    chunks = 0
    for position, record in enumerate(records):
        expected_index = expected_indices[position]
        if int(record.get("input_index", -1)) != expected_index:
            raise ValueError(f"D4.2 shard order changed at {position}")
        if record.get("video_path") != rows[expected_index].get("video_path"):
            raise ValueError(f"D4.2 shard identity changed at {position}")
        record_chunks = record.get("chunks")
        intervals = rows[expected_index].get("video_intervals")
        if not isinstance(record_chunks, list) or not isinstance(intervals, list):
            raise ValueError(f"D4.2 shard record {position} is malformed")
        if len(record_chunks) != len(intervals):
            raise ValueError(f"D4.2 chunk coverage changed at {position}")
        for chunk_index, chunk in enumerate(record_chunks):
            if not isinstance(chunk, Mapping) or int(chunk.get("chunk_index", -1)) != chunk_index:
                raise ValueError("D4.2 chunk order changed")
            hidden = chunk.get("hidden_state")
            if not isinstance(hidden, list) or len(hidden) != hidden_size:
                raise ValueError("D4.2 hidden-state coverage changed")
            if not all(math.isfinite(float(value)) for value in hidden):
                raise ValueError("D4.2 hidden state contains non-finite values")
            for name in (
                "interval",
                "model_input_frames",
                "tag_margin",
                "silent_log_probability",
                "interrupt_log_probability",
                "prompt_tokens",
                "raw_response",
                "r0_answer",
                "model_inference_seconds",
            ):
                if name not in chunk:
                    raise ValueError(f"D4.2 chunk lacks {name}")
        chunks += len(record_chunks)
    return {"sessions": len(records), "chunks": chunks}


def _without_runtime_timing(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _without_runtime_timing(item)
            for key, item in value.items()
            if key != "timing" and not str(key).endswith("_seconds")
        }
    if isinstance(value, list):
        return [_without_runtime_timing(item) for item in value]
    return value


def recover_duplicate_feature_prefix(
    records: Sequence[Mapping[str, object]],
    expected_indices: Sequence[int],
    rows: Sequence[Mapping[str, object]],
    *,
    hidden_size: int,
) -> tuple[list[Mapping[str, object]], dict[str, object]]:
    """Recover identical records interleaved by accidentally concurrent workers."""
    actual_indices = [int(record.get("input_index", -1)) for record in records]
    if len(actual_indices) == len(set(actual_indices)):
        return list(records), {
            "recovered": False,
            "original_records": len(records),
            "recovered_records": len(records),
            "discarded_duplicates": 0,
        }
    expected_positions = {index: position for position, index in enumerate(expected_indices)}
    unknown = sorted(set(actual_indices) - set(expected_positions))
    if unknown:
        raise ValueError(f"D4.2 duplicate recovery found unknown sessions: {unknown[:8]}")
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for record, input_index in zip(records, actual_indices):
        validate_feature_records(
            [record],
            [input_index],
            rows,
            hidden_size=hidden_size,
            require_complete=True,
        )
        grouped.setdefault(input_index, []).append(record)
    divergent: list[int] = []
    for input_index, duplicates in grouped.items():
        semantic_hashes = {
            object_sha256(_without_runtime_timing(record)) for record in duplicates
        }
        if len(semantic_hashes) != 1:
            divergent.append(input_index)
    if divergent:
        raise ValueError(
            "D4.2 duplicate recovery refuses semantically different records: "
            f"{sorted(divergent)[:8]}"
        )
    positions = sorted(expected_positions[index] for index in grouped)
    if positions != list(range(len(positions))):
        raise ValueError("D4.2 duplicate recovery records do not form an exact prefix")
    recovered = [grouped[index][0] for index in expected_indices[: len(positions)]]
    validate_feature_records(
        recovered,
        expected_indices,
        rows,
        hidden_size=hidden_size,
        require_complete=False,
    )
    return recovered, {
        "recovered": True,
        "original_records": len(records),
        "recovered_records": len(recovered),
        "discarded_duplicates": len(records) - len(recovered),
        "duplicate_input_indices": sorted(
            input_index for input_index, values in grouped.items() if len(values) > 1
        ),
    }


def ranking_key(summary: Mapping[str, object]) -> tuple[float, float, float, str]:
    overall = summary["overall"]
    timing = summary["timing"]
    if not isinstance(overall, Mapping) or not isinstance(timing, Mapping):
        raise ValueError("D4.2 summary lacks metrics/timing")
    return (
        -float(overall["macro_f1"]),
        -float(overall["gmean_f1"]),
        float(timing["total_model_inference_seconds"]),
        str(summary["candidate_id"]),
    )


def rank_summaries(
    summaries: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    return sorted(summaries, key=ranking_key)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
