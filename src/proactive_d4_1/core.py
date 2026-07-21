"""Frozen D4.1 grid, sampling, ranking, sharding, and resume invariants."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from proactive_r0.artifacts import write_json
from proactive_r0.core import load_jsonl

EXPERIMENT_ID = "20260720_internvl35_1b_d4_1_input_policy_search_v1"
SEED = 20260720
SEARCH_SESSIONS_PER_DOMAIN = 20
CONFIRMATION_SESSIONS_PER_DOMAIN = 20
LENGTH_QUARTILES = 4
BASELINE_PARAMETERS = {
    "max_frames": 32,
    "frames_per_interval": 16,
    "max_history_turns": 4,
    "max_new_tokens": 64,
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class InferenceParameters:
    max_frames: int
    frames_per_interval: int
    max_history_turns: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        if self.max_frames <= 0 or self.frames_per_interval <= 0:
            raise ValueError("D4.1 frame counts must be positive")
        if self.max_history_turns < 0:
            raise ValueError("D4.1 history search does not support negative windows")
        if self.max_new_tokens <= 0:
            raise ValueError("D4.1 generation length must be positive")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "InferenceParameters":
        return cls(**{name: int(value[name]) for name in BASELINE_PARAMETERS})


BASELINE = InferenceParameters.from_mapping(BASELINE_PARAMETERS)


def stable_variant_id(parameters: InferenceParameters | Mapping[str, object]) -> str:
    params = (
        parameters
        if isinstance(parameters, InferenceParameters)
        else InferenceParameters.from_mapping(parameters)
    )
    return f"d41_{object_sha256(params.to_dict())[:12]}"


def _variant(
    parameters: InferenceParameters,
    families: Iterable[str],
    *,
    origin: str = "predefined",
    components: Mapping[str, str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "variant_id": stable_variant_id(parameters),
        "parameters": parameters.to_dict(),
        "families": sorted(set(families)),
        "is_baseline": parameters == BASELINE,
        "origin": origin,
        "components": dict(components or {}),
    }


def default_variants() -> list[dict[str, object]]:
    """Return the 16 unique preregistered configurations, baseline first."""
    variants: dict[str, dict[str, object]] = {}

    def add(parameters: InferenceParameters, family: str) -> None:
        variant_id = stable_variant_id(parameters)
        if variant_id in variants:
            families = set(variants[variant_id]["families"])  # type: ignore[arg-type]
            families.add(family)
            variants[variant_id]["families"] = sorted(families)
            return
        variants[variant_id] = _variant(parameters, [family])

    for max_frames in (16, 32, 64):
        for frames_per_interval in (8, 16, 24):
            add(
                InferenceParameters(
                    max_frames=max_frames,
                    frames_per_interval=frames_per_interval,
                    max_history_turns=BASELINE.max_history_turns,
                    max_new_tokens=BASELINE.max_new_tokens,
                ),
                "visual",
            )
    for max_history_turns in (0, 2, 4, 8, 16):
        add(
            InferenceParameters(
                max_frames=BASELINE.max_frames,
                frames_per_interval=BASELINE.frames_per_interval,
                max_history_turns=max_history_turns,
                max_new_tokens=BASELINE.max_new_tokens,
            ),
            "history",
        )
    for max_new_tokens in (16, 32, 64, 96):
        add(
            InferenceParameters(
                max_frames=BASELINE.max_frames,
                frames_per_interval=BASELINE.frames_per_interval,
                max_history_turns=BASELINE.max_history_turns,
                max_new_tokens=max_new_tokens,
            ),
            "generation",
        )
    ordered = sorted(
        variants.values(),
        key=lambda item: (
            not bool(item["is_baseline"]),
            str(item["variant_id"]),
        ),
    )
    validate_variants(ordered, expected_count=16)
    return ordered


def validate_variants(
    variants: Sequence[Mapping[str, object]], expected_count: int | None = None
) -> dict[str, Mapping[str, object]]:
    if expected_count is not None and len(variants) != expected_count:
        raise ValueError(f"Expected {expected_count} D4.1 variants, got {len(variants)}")
    by_id: dict[str, Mapping[str, object]] = {}
    baseline_count = 0
    parameter_keys: set[str] = set()
    for variant in variants:
        parameters = InferenceParameters.from_mapping(
            variant["parameters"]  # type: ignore[arg-type]
        )
        variant_id = str(variant["variant_id"])
        if variant_id != stable_variant_id(parameters):
            raise ValueError(f"Unstable D4.1 variant ID: {variant_id}")
        if variant_id in by_id:
            raise ValueError(f"Duplicate D4.1 variant ID: {variant_id}")
        parameter_key = canonical_json(parameters.to_dict())
        if parameter_key in parameter_keys:
            raise ValueError("Duplicate D4.1 inference parameter tuple")
        parameter_keys.add(parameter_key)
        baseline_count += int(parameters == BASELINE)
        by_id[variant_id] = variant
    if baseline_count != 1:
        raise ValueError("D4.1 variants must contain exactly one baseline")
    return by_id


def compose_joint_variant(
    visual: Mapping[str, object],
    history: Mapping[str, object],
    generation: Mapping[str, object],
) -> dict[str, object]:
    visual_params = InferenceParameters.from_mapping(
        visual["parameters"]  # type: ignore[arg-type]
    )
    history_params = InferenceParameters.from_mapping(
        history["parameters"]  # type: ignore[arg-type]
    )
    generation_params = InferenceParameters.from_mapping(
        generation["parameters"]  # type: ignore[arg-type]
    )
    parameters = InferenceParameters(
        max_frames=visual_params.max_frames,
        frames_per_interval=visual_params.frames_per_interval,
        max_history_turns=history_params.max_history_turns,
        max_new_tokens=generation_params.max_new_tokens,
    )
    return _variant(
        parameters,
        ["joint"],
        origin="search_joint",
        components={
            "visual": str(visual["variant_id"]),
            "history": str(history["variant_id"]),
            "generation": str(generation["variant_id"]),
        },
    )


def ranking_key(summary: Mapping[str, object]) -> tuple[float, float, float, str]:
    overall = summary.get("overall")
    timing = summary.get("timing")
    if not isinstance(overall, Mapping) or not isinstance(timing, Mapping):
        raise ValueError("D4.1 ranking summary lacks overall metrics or timing")
    return (
        -float(overall["macro_f1"]),
        -float(overall["gmean_f1"]),
        float(timing["total_model_inference_seconds"]),
        str(summary["variant_id"]),
    )


def rank_summaries(
    summaries: Iterable[Mapping[str, object]], *, require_deployable: bool = False
) -> list[Mapping[str, object]]:
    selected = [
        summary
        for summary in summaries
        if not require_deployable or bool(summary.get("deployable", False))
    ]
    return sorted(selected, key=ranking_key)


def select_search_components(
    variants: Sequence[Mapping[str, object]],
    summaries: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    selected: dict[str, Mapping[str, object]] = {}
    for family in ("visual", "history", "generation"):
        candidates = [
            variant
            for variant in variants
            if family in variant.get("families", [])
        ]
        ranked = rank_summaries(
            [summaries[str(variant["variant_id"])] for variant in candidates]
        )
        if not ranked:
            raise ValueError(f"D4.1 search has no completed {family} candidate")
        selected_id = str(ranked[0]["variant_id"])
        selected[family] = next(
            variant for variant in candidates if variant["variant_id"] == selected_id
        )
    return selected


def _session_key(seed: int, *values: object) -> str:
    joined = "\0".join(str(value) for value in (seed, *values))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_sample_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int = SEED,
    sessions_per_domain: int = SEARCH_SESSIONS_PER_DOMAIN,
) -> dict[str, object]:
    """Freeze label-blind domain x within-domain length-quartile samples."""
    if any("answers" in row for row in rows):
        raise ValueError("D4.1 sampling rows must have answers stripped")
    if sessions_per_domain <= 0 or sessions_per_domain % LENGTH_QUARTILES:
        raise ValueError("D4.1 sessions per domain must divide evenly into quartiles")
    by_domain: dict[str, list[dict[str, object]]] = {}
    for input_index, row in enumerate(rows):
        domain = row.get("domain")
        video_path = row.get("video_path")
        intervals = row.get("video_intervals")
        if not isinstance(domain, str) or not domain:
            raise ValueError(f"D4.1 source row {input_index} lacks domain")
        if not isinstance(video_path, str) or not video_path:
            raise ValueError(f"D4.1 source row {input_index} lacks video_path")
        if not isinstance(intervals, list) or not intervals:
            raise ValueError(f"D4.1 source row {input_index} lacks intervals")
        by_domain.setdefault(domain, []).append(
            {
                "input_index": input_index,
                "video_path": video_path,
                "domain": domain,
                "task": row.get("task"),
                "chunks": len(intervals),
            }
        )
    if len(by_domain) != 4:
        raise ValueError(f"D4.1 protocol requires exactly four domains, got {len(by_domain)}")

    all_sessions: list[dict[str, object]] = []
    search: list[int] = []
    confirmation: list[int] = []
    per_quartile = sessions_per_domain // LENGTH_QUARTILES
    required_per_stratum = 2 * per_quartile
    strata: dict[str, dict[str, list[int]]] = {}
    for domain in sorted(by_domain):
        sessions = sorted(
            by_domain[domain],
            key=lambda item: (
                int(item["chunks"]),
                _session_key(seed, domain, item["video_path"]),
            ),
        )
        domain_strata: dict[int, list[dict[str, object]]] = {
            quartile: [] for quartile in range(LENGTH_QUARTILES)
        }
        for rank, session in enumerate(sessions):
            quartile = min(LENGTH_QUARTILES - 1, rank * LENGTH_QUARTILES // len(sessions))
            enriched = dict(session)
            enriched["length_quartile"] = quartile + 1
            all_sessions.append(enriched)
            domain_strata[quartile].append(enriched)
        strata[domain] = {}
        for quartile, sessions_in_stratum in domain_strata.items():
            if len(sessions_in_stratum) < required_per_stratum:
                raise ValueError(
                    f"D4.1 stratum {domain}/Q{quartile + 1} has "
                    f"{len(sessions_in_stratum)} sessions; needs {required_per_stratum}"
                )
            shuffled = sorted(
                sessions_in_stratum,
                key=lambda item: _session_key(
                    seed, "sample", domain, quartile, item["video_path"]
                ),
            )
            search_indices = [
                int(item["input_index"]) for item in shuffled[:per_quartile]
            ]
            confirmation_indices = [
                int(item["input_index"])
                for item in shuffled[per_quartile:required_per_stratum]
            ]
            search.extend(search_indices)
            confirmation.extend(confirmation_indices)
            strata[domain][f"Q{quartile + 1}"] = [
                *search_indices,
                *confirmation_indices,
            ]
    all_sessions.sort(key=lambda item: int(item["input_index"]))
    chunks_by_index = {
        int(session["input_index"]): int(session["chunks"]) for session in all_sessions
    }
    smoke_index = min(search, key=lambda index: (chunks_by_index[index], index))
    manifest = {
        "schema_version": 1,
        "algorithm": "domain_x_within_domain_length_quartile_sha256",
        "seed": seed,
        "labels_used_for_sampling": False,
        "sessions_per_domain_per_subset": sessions_per_domain,
        "search": {"indices": sorted(search)},
        "confirmation": {"indices": sorted(confirmation)},
        "smoke": {"indices": [smoke_index], "selection": "shortest_search_session"},
        "full": {"indices": list(range(len(rows)))},
        "strata": strata,
        "all_sessions": all_sessions,
    }
    validate_sample_manifest(manifest, rows)
    return manifest


def validate_sample_manifest(
    manifest: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    if manifest.get("labels_used_for_sampling") is not False:
        raise ValueError("D4.1 sample manifest must be label-blind")
    sessions = manifest.get("all_sessions")
    if not isinstance(sessions, list) or len(sessions) != len(rows):
        raise ValueError("D4.1 sample manifest does not cover source sessions")
    for index, (entry, row) in enumerate(zip(sessions, rows)):
        if not isinstance(entry, Mapping) or int(entry.get("input_index", -1)) != index:
            raise ValueError(f"D4.1 sample manifest order changed at {index}")
        if entry.get("video_path") != row.get("video_path"):
            raise ValueError(f"D4.1 sample manifest video changed at {index}")
        if int(entry.get("chunks", -1)) != len(row["video_intervals"]):  # type: ignore[arg-type]
            raise ValueError(f"D4.1 sample manifest length changed at {index}")
    search = list(manifest["search"]["indices"])  # type: ignore[index]
    confirmation = list(manifest["confirmation"]["indices"])  # type: ignore[index]
    if search != sorted(search) or confirmation != sorted(confirmation):
        raise ValueError("D4.1 sample indices must preserve source order")
    if len(search) != 80 or len(confirmation) != 80:
        raise ValueError("D4.1 search and confirmation must each contain 80 sessions")
    if set(search) & set(confirmation):
        raise ValueError("D4.1 search and confirmation subsets overlap")
    smoke = list(manifest["smoke"]["indices"])  # type: ignore[index]
    if len(smoke) != 1 or smoke[0] not in search:
        raise ValueError("D4.1 smoke must be one label-blind search session")
    metadata = {int(entry["input_index"]): entry for entry in sessions}
    coverage: dict[str, dict[str, dict[int, int]]] = {}
    for split_name, indices in (("search", search), ("confirmation", confirmation)):
        coverage[split_name] = {}
        for index in indices:
            entry = metadata[index]
            domain = str(entry["domain"])
            quartile = int(entry["length_quartile"])
            coverage[split_name].setdefault(domain, {}).setdefault(quartile, 0)
            coverage[split_name][domain][quartile] += 1
        for domain, quartiles in coverage[split_name].items():
            if sum(quartiles.values()) != 20 or quartiles != {1: 5, 2: 5, 3: 5, 4: 5}:
                raise ValueError(
                    f"D4.1 {split_name} coverage changed for {domain}: {quartiles}"
                )
        if len(coverage[split_name]) != 4:
            raise ValueError(f"D4.1 {split_name} does not cover all domains")
    return {"search": len(search), "confirmation": len(confirmation), "coverage": coverage}


def partition_session_indices(
    indices: Sequence[int],
    manifest: Mapping[str, object],
    num_shards: int,
) -> list[list[int]]:
    """Length-balance sessions while retaining source order inside each shard."""
    if num_shards <= 0:
        raise ValueError("D4.1 shard count must be positive")
    if len(set(indices)) != len(indices) or list(indices) != sorted(indices):
        raise ValueError("D4.1 stage indices must be unique and source ordered")
    metadata = {
        int(entry["input_index"]): entry
        for entry in manifest["all_sessions"]  # type: ignore[index]
    }
    if any(index not in metadata for index in indices):
        raise ValueError("D4.1 shard input index is absent from the manifest")
    shard_count = min(num_shards, len(indices))
    shards: list[list[int]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for index in sorted(indices, key=lambda value: (-int(metadata[value]["chunks"]), value)):
        shard = min(range(shard_count), key=lambda value: (loads[value], value))
        shards[shard].append(index)
        loads[shard] += int(metadata[index]["chunks"])
    for shard in shards:
        shard.sort()
    flattened = [index for shard in shards for index in shard]
    if sorted(flattened) != list(indices) or len(flattened) != len(set(flattened)):
        raise ValueError("D4.1 shards do not exactly cover the stage")
    return shards


def task_hash(
    *,
    experiment_config_sha256: str,
    stage: str,
    variant: Mapping[str, object],
    shard_id: int,
    session_indices: Sequence[int],
) -> str:
    return object_sha256(
        {
            "experiment_config_sha256": experiment_config_sha256,
            "stage": stage,
            "variant_id": variant["variant_id"],
            "parameters": variant["parameters"],
            "shard_id": shard_id,
            "session_indices": list(session_indices),
        }
    )


def prepare_task_directory(path: Path, task: Mapping[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    task_path = path / "task.json"
    if task_path.exists():
        existing = json.loads(task_path.read_text(encoding="utf-8"))
        if existing != task:
            raise ValueError(f"D4.1 task configuration changed on resume: {path}")
    else:
        write_json(task_path, task)
    if not (path / "status.json").exists():
        write_json(path / "status.json", {"status": "pending", "task_hash": task["task_hash"]})


def task_should_run(path: Path, expected_task_hash: str) -> bool:
    status_path = path / "status.json"
    if not status_path.exists():
        return True
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("task_hash") != expected_task_hash:
        raise ValueError(f"D4.1 task hash changed on resume: {path}")
    return status.get("status") != "complete"


def atomic_append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    """Append one complete JSON record and durably flush it before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("D4.1 append made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_shard_records(
    records: Sequence[Mapping[str, object]],
    expected_indices: Sequence[int],
    rows: Sequence[Mapping[str, object]],
    *,
    require_complete: bool,
) -> dict[str, int]:
    if len(records) > len(expected_indices):
        raise ValueError("D4.1 shard contains extra session records")
    if require_complete and len(records) != len(expected_indices):
        raise ValueError("D4.1 shard is incomplete")
    for position, record in enumerate(records):
        expected_index = expected_indices[position]
        if int(record.get("input_index", -1)) != expected_index:
            raise ValueError(f"D4.1 shard order changed at position {position}")
        if record.get("video_path") != rows[expected_index].get("video_path"):
            raise ValueError(f"D4.1 shard video identity changed at position {position}")
        prediction = record.get("prediction")
        if not isinstance(prediction, Mapping):
            raise ValueError(f"D4.1 shard record {position} has no prediction")
        answers = prediction.get("answers")
        intervals = rows[expected_index].get("video_intervals")
        if not isinstance(answers, list) or not isinstance(intervals, list):
            raise ValueError(f"D4.1 shard record {position} is structurally invalid")
        if len(answers) != len(intervals):
            raise ValueError(f"D4.1 shard answer coverage changed at position {position}")
    return {"sessions": len(records), "expected_sessions": len(expected_indices)}


def load_task_records(path: Path) -> list[dict[str, object]]:
    return load_jsonl(path) if path.exists() else []


def ensure_experiment_identity(
    experiment_dir: Path,
    *,
    config_sha256: str,
    identity: Mapping[str, object],
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    identity_path = experiment_dir / "experiment_identity.json"
    value = {**dict(identity), "experiment_config_sha256": config_sha256}
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError("D4.1 experiment directory has a different configuration hash")
    else:
        if any(experiment_dir.iterdir()):
            raise FileExistsError(
                "D4.1 output directory is non-empty but has no experiment identity"
            )
        write_json(identity_path, value)


def event(path: Path, name: str, **fields: object) -> None:
    atomic_append_jsonl(path, {"event": name, **fields})


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise ValueError("D4.1 percentile requires non-empty values and q in [0,1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
