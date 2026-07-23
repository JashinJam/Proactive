"""Frozen D6 experiment loading, hashing, and resource gates."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from proactive_d1.core import validate_fold_manifest
from proactive_d4_1.core import object_sha256
from proactive_r0.artifacts import sha256_file
from proactive_r0.core import (
    CausalInferenceConfig,
    load_jsonl,
    validate_source_rows,
)
from proactive_r0.internvl import resolve_physical_cuda_identifier

from . import EXPERIMENT_ID
from .data import rotation_indices, sanitize_model_rows
from .runtime import reference_by_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_MINIMUM_FREE_GIB = 24.0


@dataclass(frozen=True)
class ExperimentInputs:
    config: dict[str, object]
    config_sha256: str
    config_file_sha256: str
    source_rows: tuple[dict[str, object], ...]
    answer_free_rows: tuple[dict[str, object], ...]
    manifest: dict[str, object]
    fold_by_index: dict[int, int]
    references: dict[int, dict[str, object]]
    input_path: Path
    video_folder: Path
    starter_dir: Path
    model_path: Path


def resolve_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"D6 config section is not an object: {name}")
    return value


def _verify_file(path: Path, expected: object, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"D6 required {name} is missing: {path}")
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"D6 {name} SHA256 changed: {actual} != {expected}")


def load_experiment(
    config_path: Path,
    *,
    model_path_override: Path | None = None,
) -> ExperimentInputs:
    config_path = config_path.resolve()
    config = load_json_object(config_path)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("D6 experiment ID changed")
    protocol = _mapping(config.get("protocol"), "protocol")
    protocol_path = resolve_path(protocol["path"])
    _verify_file(protocol_path, protocol["sha256"], "protocol")

    data = _mapping(config.get("data"), "data")
    input_path = resolve_path(data["input"])
    video_folder = resolve_path(data["video_folder"])
    _verify_file(input_path, data["input_sha256"], "source input")
    source_rows = load_jsonl(input_path)
    validate_source_rows(source_rows, video_folder)
    if len(source_rows) != int(data["sessions"]):
        raise ValueError("D6 source session count changed")
    if sum(len(row["video_intervals"]) for row in source_rows) != int(data["chunks"]):
        raise ValueError("D6 source chunk count changed")
    answer_free_rows = sanitize_model_rows(source_rows)

    reference = _mapping(config.get("d4_2_reference"), "d4_2_reference")
    manifest_path = resolve_path(reference["fold_manifest"])
    _verify_file(manifest_path, reference["fold_manifest_sha256"], "fold manifest")
    manifest = load_json_object(manifest_path)
    if manifest.get("seed") != config["folds"]["seed"]:  # type: ignore[index]
        raise ValueError("D6 fold seed differs from frozen D4.2 manifest")
    fold_by_index = validate_fold_manifest(manifest, answer_free_rows)
    if len(fold_by_index) != 700 or set(fold_by_index.values()) != set(range(5)):
        raise ValueError("D6 fold manifest coverage changed")

    records_path = resolve_path(reference["history8_raw_generation_records"])
    _verify_file(
        records_path,
        reference["history8_raw_generation_records_sha256"],
        "history8 generation records",
    )
    references = reference_by_index(load_jsonl(records_path), len(source_rows))
    _verify_file(
        resolve_path(reference["history8_features"]),
        reference["history8_features_sha256"],
        "D4.2 history8 features",
    )
    _verify_file(
        resolve_path(reference["history8_oof_predictions"]),
        reference["history8_oof_predictions_sha256"],
        "D4.2 history8 OOF predictions",
    )

    starter = _mapping(config.get("starter_kit"), "starter_kit")
    starter_dir = resolve_path(starter["path"])
    _verify_file(resolve_path(starter["scorer_path"]), starter["scorer_sha256"], "scorer")
    model = _mapping(config.get("model"), "model")
    model_path = (model_path_override or Path(str(model["default_local_path"]))).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"D6 local model is missing: {model_path}")
    _verify_file(model_path / "model.safetensors", model["weights_sha256"], "model weights")

    frozen = _mapping(config.get("frozen_inference"), "frozen_inference")
    inference = CausalInferenceConfig(
        frames_per_interval=int(frozen["frames_per_interval"]),
        max_frames=int(frozen["max_frames"]),
        max_history_turns=int(frozen["max_history_turns"]),
        max_new_tokens=int(frozen["max_new_tokens"]),
        frame_sampling=str(frozen["frame_sampling"]),
    )
    if inference.frame_sampling != "uniform_cumulative_v1":
        raise ValueError("D6 requires the exact D4 uniform cumulative sampler")
    for fold in range(5):
        rotation_indices(manifest, answer_free_rows, fold)

    return ExperimentInputs(
        config=config,
        config_sha256=object_sha256(config),
        config_file_sha256=sha256_file(config_path),
        source_rows=tuple(source_rows),
        answer_free_rows=tuple(answer_free_rows),
        manifest=manifest,
        fold_by_index=fold_by_index,
        references=references,
        input_path=input_path,
        video_folder=video_folder,
        starter_dir=starter_dir,
        model_path=model_path,
    )


def inference_config(config: Mapping[str, object]) -> CausalInferenceConfig:
    frozen = _mapping(config.get("frozen_inference"), "frozen_inference")
    return CausalInferenceConfig(
        frames_per_interval=int(frozen["frames_per_interval"]),
        max_frames=int(frozen["max_frames"]),
        max_history_turns=int(frozen["max_history_turns"]),
        max_new_tokens=int(frozen["max_new_tokens"]),
        frame_sampling=str(frozen["frame_sampling"]),
    )


def labels_for_allowed_sessions(
    source_rows: Sequence[Mapping[str, object]], allowed_sessions: Sequence[int]
) -> dict[tuple[int, int], int]:
    """Unseal only explicitly allowed fit/calibration session labels."""
    allowed = set(allowed_sessions)
    result: dict[tuple[int, int], int] = {}
    for input_index in sorted(allowed):
        row = source_rows[input_index]
        answers = row.get("answers")
        intervals = row.get("video_intervals")
        if not isinstance(answers, list) or not isinstance(intervals, list):
            raise ValueError("D6 allowed labels are malformed")
        if len(answers) != len(intervals):
            raise ValueError("D6 allowed labels do not align with intervals")
        for chunk_index, answer in enumerate(answers):
            result[(input_index, chunk_index)] = int(
                str(answer).lstrip().startswith("$interrupt$")
            )
    if {key[0] for key in result} != allowed:
        raise ValueError("D6 allowed label sessions changed")
    return result


def gpu_resource_audit(
    device: str,
    minimum_free_gib: float,
    *,
    require_exclusive: bool = True,
) -> dict[str, object]:
    import pynvml
    import torch

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("D6 formal execution requires CUDA")
    identifier = resolve_physical_cuda_identifier(
        torch_device, os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    pynvml.nvmlInit()
    try:
        handle = (
            pynvml.nvmlDeviceGetHandleByIndex(identifier)
            if isinstance(identifier, int)
            else pynvml.nvmlDeviceGetHandleByUUID(identifier)
        )
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        foreign = [process for process in processes if int(process.pid) != os.getpid()]
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_gib = float(memory.free) / 2**30
        result = {
            "physical_identifier": identifier,
            "free_gib": free_gib,
            "total_gib": float(memory.total) / 2**30,
            "foreign_processes": [
                {
                    "pid": int(process.pid),
                    "used_memory_bytes": int(getattr(process, "usedGpuMemory", 0)),
                }
                for process in foreign
            ],
            "minimum_free_gib": minimum_free_gib,
            "require_exclusive": require_exclusive,
        }
        if foreign and require_exclusive:
            raise RuntimeError(f"D6 requires an exclusive GPU: {result}")
        if free_gib < minimum_free_gib:
            raise RuntimeError(f"D6 GPU free-memory gate failed: {result}")
        return result
    finally:
        pynvml.nvmlShutdown()
