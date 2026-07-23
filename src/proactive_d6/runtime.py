"""Causal per-session D6 training and feature extraction runtime."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from proactive_r0.internvl import check_cuda_device_occupancy
from proactive_r0.core import (
    CausalInferenceConfig,
    StarterKitSymbols,
    build_messages,
    canonicalize_response,
)

from .adapter import D6DecisionModel, D6Forward
from .data import select_uniform_causal_frames


ChunkCallback = Callable[[int, D6Forward], None]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def process_session(
    *,
    row: dict[str, object],
    input_index: int,
    video_folder: Path,
    model: D6DecisionModel,
    starter: StarterKitSymbols,
    inference: CausalInferenceConfig,
    reference: Mapping[str, object],
    memory_enabled: bool = True,
    lora_enabled: bool = True,
    record_hidden_state: bool = True,
    record_chunks: bool = True,
    callback: ChunkCallback | None = None,
    maximum_chunks: int | None = None,
) -> dict[str, object]:
    """Run one session from zero memory; callback executes before state detach."""
    if "answers" in row:
        raise ValueError("D6 model-facing session contains answers")
    if model.d6_require_exclusive_gpu:
        check_cuda_device_occupancy(model.device, require_exclusive=True)
    intervals = [
        (float(value[0]), float(value[1]))
        for value in row["video_intervals"]  # type: ignore[index]
    ]
    reference_chunks = reference.get("chunks")
    if reference.get("input_index") != input_index or not isinstance(reference_chunks, list):
        raise ValueError("D6 frozen generation reference identity changed")
    if len(reference_chunks) != len(intervals):
        raise ValueError("D6 frozen generation reference coverage changed")
    if maximum_chunks is not None and not 0 < maximum_chunks <= len(intervals):
        raise ValueError("D6 session prefix limit is outside the session")
    observed_frame_groups: list[list[object]] = []
    state = model.initial_memory_state()
    chunks: list[dict[str, object]] = []
    total_model_seconds = 0.0
    session_started = time.perf_counter()
    mode = (
        "d6_primary"
        if memory_enabled and lora_enabled
        else "d6_lora_disabled"
        if memory_enabled
        else "d6_memory_disabled"
    )
    selected_intervals = intervals if maximum_chunks is None else intervals[:maximum_chunks]
    for chunk_index, interval in enumerate(selected_intervals):
        current_frames = starter.extract_frames(
            str(video_folder / str(row["video_path"])),
            intervals=[interval],
            frames_per_interval=inference.frames_per_interval,
        )
        observed_frame_groups.append(current_frames)
        selected = select_uniform_causal_frames(
            observed_frame_groups, inference.max_frames
        )
        messages = build_messages(
            row=row,
            chunk_index=chunk_index,
            system_prompt=starter.system_prompt,
            normalize_dialog_turns=starter.normalize_dialog_turns,
            max_history_turns=inference.max_history_turns,
        )
        _synchronize(model.device)
        started = time.perf_counter()
        output = model.forward_decision(
            selected.frames,
            selected.current_interval_mask,
            messages,
            state,
            memory_enabled=memory_enabled,
            lora_enabled=lora_enabled,
        )
        _synchronize(model.device)
        elapsed = time.perf_counter() - started
        total_model_seconds += elapsed
        if callback is not None:
            callback(chunk_index, output)
        state = output.new_memory_state.detach()
        reference_chunk = reference_chunks[chunk_index]
        if not isinstance(reference_chunk, Mapping):
            raise ValueError("D6 frozen generation reference chunk is malformed")
        if int(reference_chunk.get("chunk_index", -1)) != chunk_index:
            raise ValueError("D6 frozen generation chunk order changed")
        if int(reference_chunk.get("model_input_frames", -1)) != len(selected.frames):
            raise ValueError("D6 frame selection differs from frozen history8 generation")
        raw_response = str(reference_chunk["raw_response"])
        r0_answer, r0_normalization = canonicalize_response(raw_response)
        features = output.detached_features(mode)
        record: dict[str, object] = {
            "chunk_index": chunk_index,
            "interval": list(interval),
            "current_interval_frames": len(current_frames),
            "selected_current_interval_frames": output.current_interval_frames,
            "current_interval_patch_tokens": output.current_interval_patch_tokens,
            "model_input_frames": len(selected.frames),
            "frame_source_indices": [list(value) for value in selected.source_indices],
            "raw_response": raw_response,
            "r0_answer": r0_answer,
            "r0_normalization": r0_normalization,
            "prompt_tokens": output.prompt_tokens,
            "silent_log_probability": features.silent_log_probability,
            "interrupt_log_probability": features.interrupt_log_probability,
            "tag_margin": features.tag_margin,
            "hidden_max_abs_difference": features.hidden_max_abs_difference,
            "hidden_cosine_similarity": features.hidden_cosine_similarity,
            "candidate_memory_update_max_abs_difference": output.candidate_update_max_abs_difference,
            "memory_residual_norm": float(output.residual_norm.detach().cpu()),
            "attention_entropy": float(output.attention_entropy.detach().cpu()),
            "normalized_attention_entropy": float(
                output.normalized_attention_entropy.detach().cpu()
            ),
            "decision_feature_mode": mode,
            "candidate_forward_passes": 2,
            "model_inference_seconds": elapsed,
        }
        if record_hidden_state:
            record["hidden_state"] = features.hidden_state.tolist()
        if record_chunks:
            chunks.append(record)
    return {
        "input_index": input_index,
        "video_path": row["video_path"],
        "prediction": reference["prediction"],
        "chunks": chunks,
        "timing": {
            "model_inference_seconds": total_model_seconds,
            "session_wall_seconds": time.perf_counter() - session_started,
        },
    }


def reference_by_index(
    records: Sequence[dict[str, object]], expected_sessions: int
) -> dict[int, dict[str, object]]:
    result = {int(record["input_index"]): record for record in records}
    if len(records) != expected_sessions or set(result) != set(range(expected_sessions)):
        raise ValueError("D6 frozen generation reference does not cover every session")
    return result
