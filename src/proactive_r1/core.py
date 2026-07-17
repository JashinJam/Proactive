"""Controlled multi-variant causal inference for the R1 oracle-state pilot."""

from __future__ import annotations

from pathlib import Path

from proactive_r0.core import (
    CausalInferenceConfig,
    ProactiveModel,
    StarterKitSymbols,
    canonicalize_response,
    subsample_frames,
)

from .state import STATE_VARIANTS, StateVariant, build_state_messages, render_state


def process_session_variants(
    row: dict[str, object],
    input_index: int,
    annotation: dict[str, object],
    video_folder: Path,
    model: ProactiveModel,
    starter: StarterKitSymbols,
    config: CausalInferenceConfig,
    variants: tuple[StateVariant, ...] = STATE_VARIANTS,
) -> dict[str, object]:
    """Decode frames once per chunk, then run each controlled state variant."""
    video_name = str(row["video_path"])
    video_path = video_folder / video_name
    intervals = [
        (float(interval[0]), float(interval[1]))
        for interval in row["video_intervals"]  # type: ignore[index]
    ]
    cumulative_frames: list[object] = []
    outputs: dict[str, dict[str, object]] = {
        variant: {"answers": [], "chunks": []} for variant in variants
    }

    for chunk_index, interval in enumerate(intervals):
        current_frames = starter.extract_frames(
            str(video_path),
            intervals=[interval],
            frames_per_interval=config.frames_per_interval,
        )
        cumulative_frames.extend(current_frames)
        model_frames = subsample_frames(cumulative_frames, config.max_frames)
        for variant in variants:
            state_block = render_state(annotation, chunk_index, variant)
            messages = build_state_messages(
                row=row,
                chunk_index=chunk_index,
                system_prompt=starter.system_prompt,
                normalize_dialog_turns=starter.normalize_dialog_turns,
                max_history_turns=config.max_history_turns,
                state_block=state_block,
            )
            raw_response = model.generate(
                model_frames, messages, max_new_tokens=config.max_new_tokens
            )
            answer, normalization = canonicalize_response(raw_response)
            answers = outputs[variant]["answers"]
            chunks = outputs[variant]["chunks"]
            assert isinstance(answers, list) and isinstance(chunks, list)
            answers.append(answer)
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "interval": [interval[0], interval[1]],
                    "current_interval_frames": len(current_frames),
                    "model_input_frames": len(model_frames),
                    "state_block": state_block,
                    "raw_response": raw_response,
                    "answer": answer,
                    "normalization": normalization,
                }
            )

    variants_record: dict[str, object] = {}
    for variant, output in outputs.items():
        variants_record[variant] = {
            "prediction": {"video_path": video_name, "answers": output["answers"]},
            "chunks": output["chunks"],
        }
    return {
        "input_index": input_index,
        "video_path": video_name,
        "variants": variants_record,
    }

