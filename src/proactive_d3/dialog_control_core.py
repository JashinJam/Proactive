"""Label-free dialog-policy features for the frozen D3 mechanism control."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from proactive_d1.core import LabelFreeChunk, LabeledChunk
from proactive_d1.neural_core import NeuralFeatureCache, neural_matrix
from proactive_r0.core import INTERRUPT_TAG


DIALOG_POLICY_NAMES = (
    "has_previous_chunk",
    "assistant_added_since_previous",
    "assistant_add_count_since_previous",
    "log1p_visible_assistant_turns",
    "assistant_turns_per_elapsed_chunk",
    "log1p_chunks_since_assistant_addition",
    "log1p_last_assistant_text_length",
    "last_assistant_has_interrupt_tag",
)
VARIANTS = (
    "d1_fused_replay",
    "dialog_increment_only",
    "dialog_stage_only",
    "d1_fused_plus_dialog_increment",
    "d1_fused_plus_dialog_stage",
)


def _assistant_texts(value: object) -> tuple[list[str], int, int]:
    if not isinstance(value, list):
        raise ValueError("Dialog-policy control requires each chunk dialog to be a list")
    texts: list[str] = []
    malformed = 0
    empty = 0
    for turn in value:
        if not isinstance(turn, dict):
            malformed += 1
            continue
        text = str(turn.get("text") or "").strip()
        if not text:
            empty += 1
            continue
        role = str(turn.get("role", "user")).strip().lower()
        if role == "assistant":
            texts.append(text)
    return texts, malformed, empty


def build_dialog_policy_features(
    rows: Sequence[dict[str, object]],
    chunks: Sequence[LabelFreeChunk],
) -> tuple[np.ndarray, dict[str, object]]:
    """Build causal features without accepting labels or prediction objects."""
    if any("answers" in row for row in rows):
        raise ValueError("Dialog-policy features require answer-stripped source rows")
    values: list[list[float]] = []
    cursor = 0
    malformed_turns = 0
    empty_turns = 0
    multi_add_chunks = 0
    additions = 0
    for input_index, row in enumerate(rows):
        dialog = row.get("dialog")
        intervals = row.get("video_intervals")
        if not isinstance(dialog, list) or not isinstance(intervals, list):
            raise ValueError(f"Dialog-policy source {input_index} is malformed")
        if len(dialog) != len(intervals):
            raise ValueError(f"Dialog-policy source {input_index} has unequal chunk fields")
        previous_count = 0
        chunks_since_addition = 0
        for chunk_index, current_dialog in enumerate(dialog):
            if cursor >= len(chunks):
                raise ValueError("Dialog-policy source contains extra chunks")
            chunk = chunks[cursor]
            if (chunk.input_index, chunk.chunk_index) != (input_index, chunk_index):
                raise ValueError("Dialog-policy source order differs from D1 chunks")
            assistant_texts, malformed, empty = _assistant_texts(current_dialog)
            malformed_turns += malformed
            empty_turns += empty
            current_count = len(assistant_texts)
            if chunk_index == 0:
                added_count = 0
                added = False
                chunks_since_addition = 0
            else:
                added_count = current_count - previous_count
                if added_count < 0:
                    raise ValueError(
                        f"Assistant count decreased at {(input_index, chunk_index)}"
                    )
                added = added_count > 0
                if added:
                    chunks_since_addition = 0
                    additions += 1
                else:
                    chunks_since_addition += 1
                multi_add_chunks += int(added_count > 1)
            last_text = assistant_texts[-1] if assistant_texts else ""
            values.append(
                [
                    float(chunk_index > 0),
                    float(added),
                    float(added_count),
                    math.log1p(current_count),
                    current_count / max(chunk_index, 1),
                    math.log1p(chunks_since_addition),
                    math.log1p(len(last_text)),
                    float(last_text.lstrip().startswith(INTERRUPT_TAG)),
                ]
            )
            previous_count = current_count
            cursor += 1
    if cursor != len(chunks):
        raise ValueError("Dialog-policy source does not cover every D1 chunk")
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (len(chunks), len(DIALOG_POLICY_NAMES)):
        raise ValueError("Dialog-policy matrix has the wrong shape")
    if not np.isfinite(matrix).all():
        raise ValueError("Dialog-policy matrix contains non-finite values")
    first = np.asarray([chunk.chunk_index == 0 for chunk in chunks])
    return matrix, {
        "sessions": len(rows),
        "chunks": len(chunks),
        "feature_names": list(DIALOG_POLICY_NAMES),
        "shape": list(matrix.shape),
        "malformed_turns_ignored": malformed_turns,
        "empty_turns_ignored": empty_turns,
        "multi_assistant_add_chunks": multi_add_chunks,
        "non_first_chunks_with_assistant_addition": additions,
        "first_chunk_increment_feature_abs_max": float(
            np.abs(matrix[first, :3]).max()
        ),
        "labels_read": False,
        "predictions_read": False,
        "future_chunks_read": False,
    }


def dialog_control_matrix(
    examples: Sequence[LabeledChunk],
    cache: NeuralFeatureCache,
    scalar_names: Sequence[str],
    dialog_values: np.ndarray,
    variant: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported dialog-policy variant: {variant}")
    if dialog_values.shape != (len(examples), len(DIALOG_POLICY_NAMES)):
        raise ValueError("Dialog-policy values do not align with labeled examples")
    base, base_names = neural_matrix(examples, cache, scalar_names, "fused_linear")
    increment = dialog_values[:, :2]
    increment_names = DIALOG_POLICY_NAMES[:2]
    if variant == "d1_fused_replay":
        return base, base_names
    if variant == "dialog_increment_only":
        return increment, increment_names
    if variant == "dialog_stage_only":
        return dialog_values, DIALOG_POLICY_NAMES
    if variant == "d1_fused_plus_dialog_increment":
        return np.concatenate([base, increment], axis=1), (
            *base_names,
            *increment_names,
        )
    return np.concatenate([base, dialog_values], axis=1), (
        *base_names,
        *DIALOG_POLICY_NAMES,
    )
