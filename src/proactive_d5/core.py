"""Pure causal feature construction and matrix composition for D5."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from proactive_d1.core import LabelFreeChunk, LabeledChunk
from proactive_d1.neural_core import NeuralFeatureCache
from proactive_d3.core import CausalDynamics, DYNAMIC_SCALAR_NAMES
from proactive_d3.dialog_control_core import (
    DIALOG_POLICY_NAMES,
    dialog_control_matrix,
)


ACTION_HISTORY_NAMES = (
    "action_lag2_interrupt",
    "action_lag2_available",
    "action_lag3_interrupt",
    "action_lag3_available",
    "action_lag4_interrupt",
    "action_lag4_available",
    "action_interrupt_rate_last2",
    "action_interrupt_rate_last4",
    "action_interrupt_rate_last8",
    "action_log1p_consecutive_interrupts",
    "action_log1p_consecutive_silents",
    "action_log1p_chunks_since_silent",
    "action_last2_ii",
    "action_last2_is",
    "action_last2_si",
    "action_last2_ss",
    "action_transition_rate_last4",
    "action_transition_rate_last8",
)

D5Variant = Literal[
    "d4_replay",
    "d4_plus_dynamic_scalar",
    "d4_plus_full_dynamics",
    "d4_plus_action_history",
    "d4_plus_full_dynamics_history",
]
D5_VARIANTS: tuple[D5Variant, ...] = (
    "d4_replay",
    "d4_plus_dynamic_scalar",
    "d4_plus_full_dynamics",
    "d4_plus_action_history",
    "d4_plus_full_dynamics_history",
)
PRIMARY_VARIANT: D5Variant = "d4_plus_full_dynamics_history"


@dataclass(frozen=True)
class ActionHistoryStep:
    values: np.ndarray
    previous_action_interrupt: int
    assistant_count: int
    assistant_added_count: int
    malformed_turns: int
    empty_turns: int


def _assistant_count(value: object) -> tuple[int, int, int]:
    if not isinstance(value, list):
        raise ValueError("D5 action history requires each chunk dialog to be a list")
    count = 0
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
        count += int(role == "assistant")
    return count, malformed, empty


def _window_rate(actions: Sequence[int], size: int) -> float:
    window = actions[-size:]
    return float(sum(window) / len(window)) if window else 0.0


def _transition_rate(actions: Sequence[int], size: int) -> float:
    window = actions[-size:]
    if len(window) < 2:
        return 0.0
    transitions = sum(left != right for left, right in zip(window, window[1:]))
    return float(transitions / (len(window) - 1))


def action_history_values(actions: Sequence[int]) -> np.ndarray:
    """Summarize only organizer-visible actions preceding the current chunk."""
    if any(value not in (0, 1) for value in actions):
        raise ValueError("D5 prior actions must be binary")

    def lag(offset: int) -> tuple[float, float]:
        if len(actions) < offset:
            return 0.0, 0.0
        return float(actions[-offset]), 1.0

    lag2, lag2_available = lag(2)
    lag3, lag3_available = lag(3)
    lag4, lag4_available = lag(4)
    run_length = 0
    if actions:
        last = actions[-1]
        for value in reversed(actions):
            if value != last:
                break
            run_length += 1
    consecutive_interrupts = run_length if actions and actions[-1] == 1 else 0
    consecutive_silents = run_length if actions and actions[-1] == 0 else 0
    chunks_since_silent = 0
    for value in reversed(actions):
        if value == 0:
            break
        chunks_since_silent += 1
    patterns = [0.0, 0.0, 0.0, 0.0]
    if len(actions) >= 2:
        pattern_index = {
            (1, 1): 0,
            (1, 0): 1,
            (0, 1): 2,
            (0, 0): 3,
        }
        patterns[pattern_index[(actions[-2], actions[-1])]] = 1.0
    values = np.asarray(
        [
            lag2,
            lag2_available,
            lag3,
            lag3_available,
            lag4,
            lag4_available,
            _window_rate(actions, 2),
            _window_rate(actions, 4),
            _window_rate(actions, 8),
            math.log1p(consecutive_interrupts),
            math.log1p(consecutive_silents),
            math.log1p(chunks_since_silent),
            *patterns,
            _transition_rate(actions, 4),
            _transition_rate(actions, 8),
        ],
        dtype=np.float32,
    )
    if values.shape != (len(ACTION_HISTORY_NAMES),) or not np.isfinite(values).all():
        raise ValueError("D5 action-history feature vector is invalid")
    return values


@dataclass
class OnlineActionHistoryState:
    """Consume cumulative official dialog prefixes in causal session order."""

    expected_chunk_index: int = 0
    previous_assistant_count: int | None = None
    prior_actions: list[int] = field(default_factory=list)

    def consume(self, dialog_at_chunk: object, chunk_index: int) -> ActionHistoryStep:
        if chunk_index != self.expected_chunk_index:
            raise ValueError(
                f"D5 expected chunk {self.expected_chunk_index}, got {chunk_index}"
            )
        assistant_count, malformed, empty = _assistant_count(dialog_at_chunk)
        added_count = 0
        previous_action = 0
        if self.previous_assistant_count is not None:
            added_count = assistant_count - self.previous_assistant_count
            if added_count < 0:
                raise ValueError("D5 assistant count decreased within a session")
            previous_action = int(added_count > 0)
            self.prior_actions.append(previous_action)
        values = action_history_values(self.prior_actions)
        self.previous_assistant_count = assistant_count
        self.expected_chunk_index += 1
        return ActionHistoryStep(
            values=values,
            previous_action_interrupt=previous_action,
            assistant_count=assistant_count,
            assistant_added_count=added_count,
            malformed_turns=malformed,
            empty_turns=empty,
        )


def build_action_history_features(
    rows: Sequence[dict[str, object]],
    chunks: Sequence[LabelFreeChunk],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build D5 history features without accepting targets or predictions."""
    if any("answers" in row for row in rows):
        raise ValueError("D5 action history requires answer-stripped source rows")
    values: list[np.ndarray] = []
    previous_actions: list[int] = []
    cursor = 0
    malformed_turns = 0
    empty_turns = 0
    assistant_additions = 0
    multi_assistant_additions = 0
    for input_index, row in enumerate(rows):
        dialog = row.get("dialog")
        intervals = row.get("video_intervals")
        if not isinstance(dialog, list) or not isinstance(intervals, list):
            raise ValueError(f"D5 source row {input_index} is malformed")
        if len(dialog) != len(intervals):
            raise ValueError(f"D5 source row {input_index} has unequal chunk fields")
        state = OnlineActionHistoryState()
        for chunk_index, current_dialog in enumerate(dialog):
            if cursor >= len(chunks):
                raise ValueError("D5 source contains extra chunks")
            chunk = chunks[cursor]
            if (chunk.input_index, chunk.chunk_index) != (input_index, chunk_index):
                raise ValueError("D5 source order differs from label-free chunks")
            step = state.consume(current_dialog, chunk_index)
            values.append(step.values)
            previous_actions.append(step.previous_action_interrupt)
            malformed_turns += step.malformed_turns
            empty_turns += step.empty_turns
            assistant_additions += int(step.assistant_added_count > 0)
            multi_assistant_additions += int(step.assistant_added_count > 1)
            cursor += 1
    if cursor != len(chunks):
        raise ValueError("D5 source does not cover every label-free chunk")
    matrix = np.stack(values).astype(np.float32, copy=False)
    previous = np.asarray(previous_actions, dtype=np.int8)
    expected_shape = (len(chunks), len(ACTION_HISTORY_NAMES))
    if matrix.shape != expected_shape or previous.shape != (len(chunks),):
        raise ValueError("D5 action-history arrays have the wrong shape")
    first = np.asarray([chunk.chunk_index == 0 for chunk in chunks])
    if first.any() and float(np.abs(matrix[first]).max()) != 0.0:
        raise ValueError("D5 first-chunk action history must be zero")
    return matrix, previous, {
        "sessions": len(rows),
        "chunks": len(chunks),
        "feature_names": list(ACTION_HISTORY_NAMES),
        "shape": list(matrix.shape),
        "assistant_additions": assistant_additions,
        "multi_assistant_additions": multi_assistant_additions,
        "malformed_turns_ignored": malformed_turns,
        "empty_turns_ignored": empty_turns,
        "first_chunk_feature_abs_max": float(np.abs(matrix[first]).max()),
        "labels_read": False,
        "predictions_read": False,
        "future_chunks_read": False,
    }


def d5_matrix(
    examples: Sequence[LabeledChunk],
    cache: NeuralFeatureCache,
    scalar_names: Sequence[str],
    dialog_values: np.ndarray,
    dynamics: CausalDynamics,
    action_history: np.ndarray,
    variant: D5Variant,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Compose a frozen D5 variant while preserving exact D4 replay."""
    if variant not in D5_VARIANTS:
        raise ValueError(f"Unsupported D5 variant: {variant}")
    if dialog_values.shape != (len(examples), len(DIALOG_POLICY_NAMES)):
        raise ValueError("D5 dialog values do not align with examples")
    if dynamics.scalar.shape != (len(examples), len(DYNAMIC_SCALAR_NAMES)):
        raise ValueError("D5 dynamic scalars do not align with examples")
    if dynamics.hidden_delta.shape != cache.hidden_state.shape:
        raise ValueError("D5 hidden delta does not align with neural cache")
    if action_history.shape != (len(examples), len(ACTION_HISTORY_NAMES)):
        raise ValueError("D5 action history does not align with examples")

    d4, d4_names = dialog_control_matrix(
        examples,
        cache,
        scalar_names,
        dialog_values,
        "d1_fused_plus_dialog_stage",
    )
    dynamic_scalar = dynamics.scalar[:, 1:]
    dynamic_names = DYNAMIC_SCALAR_NAMES[1:]
    hidden_names = tuple(
        f"hidden_delta_{index:04d}" for index in range(cache.hidden_state.shape[1])
    )
    if variant == "d4_replay":
        values, names = d4, d4_names
    elif variant == "d4_plus_dynamic_scalar":
        values = np.concatenate([d4, dynamic_scalar], axis=1)
        names = (*d4_names, *dynamic_names)
    elif variant == "d4_plus_full_dynamics":
        values = np.concatenate(
            [d4, dynamic_scalar, dynamics.hidden_delta], axis=1
        )
        names = (*d4_names, *dynamic_names, *hidden_names)
    elif variant == "d4_plus_action_history":
        values = np.concatenate([d4, action_history], axis=1)
        names = (*d4_names, *ACTION_HISTORY_NAMES)
    else:
        values = np.concatenate(
            [d4, dynamic_scalar, dynamics.hidden_delta, action_history], axis=1
        )
        names = (*d4_names, *dynamic_names, *hidden_names, *ACTION_HISTORY_NAMES)
    if values.shape != (len(examples), len(names)) or not np.isfinite(values).all():
        raise ValueError("D5 training matrix is invalid")
    if len(set(names)) != len(names):
        raise ValueError("D5 feature names must be unique")
    return values.astype(np.float32, copy=False), tuple(names)
