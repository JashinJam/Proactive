"""Label-free causal dynamics derived from the frozen D1 feature sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from proactive_d1.core import LabeledChunk
from proactive_d1.neural_core import NeuralFeatureCache, neural_matrix


D3Variant = Literal[
    "d1_fused_replay",
    "dynamics_scalar",
    "dynamics_hidden",
    "dynamics_fused",
]
D3_VARIANTS: tuple[D3Variant, ...] = (
    "d1_fused_replay",
    "dynamics_scalar",
    "dynamics_hidden",
    "dynamics_fused",
)
PRIMARY_VARIANT: D3Variant = "dynamics_fused"

DYNAMIC_SCALAR_NAMES = (
    "has_previous_chunk",
    "tag_margin_delta_previous",
    "tag_margin_abs_delta_previous",
    "tag_margin_delta_history_mean",
    "hidden_cosine_previous",
    "hidden_delta_rms_previous",
    "hidden_cosine_history_mean",
    "hidden_delta_rms_history_mean",
)


@dataclass(frozen=True)
class CausalDynamics:
    scalar: np.ndarray
    hidden_delta: np.ndarray


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def build_causal_dynamics(cache: NeuralFeatureCache) -> CausalDynamics:
    """Compute prefix-invariant dynamics without labels or future rows."""
    rows, hidden_size = cache.hidden_state.shape
    if rows == 0 or hidden_size == 0:
        raise ValueError("D3 requires a non-empty hidden-state sequence")
    if cache.input_index.shape != (rows,) or cache.chunk_index.shape != (rows,):
        raise ValueError("D3 cache keys do not align with hidden rows")
    if cache.tag_margin.shape != (rows,):
        raise ValueError("D3 tag margins do not align with hidden rows")
    scalar = np.zeros((rows, len(DYNAMIC_SCALAR_NAMES)), dtype=np.float32)
    hidden_delta = np.zeros((rows, hidden_size), dtype=np.float32)

    current_input: int | None = None
    previous_chunk = -1
    previous_hidden = np.zeros(hidden_size, dtype=np.float64)
    previous_margin = 0.0
    hidden_sum = np.zeros(hidden_size, dtype=np.float64)
    margin_sum = 0.0
    history_count = 0
    last_input = -1

    for row_index in range(rows):
        input_index = int(cache.input_index[row_index])
        chunk_index = int(cache.chunk_index[row_index])
        hidden = cache.hidden_state[row_index].astype(np.float64, copy=False)
        margin = float(cache.tag_margin[row_index])
        if not np.isfinite(hidden).all() or not np.isfinite(margin):
            raise ValueError(f"D3 non-finite source feature at row {row_index}")

        if input_index != current_input:
            if input_index <= last_input or chunk_index != 0:
                raise ValueError("D3 cache sessions must be increasing and begin at chunk 0")
            current_input = input_index
            last_input = input_index
            previous_chunk = -1
            previous_hidden.fill(0.0)
            previous_margin = 0.0
            hidden_sum.fill(0.0)
            margin_sum = 0.0
            history_count = 0
        elif chunk_index != previous_chunk + 1:
            raise ValueError("D3 chunks must be contiguous within every session")

        if history_count:
            history_mean = hidden_sum / history_count
            delta_previous = hidden - previous_hidden
            hidden_delta[row_index] = delta_previous.astype(np.float32)
            scalar[row_index] = np.asarray(
                [
                    1.0,
                    margin - previous_margin,
                    abs(margin - previous_margin),
                    margin - margin_sum / history_count,
                    _cosine(hidden, previous_hidden),
                    _rms(delta_previous),
                    _cosine(hidden, history_mean),
                    _rms(hidden - history_mean),
                ],
                dtype=np.float32,
            )

        hidden_sum += hidden
        margin_sum += margin
        history_count += 1
        previous_hidden = hidden.copy()
        previous_margin = margin
        previous_chunk = chunk_index

    if not np.isfinite(scalar).all() or not np.isfinite(hidden_delta).all():
        raise ValueError("D3 dynamics contain non-finite values")
    return CausalDynamics(scalar=scalar, hidden_delta=hidden_delta)


def d3_matrix(
    examples: Sequence[LabeledChunk],
    cache: NeuralFeatureCache,
    scalar_names: Sequence[str],
    dynamics: CausalDynamics,
    variant: D3Variant,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if variant not in D3_VARIANTS:
        raise ValueError(f"Unsupported D3 variant: {variant}")
    base, base_names = neural_matrix(examples, cache, scalar_names, "fused_linear")
    if dynamics.scalar.shape != (len(examples), len(DYNAMIC_SCALAR_NAMES)):
        raise ValueError("D3 dynamic scalar matrix does not align with examples")
    if dynamics.hidden_delta.shape != cache.hidden_state.shape:
        raise ValueError("D3 hidden-delta matrix does not align with cache")
    hidden_names = tuple(
        f"hidden_delta_{index:04d}" for index in range(cache.hidden_state.shape[1])
    )
    if variant == "d1_fused_replay":
        return base, base_names
    if variant == "dynamics_scalar":
        return np.concatenate([base, dynamics.scalar], axis=1), (
            *base_names,
            *DYNAMIC_SCALAR_NAMES,
        )
    if variant == "dynamics_hidden":
        return np.concatenate([base, dynamics.hidden_delta], axis=1), (
            *base_names,
            *hidden_names,
        )
    return np.concatenate([base, dynamics.scalar, dynamics.hidden_delta], axis=1), (
        *base_names,
        *DYNAMIC_SCALAR_NAMES,
        *hidden_names,
    )
