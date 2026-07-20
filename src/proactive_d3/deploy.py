"""Online causal state and session inference for the serialized D3 head."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from proactive_d1.core import (
    LinearDecisionHead,
    causal_scalar_values,
    decision_answer,
    predict_feature_values,
)
from proactive_d1.deploy import FusedProactiveModel, deployment_domains
from proactive_d1.internvl_features import NeuralDecisionFeatures
from proactive_d3.core import (
    CausalDynamics,
    DYNAMIC_SCALAR_NAMES,
    _cosine,
    _rms,
)
from proactive_r0.core import (
    CausalInferenceConfig,
    StarterKitSymbols,
    build_messages,
    canonicalize_response,
    subsample_frames,
)


def _numbered_feature_names(
    feature_names: tuple[str, ...], prefix: str
) -> tuple[str, ...]:
    """Select only canonical four-digit vector columns, not scalar name collisions."""
    return tuple(
        name
        for name in feature_names
        if name.startswith(prefix)
        and len(name) == len(prefix) + 4
        and name[len(prefix) :].isdigit()
    )


@dataclass
class OnlineCausalDynamicsState:
    hidden_size: int
    previous_hidden: np.ndarray | None = None
    previous_margin: float = 0.0
    hidden_sum: np.ndarray = field(init=False)
    margin_sum: float = 0.0
    history_count: int = 0

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("D3 online hidden size must be positive")
        self.hidden_sum = np.zeros(self.hidden_size, dtype=np.float64)

    def consume(self, hidden_state: object, tag_margin: float) -> CausalDynamics:
        hidden = hidden_state
        if hasattr(hidden, "detach"):
            hidden = hidden.detach()  # type: ignore[union-attr]
        if hasattr(hidden, "cpu"):
            hidden = hidden.cpu()  # type: ignore[union-attr]
        hidden_array = np.asarray(hidden, dtype=np.float32).reshape(-1)
        if hidden_array.shape != (self.hidden_size,):
            raise ValueError("D3 online hidden state has the wrong width")
        current = hidden_array.astype(np.float64, copy=False)
        margin = float(tag_margin)
        if not np.isfinite(current).all() or not math.isfinite(margin):
            raise ValueError("D3 online source features are non-finite")
        scalar = np.zeros(len(DYNAMIC_SCALAR_NAMES), dtype=np.float32)
        delta = np.zeros(self.hidden_size, dtype=np.float32)
        if self.history_count:
            if self.previous_hidden is None:
                raise RuntimeError("D3 online previous hidden state is missing")
            history_mean = self.hidden_sum / self.history_count
            current_delta = current - self.previous_hidden
            delta = current_delta.astype(np.float32)
            scalar = np.asarray(
                [
                    1.0,
                    margin - self.previous_margin,
                    abs(margin - self.previous_margin),
                    margin - self.margin_sum / self.history_count,
                    _cosine(current, self.previous_hidden),
                    _rms(current_delta),
                    _cosine(current, history_mean),
                    _rms(current - history_mean),
                ],
                dtype=np.float32,
            )
        self.hidden_sum += current
        self.margin_sum += margin
        self.history_count += 1
        self.previous_hidden = current.copy()
        self.previous_margin = margin
        return CausalDynamics(scalar=scalar, hidden_delta=delta)


def dynamics_feature_values(
    scalar_values: dict[str, float],
    neural: NeuralDecisionFeatures,
    state: OnlineCausalDynamicsState,
    head: LinearDecisionHead,
) -> tuple[dict[str, float], CausalDynamics]:
    base_hidden_names = _numbered_feature_names(head.feature_names, "hidden_")
    delta_names = _numbered_feature_names(head.feature_names, "hidden_delta_")
    expected_base = tuple(
        f"hidden_{index:04d}" for index in range(len(base_hidden_names))
    )
    expected_delta = tuple(
        f"hidden_delta_{index:04d}" for index in range(len(delta_names))
    )
    if base_hidden_names != expected_base or delta_names != expected_delta:
        raise ValueError("D3 deployment head has a non-canonical hidden order")
    if len(base_hidden_names) != state.hidden_size or len(delta_names) != state.hidden_size:
        raise ValueError("D3 deployment head width differs from online state")
    dynamics = state.consume(neural.hidden_state, neural.tag_margin)
    hidden = np.asarray(neural.hidden_state, dtype=np.float32).reshape(-1)
    values = dict(scalar_values)
    values["tag_margin"] = float(neural.tag_margin)
    for name, value in zip(base_hidden_names, hidden):
        values[name] = float(value)
    for index, name in enumerate(DYNAMIC_SCALAR_NAMES):
        values[name] = float(dynamics.scalar[index])
    for name, value in zip(delta_names, dynamics.hidden_delta):
        values[name] = float(value)
    missing = [name for name in head.feature_names if name not in values]
    if missing:
        raise ValueError(f"Online D3 features are missing: {missing[:8]}")
    if not all(math.isfinite(values[name]) for name in head.feature_names):
        raise ValueError("Online D3 features contain non-finite values")
    return values, dynamics


def process_session_with_dynamics_head(
    row: dict[str, object],
    input_index: int,
    video_folder: Path,
    model: FusedProactiveModel,
    starter: StarterKitSymbols,
    config: CausalInferenceConfig,
    head: LinearDecisionHead,
    record_hidden_state: bool = False,
) -> dict[str, object]:
    """Run one session while updating D3 state after every causal chunk."""
    video_name = str(row["video_path"])
    intervals = [
        (float(value[0]), float(value[1]))
        for value in row["video_intervals"]  # type: ignore[index]
    ]
    domains = deployment_domains(head)
    if str(row.get("domain", "")) not in domains:
        raise ValueError(f"Unsupported D3 deployment domain: {row.get('domain')!r}")
    hidden_names = list(_numbered_feature_names(head.feature_names, "hidden_"))
    state = OnlineCausalDynamicsState(hidden_size=len(hidden_names))
    cumulative_frames: list[object] = []
    answers: list[str] = []
    chunks: list[dict[str, object]] = []
    previous_end: float | None = None
    for chunk_index, interval in enumerate(intervals):
        current_frames = starter.extract_frames(
            str(video_folder / video_name),
            intervals=[interval],
            frames_per_interval=config.frames_per_interval,
        )
        cumulative_frames.extend(current_frames)
        model_frames = subsample_frames(cumulative_frames, config.max_frames)
        messages = build_messages(
            row=row,
            chunk_index=chunk_index,
            system_prompt=starter.system_prompt,
            normalize_dialog_turns=starter.normalize_dialog_turns,
            max_history_turns=config.max_history_turns,
        )
        raw_response = model.generate(
            model_frames, messages, max_new_tokens=config.max_new_tokens
        )
        r0_answer, r0_normalization = canonicalize_response(raw_response)
        scalar_values = causal_scalar_values(
            row=row,
            chunk_index=chunk_index,
            interval=interval,
            previous_end=previous_end,
            model_input_frames=len(model_frames),
            raw_response=raw_response,
            r0_answer=r0_answer,
            domains=domains,
            max_history_turns=config.max_history_turns,
            max_model_frames=config.max_frames,
        )
        neural = model.extract_decision_features(model_frames, messages)
        values, dynamics = dynamics_feature_values(
            scalar_values, neural, state, head
        )
        decision, logit = predict_feature_values(head, values)
        answer = decision_answer(raw_response, decision)
        answers.append(answer)
        record: dict[str, object] = {
            "chunk_index": chunk_index,
            "interval": list(interval),
            "current_interval_frames": len(current_frames),
            "model_input_frames": len(model_frames),
            "raw_response": raw_response,
            "r0_answer": r0_answer,
            "r0_normalization": r0_normalization,
            "prompt_tokens": neural.prompt_tokens,
            "silent_log_probability": neural.silent_log_probability,
            "interrupt_log_probability": neural.interrupt_log_probability,
            "tag_margin": neural.tag_margin,
            "hidden_max_abs_difference": neural.hidden_max_abs_difference,
            "hidden_cosine_similarity": neural.hidden_cosine_similarity,
            "decision_feature_mode": neural.extraction_mode,
            "candidate_forward_passes": neural.candidate_forward_passes,
            **{
                name: float(dynamics.scalar[index])
                for index, name in enumerate(DYNAMIC_SCALAR_NAMES)
            },
            "decision_logit": logit,
            "decision_threshold": head.threshold_logit,
            "decision_interrupt": decision,
            "answer": answer,
        }
        if record_hidden_state:
            record["hidden_state"] = [values[name] for name in hidden_names]
            record["hidden_delta"] = dynamics.hidden_delta.tolist()
        chunks.append(record)
        previous_end = interval[1]
    return {
        "input_index": input_index,
        "video_path": video_name,
        "prediction": {"video_path": video_name, "answers": answers},
        "chunks": chunks,
    }
