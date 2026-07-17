"""Online R0 generation plus the serialized D1 causal scalar decision head."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

from proactive_r0.core import (
    CausalInferenceConfig,
    ProactiveModel,
    StarterKitSymbols,
    build_messages,
    canonicalize_response,
    subsample_frames,
)

from .core import (
    LinearDecisionHead,
    causal_scalar_values,
    decision_answer,
    predict_feature_values,
)
from .internvl_features import NeuralDecisionFeatures


class FusedProactiveModel(ProactiveModel, Protocol):
    def extract_decision_features(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
    ) -> NeuralDecisionFeatures: ...


def deployment_domains(head: LinearDecisionHead) -> tuple[str, ...]:
    domains = tuple(
        name.removeprefix("domain=")
        for name in head.feature_names
        if name.startswith("domain=")
    )
    if not domains or len(set(domains)) != len(domains):
        raise ValueError("D1 deployment head must contain unique domain one-hot features")
    return domains


def fused_feature_values(
    scalar_values: dict[str, float],
    features: NeuralDecisionFeatures,
    head: LinearDecisionHead,
) -> dict[str, float]:
    """Combine online scalar, tag-margin, and causal hidden features in head order."""
    hidden_names = tuple(
        name for name in head.feature_names if name.startswith("hidden_")
    )
    expected_names = tuple(f"hidden_{index:04d}" for index in range(len(hidden_names)))
    if hidden_names != expected_names:
        raise ValueError("D1 fused deployment head has a non-canonical hidden order")
    if "tag_margin" not in head.feature_names or not hidden_names:
        raise ValueError("D1 fused deployment head lacks neural decision features")
    hidden = features.hidden_state
    if hasattr(hidden, "detach"):
        hidden = hidden.detach()  # type: ignore[union-attr]
    if hasattr(hidden, "cpu"):
        hidden = hidden.cpu()  # type: ignore[union-attr]
    if hasattr(hidden, "tolist"):
        hidden = hidden.tolist()  # type: ignore[union-attr]
    if not isinstance(hidden, (list, tuple)) or len(hidden) != len(hidden_names):
        raise ValueError("Online D1 hidden state differs from deployment head width")
    values = dict(scalar_values)
    values["tag_margin"] = float(features.tag_margin)
    for name, value in zip(hidden_names, hidden):
        values[name] = float(value)
    required = [name for name in head.feature_names if name not in values]
    if required:
        raise ValueError(f"Online D1 fused features missing: {required}")
    if not all(math.isfinite(values[name]) for name in head.feature_names):
        raise ValueError("Online D1 fused features contain non-finite values")
    return values


def process_session_with_scalar_head(
    row: dict[str, object],
    input_index: int,
    video_folder: Path,
    model: ProactiveModel,
    starter: StarterKitSymbols,
    config: CausalInferenceConfig,
    head: LinearDecisionHead,
) -> dict[str, object]:
    """Run one session causally while applying the serialized gate online."""
    video_name = str(row["video_path"])
    intervals = [
        (float(value[0]), float(value[1]))
        for value in row["video_intervals"]  # type: ignore[index]
    ]
    domains = deployment_domains(head)
    if str(row.get("domain", "")) not in domains:
        raise ValueError(f"Unsupported D1 deployment domain: {row.get('domain')!r}")
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
        values = causal_scalar_values(
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
        decision, logit = predict_feature_values(head, values)
        answer = decision_answer(raw_response, decision)
        answers.append(answer)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "interval": list(interval),
                "current_interval_frames": len(current_frames),
                "model_input_frames": len(model_frames),
                "raw_response": raw_response,
                "r0_answer": r0_answer,
                "r0_normalization": r0_normalization,
                "decision_logit": logit,
                "decision_threshold": head.threshold_logit,
                "decision_interrupt": decision,
                "answer": answer,
            }
        )
        previous_end = interval[1]
    return {
        "input_index": input_index,
        "video_path": video_name,
        "prediction": {"video_path": video_name, "answers": answers},
        "chunks": chunks,
    }


def process_session_with_fused_head(
    row: dict[str, object],
    input_index: int,
    video_folder: Path,
    model: FusedProactiveModel,
    starter: StarterKitSymbols,
    config: CausalInferenceConfig,
    head: LinearDecisionHead,
    record_hidden_state: bool = False,
) -> dict[str, object]:
    """Run one causal session with R0 generation and the fused neural gate."""
    video_name = str(row["video_path"])
    intervals = [
        (float(value[0]), float(value[1]))
        for value in row["video_intervals"]  # type: ignore[index]
    ]
    domains = deployment_domains(head)
    if str(row.get("domain", "")) not in domains:
        raise ValueError(f"Unsupported D1 deployment domain: {row.get('domain')!r}")
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
        values = fused_feature_values(scalar_values, neural, head)
        decision, logit = predict_feature_values(head, values)
        answer = decision_answer(raw_response, decision)
        answers.append(answer)
        chunk_record: dict[str, object] = {
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
            "decision_logit": logit,
            "decision_threshold": head.threshold_logit,
            "decision_interrupt": decision,
            "answer": answer,
        }
        if record_hidden_state:
            hidden_names = [
                name for name in head.feature_names if name.startswith("hidden_")
            ]
            chunk_record["hidden_state"] = [values[name] for name in hidden_names]
        chunks.append(chunk_record)
        previous_end = interval[1]
    return {
        "input_index": input_index,
        "video_path": video_name,
        "prediction": {"video_path": video_name, "answers": answers},
        "chunks": chunks,
    }
