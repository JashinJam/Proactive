"""Online causal dialog-stage features and D4 session inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from proactive_d1.core import (
    LinearDecisionHead,
    causal_scalar_values,
    decision_answer,
    predict_feature_values,
)
from proactive_d1.deploy import FusedProactiveModel, deployment_domains
from proactive_d1.internvl_features import NeuralDecisionFeatures
from proactive_d3.dialog_control_core import DIALOG_POLICY_NAMES
from proactive_r0.core import (
    CausalInferenceConfig,
    INTERRUPT_TAG,
    StarterKitSymbols,
    build_messages,
    canonicalize_response,
    subsample_frames,
)


def _assistant_texts(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("D4 online chunk dialog must be a list")
    result: list[str] = []
    for turn in value:
        if not isinstance(turn, dict):
            continue
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        if str(turn.get("role", "user")).strip().lower() == "assistant":
            result.append(text)
    return result


@dataclass
class OnlineDialogPolicyState:
    previous_count: int = 0
    chunks_since_addition: int = 0
    next_chunk_index: int = 0

    def consume(self, current_dialog: object, chunk_index: int) -> np.ndarray:
        if chunk_index != self.next_chunk_index:
            raise ValueError("D4 online dialog chunks must be contiguous")
        texts = _assistant_texts(current_dialog)
        current_count = len(texts)
        if chunk_index == 0:
            added_count = 0
            added = False
            self.chunks_since_addition = 0
        else:
            added_count = current_count - self.previous_count
            if added_count < 0:
                raise ValueError("D4 online assistant count decreased")
            added = added_count > 0
            if added:
                self.chunks_since_addition = 0
            else:
                self.chunks_since_addition += 1
        last_text = texts[-1] if texts else ""
        values = np.asarray(
            [
                float(chunk_index > 0),
                float(added),
                float(added_count),
                math.log1p(current_count),
                current_count / max(chunk_index, 1),
                math.log1p(self.chunks_since_addition),
                math.log1p(len(last_text)),
                float(last_text.lstrip().startswith(INTERRUPT_TAG)),
            ],
            dtype=np.float32,
        )
        self.previous_count = current_count
        self.next_chunk_index += 1
        return values


def _hidden_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        name
        for name in feature_names
        if name.startswith("hidden_")
        and len(name) == len("hidden_") + 4
        and name[len("hidden_") :].isdigit()
    )


def dialog_stage_feature_values(
    scalar_values: dict[str, float],
    neural: NeuralDecisionFeatures,
    dialog_values: Sequence[float],
    head: LinearDecisionHead,
) -> dict[str, float]:
    hidden_names = _hidden_names(head.feature_names)
    expected = tuple(f"hidden_{index:04d}" for index in range(len(hidden_names)))
    if hidden_names != expected or not hidden_names:
        raise ValueError("D4 deployment head has a non-canonical hidden order")
    if "tag_margin" not in head.feature_names:
        raise ValueError("D4 deployment head lacks tag margin")
    if tuple(name for name in head.feature_names if name in DIALOG_POLICY_NAMES) != DIALOG_POLICY_NAMES:
        raise ValueError("D4 deployment head has a non-canonical dialog feature order")
    hidden = neural.hidden_state
    if hasattr(hidden, "detach"):
        hidden = hidden.detach()  # type: ignore[union-attr]
    if hasattr(hidden, "cpu"):
        hidden = hidden.cpu()  # type: ignore[union-attr]
    hidden_array = np.asarray(hidden, dtype=np.float32).reshape(-1)
    if hidden_array.shape != (len(hidden_names),):
        raise ValueError("D4 online hidden width differs from the head")
    dialog_array = np.asarray(dialog_values, dtype=np.float32).reshape(-1)
    if dialog_array.shape != (len(DIALOG_POLICY_NAMES),):
        raise ValueError("D4 online dialog feature width changed")
    values = dict(scalar_values)
    values["tag_margin"] = float(neural.tag_margin)
    for name, value in zip(hidden_names, hidden_array):
        values[name] = float(value)
    for name, value in zip(DIALOG_POLICY_NAMES, dialog_array):
        values[name] = float(value)
    missing = [name for name in head.feature_names if name not in values]
    if missing:
        raise ValueError(f"D4 online features are missing: {missing[:8]}")
    if not all(math.isfinite(values[name]) for name in head.feature_names):
        raise ValueError("D4 online features contain non-finite values")
    return values


def process_session_with_dialog_stage_head(
    row: dict[str, object],
    input_index: int,
    video_folder: Path,
    model: FusedProactiveModel,
    starter: StarterKitSymbols,
    config: CausalInferenceConfig,
    head: LinearDecisionHead,
    record_hidden_state: bool = False,
) -> dict[str, object]:
    """Run one session with the serialized D4 dialog-stage head."""
    video_name = str(row["video_path"])
    intervals = [
        (float(value[0]), float(value[1]))
        for value in row["video_intervals"]  # type: ignore[index]
    ]
    dialog = row.get("dialog")
    if not isinstance(dialog, list) or len(dialog) != len(intervals):
        raise ValueError("D4 deployment dialog does not cover all chunks")
    domains = deployment_domains(head)
    if str(row.get("domain", "")) not in domains:
        raise ValueError(f"Unsupported D4 deployment domain: {row.get('domain')!r}")
    state = OnlineDialogPolicyState()
    hidden_names = _hidden_names(head.feature_names)
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
        dialog_vector = state.consume(dialog[chunk_index], chunk_index)
        values = dialog_stage_feature_values(
            scalar_values, neural, dialog_vector, head
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
                name: float(dialog_vector[index])
                for index, name in enumerate(DIALOG_POLICY_NAMES)
            },
            "decision_logit": logit,
            "decision_threshold": head.threshold_logit,
            "decision_interrupt": decision,
            "answer": answer,
        }
        if record_hidden_state:
            record["hidden_state"] = [values[name] for name in hidden_names]
        chunks.append(record)
        previous_end = interval[1]
    return {
        "input_index": input_index,
        "video_path": video_name,
        "prediction": {"video_path": video_name, "answers": answers},
        "chunks": chunks,
    }
