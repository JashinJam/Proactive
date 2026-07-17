"""Prompt, state, and output controls for fixed-gate utterance generation."""

from __future__ import annotations

import copy
from typing import Sequence

from proactive_r0.core import INTERRUPT_TAG, SILENT_TAG
from proactive_u0.core import FALLBACK_ANSWER


CONTROLLED_GENERATION_SUFFIX = """

[Controlled content-generation pass]
The interrupt decision has already been made and the assistant response is
prefilled with $interrupt$. Continue that prefix with exactly one concise,
grounded, actionable English utterance for the current moment. Do not emit
another decision tag. Do not claim an action is complete unless the visible
evidence supports it.
""".strip()


def controlled_messages(
    messages: Sequence[dict[str, str]], state_block: str | None = None
) -> list[dict[str, str]]:
    result = copy.deepcopy(list(messages))
    if not result or result[0].get("role") != "system":
        raise ValueError("U1 messages must begin with the system prompt")
    suffix = CONTROLLED_GENERATION_SUFFIX
    if state_block:
        suffix += "\n\n" + state_block.strip()
    result[0]["content"] = result[0]["content"].rstrip() + "\n\n" + suffix
    return result


def normalize_continuation(continuation: object) -> dict[str, object]:
    raw = str(continuation).strip()
    content = raw
    reason: str | None = None
    extra_interrupt_tag = False
    generated_silent_tag = False
    if content.startswith(INTERRUPT_TAG):
        extra_interrupt_tag = True
        content = content[len(INTERRUPT_TAG) :].strip()
        reason = "removed_repeated_interrupt_tag"
    elif content.startswith(SILENT_TAG):
        generated_silent_tag = True
        content = ""
        reason = "generated_silent_after_forced_interrupt_prefix"
    if not content:
        answer = FALLBACK_ANSWER
        if reason is None:
            reason = "empty_continuation_used_fallback"
    else:
        answer = f"{INTERRUPT_TAG}{content}"
    return {
        "raw_continuation": raw,
        "content": content,
        "answer": answer,
        "normalization": reason,
        "extra_interrupt_tag": extra_interrupt_tag,
        "generated_silent_tag": generated_silent_tag,
        "empty_continuation": not bool(raw),
        "used_fallback": answer == FALLBACK_ANSWER,
    }


def _step_text(steps: Sequence[dict[str, object]], step_id: object) -> str:
    if step_id in (None, ""):
        return "none"
    matches = [str(step["text"]) for step in steps if step.get("id") == step_id]
    if len(matches) != 1:
        raise ValueError(f"Oracle state references unknown step ID: {step_id!r}")
    return matches[0]


def oracle_state_block(
    annotation: dict[str, object], sample_id: str, variant: str
) -> str:
    if annotation.get("status") != "complete":
        raise ValueError(f"Oracle annotation is not complete for {sample_id}")
    steps = annotation.get("steps")
    states = annotation.get("sampled_chunk_states")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Oracle annotation has no steps for {sample_id}")
    if not isinstance(states, list):
        raise ValueError(f"Oracle annotation has no sampled states for {sample_id}")
    matches = [state for state in states if state.get("sample_id") == sample_id]
    if len(matches) != 1:
        raise ValueError(f"Oracle annotation state mismatch for {sample_id}")
    state = matches[0]
    current = _step_text(steps, state.get("current_step_id"))
    next_step = _step_text(steps, state.get("next_step_id"))
    common = [
        "[Answer-blind oracle procedural state]",
        f"Current step: {current}",
        f"Next step: {next_step}",
    ]
    if variant == "forced_oracle_step":
        return "\n".join(common)
    if variant != "forced_oracle_full":
        raise ValueError(f"Unsupported oracle U1 variant: {variant}")
    completion = state.get("completion_evidence")
    error = state.get("incompletion_or_error_evidence")
    if not isinstance(completion, list) or not isinstance(error, list):
        raise ValueError(f"Oracle evidence fields must be lists for {sample_id}")
    recovery = str(state.get("recovery_action") or "none")
    return "\n".join(
        [
            *common,
            f"Progress: {state.get('progress')}",
            "Observed completion evidence: "
            + ("; ".join(str(value) for value in completion) or "none"),
            "Observed incomplete/error evidence: "
            + ("; ".join(str(value) for value in error) or "none"),
            f"Recovery action: {recovery}",
        ]
    )


def validate_oracle_annotations(
    annotations: Sequence[dict[str, object]],
    samples: Sequence[dict[str, object]],
) -> dict[str, object]:
    allowed_progress = {
        "not_started",
        "ongoing",
        "complete",
        "deviated",
        "recovered",
    }
    forbidden_fragments = (
        "$interrupt$",
        "$silent$",
        "should interrupt",
        "should speak",
    )

    def inspect_text(value: object, path: str) -> None:
        if isinstance(value, str):
            lowered = value.lower()
            for forbidden in forbidden_fragments:
                if forbidden in lowered:
                    raise ValueError(
                        f"Forbidden target marker {forbidden!r} in oracle {path}"
                    )
        elif isinstance(value, dict):
            for key, nested in value.items():
                inspect_text(nested, f"{path}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                inspect_text(nested, f"{path}[{index}]")

    expected = {str(sample["sample_id"]): sample for sample in samples}
    seen: set[str] = set()
    inputs = {int(sample["input_index"]) for sample in samples}
    annotation_inputs: set[int] = set()
    for annotation_index, annotation in enumerate(annotations):
        inspect_text(annotation, f"annotations[{annotation_index}]")
        if annotation.get("status") != "complete":
            raise ValueError("Every supplied U1 oracle annotation must be complete")
        input_index = int(annotation["input_index"])
        if input_index not in inputs:
            raise ValueError(f"Oracle annotation is outside selected inputs: {input_index}")
        if input_index in annotation_inputs:
            raise ValueError(f"Duplicate oracle annotation input: {input_index}")
        annotation_inputs.add(input_index)
        steps = annotation.get("steps")
        states = annotation.get("sampled_chunk_states")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"Oracle annotation {input_index} has no static plan")
        if not isinstance(states, list) or not states:
            raise ValueError(f"Oracle annotation {input_index} has no states")
        step_ids = [str(step.get("id", "")) for step in steps]
        if any(not value for value in step_ids) or len(set(step_ids)) != len(step_ids):
            raise ValueError(f"Oracle annotation {input_index} has invalid step IDs")
        for state in states:
            sample_id = str(state.get("sample_id", ""))
            sample = expected.get(sample_id)
            if sample is None:
                raise ValueError(f"Unknown oracle sample ID: {sample_id}")
            if int(sample["input_index"]) != input_index:
                raise ValueError(f"Oracle sample/input mismatch: {sample_id}")
            if sample_id in seen:
                raise ValueError(f"Duplicate oracle sample state: {sample_id}")
            seen.add(sample_id)
            if int(state["chunk_index"]) != int(sample["chunk_index"]):
                raise ValueError(f"Oracle chunk mismatch: {sample_id}")
            observed = float(state["observed_through_sec"])
            expected_end = float(sample["observed_through_sec"])
            if abs(observed - expected_end) > 1e-6:
                raise ValueError(f"Oracle timestamp mismatch: {sample_id}")
            if state.get("progress") not in allowed_progress:
                raise ValueError(f"Oracle progress is invalid: {sample_id}")
            if str(state.get("current_step_id")) not in step_ids:
                raise ValueError(f"Oracle current step is invalid: {sample_id}")
            next_id = state.get("next_step_id")
            if next_id not in (None, "") and str(next_id) not in step_ids:
                raise ValueError(f"Oracle next step is invalid: {sample_id}")
            confidence = float(state["confidence"])
            if not 0 <= confidence <= 1:
                raise ValueError(f"Oracle confidence is invalid: {sample_id}")
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        extra = sorted(seen - set(expected))
        raise ValueError(f"Oracle sample coverage mismatch; missing={missing}, extra={extra}")
    return {
        "sessions": len(annotation_inputs),
        "sampled_states": len(seen),
        "causal_timestamps_exact": True,
        "forbidden_target_markers_absent": True,
    }


def validate_decision_invariance(
    reference: Sequence[dict[str, object]], candidate: Sequence[dict[str, object]]
) -> None:
    if len(reference) != len(candidate):
        raise ValueError("U1 prediction row counts differ")
    for row_index, (left, right) in enumerate(zip(reference, candidate)):
        if left.get("video_path") != right.get("video_path"):
            raise ValueError(f"U1 video order differs at row {row_index}")
        left_answers = left.get("answers")
        right_answers = right.get("answers")
        if not isinstance(left_answers, list) or not isinstance(right_answers, list):
            raise ValueError(f"U1 answers malformed at row {row_index}")
        if len(left_answers) != len(right_answers):
            raise ValueError(f"U1 answer counts differ at row {row_index}")
        for chunk_index, (left_answer, right_answer) in enumerate(
            zip(left_answers, right_answers)
        ):
            left_interrupt = str(left_answer).startswith(INTERRUPT_TAG)
            right_interrupt = str(right_answer).startswith(INTERRUPT_TAG)
            if left_interrupt != right_interrupt:
                raise ValueError(
                    f"U1 decision changed at {(row_index, chunk_index)}"
                )
