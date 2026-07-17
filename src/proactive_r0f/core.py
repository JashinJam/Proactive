"""Response-intent repair for non-empty malformed proactive generations."""

from __future__ import annotations

from proactive_r0.core import (
    EMPTY_INTERRUPT_UTTERANCE,
    INTERRUPT_TAG,
    SILENT_TAG,
)


def repair_response_intent(raw_response: object) -> tuple[str, str | None]:
    """Canonicalize explicit tags and treat other non-empty speech as interrupt.

    This rule does not inspect labels. Its selection after analysis of the public
    validation errors still makes the resulting experiment val-supervised.
    """
    stripped = str(raw_response).lstrip()
    if stripped.startswith(INTERRUPT_TAG):
        utterance = stripped[len(INTERRUPT_TAG) :].strip()
        if utterance:
            return f"{INTERRUPT_TAG}{utterance}", None
        return (
            f"{INTERRUPT_TAG}{EMPTY_INTERRUPT_UTTERANCE}",
            "empty_interrupt_utterance",
        )
    if stripped.startswith(SILENT_TAG):
        reason = None if stripped == SILENT_TAG else "trimmed_silent_suffix"
        return SILENT_TAG, reason
    if not stripped:
        return SILENT_TAG, "empty_raw_response_kept_silent"
    return (
        f"{INTERRUPT_TAG}{stripped}",
        "malformed_nonempty_repaired_as_interrupt",
    )

