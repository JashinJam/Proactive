"""Frozen D6 folds, labels, and causal frame provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from proactive_d1.core import strip_answers, validate_fold_manifest


@dataclass(frozen=True)
class SelectedFrames:
    frames: tuple[object, ...]
    current_interval_mask: tuple[bool, ...]
    source_indices: tuple[tuple[int, int], ...]


def select_uniform_causal_frames(
    interval_frames: Sequence[Sequence[object]], max_frames: int
) -> SelectedFrames:
    """Match D4 uniform cumulative sampling while retaining interval identity."""
    if not interval_frames or max_frames <= 0:
        raise ValueError("D6 frame groups and max_frames must be non-empty")
    references = [
        (interval_index, frame_index, frame)
        for interval_index, frames in enumerate(interval_frames)
        for frame_index, frame in enumerate(frames)
    ]
    if not references:
        raise ValueError("D6 observed intervals contain no frames")
    if len(references) > max_frames:
        stride = len(references) / max_frames
        references = [references[int(index * stride)] for index in range(max_frames)]
    current_index = len(interval_frames) - 1
    mask = tuple(reference[0] == current_index for reference in references)
    if not any(mask):
        raise ValueError("D6 uniform frame cap removed every current-interval frame")
    return SelectedFrames(
        frames=tuple(reference[2] for reference in references),
        current_interval_mask=mask,
        source_indices=tuple((reference[0], reference[1]) for reference in references),
    )


def sanitize_model_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    sanitized = strip_answers(rows)
    if any("answers" in row for row in sanitized):
        raise RuntimeError("D6 answer stripping failed")
    return sanitized


def gold_labels(rows: Sequence[Mapping[str, object]]) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for input_index, row in enumerate(rows):
        answers = row.get("answers")
        intervals = row.get("video_intervals")
        if not isinstance(answers, list) or not isinstance(intervals, list):
            raise ValueError("D6 labels require aligned answers and intervals")
        if len(answers) != len(intervals):
            raise ValueError("D6 labels and intervals differ")
        for chunk_index, answer in enumerate(answers):
            result[(input_index, chunk_index)] = int(
                str(answer).lstrip().startswith("$interrupt$")
            )
    return result


def rotation_indices(
    manifest: Mapping[str, object],
    answer_free_rows: Sequence[dict[str, object]],
    test_fold: int,
) -> dict[str, object]:
    fold_by_index = validate_fold_manifest(dict(manifest), answer_free_rows)
    folds = int(manifest["folds"])
    if not 0 <= test_fold < folds:
        raise ValueError("D6 test fold is outside the manifest")
    calibration_fold = (test_fold + 1) % folds
    fit_folds = sorted(set(range(folds)) - {test_fold, calibration_fold})
    result = {
        "test_fold": test_fold,
        "calibration_fold": calibration_fold,
        "fit_folds": fit_folds,
        "fit": [
            index for index in range(len(answer_free_rows)) if fold_by_index[index] in fit_folds
        ],
        "calibration": [
            index
            for index in range(len(answer_free_rows))
            if fold_by_index[index] == calibration_fold
        ],
        "test": [
            index
            for index in range(len(answer_free_rows))
            if fold_by_index[index] == test_fold
        ],
        "fold_by_index": fold_by_index,
    }
    assigned = set(result["fit"]) | set(result["calibration"]) | set(result["test"])
    if assigned != set(range(len(answer_free_rows))):
        raise ValueError("D6 rotation does not cover every session")
    return result


def labels_for_sessions(
    labels: Mapping[tuple[int, int], int], session_indices: Sequence[int]
) -> list[int]:
    selected = set(session_indices)
    return [
        label
        for (input_index, _), label in sorted(labels.items())
        if input_index in selected
    ]

