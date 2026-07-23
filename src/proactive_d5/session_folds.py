"""Validation and audit helpers for the frozen D4 session-level folds."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from proactive_d1.core import validate_fold_manifest


def validate_session_fold_manifest(
    manifest: Mapping[str, object], rows: Sequence[dict[str, object]]
) -> tuple[dict[int, int], dict[str, object]]:
    """Validate the D4 split contract and summarize its session/chunk balance."""
    fold_by_index = validate_fold_manifest(dict(manifest), rows)
    folds = int(manifest["folds"])
    fold_summary = {
        fold: {"sessions": 0, "chunks": 0, "domains": Counter()}
        for fold in range(folds)
    }
    for input_index, row in enumerate(rows):
        fold = fold_by_index[input_index]
        summary = fold_summary[fold]
        summary["sessions"] += 1
        summary["chunks"] += len(row["video_intervals"])  # type: ignore[arg-type]
        summary["domains"][str(row["domain"])] += 1
    return fold_by_index, {
        "algorithm": manifest["algorithm"],
        "seed": manifest["seed"],
        "labels_used_for_assignment": manifest["labels_used_for_assignment"],
        "folds": {
            str(fold): {
                "sessions": value["sessions"],
                "chunks": value["chunks"],
                "domains": dict(sorted(value["domains"].items())),  # type: ignore[union-attr]
            }
            for fold, value in fold_summary.items()
        },
    }
