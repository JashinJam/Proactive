"""Evaluate frozen S0 predictions after both target-isolated views complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl
from proactive_state_s0.core import (
    CANDIDATE_LABELS,
    STATE_TARGETS,
    grouped_composite,
    multiclass_metrics,
    paired_session_bootstrap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def evaluate_view(
    records: Sequence[dict[str, object]], targets: Sequence[dict[str, object]]
) -> dict[str, object]:
    by_id = {str(row["sample_id"]): row for row in records}
    target_by_id = {str(row["sample_id"]): row for row in targets}
    if by_id.keys() != target_by_id.keys() or len(by_id) != 80:
        raise ValueError("S0 predictions and targets do not cover the same 80 states")
    aligned: list[dict[str, object]] = []
    target_metrics: dict[str, object] = {}
    for target in STATE_TARGETS:
        gold = [str(target_by_id[sample_id][target]) for sample_id in sorted(by_id)]
        predicted = [
            str(by_id[sample_id]["predictions"][target]["label"])  # type: ignore[index]
            for sample_id in sorted(by_id)
        ]
        target_metrics[target] = multiclass_metrics(
            gold, predicted, CANDIDATE_LABELS[target]
        )
    for sample_id in sorted(by_id):
        record = by_id[sample_id]
        target = target_by_id[sample_id]
        correct = {
            name: int(
                str(record["predictions"][name]["label"]) == str(target[name])  # type: ignore[index]
            )
            for name in STATE_TARGETS
        }
        confidences = [
            float(record["predictions"][name]["confidence"])  # type: ignore[index]
            for name in STATE_TARGETS
        ]
        entropies = [
            float(record["predictions"][name]["entropy"])  # type: ignore[index]
            for name in STATE_TARGETS
        ]
        aligned.append(
            {
                "sample_id": sample_id,
                "input_index": int(target["input_index"]),
                "domain": target["domain"],
                "position_bin": target["position_bin"],
                "correct": correct,
                "composite_correctness": sum(correct.values()) / len(correct),
                "joint_step_progress_correct": int(correct["step"] and correct["progress"]),
                "mean_confidence": float(np.mean(confidences)),
                "mean_entropy": float(np.mean(entropies)),
                "step_absolute_ordinal_error": abs(
                    int(str(record["predictions"]["step"]["label"])[1:])  # type: ignore[index]
                    - int(target["step_ordinal"])
                ),
            }
        )
    mean_macro = float(
        np.mean([float(target_metrics[name]["macro_f1"]) for name in STATE_TARGETS])  # type: ignore[index]
    )
    if mean_macro >= 0.45:
        band = "strong_zero_shot_signal"
    elif mean_macro >= 0.35:
        band = "weak_but_usable_signal"
    else:
        band = "insufficient_zero_shot_signal"
    correct_rows = [row for row in aligned if float(row["composite_correctness"]) == 1.0]
    wrong_rows = [row for row in aligned if float(row["composite_correctness"]) < 1.0]
    return {
        "view": records[0]["view"],
        "states": len(aligned),
        "sessions": len({int(row["input_index"]) for row in aligned}),
        "targets": target_metrics,
        "step_ordinal_mae": float(
            np.mean([float(row["step_absolute_ordinal_error"]) for row in aligned])
        ),
        "joint_step_progress_accuracy": float(
            np.mean([float(row["joint_step_progress_correct"]) for row in aligned])
        ),
        "mean_composite_correctness": float(
            np.mean([float(row["composite_correctness"]) for row in aligned])
        ),
        "mean_task_macro_f1": mean_macro,
        "interpretation_band": band,
        "by_domain": grouped_composite(aligned, "domain"),
        "by_position": grouped_composite(aligned, "position_bin"),
        "confidence_diagnostics": {
            "mean_confidence": float(np.mean([float(row["mean_confidence"]) for row in aligned])),
            "mean_entropy": float(np.mean([float(row["mean_entropy"]) for row in aligned])),
            "fully_correct_states": len(correct_rows),
            "mean_confidence_fully_correct": (
                float(np.mean([float(row["mean_confidence"]) for row in correct_rows]))
                if correct_rows
                else None
            ),
            "mean_confidence_not_fully_correct": (
                float(np.mean([float(row["mean_confidence"]) for row in wrong_rows]))
                if wrong_rows
                else None
            ),
        },
        "aligned_rows": aligned,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--official-dir", required=True)
    parser.add_argument("--no-assistant-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    target_config = dict(config["targets"])  # type: ignore[arg-type]
    target_path = _resolve(target_config["path"])
    if sha256_file(target_path) != target_config["sha256"]:
        raise ValueError("S0 target fingerprint changed")
    targets = load_jsonl(target_path)
    official_dir = _resolve(args.official_dir)
    no_assistant_dir = _resolve(args.no_assistant_dir)
    official_path = official_dir / "state_predictions.jsonl"
    no_assistant_path = no_assistant_dir / "state_predictions.jsonl"
    official_records = load_jsonl(official_path)
    no_assistant_records = load_jsonl(no_assistant_path)
    required_calibration = str(config.get("required_calibration_mode", "none"))
    for name, records in (
        ("official_dialog", official_records),
        ("no_assistant_history", no_assistant_records),
    ):
        if not records or {str(row.get("view")) for row in records} != {name}:
            raise ValueError(f"S0 prediction view mismatch for {name}")
        if {str(row.get("calibration_mode", "none")) for row in records} != {
            required_calibration
        }:
            raise ValueError(f"S0 calibration mode mismatch for {name}")
    official = evaluate_view(official_records, targets)
    no_assistant = evaluate_view(no_assistant_records, targets)
    bootstrap_config = dict(config["bootstrap"])  # type: ignore[arg-type]
    comparison = paired_session_bootstrap(
        official["aligned_rows"],  # type: ignore[arg-type]
        no_assistant["aligned_rows"],  # type: ignore[arg-type]
        int(bootstrap_config["repetitions"]),
        int(bootstrap_config["seed"]),
    )
    official.pop("aligned_rows")
    no_assistant.pop("aligned_rows")
    result = {
        "schema_version": 1,
        "status": "complete S0 evaluation after frozen predictions",
        "classification": (
            "oracle-plan zero-shot dynamic-state feasibility; state-label-aware "
            "protocol, not hidden-test or submission-promotion evidence"
        ),
        "views": {
            "official_dialog": official,
            "no_assistant_history": no_assistant,
        },
        "paired_official_minus_no_assistant": comparison,
        "artifacts": {
            "evaluation_config_sha256": sha256_file(config_path),
            "targets_sha256": sha256_file(target_path),
            "official_predictions_sha256": sha256_file(official_path),
            "no_assistant_predictions_sha256": sha256_file(no_assistant_path),
        },
    }
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
