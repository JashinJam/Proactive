"""Validate the repository's leaderboard experiment artifact contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from proactive_r0.artifacts import sha256_file


REQUIRED_ARTIFACTS = (
    "README.md",
    "config.json",
    "command.sh",
    "environment.txt",
    "code_state.txt",
    "data_manifest.json",
    "metrics.json",
    "predictions.jsonl",
    "run.log",
)


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _audit_predictions(path: Path) -> dict[str, object]:
    sessions = 0
    chunks = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Prediction line {line_number} is not an object")
            answers = row.get("answers")
            if not isinstance(row.get("video_path"), str) or not isinstance(
                answers, list
            ):
                raise ValueError(
                    f"Prediction line {line_number} must contain video_path/answers"
                )
            sessions += 1
            chunks += len(answers)
    if sessions == 0:
        raise ValueError("Prediction file is empty")
    return {
        "sessions": sessions,
        "chunks": chunks,
        "sha256": sha256_file(path),
    }


def audit_experiment_contract(experiment_dir: Path) -> dict[str, object]:
    """Return a structured audit; callers decide whether errors are fatal."""
    root = experiment_dir.resolve()
    errors: list[str] = []
    artifacts: dict[str, object] = {}
    if not root.is_dir():
        return {
            "status": "failed",
            "experiment_dir": str(root),
            "errors": ["experiment directory does not exist"],
        }

    for name in REQUIRED_ARTIFACTS:
        path = root / name
        if not path.is_file():
            errors.append(f"missing required artifact: {name}")
            continue
        size = path.stat().st_size
        if size == 0:
            errors.append(f"empty required artifact: {name}")
            continue
        artifacts[name] = {"bytes": size, "sha256": sha256_file(path)}

    for name in ("config.json", "data_manifest.json", "metrics.json"):
        path = root / name
        if path.is_file() and path.stat().st_size:
            try:
                _load_json_object(path)
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(f"invalid {name}: {exc}")

    predictions_path = root / "predictions.jsonl"
    prediction_audit: dict[str, object] | None = None
    if predictions_path.is_file() and predictions_path.stat().st_size:
        try:
            prediction_audit = _audit_predictions(predictions_path)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"invalid predictions.jsonl: {exc}")

    return {
        "status": "passed" if not errors else "failed",
        "experiment_dir": str(root),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "artifacts": artifacts,
        "predictions": prediction_audit,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args(argv)
    audit = audit_experiment_contract(args.experiment_dir)
    print(json.dumps(audit, indent=2, ensure_ascii=True))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
