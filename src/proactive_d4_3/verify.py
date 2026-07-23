"""Verify online history8 inference against the frozen D4.1/D4.2 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from proactive_d3.dialog_control_core import DIALOG_POLICY_NAMES
from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _checked_path(spec: Mapping[str, object]) -> Path:
    path = _resolve(spec["path"])
    actual = sha256_file(path)
    if actual != str(spec["sha256"]):
        raise ValueError(f"D4.3 reference hash mismatch: {path}: {actual}")
    return path


def verify(deployment_dir: Path, config_path: Path) -> dict[str, object]:
    config = _load_object(config_path)
    smoke = dict(config["smoke"])  # type: ignore[arg-type]
    tolerance = float(smoke["numeric_abs_tolerance"])
    expected_indices = [int(value) for value in smoke["session_indices"]]  # type: ignore[arg-type]
    generation_path = _checked_path(smoke["reference_generation_records"])  # type: ignore[arg-type]
    cache_path = _checked_path(smoke["reference_neural_cache"])  # type: ignore[arg-type]
    final_records_path = _checked_path(smoke["reference_final_records"])  # type: ignore[arg-type]
    final_predictions_path = _checked_path(smoke["reference_final_predictions"])  # type: ignore[arg-type]

    deployment = load_jsonl(deployment_dir / "session_records.jsonl")
    if [int(row["input_index"]) for row in deployment] != expected_indices:
        raise ValueError("D4.3 deployment session indices differ from the protocol")
    generation = {
        int(row["input_index"]): row for row in load_jsonl(generation_path)
    }
    final_records = {
        (int(row["input_index"]), int(row["chunk_index"])): row
        for row in load_jsonl(final_records_path)
    }
    final_predictions = load_jsonl(final_predictions_path)
    with np.load(cache_path, allow_pickle=False) as arrays:
        cache_rows = {
            (int(input_index), int(chunk_index)): row_index
            for row_index, (input_index, chunk_index) in enumerate(
                zip(arrays["input_index"], arrays["chunk_index"])
            )
        }
        hidden = arrays["hidden_state"].astype(np.float32, copy=False)
        tag_margin = arrays["tag_margin"].astype(np.float32, copy=False)
        silent_logp = arrays["silent_log_probability"].astype(np.float32, copy=False)
        interrupt_logp = arrays["interrupt_log_probability"].astype(
            np.float32, copy=False
        )
        prompt_tokens = arrays["prompt_tokens"].astype(np.int32, copy=False)

        exact = {
            "raw_response": 0,
            "prompt_tokens": 0,
            "model_input_frames": 0,
            "dialog_features": 0,
            "decision": 0,
            "answer": 0,
        }
        within_tolerance = {
            "tag_margin": 0,
            "silent_log_probability": 0,
            "interrupt_log_probability": 0,
            "hidden_state": 0,
            "logit": 0,
        }
        maxima = {name: 0.0 for name in within_tolerance}
        chunks = 0
        session_seconds: list[float] = []
        for session in deployment:
            input_index = int(session["input_index"])
            reference_session = generation[input_index]
            online_chunks = session["chunks"]
            reference_chunks = reference_session["chunks"]
            if len(online_chunks) != len(reference_chunks):
                raise ValueError(f"D4.3 chunk count mismatch for session {input_index}")
            session_seconds.append(float(session["session_wall_time_seconds"]))
            for chunk_index, (online, reference) in enumerate(
                zip(online_chunks, reference_chunks)
            ):
                key = (input_index, chunk_index)
                cache_index = cache_rows[key]
                final = final_records[key]
                exact["raw_response"] += int(
                    online["raw_response"] == reference["raw_response"]
                )
                exact["prompt_tokens"] += int(
                    int(online["prompt_tokens"]) == int(prompt_tokens[cache_index])
                )
                exact["model_input_frames"] += int(
                    int(online["model_input_frames"])
                    == int(reference["model_input_frames"])
                )
                dialog_difference = max(
                    abs(float(online[name]) - float(reference[name]))
                    for name in DIALOG_POLICY_NAMES
                )
                exact["dialog_features"] += int(dialog_difference == 0.0)
                exact["decision"] += int(
                    int(online["decision_interrupt"])
                    == int(final["predicted_interrupt"])
                )
                exact["answer"] += int(
                    online["answer"]
                    == final_predictions[input_index]["answers"][chunk_index]
                )
                online_hidden = np.asarray(online["hidden_state"], dtype=np.float32)
                differences = {
                    "tag_margin": abs(
                        float(online["tag_margin"]) - float(tag_margin[cache_index])
                    ),
                    "silent_log_probability": abs(
                        float(online["silent_log_probability"])
                        - float(silent_logp[cache_index])
                    ),
                    "interrupt_log_probability": abs(
                        float(online["interrupt_log_probability"])
                        - float(interrupt_logp[cache_index])
                    ),
                    "hidden_state": float(
                        np.max(np.abs(online_hidden - hidden[cache_index]))
                    ),
                    "logit": abs(
                        float(online["decision_logit"]) - float(final["logit"])
                    ),
                }
                for name, difference in differences.items():
                    maxima[name] = max(maxima[name], difference)
                    within_tolerance[name] += int(difference <= tolerance)
                chunks += 1

    runtime = _load_object(deployment_dir / "runtime.json")
    expected_chunks = int(smoke["expected_chunks"])
    expected_sessions = int(smoke["expected_sessions"])
    max_session_seconds = max(session_seconds, default=float("inf"))
    passed = (
        len(deployment) == expected_sessions
        and chunks == expected_chunks
        and all(value == chunks for value in exact.values())
        and all(value == chunks for value in within_tolerance.values())
        and max_session_seconds <= float(smoke["max_session_seconds"])
        and int(runtime["total_parameters"]) <= 2_000_000_000
    )
    return {
        "status": "pass" if passed else "fail",
        "classification": "D4.3 GPU history8 equivalence smoke; not a performance estimate",
        "sessions": len(deployment),
        "chunks": chunks,
        "exact_match_counts": exact,
        "within_tolerance_counts": within_tolerance,
        "max_abs_differences": maxima,
        "numeric_abs_tolerance": tolerance,
        "session_wall_time_seconds": session_seconds,
        "max_session_wall_time_seconds": max_session_seconds,
        "peak_gpu_memory_bytes": runtime.get("peak_gpu_memory_bytes"),
        "total_parameters": runtime["total_parameters"],
        "artifacts": {
            "config_sha256": sha256_file(config_path),
            "deployment_records_sha256": sha256_file(
                deployment_dir / "session_records.jsonl"
            ),
            "deployment_predictions_sha256": sha256_file(
                deployment_dir / "predictions.jsonl"
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    deployment_dir = _resolve(args.deployment_dir)
    result = verify(deployment_dir, _resolve(args.config))
    write_json(deployment_dir / "equivalence_audit.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
