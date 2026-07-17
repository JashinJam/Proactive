"""Compare a D1 deployment optimization against its sequential reference."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_PATTERN = re.compile(
    r"Session (\d+)/(\d+) complete: chunks=(\d+) elapsed=([0-9.]+)s"
)


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _session_times(path: Path) -> list[dict[str, float | int]]:
    values = [
        {
            "position": int(match.group(1)) - 1,
            "sessions": int(match.group(2)),
            "chunks": int(match.group(3)),
            "elapsed_seconds": float(match.group(4)),
        }
        for match in SESSION_PATTERN.finditer(path.read_text(encoding="utf-8"))
    ]
    if not values or [value["position"] for value in values] != list(
        range(len(values))
    ):
        raise ValueError(f"Incomplete or misordered session timing log: {path}")
    return values


def compare(reference_dir: Path, candidate_dir: Path) -> dict[str, object]:
    reference_runtime = _load_json(reference_dir / "runtime.json")
    candidate_runtime = _load_json(candidate_dir / "runtime.json")
    candidate_config = _load_json(candidate_dir / "config.json")
    candidate_audit = _load_json(candidate_dir / "consistency_audit.json")
    reference_records = load_jsonl(reference_dir / "session_records.jsonl")
    candidate_records = load_jsonl(candidate_dir / "session_records.jsonl")
    if len(reference_records) != len(candidate_records):
        raise ValueError("D1 deployment mode session counts differ")
    reference_times = _session_times(reference_dir / "run.log")
    candidate_times = _session_times(candidate_dir / "run.log")
    if len(reference_times) != len(candidate_times) or len(reference_times) != len(
        reference_records
    ):
        raise ValueError("D1 deployment mode timing/session coverage differs")

    per_session: list[dict[str, object]] = []
    for reference_record, candidate_record, reference_time, candidate_time in zip(
        reference_records, candidate_records, reference_times, candidate_times
    ):
        identity = (
            reference_record.get("input_index"),
            reference_record.get("video_path"),
        )
        if identity != (
            candidate_record.get("input_index"),
            candidate_record.get("video_path"),
        ):
            raise ValueError("D1 deployment mode session identities differ")
        if reference_time["chunks"] != candidate_time["chunks"]:
            raise ValueError("D1 deployment mode session chunk counts differ")
        reference_elapsed = float(reference_time["elapsed_seconds"])
        candidate_elapsed = float(candidate_time["elapsed_seconds"])
        per_session.append(
            {
                "input_index": identity[0],
                "video_path": identity[1],
                "chunks": reference_time["chunks"],
                "reference_elapsed_seconds": reference_elapsed,
                "candidate_elapsed_seconds": candidate_elapsed,
                "improvement_fraction": 1.0 - candidate_elapsed / reference_elapsed,
            }
        )

    reference_compute = sum(
        float(value["reference_elapsed_seconds"]) for value in per_session
    )
    candidate_compute = sum(
        float(value["candidate_elapsed_seconds"]) for value in per_session
    )
    reference_wall = float(reference_runtime["wall_time_seconds"])
    candidate_wall = float(candidate_runtime["wall_time_seconds"])
    reference_peak = int(reference_runtime["peak_gpu_memory_bytes"])
    candidate_peak = int(candidate_runtime["peak_gpu_memory_bytes"])
    compute_improvement = 1.0 - candidate_compute / reference_compute
    wall_improvement = 1.0 - candidate_wall / reference_wall
    memory_increase = candidate_peak / reference_peak - 1.0

    gate = candidate_config.get("equivalence_gate")
    if not isinstance(gate, dict):
        raise ValueError("Candidate deployment config has no equivalence gate")
    minimum_improvement = float(
        gate.get(
            "session_compute_time_improvement_fraction_min",
            0.0 if gate.get("latency_must_improve") else -1.0e9,
        )
    )
    maximum_memory_increase = float(
        gate.get("peak_memory_increase_fraction_max", 1.0e9)
    )
    prediction_exact = (
        reference_dir.joinpath("predictions.jsonl").read_bytes()
        == candidate_dir.joinpath("predictions.jsonl").read_bytes()
    )
    metrics_exact = (
        reference_dir.joinpath("metrics.json").read_bytes()
        == candidate_dir.joinpath("metrics.json").read_bytes()
    )
    equivalence_passed = (
        candidate_audit.get("status") == "pass"
        and int(candidate_audit["decision_exact_matches"])
        == int(candidate_audit["chunks"])
        and int(candidate_audit["answer_exact_matches"])
        == int(candidate_audit["chunks"])
        and prediction_exact
        and metrics_exact
    )
    latency_passed = compute_improvement >= minimum_improvement
    memory_passed = memory_increase <= maximum_memory_increase
    promotion_passed = equivalence_passed and latency_passed and memory_passed
    return {
        "status": "promote" if promotion_passed else "reject",
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "candidate_mode": candidate_runtime.get("decision_feature_mode"),
        "sessions": len(per_session),
        "chunks": sum(int(value["chunks"]) for value in per_session),
        "equivalence": {
            "passed": equivalence_passed,
            "predictions_byte_exact": prediction_exact,
            "metrics_byte_exact": metrics_exact,
            "consistency_audit": candidate_audit,
        },
        "latency": {
            "reference_compute_seconds": reference_compute,
            "candidate_compute_seconds": candidate_compute,
            "compute_improvement_fraction": compute_improvement,
            "minimum_improvement_fraction": minimum_improvement,
            "passed": latency_passed,
            "reference_wall_seconds": reference_wall,
            "candidate_wall_seconds": candidate_wall,
            "wall_improvement_fraction": wall_improvement,
            "candidate_faster_sessions": sum(
                float(value["improvement_fraction"]) > 0 for value in per_session
            ),
        },
        "memory": {
            "reference_peak_bytes": reference_peak,
            "candidate_peak_bytes": candidate_peak,
            "increase_fraction": memory_increase,
            "maximum_increase_fraction": maximum_memory_increase,
            "passed": memory_passed,
        },
        "promotion_gate_passed": promotion_passed,
        "per_session": per_session,
        "artifacts": {
            "reference_predictions_sha256": sha256_file(
                reference_dir / "predictions.jsonl"
            ),
            "candidate_predictions_sha256": sha256_file(
                candidate_dir / "predictions.jsonl"
            ),
            "reference_metrics_sha256": sha256_file(reference_dir / "metrics.json"),
            "candidate_metrics_sha256": sha256_file(candidate_dir / "metrics.json"),
            "candidate_consistency_audit_sha256": sha256_file(
                candidate_dir / "consistency_audit.json"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    candidate_dir = _resolve(args.candidate_dir)
    result = compare(_resolve(args.reference_dir), candidate_dir)
    output_path = _resolve(args.output) if args.output else candidate_dir / "comparison.json"
    write_json(output_path, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
