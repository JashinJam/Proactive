"""Create target-isolated S0 oracle-plan inputs and evaluation targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl, write_jsonl
from proactive_u1.core import validate_oracle_annotations


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def prepare(
    sample_paths: list[Path], oracle_path: Path, output_dir: Path
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"S0 prepared directory already exists: {output_dir}")
    samples = [row for path in sample_paths for row in load_jsonl(path)]
    samples.sort(key=lambda row: (int(row["input_index"]), int(row["chunk_index"])))
    if len(samples) != 80 or len({str(row["sample_id"]) for row in samples}) != 80:
        raise ValueError("S0 requires exactly 80 unique formal samples")
    annotations = json.loads(oracle_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, list):
        raise ValueError("S0 oracle annotations must be a JSON array")
    validation = validate_oracle_annotations(annotations, samples)
    by_input = {int(row["input_index"]): row for row in annotations}
    inputs: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    for sample in samples:
        input_index = int(sample["input_index"])
        annotation = by_input[input_index]
        states = annotation["sampled_chunk_states"]
        state = next(
            value
            for value in states
            if value["sample_id"] == sample["sample_id"]
        )
        steps = annotation["steps"]
        step_ids = [str(value["id"]) for value in steps]
        current_step = str(state["current_step_id"])
        inputs.append(
            {
                **sample,
                "goal": annotation["goal"],
                "steps": steps,
                "plan_provenance": {
                    "inputs": ["task", "query"],
                    "video_read": False,
                    "oracle_static_plan": True,
                },
            }
        )
        targets.append(
            {
                "sample_id": sample["sample_id"],
                "input_index": input_index,
                "chunk_index": sample["chunk_index"],
                "domain": sample["domain"],
                "position_bin": sample["position_bin"],
                "step": current_step,
                "step_ordinal": step_ids.index(current_step) + 1,
                "progress": state["progress"],
                "error": (
                    "present"
                    if state["incompletion_or_error_evidence"]
                    else "absent"
                ),
                "annotator_confidence": state["confidence"],
            }
        )
    output_dir.mkdir(parents=True)
    inputs_path = output_dir / "inputs.jsonl"
    targets_path = output_dir / "targets.jsonl"
    write_jsonl(inputs_path, inputs)
    write_jsonl(targets_path, targets)
    manifest = {
        "schema_version": 1,
        "status": "complete target-isolated S0 preparation",
        "sessions": len({int(row["input_index"]) for row in inputs}),
        "states": len(inputs),
        "prediction_runner_reads_targets": False,
        "inputs_contain_dynamic_state_targets": False,
        "inputs_contain_current_or_future_answers": False,
        "inputs_contain_oracle_static_plan": True,
        "validation": validation,
        "sources": {
            "samples": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in sample_paths
            ],
            "oracle": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)},
        },
        "artifacts": {
            "inputs": {"path": str(inputs_path), "sha256": sha256_file(inputs_path)},
            "targets": {"path": str(targets_path), "sha256": sha256_file(targets_path)},
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", action="append", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = prepare(
        [_resolve(value) for value in args.samples],
        _resolve(args.oracle),
        _resolve(args.output_dir),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

