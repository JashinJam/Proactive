"""Merge and validate isolated U1 formal-blind oracle annotation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl
from proactive_u1.core import validate_oracle_annotations


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ANNOTATION_TYPE = "formal_blind_evaluation_only_oracle_non_deployable"


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_annotations(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Oracle annotation shard must be a JSON array: {path}")
    return value


def finalize_oracle(
    sample_parts: Sequence[Path],
    annotation_parts: Sequence[Path],
    expected_sessions: int = 20,
    expected_states: int = 80,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(sample_parts) != len(annotation_parts) or not sample_parts:
        raise ValueError("Sample and annotation shard counts must be equal and non-zero")
    samples = [row for path in sample_parts for row in load_jsonl(path)]
    annotations = [row for path in annotation_parts for row in _load_annotations(path)]
    sample_ids = [str(row["sample_id"]) for row in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Formal-blind sample shards overlap")
    input_ids = [int(row["input_index"]) for row in annotations]
    if len(input_ids) != len(set(input_ids)):
        raise ValueError("Formal-blind annotation shards overlap")
    if len(set(input_ids)) != expected_sessions or len(samples) != expected_states:
        raise ValueError(
            "Formal-blind coverage differs from the frozen protocol: "
            f"sessions={len(set(input_ids))}, states={len(samples)}"
        )
    for annotation in annotations:
        provenance = annotation.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"Formal annotation lacks provenance: {annotation.get('input_index')}")
        if provenance.get("annotation_type") != FORMAL_ANNOTATION_TYPE:
            raise ValueError(
                f"Formal annotation has wrong provenance: {annotation.get('input_index')}"
            )
        excluded = {str(value) for value in provenance.get("excluded_inputs", [])}
        required_exclusions = {
            "answers",
            "future_dialog",
            "future_video",
            "model_outputs",
            "R0/D1 errors",
        }
        if not required_exclusions.issubset(excluded):
            raise ValueError(
                f"Formal annotation exclusions are incomplete: {annotation.get('input_index')}"
            )
    validation = validate_oracle_annotations(annotations, samples)
    ordered = sorted(annotations, key=lambda row: int(row["input_index"]))
    return ordered, {
        **validation,
        "formal_blind_provenance": True,
        "expected_sessions": expected_sessions,
        "expected_states": expected_states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-part", action="append", required=True)
    parser.add_argument("--annotation-part", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    sample_paths = [_resolve(value) for value in args.sample_part]
    annotation_paths = [_resolve(value) for value in args.annotation_part]
    rows, validation = finalize_oracle(sample_paths, annotation_paths)
    output_path = _resolve(args.output)
    manifest_path = _resolve(args.manifest)
    write_json(output_path, rows)
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "annotation_protocol": FORMAL_ANNOTATION_TYPE,
            "sample_parts": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in sample_paths
            ],
            "annotation_parts": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in annotation_paths
            ],
            "merged_oracle_states": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
            },
            "validation": validation,
        },
    )
    print(json.dumps({"output": str(output_path), "validation": validation}, sort_keys=True))


if __name__ == "__main__":
    main()
