"""Validate and fingerprint frozen S1 train or held-out annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proactive_r0.artifacts import write_json
from proactive_state_s1.core import (
    load_json,
    load_jsonl,
    sha256_file,
    validate_collection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--split", choices=("train", "heldout"), required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    sessions_path = _resolve(args.sessions)
    annotations_path = _resolve(args.annotations)
    manifest_path = _resolve(args.manifest)
    annotations = load_json(annotations_path)
    if not isinstance(annotations, list) or not all(
        isinstance(row, dict) for row in annotations
    ):
        raise ValueError("S1 annotations must be a JSON list of objects")
    summary = validate_collection(
        annotations, load_jsonl(sessions_path), expected_split=args.split
    )
    manifest = {
        "schema_version": 1,
        "status": "complete and strictly validated",
        "split": args.split,
        **summary,
        "sources": {
            "sessions": {"path": str(sessions_path), "sha256": sha256_file(sessions_path)},
            "annotations": {
                "path": str(annotations_path),
                "sha256": sha256_file(annotations_path),
            },
            "protocol": {
                "path": str(PROJECT_ROOT / "annotations/state_s1_decoder_v1/PROTOCOL.md"),
                "sha256": sha256_file(
                    PROJECT_ROOT / "annotations/state_s1_decoder_v1/PROTOCOL.md"
                ),
            },
        },
        "targets": {
            "step": list(("s1", "s2", "s3", "s4")),
            "progress": list(
                ("not_started", "ongoing", "complete", "deviated", "recovered")
            ),
            "error_present": "bool(incompletion_or_error_evidence)",
        },
    }
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
