"""Render exactly one explicit S1 interval after static plans are frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proactive_state_s1.core import load_json, load_jsonl, load_optional_jsonl, sha256_file
from proactive_u1.contact_sheet import make_contact_sheet


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--plan-manifest", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--input-index", type=int, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    sessions_path = _resolve(args.sessions)
    manifest_path = _resolve(args.plan_manifest)
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or not str(manifest.get("status", "")).startswith(
        "query-only plans frozen"
    ):
        raise ValueError("A complete S1 plan-freeze manifest is required")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not isinstance(sources.get("sessions"), dict):
        raise ValueError("Malformed S1 plan-freeze manifest")
    if sources["sessions"].get("sha256") != sha256_file(sessions_path):
        raise ValueError("S1 sessions changed after plan freeze")
    selected = [
        row for row in load_jsonl(sessions_path)
        if int(row["input_index"]) == args.input_index
    ]
    if len(selected) != 1:
        raise ValueError(f"Unknown or duplicate S1 input_index {args.input_index}")
    row = selected[0]
    chunks = row.get("chunks")
    if not isinstance(chunks, list) or not 0 <= args.chunk_index < len(chunks):
        raise ValueError("S1 chunk_index is outside the selected session")
    records_path = _resolve(args.records)
    existing = [
        value for value in load_optional_jsonl(records_path)
        if int(value.get("input_index", -1)) == args.input_index
    ]
    if args.chunk_index != len(existing):
        raise ValueError(
            f"input {args.input_index}: only next unrecorded chunk {len(existing)} may be rendered"
        )
    chunk = chunks[args.chunk_index]
    if not isinstance(chunk, dict) or chunk.get("chunk_index") != args.chunk_index:
        raise ValueError("Malformed or non-contiguous S1 chunk")
    interval = chunk.get("interval")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("Malformed S1 interval")
    video_path = _resolve(args.video_dir) / str(row["video_path"])
    output_path = _resolve(args.output)
    make_contact_sheet(
        video_path,
        output_path,
        float(interval[0]),
        float(interval[1]),
        args.frames,
        4,
        320,
    )
    result = {
        "input_index": args.input_index,
        "chunk_index": args.chunk_index,
        "interval": interval,
        "query": row["query"],
        "task": row["task"],
        "prior_dialog": chunk["prior_dialog"],
        "output": str(output_path),
        "sha256": sha256_file(output_path),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
