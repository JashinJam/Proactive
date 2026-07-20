"""Select and sanitize the label-independent S1 annotation sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = "20260717-state-s1-v1"
DOMAINS = ("Arts and Crafts", "Chef", "Handyman", "Tutorial")
U1_FORMAL_INDICES = {
    19, 39, 56, 66, 117, 135, 192, 196, 244, 263,
    293, 342, 369, 434, 496, 546, 580, 597, 672, 673,
}
R1_PILOT_INDICES = {14, 123, 326, 687}
HELDOUT_BANDS = {
    "Arts and Crafts": ("short", "middle"),
    "Chef": ("middle", "long"),
    "Handyman": ("short", "long"),
    "Tutorial": ("short", "middle"),
}


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _rank_key(domain: str, row: dict[str, object]) -> str:
    payload = f"{SEED}\0{domain}\0{row['video_path']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    excluded = U1_FORMAL_INDICES | R1_PILOT_INDICES
    for domain in DOMAINS:
        candidates = [
            {**row, "input_index": index, "chunks": len(row["video_intervals"])}
            for index, row in enumerate(rows)
            if row.get("domain") == domain and index not in excluded
        ]
        ordered = sorted(
            candidates,
            key=lambda row: (int(row["chunks"]), str(row["video_path"])),
        )
        bands = np.array_split(np.asarray(ordered, dtype=object), 3)
        counts = (3, 3, 2)
        domain_selection: list[dict[str, object]] = []
        for band_name, band, count in zip(("short", "middle", "long"), bands, counts):
            choices = sorted(
                band.tolist(), key=lambda row: _rank_key(domain, row)
            )[:count]
            for row in choices:
                row["length_band"] = band_name
            domain_selection.extend(choices)
        held_out: set[int] = set()
        for band_name in HELDOUT_BANDS[domain]:
            choices = [row for row in domain_selection if row["length_band"] == band_name]
            held_out.add(
                int(min(choices, key=lambda row: _rank_key(domain + "-heldout", row))["input_index"])
            )
        for row in domain_selection:
            row["state_split"] = (
                "heldout" if int(row["input_index"]) in held_out else "train"
            )
            selected.append(row)
    return sorted(selected, key=lambda row: int(row["input_index"]))


def _sanitize(row: dict[str, object]) -> dict[str, object]:
    intervals = row["video_intervals"]
    dialogs = row["dialog"]
    chunks = []
    for chunk_index, interval in enumerate(intervals):
        chunks.append(
            {
                "chunk_index": chunk_index,
                "interval": interval,
                "observed_through_sec": float(interval[1]),
                "video_intervals_so_far": intervals[: chunk_index + 1],
                "prior_dialog": dialogs[chunk_index],
            }
        )
    return {
        "input_index": row["input_index"],
        "video_path": row["video_path"],
        "query": row["query"],
        "task": row["task"],
        "domain": row["domain"],
        "length_band": row["length_band"],
        "state_split": row["state_split"],
        "chunks": chunks,
    }


def _template(row: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "incomplete",
        "input_index": row["input_index"],
        "video_path": row["video_path"],
        "query": row["query"],
        "task": row["task"],
        "domain": row["domain"],
        "length_band": row["length_band"],
        "state_split": row["state_split"],
        "provenance": {
            "plan_inputs": ["task", "query"],
            "chunk_inputs": [
                "task", "query", "dialog_at_chunk", "video_intervals_so_far"
            ],
            "excluded_inputs": [
                "answers", "future_dialog", "future_video", "model_outputs",
                "R0/D1/D3 errors", "ratings", "existing_oracle_states"
            ],
            "annotation_type": "s1_training_or_heldout_causal_state_supervision"
        },
        "goal": "",
        "steps": [
            {
                "id": f"s{index}",
                "text": "",
                "completion_cues": [],
                "incompletion_cues": [],
            }
            for index in range(1, 5)
        ],
        "chunk_states": [
            {
                "chunk_index": chunk["chunk_index"],
                "observed_through_sec": chunk["observed_through_sec"],
                "current_step_id": "",
                "progress": "",
                "completion_evidence": [],
                "incompletion_or_error_evidence": [],
                "next_step_id": "",
                "recovery_action": "",
                "confidence": None,
            }
            for chunk in row["chunks"]  # type: ignore[union-attr]
        ],
    }


def prepare(source_path: Path, protocol_path: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"S1 prepared directory already exists: {output_dir}")
    rows = load_jsonl(source_path)
    if len(rows) != 700:
        raise ValueError("S1 selection requires the frozen 700-session source")
    label_free = [{key: value for key, value in row.items() if key != "answers"} for row in rows]
    selected = _select(label_free)
    if len(selected) != 32:
        raise ValueError("S1 selection did not produce 32 sessions")
    sanitized = [_sanitize(row) for row in selected]
    templates = [_template(row) for row in sanitized]
    output_dir.mkdir(parents=True)
    sessions_path = output_dir / "sessions.jsonl"
    template_path = output_dir / "annotation_template.json"
    write_jsonl(sessions_path, sanitized)
    write_json(template_path, templates)
    by_split = {
        split: [row for row in sanitized if row["state_split"] == split]
        for split in ("train", "heldout")
    }
    manifest = {
        "schema_version": 1,
        "status": "complete label-independent S1 annotation preparation",
        "selection_seed": SEED,
        "labels_or_answers_read": False,
        "model_outputs_or_errors_read": False,
        "ratings_read": False,
        "excluded_u1_formal_indices": sorted(U1_FORMAL_INDICES),
        "excluded_r1_pilot_indices": sorted(R1_PILOT_INDICES),
        "sessions": len(sanitized),
        "states": sum(len(row["chunks"]) for row in sanitized),
        "splits": {
            split: {
                "sessions": len(values),
                "states": sum(len(row["chunks"]) for row in values),
                "by_domain": {
                    domain: sum(row["domain"] == domain for row in values)
                    for domain in DOMAINS
                },
                "by_length_band": {
                    band: sum(row["length_band"] == band for row in values)
                    for band in ("short", "middle", "long")
                },
            }
            for split, values in by_split.items()
        },
        "selected_sessions": [
            {
                "input_index": row["input_index"],
                "video_path": row["video_path"],
                "domain": row["domain"],
                "length_band": row["length_band"],
                "state_split": row["state_split"],
                "chunks": len(row["chunks"]),
            }
            for row in sanitized
        ],
        "sources": {
            "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        },
        "artifacts": {
            "sessions": {"path": str(sessions_path), "sha256": sha256_file(sessions_path)},
            "template": {"path": str(template_path), "sha256": sha256_file(template_path)},
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = prepare(
        _resolve(args.source), _resolve(args.protocol), _resolve(args.output_dir)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
