"""Freeze the label-independent five-fold S1 train-session split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from proactive_r0.artifacts import sha256_file, write_json
from proactive_state_s1.core import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = "20260718-state-s1-cv-v1"
FOLDS = 5


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def create_manifest(sessions_path: Path) -> dict[str, object]:
    sessions = [
        row for row in load_jsonl(sessions_path) if row.get("state_split") == "train"
    ]
    by_domain: dict[str, list[dict[str, object]]] = {}
    for row in sessions:
        by_domain.setdefault(str(row["domain"]), []).append(row)
    assignments: dict[int, int] = {}
    for domain, rows in sorted(by_domain.items()):
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{SEED}\0{domain}\0{row['video_path']}".encode("utf-8")
            ).hexdigest(),
        )
        for rank, row in enumerate(ranked):
            assignments[int(row["input_index"])] = rank % FOLDS
    return {
        "schema_version": 1,
        "status": "frozen label-independent S1 train-session CV split",
        "seed": SEED,
        "folds": FOLDS,
        "algorithm": "domain_stratified_sha256_rank_modulo_folds",
        "labels_answers_model_outputs_errors_ratings_read": False,
        "heldout_annotations_read": False,
        "sessions_source": {
            "path": str(sessions_path),
            "sha256": sha256_file(sessions_path),
        },
        "model_protocol": {
            "path": str(
                PROJECT_ROOT / "annotations/state_s1_decoder_v1/MODEL_PROTOCOL_v1.md"
            ),
            "sha256": sha256_file(
                PROJECT_ROOT / "annotations/state_s1_decoder_v1/MODEL_PROTOCOL_v1.md"
            ),
        },
        "sessions": [
            {
                "input_index": int(row["input_index"]),
                "video_path": row["video_path"],
                "domain": row["domain"],
                "fold": assignments[int(row["input_index"])],
                "chunks": len(row["chunks"]),
            }
            for row in sessions
        ],
        "fold_summary": {
            str(fold): {
                "sessions": sum(value == fold for value in assignments.values()),
                "by_domain": {
                    domain: sum(
                        assignments[int(row["input_index"])] == fold
                        for row in rows
                    )
                    for domain, rows in sorted(by_domain.items())
                },
            }
            for fold in range(FOLDS)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite S1 CV split: {output}")
    manifest = create_manifest(_resolve(args.sessions))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
