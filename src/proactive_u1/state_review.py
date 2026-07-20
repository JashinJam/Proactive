"""Build the frozen three-way blind U1 oracle-state review package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl, write_jsonl
from proactive_u1.analyze import _write_ratings, build_blind_multivariant


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_VARIANTS = (
    "forced_no_state",
    "forced_oracle_step",
    "forced_oracle_full",
)
STATE_REVIEW_SEED = "20260717-u1-state-review-v1"


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--content-records", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", default=STATE_REVIEW_SEED)
    args = parser.parse_args()
    sample_path = _resolve(args.samples)
    content_paths = [_resolve(value) for value in args.content_records]
    samples = load_jsonl(sample_path)
    content = [row for path in content_paths for row in load_jsonl(path)]
    blind, key = build_blind_multivariant(
        samples, content, STATE_VARIANTS, args.seed
    )
    if len(samples) != 80 or len(blind) != 240 or len(key) != 240:
        raise ValueError(
            f"State review must contain 80 samples/240 candidates, got "
            f"{len(samples)}/{len(blind)}/{len(key)}"
        )
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = output_dir / "state_review_blind.jsonl"
    key_path = output_dir / "state_review_key.jsonl"
    ratings_path = output_dir / "state_ratings_template.csv"
    write_jsonl(blind_path, blind)
    write_jsonl(key_path, key)
    _write_ratings(ratings_path, blind)
    manifest = {
        "schema_version": 1,
        "status": "complete; human ratings pending",
        "seed": args.seed,
        "variants": list(STATE_VARIANTS),
        "samples": len(samples),
        "candidates": len(blind),
        "reviewer_rows": len(blind) * 2,
        "sample_source": {"path": str(sample_path), "sha256": sha256_file(sample_path)},
        "content_sources": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in content_paths
        ],
        "blind_sha256": sha256_file(blind_path),
        "key_sha256": sha256_file(key_path),
        "ratings_template_sha256": sha256_file(ratings_path),
        "ratings_from_interface_package_reused": False,
        "frozen_protocol": {
            "path": str(PROJECT_ROOT / "annotations/u1_forced_generation_v1/PROTOCOL.md"),
            "sha256": sha256_file(
                PROJECT_ROOT / "annotations/u1_forced_generation_v1/PROTOCOL.md"
            ),
        },
        "code": {
            "state_review_sha256": sha256_file(Path(__file__).resolve()),
            "ratings_sha256": sha256_file(PROJECT_ROOT / "src/proactive_u1/ratings.py"),
            "state_ratings_sha256": sha256_file(
                PROJECT_ROOT / "src/proactive_u1/state_ratings.py"
            ),
        },
    }
    write_json(output_dir / "state_review_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
