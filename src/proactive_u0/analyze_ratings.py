"""Run the frozen U0 two-reviewer aggregation and disagreement audit."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    write_json,
)
from proactive_r0.core import load_jsonl, write_jsonl
from proactive_u0.core import finite_json
from proactive_u0.ratings import analyze_ratings, validate_ratings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _check_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"U0 ratings SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Ratings CSV has no header: {path}")
        required = {
            "review_id",
            "reviewer_slot",
            "should_interrupt",
            "decision_confidence_1_5",
            "timeliness_1_5",
            "correctness_1_5",
            "specificity_1_5",
            "actionability_1_5",
            "groundedness_1_5",
            "plan_consistency_1_5",
            "conciseness_1_5",
            "safety_1_5",
            "generic_flag",
            "hallucination_flag",
            "unsafe_flag",
            "primary_error_type",
        }
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Ratings CSV is missing columns {missing}: {path}")
        return [dict(row) for row in reader]


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _artifact_manifest(output_dir: Path, names: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifacts": {
            name: {
                "bytes": (output_dir / name).stat().st_size,
                "sha256": sha256_file(output_dir / name),
            }
            for name in names
        },
    }


def run(config_path: Path, output_dir: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"U0 ratings output directory is not empty: {output_dir}")

    protocol_path = _resolve(config["protocol"]["path"])
    _check_hash(protocol_path, str(config["protocol"]["sha256"]))
    source_paths = {
        name: _resolve(value["path"]) for name, value in config["sources"].items()
    }
    source_hashes = {
        name: _check_hash(source_paths[name], str(value["sha256"]))
        for name, value in config["sources"].items()
    }
    rating_rows = [
        row
        for name in ("reviewer_a", "reviewer_b")
        for row in _read_csv(source_paths[name])
    ]
    blind_rows = load_jsonl(source_paths["blind_items"])
    key_rows = load_jsonl(source_paths["review_key"])
    parsed = validate_ratings(
        rating_rows,
        blind_rows,
        key_rows,
        expected_items=int(config["analysis"]["expected_items"]),
    )
    analysis, item_records, disagreements = analyze_ratings(
        parsed,
        bootstrap_seed=int(config["analysis"]["bootstrap_seed"]),
        bootstrap_samples=int(config["analysis"]["bootstrap_samples"]),
    )
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "protocol_sha256": config["protocol"]["sha256"],
        "source_hashes": source_hashes,
        "raw_ratings_locked_before_keyed_analysis": True,
        "raw_ratings_modified": False,
        **analysis,
    }
    finite_json(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_config = copy.deepcopy(config)
    effective_config["runtime"] = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "config.json", effective_config)
    write_json(output_dir / "analysis.json", result)
    write_jsonl(output_dir / "item_records.jsonl", item_records)
    write_jsonl(output_dir / "disagreement_cases.jsonl", disagreements)
    write_json(
        output_dir / "data_manifest.json",
        {
            "schema_version": 1,
            "source_hashes": source_hashes,
            "ratings_are_original_locked_exports": True,
            "review_key_read_only_after_rating_hash_validation": True,
            "public_validation_hard_stratum_sample": True,
            "population_prevalence_claim_allowed": False,
            "official_scorer_invoked": False,
            "external_data_used": False,
        },
    )
    write_json(output_dir / "environment.txt", environment_snapshot())
    write_json(
        output_dir / "code_state.txt",
        code_snapshot(
            PROJECT_ROOT,
            [
                config_path,
                protocol_path,
                PROJECT_ROOT / "src/proactive_u0/core.py",
                PROJECT_ROOT / "src/proactive_u0/ratings.py",
                PROJECT_ROOT / "src/proactive_u0/analyze_ratings.py",
                PROJECT_ROOT / "src/proactive_u0/tests/test_ratings.py",
            ],
        ),
    )
    _write_text(
        output_dir / "command.sh",
        " ".join(["PYTHONNOUSERSITE=1", "PYTHONPATH=src", *sys.argv]) + "\n",
    )
    (output_dir / "command.sh").chmod(0o755)

    composite = result["overall"]["content_composite"]
    timeliness = result["overall"]["scores"]["timeliness_1_5"]
    _write_text(
        output_dir / "README.md",
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "Status: complete U0 two-reviewer hard-stratum analysis.",
                "",
                f"Items/reviewer rows: {result['validation']['items']}/"
                f"{result['validation']['reviewer_rows']}.",
                f"Pair-average timeliness: {timeliness['pair_average']['estimate']:.4f}.",
                f"Pair-average spoken content composite: "
                f"{composite['pair_average']['estimate']:.4f}.",
                f"Adjudication-list items: {result['disagreement']['items']}.",
                "",
                "The sample is deliberately stratified and must not be used to estimate "
                "full-validation prevalence.",
                "",
            ]
        ),
    )
    _write_text(
        output_dir / "run.log",
        json.dumps(
            {
                "status": result["status"],
                "validation": result["validation"],
                "disagreement": result["disagreement"],
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
    )
    artifact_names = [
        "README.md",
        "analysis.json",
        "code_state.txt",
        "command.sh",
        "config.json",
        "data_manifest.json",
        "disagreement_cases.jsonl",
        "environment.txt",
        "item_records.jsonl",
        "run.log",
    ]
    manifest = _artifact_manifest(output_dir, artifact_names)
    write_json(output_dir / "artifact_manifest.json", manifest)
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(_resolve(args.config), _resolve(args.output_dir))


if __name__ == "__main__":
    main()
