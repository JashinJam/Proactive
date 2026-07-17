"""Build and officially score the full R0 response-intent repair ablation."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    write_json,
)
from proactive_r0.core import load_jsonl, validate_prediction_rows, write_jsonl
from proactive_r0.run import _run_official_scorer, _validate_static_files

from .core import repair_response_intent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "r0f_internvl35_1b_response_intent_repair.json"


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("R0-F config must be an object")
    return value


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_r0f.run", *argv])
    content = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n"
    )
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _tracked_paths(config_path: Path) -> list[Path]:
    return [
        *sorted((PROJECT_ROOT / "src" / "proactive_r0").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_r0f").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_r0f" / "tests").glob("*.py")),
        config_path,
        PROJECT_ROOT / "configs" / "r0_internvl35_1b_no_plan.json",
        PROJECT_ROOT / "models" / "internvl35_1b_hf.json",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
        PROJECT_ROOT / "C1_SPEC.md",
        PROJECT_ROOT / "Agent.md",
    ]


def _readme(
    config: dict[str, object],
    metrics: dict[str, object],
    diagnostics: dict[str, object],
    reference_metrics: dict[str, object],
) -> str:
    overall = metrics["overall"]
    reference = reference_metrics["overall"]
    assert isinstance(overall, dict) and isinstance(reference, dict)
    return "\n".join(
        [
            f"# {config['experiment_id']}",
            "",
            "Status: **complete full public-validation format ablation**",
            "",
            str(config["hypothesis"]),
            "",
            "## Result",
            "",
            f"- Official Macro F1: `{overall['macro_f1']}` (R0 `{reference['macro_f1']}`)",
            f"- Interrupt P/R/F1: `{overall['interrupt_precision']}` / `{overall['interrupt_recall']}` / `{overall['interrupt_f1']}`",
            f"- Silent P/R/F1: `{overall['silent_precision']}` / `{overall['silent_recall']}` / `{overall['silent_f1']}`",
            f"- TP/FP/TN/FN: `{overall['tp']}` / `{overall['fp']}` / `{overall['tn']}` / `{overall['fn']}`",
            f"- Repaired non-empty malformed responses: `{diagnostics['repaired_nonempty_malformed']}`",
            f"- Predictions SHA256: `{diagnostics['predictions_sha256']}`",
            "",
            "## Interpretation Boundary",
            "",
            "The repair function does not read labels, but this rule was selected after public-validation error analysis. The result is val-supervised and is not held-out evidence.",
            "",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    started_at = time.monotonic()
    config_path = _resolve(args.config)
    config = _load_config(config_path)
    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    reference_config = dict(config["r0_reference"])  # type: ignore[arg-type]
    input_path = _resolve(data_config["input"])
    starter_dir = _resolve(starter_config["path"])
    r0_dir = _resolve(reference_config["experiment_dir"])
    output_dir = _resolve(args.output_dir or f"output/experiments/{config['experiment_id']}")
    if output_dir.exists():
        raise FileExistsError(f"R0-F output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    fingerprints = _validate_static_files(config, input_path, starter_dir)
    reference_files = {
        "predictions_sha256": sha256_file(r0_dir / "predictions.jsonl"),
        "session_records_sha256": sha256_file(r0_dir / "session_records.jsonl"),
        "metrics_sha256": sha256_file(r0_dir / "metrics.json"),
    }
    expected_reference = {
        key: reference_config[key] for key in reference_files
    }
    if reference_files != expected_reference:
        raise ValueError(f"Frozen R0 reference mismatch: {reference_files}")

    source_rows = load_jsonl(input_path)
    reference_records = load_jsonl(r0_dir / "session_records.jsonl")
    if len(source_rows) != 700 or len(reference_records) != 700:
        raise ValueError("R0-F requires the complete 700-session R0 artifact")
    repaired_records: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    changed_decisions = 0
    total_chunks = 0
    for expected_index, (source, reference) in enumerate(zip(source_rows, reference_records)):
        if reference.get("input_index") != expected_index:
            raise ValueError(f"R0 record order mismatch at {expected_index}")
        if reference.get("video_path") != source.get("video_path"):
            raise ValueError(f"R0 video identity mismatch at {expected_index}")
        reference_prediction = reference.get("prediction")
        chunks = reference.get("chunks")
        if not isinstance(reference_prediction, dict) or not isinstance(chunks, list):
            raise ValueError(f"R0 record {expected_index} is malformed")
        original_answers = reference_prediction.get("answers")
        if not isinstance(original_answers, list) or len(original_answers) != len(chunks):
            raise ValueError(f"R0 record {expected_index} answer/chunk mismatch")
        repaired_answers: list[str] = []
        repaired_chunks: list[dict[str, object]] = []
        for chunk_index, (original_answer, chunk) in enumerate(zip(original_answers, chunks)):
            if not isinstance(chunk, dict) or "raw_response" not in chunk:
                raise ValueError(f"R0 record {expected_index} chunk {chunk_index} lacks raw text")
            repaired_answer, reason = repair_response_intent(chunk["raw_response"])
            repaired_answers.append(repaired_answer)
            if reason:
                reasons[reason] += 1
            if repaired_answer.startswith("$interrupt$") != str(original_answer).startswith("$interrupt$"):
                changed_decisions += 1
            repaired_chunks.append(
                {
                    "chunk_index": chunk_index,
                    "interval": chunk["interval"],
                    "raw_response": chunk["raw_response"],
                    "r0_answer": original_answer,
                    "answer": repaired_answer,
                    "repair_reason": reason,
                }
            )
            total_chunks += 1
        prediction = {"video_path": source["video_path"], "answers": repaired_answers}
        predictions.append(prediction)
        repaired_records.append(
            {
                "input_index": expected_index,
                "video_path": source["video_path"],
                "prediction": prediction,
                "chunks": repaired_chunks,
            }
        )
    validation = validate_prediction_rows(source_rows, predictions)
    if validation["sessions"] != 700 or validation["chunks"] != 9935 or total_chunks != 9935:
        raise ValueError(f"Unexpected R0-F full-set shape: {validation}")

    write_jsonl(output_dir / "session_records.jsonl", repaired_records)
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    diagnostics = {
        **validation,
        "predicted_interrupt_rate": validation["interrupts"] / validation["chunks"],
        "changed_binary_decisions_vs_r0": changed_decisions,
        "repaired_nonempty_malformed": reasons[
            "malformed_nonempty_repaired_as_interrupt"
        ],
        "repair_reason_counts": dict(sorted(reasons.items())),
        "predictions_sha256": sha256_file(predictions_path),
    }
    write_json(output_dir / "diagnostics.json", diagnostics)

    # Prediction bytes and diagnostics are durable before the scorer reads gold.
    metrics_path = output_dir / "metrics.json"
    _run_official_scorer(
        starter_dir,
        input_path,
        predictions_path,
        metrics_path,
        output_dir / "scorer.log",
    )
    metrics = _load_config(metrics_path)
    reference_metrics = _load_config(r0_dir / "metrics.json")
    overall = metrics["overall"]
    reference_overall = reference_metrics["overall"]
    assert isinstance(overall, dict) and isinstance(reference_overall, dict)
    comparison = {
        "r0_frozen": reference_overall,
        "r0f_response_intent_repair": overall,
        "delta": {
            key: round(float(overall[key]) - float(reference_overall[key]), 4)
            for key in (
                "macro_f1",
                "gmean_f1",
                "interrupt_precision",
                "interrupt_recall",
                "interrupt_f1",
                "silent_precision",
                "silent_recall",
                "silent_f1",
            )
        },
    }
    write_json(output_dir / "comparison.json", comparison)
    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "reference_dir": str(r0_dir),
        "output_dir": str(output_dir),
        "model_inference_rerun": False,
    }
    write_json(output_dir / "config.json", effective)
    _write_command(output_dir / "command.sh", raw_argv)
    write_json(output_dir / "environment.txt", environment_snapshot())
    write_json(output_dir / "code_state.txt", code_snapshot(PROJECT_ROOT, _tracked_paths(config_path)))
    write_json(
        output_dir / "data_manifest.json",
        {
            "source": {
                "input_path": str(input_path),
                "input_sha256": fingerprints["input_sha256"],
                "sessions": 700,
                "chunks": 9935,
                "top_level_license": data_config["license"],
            },
            "r0_reference": {"path": str(r0_dir), **reference_files},
            "starter_kit_sha256": fingerprints,
            "supervision": {
                "repair_function_reads_gold_labels": False,
                "rule_family_selected_after_public_validation_analysis": True,
                "classification": "val-supervised",
            },
        },
    )
    runtime = {
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started_at, 3),
        "model_inference_rerun": False,
        "gpu_used": False,
        "sessions": 700,
        "chunks": 9935,
    }
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "README.md").write_text(
        _readme(config, metrics, diagnostics, reference_metrics), encoding="utf-8"
    )
    print(json.dumps({"overall": overall, "diagnostics": diagnostics}, sort_keys=True))


if __name__ == "__main__":
    main()
