"""Reproduce the D4.2 history8 baseline on its frozen session-level folds."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from proactive_d1.core import prediction_rows, strip_answers
from proactive_d1.neural_core import cross_validate_neural_matrix
from proactive_d4_1.compare import (
    flatten_decisions,
    load_official_scorer,
    stratified_statistics,
)
from proactive_d4_1.core import object_sha256
from proactive_d4_2.evaluate import build_candidate_matrix, length_quartiles
from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    write_json,
)
from proactive_r0.core import load_jsonl, validate_prediction_rows, write_jsonl
from proactive_r0.run import _run_official_scorer

from .session_folds import validate_session_fold_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("proactive_d5.session_baseline")


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _check_hash(path: Path, expected: object) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"D5 frozen artifact mismatch: {path}: {actual}")
    return actual


def _write_command(path: Path, argv: Sequence[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d5.run_session_baseline", *argv])
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _configure_logging(output_dir: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def _tracked_paths(config_path: Path) -> list[Path]:
    return [
        *sorted((PROJECT_ROOT / "src" / "proactive_d1").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d3").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d4_2").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d5").glob("*.py")),
        config_path,
        PROJECT_ROOT / "Agent.md",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
    ]


def _predicted_interrupt_rate(predictions: Sequence[Mapping[str, object]]) -> float:
    answers = [
        str(answer)
        for row in predictions
        for answer in row["answers"]  # type: ignore[index]
    ]
    return sum(answer.startswith("$interrupt$") for answer in answers) / len(answers)


def run(config_path: Path, output_dir: Path, raw_argv: Sequence[str]) -> dict[str, object]:
    started = time.monotonic()
    config_path = config_path.resolve()
    config = _load_object(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"D5 session baseline output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(output_dir)
    write_json(output_dir / "config.json", config)
    _write_command(output_dir / "command.sh", raw_argv)
    write_json(output_dir / "environment.txt", environment_snapshot())
    write_json(
        output_dir / "code_state.txt",
        code_snapshot(PROJECT_ROOT, _tracked_paths(config_path)),
    )

    protocol = dict(config["protocol"])  # type: ignore[arg-type]
    data = dict(config["data"])  # type: ignore[arg-type]
    starter = dict(config["starter_kit"])  # type: ignore[arg-type]
    history8 = dict(config["history8"])  # type: ignore[arg-type]
    reference = dict(config["d4_2_reference"])  # type: ignore[arg-type]
    folds_config = dict(config["folds"])  # type: ignore[arg-type]
    training = dict(config["training"])  # type: ignore[arg-type]
    input_path = _resolve(data["input"])
    starter_dir = _resolve(starter["path"])
    protocol_path = _resolve(protocol["path"])
    records_spec = dict(history8["generation_records"])  # type: ignore[arg-type]
    cache_spec = dict(history8["neural_cache"])  # type: ignore[arg-type]
    records_path = _resolve(records_spec["path"])
    cache_path = _resolve(cache_spec["path"])
    manifest_source = _resolve(folds_config["manifest"])
    reference_predictions = _resolve(reference["predictions"])
    reference_metrics = _resolve(reference["metrics"])
    fingerprints = {
        "input": _check_hash(input_path, data["input_sha256"]),
        "protocol": _check_hash(protocol_path, protocol["sha256"]),
        "generation_records": _check_hash(records_path, records_spec["sha256"]),
        "neural_cache": _check_hash(cache_path, cache_spec["sha256"]),
        "fold_manifest": _check_hash(
            manifest_source, folds_config["manifest_sha256"]
        ),
        "reference_predictions": _check_hash(
            reference_predictions, reference["predictions_sha256"]
        ),
        "reference_metrics": _check_hash(
            reference_metrics, reference["metrics_sha256"]
        ),
        "official_scorer": _check_hash(
            starter_dir / "run_evaluation.py", starter["scorer_py_sha256"]
        ),
    }
    source_rows = load_jsonl(input_path)
    answer_free_rows = strip_answers(source_rows)
    manifest_path = output_dir / "fold_manifest.json"
    shutil.copyfile(manifest_source, manifest_path)
    manifest = _load_object(manifest_path)
    fold_by_index, split_audit = validate_session_fold_manifest(
        manifest, answer_free_rows
    )
    records = load_jsonl(records_path)
    candidate = {
        "name": "history8_session_baseline",
        "parameters": history8["parameters"],
    }
    examples, values, names, dialog_audit = build_candidate_matrix(
        source_rows=source_rows,
        answer_free_rows=answer_free_rows,
        records=records,
        fold_by_index=fold_by_index,
        candidate=candidate,
        cache_path=cache_path,
        config=config,
    )
    LOGGER.info(
        "Built D4-fold history8 matrix: sessions=%d chunks=%d features=%d",
        len(source_rows),
        len(examples),
        len(names),
    )
    decisions, fold_details = cross_validate_neural_matrix(
        examples,
        values,
        names,
        folds=int(folds_config["count"]),
        calibration_fold_offset=int(folds_config["calibration_fold_offset"]),
        seed=int(training["seed"]),
        max_iterations=int(training["max_iterations"]),
        l2_weights=[float(value) for value in training["l2_weights"]],  # type: ignore[index]
        l2_reduction=str(training["l2_reduction"]),  # type: ignore[arg-type]
    )
    predictions = prediction_rows(examples, decisions)
    validation = validate_prediction_rows(source_rows, predictions)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    write_jsonl(predictions_path, predictions)
    _run_official_scorer(
        starter_dir,
        input_path,
        predictions_path,
        metrics_path,
        output_dir / "scorer.log",
    )
    metrics = _load_object(metrics_path)
    predictions_sha256 = sha256_file(predictions_path)
    metrics_sha256 = sha256_file(metrics_path)
    if predictions_sha256 != str(reference["predictions_sha256"]):
        raise ValueError(
            "D5 session baseline did not reproduce D4.2 history8 predictions: "
            f"{predictions_sha256}"
        )
    if metrics_sha256 != str(reference["metrics_sha256"]):
        raise ValueError(
            "D5 session baseline did not reproduce D4.2 history8 metrics: "
            f"{metrics_sha256}"
        )
    scorer = load_official_scorer(starter_dir, str(starter["scorer_py_sha256"]))
    flattened = flatten_decisions(
        source_rows,
        predictions,
        list(range(len(source_rows))),
        length_quartiles(answer_free_rows),
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "classification": config["evaluation"]["classification"],  # type: ignore[index]
        "overall": metrics["overall"],
        "predicted_interrupt_rate": _predicted_interrupt_rate(predictions),
        "sessions": validation["sessions"],
        "chunks": validation["chunks"],
        "feature_count": len(names),
        "head_parameters": len(names) + 1,
        "total_parameters": config["model_accounting"]["total_parameters"],  # type: ignore[index]
        "fold_details": fold_details,
        "split_audit": split_audit,
        "dialog_feature_audit": dialog_audit,
        "stratified": stratified_statistics(scorer, flattened),
        "fold_manifest_sha256": sha256_file(manifest_path),
        "predictions_sha256": predictions_sha256,
        "metrics_sha256": metrics_sha256,
        "d4_2_reference_reproduced": True,
        "config_object_sha256": object_sha256(config),
        "wall_time_seconds": time.monotonic() - started,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "data_manifest.json",
        {
            "data": data,
            "starter_kit": starter,
            "protocol": protocol,
            "history8": history8,
            "d4_2_reference": reference,
            "fingerprints": fingerprints,
            "split": folds_config,
            "split_audit": split_audit,
            "labels_used_for_feature_inference": False,
            "external_data_used": False,
            "supervision": config["evaluation"]["classification"],  # type: ignore[index]
        },
    )
    overall = metrics["overall"]
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "状态：已完成 D4 session-level folds 上的 history8 五折 OOF 基线。",
                "",
                f"- Sessions/chunks: `{validation['sessions']}/{validation['chunks']}`",
                f"- Fold algorithm: `{split_audit['algorithm']}`",
                f"- D4.2 predictions reproduced: `{summary['d4_2_reference_reproduced']}`",
                f"- Official Macro/G-mean: `{overall['macro_f1']:.4f}/{overall['gmean_f1']:.4f}`",
                f"- Predictions SHA256: `{summary['predictions_sha256']}`",
                "- 证据类型：D4.2 session-fold exact replay；仍是 val-supervised，不是独立泛化证据。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    LOGGER.info("D4-fold history8 OOF complete: Macro=%.4f", overall["macro_f1"])
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/d5_internvl35_1b_session_history8_baseline_v1.json",
    )
    parser.add_argument("--experiment-dir")
    args = parser.parse_args(raw_argv)
    config_path = _resolve(args.config)
    config = _load_object(config_path)
    output_dir = _resolve(
        args.experiment_dir or f"output/experiments/{config['experiment_id']}"
    )
    result = run(config_path, output_dir, raw_argv)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
