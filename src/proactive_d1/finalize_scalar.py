"""Fit and serialize one deployable D1 scalar head after OOF promotion."""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from proactive_r0.artifacts import code_snapshot, environment_snapshot, sha256_file, write_json
from proactive_r0.core import load_jsonl, validate_prediction_rows, write_jsonl
from proactive_r0.run import _run_official_scorer, _validate_static_files

from .core import (
    LinearDecisionHead,
    attach_gold_labels,
    binary_metrics,
    build_label_free_chunks,
    feature_names,
    fit_linear_logistic,
    load_decision_head,
    matrix,
    predict_feature_values,
    prediction_rows,
    serialize_decision_head,
    strip_answers,
    validate_fold_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "d1_internvl35_1b_scalar_final.json"


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _check_hash(path: Path, expected: object) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"Frozen artifact mismatch for {path}: {actual} != {expected}")
    return actual


def _tracked_paths(config_path: Path) -> list[Path]:
    return [
        *sorted((PROJECT_ROOT / "src" / "proactive_r0").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_r0f").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d1").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d1" / "tests").glob("*.py")),
        config_path,
        PROJECT_ROOT / "models" / "internvl35_1b_hf.json",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
        PROJECT_ROOT / "C1_SPEC.md",
        PROJECT_ROOT / "Agent.md",
    ]


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d1.finalize_scalar", *argv])
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


def _verify_r0(reference: dict[str, object]) -> tuple[Path, dict[str, str]]:
    directory = _resolve(reference["experiment_dir"])
    hashes = {
        "predictions_sha256": _check_hash(
            directory / "predictions.jsonl", reference["predictions_sha256"]
        ),
        "session_records_sha256": _check_hash(
            directory / "session_records.jsonl", reference["session_records_sha256"]
        ),
        "metrics_sha256": _check_hash(
            directory / "metrics.json", reference["metrics_sha256"]
        ),
    }
    return directory, hashes


def _verify_oof(reference: dict[str, object]) -> tuple[Path, dict[str, str]]:
    directory = _resolve(reference["experiment_dir"])
    variant = str(reference["feature_variant"])
    files = {
        "split_manifest_sha256": directory / "split_manifest.json",
        "comparison_sha256": directory / "comparison.json",
        "diagnostics_sha256": directory / "variants" / variant / "diagnostics.json",
        "predictions_sha256": directory / "variants" / variant / "predictions.jsonl",
        "metrics_sha256": directory / "variants" / variant / "metrics.json",
    }
    return directory, {
        key: _check_hash(path, reference[key]) for key, path in files.items()
    }


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    args = parser.parse_args(raw_argv)
    started_at = time.monotonic()

    config_path = _resolve(args.config)
    config = _load_json(config_path)
    required = {
        "experiment_id",
        "model",
        "data",
        "starter_kit",
        "r0_reference",
        "oof_reference",
        "features",
        "training",
        "threshold",
        "validation_policy",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"D1 final config missing keys: {sorted(missing)}")

    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    model_config = dict(config["model"])  # type: ignore[arg-type]
    feature_config = dict(config["features"])  # type: ignore[arg-type]
    training_config = dict(config["training"])  # type: ignore[arg-type]
    threshold_config = dict(config["threshold"])  # type: ignore[arg-type]
    r0_reference = dict(config["r0_reference"])  # type: ignore[arg-type]
    oof_reference = dict(config["oof_reference"])  # type: ignore[arg-type]

    input_path = _resolve(data_config["input"])
    starter_dir = _resolve(starter_config["path"])
    output_dir = _resolve(args.output_dir or f"output/experiments/{config['experiment_id']}")
    if output_dir.exists():
        raise FileExistsError(f"D1 final output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    static_hashes = _validate_static_files(config, input_path, starter_dir)
    r0_dir, r0_hashes = _verify_r0(r0_reference)
    oof_dir, oof_hashes = _verify_oof(oof_reference)
    source_rows = load_jsonl(input_path)
    r0_records = load_jsonl(r0_dir / "session_records.jsonl")
    if len(source_rows) != 700 or len(r0_records) != 700:
        raise ValueError("D1 final refit requires all 700 aligned development sessions")

    manifest = _load_json(oof_dir / "split_manifest.json")
    fold_by_index = validate_fold_manifest(manifest, strip_answers(source_rows))
    label_free = build_label_free_chunks(
        strip_answers(source_rows),
        r0_records,
        fold_by_index,
        max_history_turns=int(feature_config["max_history_turns"]),
        max_model_frames=int(feature_config["max_model_frames"]),
    )
    if len(label_free) != 9935:
        raise ValueError(f"Unexpected D1 final chunk count: {len(label_free)}")

    variant = str(feature_config["variant"])
    if variant != str(oof_reference["feature_variant"]):
        raise ValueError("D1 final variant differs from frozen OOF selection")
    domains = sorted({feature.domain for feature in label_free})
    names = feature_names(variant, domains)  # type: ignore[arg-type]
    diagnostics = _load_json(oof_dir / "variants" / variant / "diagnostics.json")
    if tuple(diagnostics.get("feature_names", ())) != names:
        raise ValueError("D1 final feature order differs from frozen OOF diagnostics")
    fold_details = diagnostics.get("fold_details")
    if not isinstance(fold_details, list) or len(fold_details) != 5:
        raise ValueError("D1 final threshold requires five frozen OOF folds")
    thresholds = [float(detail["threshold_logit"]) for detail in fold_details]
    if threshold_config.get("selection") != "median_of_five_frozen_oof_calibration_thresholds":
        raise ValueError("Unsupported D1 final threshold policy")
    threshold = statistics.median(thresholds)

    examples = attach_gold_labels(label_free, source_rows)
    values, labels = matrix(examples, names)
    model = fit_linear_logistic(
        values,
        labels,
        seed=int(training_config["seed"]),
        max_iterations=int(training_config["max_iterations"]),
        l2_weight=float(training_config["l2_weight"]),
    )
    head = LinearDecisionHead(
        feature_names=names,
        model=model,
        threshold_logit=threshold,
    )
    head_payload = serialize_decision_head(
        head,
        {
            "experiment_id": config["experiment_id"],
            "classification": config["validation_policy"],
            "model": model_config,
            "feature_variant": variant,
            "max_history_turns": feature_config["max_history_turns"],
            "max_model_frames": feature_config["max_model_frames"],
            "fit_sessions": 700,
            "fit_chunks": 9935,
            "threshold_source_values": thresholds,
            "threshold_full_fit_predictions_used": False,
            "oof_reference": oof_reference,
        },
    )
    head_path = output_dir / "decision_head.json"
    write_json(head_path, head_payload)

    loaded = load_decision_head(_load_json(head_path))
    decisions: dict[tuple[int, int], int] = {}
    logits: dict[tuple[int, int], float] = {}
    for example in examples:
        decision, logit = predict_feature_values(loaded, example.feature.values)
        decisions[example.key] = decision
        logits[example.key] = logit
    train_fit_internal = binary_metrics(
        labels, [decisions[example.key] for example in examples]
    )
    predictions = prediction_rows(examples, decisions)
    validation = validate_prediction_rows(source_rows, predictions)
    predictions_path = output_dir / "train_fit_predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    write_jsonl(
        output_dir / "train_fit_records.jsonl",
        [
            {
                "input_index": example.feature.input_index,
                "video_path": example.feature.video_path,
                "chunk_index": example.feature.chunk_index,
                "logit": logits[example.key],
                "threshold_logit": threshold,
                "predicted_interrupt": decisions[example.key],
                "gold_interrupt": example.gold_interrupt,
            }
            for example in examples
        ],
    )
    metrics_path = output_dir / "train_fit_metrics.json"
    _run_official_scorer(
        starter_dir,
        input_path,
        predictions_path,
        metrics_path,
        output_dir / "scorer.log",
    )
    official = _load_json(metrics_path)

    result = {
        **validation,
        "status": "complete deployable scalar refit",
        "classification": config["validation_policy"],
        "feature_variant": variant,
        "feature_names": names,
        "head_parameters": len(names) + 1,
        "threshold_logit": threshold,
        "frozen_oof_thresholds": thresholds,
        "train_fit_internal": train_fit_internal,
        "train_fit_official": official["overall"],
        "oof_reference_macro_f1": oof_reference["oof_macro_f1"],
        "head_sha256": sha256_file(head_path),
        "train_fit_predictions_sha256": sha256_file(predictions_path),
    }
    write_json(output_dir / "diagnostics.json", result)

    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "input_path": str(input_path),
        "r0_reference_dir": str(r0_dir),
        "oof_reference_dir": str(oof_dir),
        "output_dir": str(output_dir),
        "gpu_used": False,
        "model_inference_rerun": False,
    }
    write_json(output_dir / "config.json", effective)
    write_json(output_dir / "environment.txt", environment_snapshot())
    write_json(output_dir / "code_state.txt", code_snapshot(PROJECT_ROOT, _tracked_paths(config_path)))
    write_json(
        output_dir / "data_manifest.json",
        {
            "source": {"path": str(input_path), "sha256": static_hashes["input_sha256"]},
            "r0_reference": {"path": str(r0_dir), **r0_hashes},
            "oof_reference": {"path": str(oof_dir), **oof_hashes},
            "starter_kit_sha256": static_hashes,
            "supervision": config["validation_policy"],
        },
    )
    _write_command(output_dir / "command.sh", raw_argv)
    runtime = {
        "status": "complete deployable scalar refit",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started_at, 3),
        "gpu_used": False,
        "model_inference_rerun": False,
        "sessions": 700,
        "chunks": 9935,
        "head_parameters": len(names) + 1,
        "total_parameters": int(model_config["total_parameters"]) + len(names) + 1,
    }
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "Status: **complete deployable scalar refit**",
                "",
                "This directory serializes one full-public-development D1 scalar head.",
                "Its train-fit score is a sanity check, not generalization evidence; the",
                f"frozen OOF reference remains Macro F1 `{oof_reference['oof_macro_f1']}`.",
                "",
                f"- Feature variant: `{variant}`",
                f"- Head parameters: `{len(names) + 1}`",
                f"- Frozen median threshold: `{threshold}`",
                f"- Decision head SHA256: `{result['head_sha256']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
