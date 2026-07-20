"""Fit and serialize one full-development D3 dynamics_fused decision head."""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from proactive_d1.core import (
    LinearDecisionHead,
    attach_gold_labels,
    binary_metrics,
    build_label_free_chunks,
    feature_names,
    fit_linear_logistic,
    load_decision_head,
    predict_logits,
    prediction_rows,
    serialize_decision_head,
    strip_answers,
    validate_fold_manifest,
)
from proactive_d1.neural_core import load_aligned_neural_cache
from proactive_d3.core import (
    DYNAMIC_SCALAR_NAMES,
    PRIMARY_VARIANT,
    build_causal_dynamics,
    d3_matrix,
)
from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    write_json,
)
from proactive_r0.core import load_jsonl, validate_prediction_rows, write_jsonl
from proactive_r0.run import _run_official_scorer, _validate_static_files


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/d3_internvl35_1b_causal_dynamics_final.json"


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
        raise ValueError(f"Frozen D3 final artifact mismatch for {path}")
    return actual


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d3.finalize", *argv])
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    args = parser.parse_args(raw_argv)
    started = time.monotonic()
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    data = dict(config["data"])  # type: ignore[arg-type]
    starter = dict(config["starter_kit"])  # type: ignore[arg-type]
    model_config = dict(config["model"])  # type: ignore[arg-type]
    r0_reference = dict(config["r0_reference"])  # type: ignore[arg-type]
    split_reference = dict(config["split_reference"])  # type: ignore[arg-type]
    cache_config = dict(config["neural_cache"])  # type: ignore[arg-type]
    oof_reference = dict(config["oof_reference"])  # type: ignore[arg-type]
    features_config = dict(config["features"])  # type: ignore[arg-type]
    training = dict(config["training"])  # type: ignore[arg-type]
    threshold_config = dict(config["threshold"])  # type: ignore[arg-type]

    input_path = _resolve(data["input"])
    starter_dir = _resolve(starter["path"])
    r0_dir = _resolve(r0_reference["experiment_dir"])
    split_dir = _resolve(split_reference["experiment_dir"])
    split_path = split_dir / "split_manifest.json"
    cache_dir = _resolve(cache_config["path"])
    cache_path = cache_dir / "features.npz"
    oof_dir = _resolve(oof_reference["experiment_dir"])
    oof_variant_dir = oof_dir / "variants" / str(oof_reference["variant"])
    output_dir = _resolve(args.output_dir or f"output/experiments/{config['experiment_id']}")
    if output_dir.exists():
        raise FileExistsError(f"D3 final output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    static_hashes = _validate_static_files(config, input_path, starter_dir)
    frozen_hashes = {
        "r0_session_records": _check_hash(
            r0_dir / "session_records.jsonl", r0_reference["session_records_sha256"]
        ),
        "split_manifest": _check_hash(
            split_path, split_reference["manifest_sha256"]
        ),
        "neural_features": _check_hash(
            cache_path, cache_config["features_sha256"]
        ),
        "neural_summary": _check_hash(
            cache_dir / "summary.json", cache_config["summary_sha256"]
        ),
        "oof_comparison": _check_hash(
            oof_dir / "comparison.json", oof_reference["comparison_sha256"]
        ),
        "oof_diagnostics": _check_hash(
            oof_variant_dir / "diagnostics.json", oof_reference["diagnostics_sha256"]
        ),
        "oof_predictions": _check_hash(
            oof_variant_dir / "predictions.jsonl", oof_reference["predictions_sha256"]
        ),
        "oof_metrics": _check_hash(
            oof_variant_dir / "metrics.json", oof_reference["metrics_sha256"]
        ),
    }
    if oof_reference.get("promotion_gate_passed") is not True:
        raise ValueError("D3 final requires a promoted OOF reference")
    cache_summary = _load_json(cache_dir / "summary.json")
    if cache_summary.get("labels_read_or_stored") is not False:
        raise ValueError("D3 final requires a label-free cache")

    source_rows = load_jsonl(input_path)
    r0_records = load_jsonl(r0_dir / "session_records.jsonl")
    fold_by_index = validate_fold_manifest(
        _load_json(split_path), strip_answers(source_rows)
    )
    label_free = build_label_free_chunks(
        strip_answers(source_rows),
        r0_records,
        fold_by_index,
        max_history_turns=int(features_config["max_history_turns"]),
        max_model_frames=int(features_config["max_model_frames"]),
    )
    examples = attach_gold_labels(label_free, source_rows)
    cache = load_aligned_neural_cache(
        cache_path, examples, hidden_size=int(features_config["hidden_size"])
    )
    dynamics = build_causal_dynamics(cache)
    scalar_names = feature_names(
        str(features_config["scalar_variant"]),
        sorted({feature.domain for feature in label_free}),  # type: ignore[arg-type]
    )
    if str(features_config["variant"]) != PRIMARY_VARIANT:
        raise ValueError("D3 final must use the preregistered primary variant")
    if tuple(features_config["dynamic_scalars"]) != DYNAMIC_SCALAR_NAMES:  # type: ignore[arg-type]
        raise ValueError("D3 final dynamic scalar order changed")
    values, names = d3_matrix(
        examples, cache, scalar_names, dynamics, PRIMARY_VARIANT
    )
    if len(names) != int(features_config["feature_count"]):
        raise ValueError("D3 final feature count changed")

    oof_diagnostics = _load_json(oof_variant_dir / "diagnostics.json")
    fold_details = oof_diagnostics.get("fold_details")
    if not isinstance(fold_details, list) or len(fold_details) != 5:
        raise ValueError("D3 final requires five frozen OOF fold details")
    selected_l2 = [float(detail["selected_l2_weight"]) for detail in fold_details]
    thresholds = [float(detail["threshold_logit"]) for detail in fold_details]
    if training.get("l2_selection") != "median_of_five_frozen_oof_selected_l2":
        raise ValueError("D3 final L2 policy changed")
    l2_weight = statistics.median(selected_l2)
    if l2_weight != float(training["expected_l2_weight"]):
        raise ValueError("D3 final L2 differs from the frozen OOF median")
    if threshold_config.get("selection") != "median_of_five_frozen_oof_calibration_thresholds":
        raise ValueError("D3 final threshold policy changed")
    threshold = statistics.median(thresholds)
    if abs(threshold - float(threshold_config["expected_threshold_logit"])) > 1e-15:
        raise ValueError("D3 final threshold differs from the frozen OOF median")

    labels = [example.gold_interrupt for example in examples]
    model = fit_linear_logistic(
        values,
        labels,
        seed=int(training["seed"]),
        max_iterations=int(training["max_iterations"]),
        l2_weight=l2_weight,
        l2_reduction=str(training["l2_reduction"]),  # type: ignore[arg-type]
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
            "feature_variant": PRIMARY_VARIANT,
            "scalar_variant": features_config["scalar_variant"],
            "dynamic_scalar_names": list(DYNAMIC_SCALAR_NAMES),
            "hidden_delta": features_config["hidden_delta"],
            "hidden_size": features_config["hidden_size"],
            "max_history_turns": features_config["max_history_turns"],
            "max_model_frames": features_config["max_model_frames"],
            "fit_sessions": 700,
            "fit_chunks": 9935,
            "l2_weight": l2_weight,
            "l2_source_values": selected_l2,
            "threshold_source_values": thresholds,
            "threshold_full_fit_predictions_used": False,
            "oof_reference": oof_reference,
            "neural_cache_sha256": cache_config["features_sha256"],
        },
    )
    head_path = output_dir / "decision_head.json"
    write_json(head_path, head_payload)
    loaded = load_decision_head(_load_json(head_path))
    if loaded.feature_names != names:
        raise ValueError("Reloaded D3 head changed feature order")
    logits_list = predict_logits(loaded.model, values)
    decisions = {
        example.key: int(logit >= loaded.threshold_logit)
        for example, logit in zip(examples, logits_list)
    }
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
                "logit": float(logit),
                "threshold_logit": threshold,
                "predicted_interrupt": decisions[example.key],
                "gold_interrupt": example.gold_interrupt,
                "tag_margin": float(cache.tag_margin[index]),
                **{
                    name: float(dynamics.scalar[index, scalar_index])
                    for scalar_index, name in enumerate(DYNAMIC_SCALAR_NAMES)
                },
            }
            for index, (example, logit) in enumerate(zip(examples, logits_list))
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
        "status": "complete deployable D3 dynamics refit",
        "classification": config["validation_policy"],
        "feature_variant": PRIMARY_VARIANT,
        "feature_count": len(names),
        "head_parameters": len(names) + 1,
        "l2_weight": l2_weight,
        "frozen_oof_selected_l2": selected_l2,
        "threshold_logit": threshold,
        "frozen_oof_thresholds": thresholds,
        "train_fit_internal": binary_metrics(
            labels, [decisions[example.key] for example in examples]
        ),
        "train_fit_official": official["overall"],
        "oof_reference_macro_f1": oof_reference["oof_macro_f1"],
        "head_sha256": sha256_file(head_path),
        "train_fit_predictions_sha256": sha256_file(predictions_path),
        "train_fit_records_sha256": sha256_file(output_dir / "train_fit_records.jsonl"),
    }
    write_json(output_dir / "diagnostics.json", result)
    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "input_path": str(input_path),
        "cache_path": str(cache_path),
        "oof_reference_dir": str(oof_dir),
        "output_dir": str(output_dir),
        "gpu_used": False,
        "model_inference_rerun": False,
    }
    write_json(output_dir / "config.json", effective)
    write_json(output_dir / "environment.txt", environment_snapshot())
    write_json(
        output_dir / "code_state.txt",
        code_snapshot(
            PROJECT_ROOT,
            [
                *sorted((PROJECT_ROOT / "src/proactive_d1").glob("*.py")),
                *sorted((PROJECT_ROOT / "src/proactive_d3").glob("*.py")),
                *sorted((PROJECT_ROOT / "src/proactive_d3/tests").glob("*.py")),
                config_path,
                PROJECT_ROOT / "CURRENT_ROUTE.md",
                PROJECT_ROOT / "Agent.md",
            ],
        ),
    )
    write_json(
        output_dir / "data_manifest.json",
        {
            "source": {"path": str(input_path), "sha256": static_hashes["input_sha256"]},
            "frozen_hashes": frozen_hashes,
            "starter_kit_sha256": static_hashes,
            "dynamic_features_read_labels": False,
            "dynamic_features_read_future_rows": False,
            "supervision": config["validation_policy"],
            "external_data_used": False,
        },
    )
    _write_command(output_dir / "command.sh", raw_argv)
    runtime = {
        "status": "complete deployable D3 dynamics refit",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "gpu_used": False,
        "model_inference_rerun": False,
        "sessions": 700,
        "chunks": 9935,
        "head_parameters": len(names) + 1,
        "total_parameters": int(model_config["total_parameters"]) + len(names) + 1,
    }
    if runtime["total_parameters"] > 2_000_000_000:
        raise ValueError("D3 deployment system exceeds Small")
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "Status: complete full-public-development D3 refit.",
                "",
                "Train-fit metrics are sanity only; OOF Macro 0.6690 is the development estimate.",
                f"Head parameters: {len(names) + 1}.",
                f"Decision head SHA256: {result['head_sha256']}.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
