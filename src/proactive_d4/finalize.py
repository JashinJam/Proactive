"""Fit and serialize the single frozen D4 dialog-stage candidate."""

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
from proactive_d3.dialog_control_core import (
    DIALOG_POLICY_NAMES,
    build_dialog_policy_features,
    dialog_control_matrix,
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
SOURCE_VARIANT = "d1_fused_plus_dialog_stage"


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
        raise ValueError(f"Frozen D4 artifact mismatch for {path}: {actual} != {expected}")
    return actual


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d4.finalize", *argv])
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def run(config_path: Path, output_dir: Path, raw_argv: list[str]) -> dict[str, object]:
    started = time.monotonic()
    config = _load_json(config_path)
    if output_dir.exists():
        raise FileExistsError(f"D4 final output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    protocol = dict(config["protocol"])
    data = dict(config["data"])
    starter = dict(config["starter_kit"])
    model_config = dict(config["model"])
    sources = dict(config["sources"])
    oof_reference = dict(config["oof_reference"])
    features = dict(config["features"])
    training = dict(config["training"])
    threshold_config = dict(config["threshold"])

    protocol_path = _resolve(protocol["path"])
    _check_hash(protocol_path, protocol["sha256"])
    input_path = _resolve(data["input"])
    starter_dir = _resolve(starter["path"])
    static_hashes = _validate_static_files(config, input_path, starter_dir)
    source_paths = {name: _resolve(value["path"]) for name, value in sources.items()}
    source_hashes = {
        name: _check_hash(source_paths[name], value["sha256"])
        for name, value in sources.items()
    }
    oof_dir = _resolve(oof_reference["experiment_dir"])
    oof_variant_dir = oof_dir / "variants" / str(oof_reference["variant"])
    oof_hashes = {
        "comparison": _check_hash(
            oof_dir / "comparison.json", oof_reference["comparison_sha256"]
        ),
        "diagnostics": _check_hash(
            oof_variant_dir / "diagnostics.json",
            oof_reference["diagnostics_sha256"],
        ),
        "predictions": _check_hash(
            oof_variant_dir / "predictions.jsonl",
            oof_reference["predictions_sha256"],
        ),
        "metrics": _check_hash(
            oof_variant_dir / "metrics.json", oof_reference["metrics_sha256"]
        ),
    }
    if oof_reference.get("diagnostic_non_promotable") is not True:
        raise ValueError("D4 must preserve the D3-D non-promotable classification")
    if str(features["source_oof_variant"]) != SOURCE_VARIANT:
        raise ValueError("D4 source OOF variant changed")
    if tuple(features["dialog_policy_names"]) != DIALOG_POLICY_NAMES:
        raise ValueError("D4 dialog-policy feature order changed")

    source_rows = load_jsonl(input_path)
    label_free_rows = strip_answers(source_rows)
    split_manifest = _load_json(source_paths["split_manifest"])
    folds = validate_fold_manifest(split_manifest, label_free_rows)
    label_free = build_label_free_chunks(
        label_free_rows,
        load_jsonl(source_paths["r0_session_records"]),
        folds,
        max_history_turns=int(features["max_history_turns"]),
        max_model_frames=int(features["max_model_frames"]),
    )
    dialog_values, dialog_audit = build_dialog_policy_features(
        label_free_rows, label_free
    )
    examples = attach_gold_labels(label_free, source_rows)
    cache = load_aligned_neural_cache(
        source_paths["neural_features"],
        examples,
        hidden_size=int(features["hidden_size"]),
    )
    scalar_names = feature_names(
        str(features["scalar_variant"]),
        sorted({feature.domain for feature in label_free}),
    )
    values, names = dialog_control_matrix(
        examples, cache, scalar_names, dialog_values, SOURCE_VARIANT
    )
    if len(names) != int(features["feature_count"]):
        raise ValueError("D4 feature count changed")

    oof_diagnostics = _load_json(oof_variant_dir / "diagnostics.json")
    if float(oof_diagnostics["official_metrics"]["macro_f1"]) != float(
        oof_reference["oof_macro_f1"]
    ):
        raise ValueError("D4 OOF metric differs from the frozen reference")
    fold_details = oof_diagnostics.get("fold_details")
    if not isinstance(fold_details, list) or len(fold_details) != 5:
        raise ValueError("D4 requires five frozen OOF fold details")
    selected_l2 = [float(detail["selected_l2_weight"]) for detail in fold_details]
    thresholds = [float(detail["threshold_logit"]) for detail in fold_details]
    if training.get("l2_selection") != "median_of_five_frozen_oof_selected_l2":
        raise ValueError("D4 L2 policy changed")
    l2_weight = statistics.median(selected_l2)
    if l2_weight != float(training["expected_l2_weight"]):
        raise ValueError("D4 L2 differs from the frozen OOF median")
    if threshold_config.get("selection") != "median_of_five_frozen_oof_calibration_thresholds":
        raise ValueError("D4 threshold policy changed")
    threshold = statistics.median(thresholds)
    if abs(threshold - float(threshold_config["expected_threshold_logit"])) > 1e-15:
        raise ValueError("D4 threshold differs from the frozen OOF median")

    labels = [example.gold_interrupt for example in examples]
    model = fit_linear_logistic(
        values,
        labels,
        seed=int(training["seed"]),
        max_iterations=int(training["max_iterations"]),
        l2_weight=l2_weight,
        l2_reduction=str(training["l2_reduction"]),
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
            "feature_variant": features["variant"],
            "source_oof_variant": SOURCE_VARIANT,
            "scalar_variant": features["scalar_variant"],
            "dialog_policy_names": list(DIALOG_POLICY_NAMES),
            "hidden_size": features["hidden_size"],
            "max_history_turns": features["max_history_turns"],
            "max_model_frames": features["max_model_frames"],
            "fit_sessions": 700,
            "fit_chunks": 9935,
            "l2_weight": l2_weight,
            "l2_source_values": selected_l2,
            "threshold_source_values": thresholds,
            "threshold_full_fit_predictions_used": False,
            "oof_reference": oof_reference,
            "neural_cache_sha256": sources["neural_features"]["sha256"],
        },
    )
    head_path = output_dir / "decision_head.json"
    write_json(head_path, head_payload)
    loaded = load_decision_head(_load_json(head_path))
    if loaded.feature_names != names:
        raise ValueError("Reloaded D4 head changed feature order")
    logits = predict_logits(loaded.model, values)
    decisions = {
        example.key: int(logit >= loaded.threshold_logit)
        for example, logit in zip(examples, logits)
    }
    predictions = prediction_rows(examples, decisions)
    validation = validate_prediction_rows(source_rows, predictions)
    predictions_path = output_dir / "train_fit_predictions.jsonl"
    records_path = output_dir / "train_fit_records.jsonl"
    write_jsonl(predictions_path, predictions)
    write_jsonl(
        records_path,
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
                    name: float(dialog_values[index, feature_index])
                    for feature_index, name in enumerate(DIALOG_POLICY_NAMES)
                },
            }
            for index, (example, logit) in enumerate(zip(examples, logits))
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
        "status": "complete D4 dialog-stage full-development refit",
        "classification": config["validation_policy"],
        "feature_variant": features["variant"],
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
        "oof_reference_is_non_promotable_diagnostic": True,
        "head_sha256": sha256_file(head_path),
        "train_fit_predictions_sha256": sha256_file(predictions_path),
        "train_fit_records_sha256": sha256_file(records_path),
    }
    write_json(output_dir / "diagnostics.json", result)
    write_json(output_dir / "feature_audit.json", dialog_audit)
    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "input_path": str(input_path),
        "cache_path": str(source_paths["neural_features"]),
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
                config_path,
                protocol_path,
                PROJECT_ROOT / "src/proactive_d1/core.py",
                PROJECT_ROOT / "src/proactive_d1/neural_core.py",
                PROJECT_ROOT / "src/proactive_d3/dialog_control_core.py",
                PROJECT_ROOT / "src/proactive_d4/finalize.py",
                PROJECT_ROOT / "src/proactive_d4/deploy.py",
                PROJECT_ROOT / "src/proactive_d4/verify.py",
                PROJECT_ROOT / "src/proactive_d4/tests/test_deploy.py",
                PROJECT_ROOT / "CURRENT_ROUTE.md",
                PROJECT_ROOT / "Agent.md",
            ],
        ),
    )
    write_json(
        output_dir / "data_manifest.json",
        {
            "source": {"path": str(input_path), "sha256": static_hashes["input_sha256"]},
            "source_hashes": source_hashes,
            "oof_hashes": oof_hashes,
            "starter_kit_sha256": static_hashes,
            "protocol_sha256": protocol["sha256"],
            "dialog_features_read_labels": False,
            "dialog_features_read_predictions": False,
            "dialog_features_read_future_rows": False,
            "supervision": config["validation_policy"],
            "external_data_used": False,
        },
    )
    _write_command(output_dir / "command.sh", raw_argv)
    runtime = {
        "status": result["status"],
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
        raise ValueError("D4 deployment system exceeds Small")
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "状态：已完成 D4 全 public-development refit。",
                "",
                "Full-fit 指标仅作训练闭环检查；OOF 0.6846 仍是非晋级诊断证据。",
                f"Head parameters: {len(names) + 1}.",
                f"Decision head SHA256: {result['head_sha256']}.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    raw_argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/d4_internvl35_1b_dialog_stage_final_v1.json"
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args(raw_argv)
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    output_dir = _resolve(
        args.output_dir or f"output/experiments/{config['experiment_id']}"
    )
    run(config_path, output_dir, raw_argv)


if __name__ == "__main__":
    main()
