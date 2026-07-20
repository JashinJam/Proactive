"""Evaluate frozen S1 heads once after their train-only artifact hash is fixed."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import write_jsonl
from proactive_state_s1.core import load_json
from proactive_state_s1.model import (
    FEATURE_VARIANTS,
    TARGETS,
    TARGET_CLASSES,
    align_matrix,
    build_feature_bundle,
    classification_metrics,
    composite_correctness,
    load_targets,
    predict_head,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _object(path: Path) -> dict[str, object]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def session_bootstrap_delta(
    input_index: np.ndarray,
    candidate_correct: np.ndarray,
    reference_correct: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    sessions = np.asarray(sorted(set(input_index.tolist())), dtype=np.int64)
    candidate_by_session = np.asarray(
        [candidate_correct[input_index == value].mean() for value in sessions]
    )
    reference_by_session = np.asarray(
        [reference_correct[input_index == value].mean() for value in sessions]
    )
    observed = float(np.mean(candidate_by_session - reference_by_session))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(sessions), size=(repetitions, len(sessions)))
    deltas = (candidate_by_session - reference_by_session)[sampled].mean(axis=1)
    return {
        "observed_session_mean_delta": observed,
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "positive_probability": float(np.mean(deltas > 0)),
    }


def _variant_metrics(targets: object, predictions: dict[str, np.ndarray]) -> dict[str, object]:
    task_metrics = {
        name: classification_metrics(
            targets.values[name], predictions[name], len(TARGET_CLASSES[name])  # type: ignore[attr-defined]
        )
        for name in TARGETS
    }
    correct = np.stack(
        [predictions[name] == targets.values[name] for name in TARGETS], axis=1  # type: ignore[attr-defined]
    )
    result: dict[str, object] = {
        "tasks": task_metrics,
        "mean_task_macro_f1": float(
            np.mean([task_metrics[name]["macro_f1"] for name in TARGETS])
        ),
        "composite_correctness": composite_correctness(targets, predictions),  # type: ignore[arg-type]
        "joint_step_progress_accuracy": float(
            np.mean(
                (predictions["step"] == targets.values["step"])  # type: ignore[attr-defined]
                & (predictions["progress"] == targets.values["progress"])  # type: ignore[attr-defined]
            )
        ),
        "fully_correct_three_field_accuracy": float(np.mean(np.all(correct, axis=1))),
        "step_ordinal_mae": float(
            np.mean(np.abs(predictions["step"] - targets.values["step"]))  # type: ignore[attr-defined]
        ),
        "predicted_distribution": {
            name: {
                str(TARGET_CLASSES[name][class_index]): int(
                    np.sum(predictions[name] == class_index)
                )
                for class_index in range(len(TARGET_CLASSES[name]))
            }
            for name in TARGETS
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-run-manifest", required=True)
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--heldout-annotations", required=True)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    config_path = _resolve(args.config)
    train_manifest_path = _resolve(args.train_run_manifest)
    model_path = _resolve(args.model_bundle)
    annotations_path = _resolve(args.heldout_annotations)
    annotation_manifest_path = _resolve(args.heldout_manifest)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"S1 held-out output already exists: {output_dir}")
    config = _object(config_path)
    train_manifest = _object(train_manifest_path)
    if train_manifest.get("heldout_annotations_read") is not False:
        raise ValueError("S1 train artifact does not certify held-out isolation")
    trained_model = dict(train_manifest["artifacts"])["model_bundle"]  # type: ignore[index]
    if not isinstance(trained_model, dict) or trained_model.get("sha256") != sha256_file(model_path):
        raise ValueError("S1 model bundle differs from frozen train artifact")
    annotation_manifest = _object(annotation_manifest_path)
    if annotation_manifest.get("split") != "heldout":
        raise ValueError("S1 evaluation requires a validated held-out manifest")
    annotation_source = dict(annotation_manifest["sources"])["annotations"]  # type: ignore[index]
    if not isinstance(annotation_source, dict) or annotation_source.get("sha256") != sha256_file(annotations_path):
        raise ValueError("S1 held-out annotations differ from validation manifest")
    model_bundle = _object(model_path)
    if not str(model_bundle.get("status", "")).startswith("frozen train-only"):
        raise ValueError("S1 model bundle is not a frozen train-only artifact")
    data = dict(config["data"])  # type: ignore[arg-type]
    targets = load_targets(annotations_path, _resolve(str(data["sessions"])), "heldout")
    feature_bundle = build_feature_bundle(config, PROJECT_ROOT)

    predictions: dict[str, dict[str, np.ndarray]] = {}
    results: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for variant in FEATURE_VARIANTS:
        variant_model = dict(model_bundle["variants"])[variant]  # type: ignore[index]
        if not isinstance(variant_model, dict):
            raise ValueError(f"Malformed S1 model variant {variant}")
        values = align_matrix(feature_bundle, targets, variant)
        heads = variant_model["heads"]
        assert isinstance(heads, dict)
        predictions[variant] = {
            name: predict_head(heads[name], values)  # type: ignore[arg-type]
            for name in TARGETS
        }
        results[variant] = _variant_metrics(targets, predictions[variant])
        for index, key in enumerate(targets.keys):
            rows.append(
                {
                    "variant": variant,
                    "input_index": int(key[0]),
                    "chunk_index": int(key[1]),
                    "domain": str(targets.domains[index]),
                    "gold": {name: int(targets.values[name][index]) for name in TARGETS},
                    "prediction": {
                        name: int(predictions[variant][name][index]) for name in TARGETS
                    },
                }
            )

    candidate = predictions["d3_dynamics"]
    reference = predictions["temporal_only"]
    candidate_correct = np.stack(
        [candidate[name] == targets.values[name] for name in TARGETS], axis=1
    ).mean(axis=1)
    reference_correct = np.stack(
        [reference[name] == targets.values[name] for name in TARGETS], axis=1
    ).mean(axis=1)
    evaluation = dict(config["evaluation"])  # type: ignore[arg-type]
    bootstrap = session_bootstrap_delta(
        targets.input_index,
        candidate_correct,
        reference_correct,
        int(evaluation["bootstrap_repetitions"]),
        int(evaluation["bootstrap_seed"]),
    )
    by_domain: dict[str, dict[str, float]] = {}
    for domain in sorted(set(targets.domains.tolist())):
        mask = targets.domains == domain
        candidate_value = float(candidate_correct[mask].mean())
        reference_value = float(reference_correct[mask].mean())
        by_domain[str(domain)] = {
            "candidate_composite": candidate_value,
            "temporal_composite": reference_value,
            "delta": candidate_value - reference_value,
        }
    candidate_metrics = results["d3_dynamics"]
    reference_metrics = results["temporal_only"]
    assert isinstance(candidate_metrics, dict) and isinstance(reference_metrics, dict)
    gate_config = dict(evaluation["gate"])  # type: ignore[arg-type]
    composite_delta = float(candidate_metrics["composite_correctness"]) - float(
        reference_metrics["composite_correctness"]
    )
    checks = {
        "mean_task_macro_f1": float(candidate_metrics["mean_task_macro_f1"])
        >= float(gate_config["minimum_mean_task_macro_f1"]),
        "composite_delta_vs_temporal": composite_delta
        >= float(gate_config["minimum_composite_delta_vs_temporal"]),
        "bootstrap_ci95_low_positive": float(bootstrap["ci95_low"]) > 0,
        "positive_domains": sum(value["delta"] > 0 for value in by_domain.values())
        >= int(gate_config["minimum_positive_domains"]),
        "step_macro_f1": float(candidate_metrics["tasks"]["step"]["macro_f1"])  # type: ignore[index]
        >= float(gate_config["minimum_step_macro_f1"]),
        "progress_macro_f1": float(candidate_metrics["tasks"]["progress"]["macro_f1"])  # type: ignore[index]
        >= float(gate_config["minimum_progress_macro_f1"]),
    }
    summary = {
        "schema_version": 1,
        "status": "complete one-shot S1 held-out evaluation",
        "variants": results,
        "d3_vs_temporal": {
            "composite_delta": composite_delta,
            "session_bootstrap": bootstrap,
            "by_domain": by_domain,
        },
        "gate": {"checks": checks, "passed": all(checks.values())},
    }
    output_dir.mkdir(parents=True)
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.jsonl"
    write_json(metrics_path, summary)
    write_jsonl(predictions_path, rows)
    manifest = {
        "schema_version": 1,
        "status": "complete one-shot held-out evaluation",
        "wall_seconds": time.monotonic() - started,
        "sources": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "train_run_manifest": {"path": str(train_manifest_path), "sha256": sha256_file(train_manifest_path)},
            "model_bundle": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "heldout_annotations": {"path": str(annotations_path), "sha256": sha256_file(annotations_path)},
            "heldout_manifest": {"path": str(annotation_manifest_path), "sha256": sha256_file(annotation_manifest_path)},
        },
        "artifacts": {
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path)},
        },
    }
    write_json(output_dir / "evaluation_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
