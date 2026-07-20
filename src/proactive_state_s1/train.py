"""Run train-only S1 CV, select shared L2, and freeze final state heads."""

from __future__ import annotations

import argparse
import json
import platform
import sys
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
    fit_head,
    load_targets,
    predict_head,
    validate_l2_grid,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_object(path: Path) -> dict[str, object]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    config_path = _resolve(args.config)
    annotations_path = _resolve(args.train_annotations)
    annotation_manifest_path = _resolve(args.train_manifest)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"S1 training output already exists: {output_dir}")
    if "heldout" in str(annotations_path).lower() or "heldout" in str(annotation_manifest_path).lower():
        raise ValueError("S1 train command refuses paths named heldout")
    config = _load_object(config_path)
    annotation_manifest = _load_object(annotation_manifest_path)
    if annotation_manifest.get("split") != "train":
        raise ValueError("S1 train command requires a validated train manifest")
    annotation_source = dict(annotation_manifest["sources"])["annotations"]  # type: ignore[index]
    if not isinstance(annotation_source, dict) or annotation_source.get("sha256") != sha256_file(annotations_path):
        raise ValueError("S1 train annotation hash differs from validation manifest")
    data = dict(config["data"])  # type: ignore[arg-type]
    sessions_path = _resolve(str(data["sessions"]))
    cv_path = _resolve(str(data["cv_split"]))
    cv_manifest = _load_object(cv_path)
    if cv_manifest.get("heldout_annotations_read") is not False:
        raise ValueError("S1 CV split does not certify held-out isolation")
    fold_by_input = {
        int(row["input_index"]): int(row["fold"])
        for row in cv_manifest["sessions"]  # type: ignore[union-attr]
    }
    training = dict(config["training"])  # type: ignore[arg-type]
    l2_values = validate_l2_grid(training["l2_values"])  # type: ignore[arg-type]
    targets = load_targets(annotations_path, sessions_path, "train")
    folds = np.asarray([fold_by_input[int(value)] for value in targets.input_index])
    if set(folds.tolist()) != set(range(int(training["folds"]))):
        raise ValueError("S1 train targets do not cover all frozen folds")
    feature_bundle = build_feature_bundle(config, PROJECT_ROOT)

    cv_results: dict[str, object] = {}
    model_bundle: dict[str, object] = {
        "schema_version": 1,
        "status": "frozen train-only S1 final heads; held-out annotations unread",
        "targets": {name: list(TARGET_CLASSES[name]) for name in TARGETS},
        "variants": {},
    }
    oof_rows: list[dict[str, object]] = []
    for variant in FEATURE_VARIANTS:
        values = align_matrix(feature_bundle, targets, variant)
        candidates: list[dict[str, object]] = []
        candidate_predictions: list[dict[str, np.ndarray]] = []
        for l2 in l2_values:
            predictions = {
                name: np.full(len(targets.keys), -1, dtype=np.int64) for name in TARGETS
            }
            fold_details: list[dict[str, object]] = []
            for fold in range(int(training["folds"])):
                fit = folds != fold
                test = folds == fold
                detail: dict[str, object] = {
                    "fold": fold,
                    "fit_sessions": len(set(targets.input_index[fit].tolist())),
                    "test_sessions": len(set(targets.input_index[test].tolist())),
                    "fit_states": int(fit.sum()),
                    "test_states": int(test.sum()),
                    "targets": {},
                }
                for name in TARGETS:
                    model = fit_head(values[fit], targets.values[name][fit], l2)
                    predictions[name][test] = predict_head(model, values[test])
                    detail["targets"][name] = {  # type: ignore[index]
                        "fit_classes": model["classes"],
                        "n_iter": model["n_iter"],
                    }
                fold_details.append(detail)
            metrics = {
                name: classification_metrics(
                    targets.values[name], predictions[name], len(TARGET_CLASSES[name])
                )
                for name in TARGETS
            }
            mean_macro = float(np.mean([metrics[name]["macro_f1"] for name in TARGETS]))
            candidates.append(
                {
                    "l2": l2,
                    "mean_task_macro_f1": mean_macro,
                    "metrics": metrics,
                    "folds": fold_details,
                }
            )
            candidate_predictions.append(predictions)
        best_index = max(
            range(len(candidates)),
            key=lambda index: (
                round(float(candidates[index]["mean_task_macro_f1"]), 12),
                float(candidates[index]["l2"]),
            ),
        )
        selected = candidates[best_index]
        selected_predictions = candidate_predictions[best_index]
        final_heads = {
            name: fit_head(values, targets.values[name], float(selected["l2"]))
            for name in TARGETS
        }
        model_bundle["variants"][variant] = {  # type: ignore[index]
            "feature_count": values.shape[1],
            "feature_names": list(feature_bundle.names[variant]),
            "selected_l2": selected["l2"],
            "heads": final_heads,
        }
        cv_results[variant] = {
            "selected_l2": selected["l2"],
            "selected_mean_task_macro_f1": selected["mean_task_macro_f1"],
            "selected_metrics": selected["metrics"],
            "grid": candidates,
        }
        for index, key in enumerate(targets.keys):
            oof_rows.append(
                {
                    "variant": variant,
                    "input_index": int(key[0]),
                    "chunk_index": int(key[1]),
                    "fold": int(folds[index]),
                    "gold": {name: int(targets.values[name][index]) for name in TARGETS},
                    "prediction": {
                        name: int(selected_predictions[name][index]) for name in TARGETS
                    },
                }
            )

    output_dir.mkdir(parents=True)
    model_path = output_dir / "model_bundle.json"
    cv_path_out = output_dir / "cv_results.json"
    oof_path = output_dir / "oof_predictions.jsonl"
    write_json(model_path, model_bundle)
    write_json(cv_path_out, cv_results)
    write_jsonl(oof_path, oof_rows)
    manifest = {
        "schema_version": 1,
        "status": "train-only CV and final refit complete; held-out annotations unread",
        "wall_seconds": time.monotonic() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "sources": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "train_annotations": {"path": str(annotations_path), "sha256": sha256_file(annotations_path)},
            "train_manifest": {"path": str(annotation_manifest_path), "sha256": sha256_file(annotation_manifest_path)},
            "cv_split": {"path": str(cv_path), "sha256": sha256_file(cv_path)},
        },
        "heldout_annotation_path_accepted_by_command": False,
        "heldout_annotations_read": False,
        "artifacts": {
            "model_bundle": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "cv_results": {"path": str(cv_path_out), "sha256": sha256_file(cv_path_out)},
            "oof_predictions": {"path": str(oof_path), "sha256": sha256_file(oof_path)},
        },
    }
    write_json(output_dir / "train_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
