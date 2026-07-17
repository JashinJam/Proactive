"""Run the frozen five-fold final-MLP adapter and fused-head OOF protocol."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shlex
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from proactive_d1.core import (
    attach_gold_labels,
    binary_metrics,
    build_label_free_chunks,
    feature_names,
    fit_linear_logistic,
    metrics_for_subset,
    paired_session_bootstrap,
    predict_logits,
    prediction_rows,
    select_threshold,
    strip_answers,
    validate_fold_manifest,
)
from proactive_d1.internvl_features import InternVLDecisionFeatureExtractor
from proactive_d1.neural_core import cross_validate_neural_matrix
from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    verify_model_snapshot,
    write_json,
)
from proactive_r0.core import INTERRUPT_TAG, load_jsonl, validate_prediction_rows, write_jsonl
from proactive_r0.run import _run_official_scorer, _validate_static_files

from .final_mlp_lora import configure_final_mlp_lora, internvl_lora_components
from .final_mlp_training import (
    export_adapter_features,
    load_final_mlp_cache_arrays,
    restore_trainable_state,
    train_adapter_fold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/d2_internvl35_1b_final_mlp_lora_oof.json"
VARIANTS: tuple[str, ...] = (
    "adapted_tag_only",
    "adapted_hidden_linear",
    "adapted_fused_linear",
)
LOGGER = logging.getLogger("proactive_d2.run_final_mlp_oof")
TEST_LABEL_SENTINEL = -1


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
        raise ValueError(f"Frozen D2 artifact mismatch for {path}: {actual} != {expected}")
    return actual


def _configure_logging(output_dir: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    _close_logging()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def _close_logging() -> None:
    for handler in LOGGER.handlers:
        handler.flush()
        handler.close()
    LOGGER.handlers.clear()


@contextmanager
def _atomic_output_directory(final_output_dir: Path) -> Iterator[Path]:
    """Keep a failed attempt diagnostic-only and publish success with one rename."""
    if final_output_dir.exists():
        raise FileExistsError(
            f"Final-MLP OOF output already exists: {final_output_dir}"
        )
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = final_output_dir.with_name(
        f"{final_output_dir.name}.incomplete-{os.getpid()}"
    )
    if staging_dir.exists():
        raise FileExistsError(
            f"Final-MLP OOF diagnostic output already exists: {staging_dir}"
        )
    staging_dir.mkdir()
    try:
        yield staging_dir
        _close_logging()
        if final_output_dir.exists():
            raise FileExistsError(
                f"Final-MLP OOF output appeared during the run: {final_output_dir}"
            )
        staging_dir.rename(final_output_dir)
    except BaseException as error:
        if LOGGER.handlers:
            LOGGER.error(
                "Final-MLP OOF failed; diagnostics remain at %s",
                staging_dir,
                exc_info=(type(error), error, error.__traceback__),
            )
        if staging_dir.exists():
            try:
                write_json(
                    staging_dir / "failure.json",
                    {
                        "status": "incomplete",
                        "failed_at": datetime.now().astimezone().isoformat(),
                        "pid": os.getpid(),
                        "final_output_dir": str(final_output_dir),
                        "diagnostic_output_dir": str(staging_dir),
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                    },
                )
            except Exception:
                pass
        _close_logging()
        raise


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join(
        [sys.executable, "-m", "proactive_d2.run_final_mlp_oof", *argv]
    )
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _tracked_paths(config_path: Path) -> list[Path]:
    return [
        *sorted((PROJECT_ROOT / "src/proactive_r0").glob("*.py")),
        *sorted((PROJECT_ROOT / "src/proactive_d1").glob("*.py")),
        *sorted((PROJECT_ROOT / "src/proactive_d2").glob("*.py")),
        *sorted((PROJECT_ROOT / "src/proactive_d2/tests").glob("*.py")),
        config_path,
        PROJECT_ROOT / "models/internvl35_1b_hf.json",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
        PROJECT_ROOT / "C1_SPEC.md",
        PROJECT_ROOT / "Agent.md",
    ]


def _validate_protocol_config(config: dict[str, object]) -> None:
    expected: dict[str, dict[str, object]] = {
        "cache": {
            "answers_removed_before_extraction": True,
        },
        "split": {
            "folds": 5,
            "fit_folds_per_rotation": 3,
            "calibration_folds_per_rotation": 1,
            "test_folds_per_rotation": 1,
            "calibration_fold_offset": 1,
            "unit": "session",
        },
        "training": {
            "primary_loss": (
                "class_balanced_bce_on_interrupt_minus_silent_full_tag_log_probability"
            ),
            "optimizer": "adamw",
            "pad_evaluation_batches_to_fixed_size": True,
            "include_zero_adapter_epoch": True,
            "early_stopping_metric": "class_balanced_calibration_bce",
            "log_softmax_dtype": "float32",
            "cache_compute_dtype": "bfloat16",
            "one_configuration_only": True,
        },
        "decision_head": {
            "primary_variant": "adapted_fused_linear",
            "classifier": "class_balanced_linear_logistic_regression",
            "optimizer": "lbfgs",
            "l2_reduction": "sum",
            "l2_selection": "calibration_fold_macro_f1",
            "threshold_selection": "exact_calibration_fold_macro_f1",
            "frozen_output_utterances": (
                "reuse the frozen R0 raw response; adapter is disabled for generation"
            ),
        },
        "controls": {
            "source_order_preserved": True,
            "test_labels_read_only_after_each_fold_predictions_are_frozen": True,
            "full_sequence_adapter_forward": (
                "diagnostic only; not used for training or deployment because BF16 "
                "batch-shape rounding differs"
            ),
            "utterance_generation_adapter_state": "disabled",
        },
        "evaluation": {
            "official_scorer": "data/starter_kit/run_evaluation.py",
            "primary_metric": "macro_f1",
        },
        "validation_policy": {
            "cache_generation_reads_labels": False,
            "adapter_fit_reads_fit_fold_public_validation_labels": True,
            "adapter_early_stopping_reads_only_calibration_fold_labels": True,
            "linear_head_fit_reads_fit_fold_labels": True,
            "l2_and_threshold_read_only_calibration_fold_labels": True,
            "test_fold_labels_used_only_after_predictions": True,
            "classification": (
                "single-configuration session-level OOF val-supervised "
                "representation-adaptation study; not hidden-test evidence"
            ),
        },
    }
    for section_name, expected_values in expected.items():
        section = config.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"Final-MLP protocol section is invalid: {section_name}")
        for key, expected_value in expected_values.items():
            actual_value = section.get(key)
            if actual_value != expected_value or type(actual_value) is not type(
                expected_value
            ):
                raise ValueError(
                    "Frozen final-MLP protocol changed: "
                    f"{section_name}.{key}={actual_value!r}, expected {expected_value!r}"
                )

    promotion = config["evaluation"].get("promotion")  # type: ignore[union-attr]
    if not isinstance(promotion, dict):
        raise ValueError("Final-MLP promotion protocol is invalid")
    for key in (
        "require_positive_session_bootstrap_lower_bound",
        "require_non_first_chunk_gain",
        "require_both_class_f1_nonzero",
    ):
        if type(promotion.get(key)) is not bool:
            raise ValueError(f"Final-MLP promotion flag must be boolean: {key}")


def _restricted_fold_labels(
    labels: np.ndarray,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
) -> np.ndarray:
    restricted = np.asarray(labels, dtype=np.int64).copy()
    restricted[test_indices] = TEST_LABEL_SENTINEL
    _assert_restricted_fold_labels(
        restricted, fit_indices, calibration_indices, test_indices
    )
    return restricted


def _assert_restricted_fold_labels(
    labels: np.ndarray,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("Fold labels must be a one-dimensional array")
    groups = {
        "fit": np.asarray(fit_indices, dtype=np.int64),
        "calibration": np.asarray(calibration_indices, dtype=np.int64),
        "test": np.asarray(test_indices, dtype=np.int64),
    }
    for name, indices in groups.items():
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"Final-MLP {name} indices must be non-empty and 1D")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"Final-MLP {name} indices contain duplicates")
        if int(indices.min()) < 0 or int(indices.max()) >= len(values):
            raise ValueError(f"Final-MLP {name} indices are out of range")
    if (
        np.intersect1d(groups["fit"], groups["calibration"]).size
        or np.intersect1d(groups["fit"], groups["test"]).size
        or np.intersect1d(groups["calibration"], groups["test"]).size
    ):
        raise ValueError("Final-MLP fit/calibration/test indices must be disjoint")
    covered = np.concatenate(tuple(groups.values()))
    if np.unique(covered).size != len(values):
        raise ValueError("Final-MLP fold indices must cover every label exactly once")
    for name in ("fit", "calibration"):
        if not np.isin(values[groups[name]], (0, 1)).all():
            raise ValueError(f"Final-MLP {name} labels must be binary")
    if not np.all(values[groups["test"]] == TEST_LABEL_SENTINEL):
        raise ValueError("Final-MLP test labels must remain sentinel-masked")


def _verified_d1_macro_f1(
    metrics: dict[str, object], configured_macro_f1: object
) -> float:
    overall = metrics.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("Verified D1 metrics have no overall object")
    value = overall.get("macro_f1")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Verified D1 Macro F1 is not numeric")
    macro_f1 = float(value)
    configured = float(configured_macro_f1)
    if not math.isfinite(macro_f1) or not 0.0 <= macro_f1 <= 1.0:
        raise ValueError("Verified D1 Macro F1 is outside [0, 1]")
    if not math.isclose(macro_f1, configured, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Configured D1 Macro F1 differs from verified metrics: {configured} != {macro_f1}"
        )
    return macro_f1


def _promotion_checks(
    *,
    candidate_macro_f1: float,
    d1_macro_f1: float,
    bootstrap_lower_bound: float,
    positive_folds: int,
    non_first_macro_f1: float,
    d1_non_first_macro_f1: float,
    interrupt_f1: float,
    silent_f1: float,
    config: dict[str, object],
) -> dict[str, bool]:
    for key in (
        "require_positive_session_bootstrap_lower_bound",
        "require_non_first_chunk_gain",
        "require_both_class_f1_nonzero",
    ):
        if type(config.get(key)) is not bool:
            raise ValueError(f"Final-MLP promotion flag must be boolean: {key}")
    return {
        "minimum_macro_delta": candidate_macro_f1 - d1_macro_f1
        >= float(config["min_delta_macro_f1_vs_d1"]),
        "positive_bootstrap_lower_bound": (
            bootstrap_lower_bound > 0
            if config["require_positive_session_bootstrap_lower_bound"]
            else True
        ),
        "minimum_positive_folds": positive_folds
        >= int(config["min_positive_folds"]),
        "non_first_chunk_gain": (
            non_first_macro_f1 > d1_non_first_macro_f1
            if config["require_non_first_chunk_gain"]
            else True
        ),
        "both_classes_nonzero": (
            interrupt_f1 > 0 and silent_f1 > 0
            if config["require_both_class_f1_nonzero"]
            else True
        ),
    }


def _prediction_decisions(
    predictions: Sequence[dict[str, object]],
) -> dict[tuple[int, int], int]:
    decisions: dict[tuple[int, int], int] = {}
    for input_index, row in enumerate(predictions):
        answers = row.get("answers")
        if not isinstance(answers, list):
            raise ValueError(f"Prediction row {input_index} has no answers")
        for chunk_index, answer in enumerate(answers):
            decisions[(input_index, chunk_index)] = int(
                str(answer).lstrip().startswith(INTERRUPT_TAG)
            )
    return decisions


def _variant_matrix(
    variant: str,
    scalar: np.ndarray,
    scalar_names: Sequence[str],
    margin: np.ndarray,
    hidden: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    hidden_names = tuple(f"hidden_{index:04d}" for index in range(hidden.shape[1]))
    if variant == "adapted_tag_only":
        return margin[:, None], ("adapted_tag_margin",)
    if variant == "adapted_hidden_linear":
        return hidden, hidden_names
    if variant == "adapted_fused_linear":
        return np.concatenate([scalar, margin[:, None], hidden], axis=1), (
            *scalar_names,
            "adapted_tag_margin",
            *hidden_names,
        )
    raise ValueError(f"Unknown final-MLP OOF variant: {variant}")


def _model_summary(names: Sequence[str], model: object) -> dict[str, object]:
    weight = np.asarray(model.weight, dtype=np.float64)  # type: ignore[attr-defined]
    ranked = np.argsort(np.abs(weight))[::-1][:20]
    return {
        "weight_l2_norm": float(np.linalg.norm(weight)),
        "weight_abs_max": float(np.abs(weight).max()),
        "bias": float(model.bias),  # type: ignore[attr-defined]
        "train_loss": float(model.train_loss),  # type: ignore[attr-defined]
        "top_standardized_coefficients": [
            {"name": names[index], "weight": float(weight[index])}
            for index in ranked
        ],
    }


def _fit_head_for_fold(
    values: np.ndarray,
    names: Sequence[str],
    labels: np.ndarray,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    l2_weights: Sequence[float],
    seed: int,
    max_iterations: int,
    test_fold: int,
) -> tuple[np.ndarray, dict[str, object]]:
    _assert_restricted_fold_labels(
        labels, fit_indices, calibration_indices, test_indices
    )
    candidates: list[tuple[float, object, float, dict[str, float | int]]] = []
    for grid_index, l2_weight in enumerate(l2_weights):
        model = fit_linear_logistic(
            values[fit_indices],
            labels[fit_indices],
            seed=seed + test_fold * 100 + grid_index,
            max_iterations=max_iterations,
            l2_weight=float(l2_weight),
            l2_reduction="sum",
        )
        calibration_logits = predict_logits(model, values[calibration_indices])
        threshold, metrics = select_threshold(
            calibration_logits, labels[calibration_indices].tolist()
        )
        candidates.append((float(l2_weight), model, threshold, metrics))
    selected_l2, selected_model, threshold, calibration_metrics = max(
        candidates,
        key=lambda item: (
            float(item[3]["macro_f1"]),
            -item[0],
            -abs(item[2]),
        ),
    )
    test_logits = np.asarray(
        predict_logits(selected_model, values[test_indices]), dtype=np.float64
    )
    test_predictions = np.asarray(test_logits >= threshold, dtype=np.int64)
    details = {
        "selected_l2_weight": selected_l2,
        "threshold_logit": threshold,
        "calibration_metrics": calibration_metrics,
        "calibration_grid": [
            {
                "l2_weight": l2_weight,
                "macro_f1": metrics["macro_f1"],
                "threshold_logit": candidate_threshold,
            }
            for l2_weight, _, candidate_threshold, metrics in candidates
        ],
        "feature_count": len(names),
        "head_parameters": len(names) + 1,
        "model": _model_summary(names, selected_model),
    }
    return test_predictions, details


def _adapter_initial_state(peft_model: object) -> dict[str, object]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    }


def _add_peft_version(environment: dict[str, object]) -> dict[str, object]:
    import peft

    result = json.loads(json.dumps(environment))
    packages = dict(result["packages"])  # type: ignore[arg-type]
    packages["peft"] = peft.__version__
    result["packages"] = packages
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    return parser.parse_args(argv)


def _run_main(
    argv: list[str], *, output_dir: Path, final_output_dir: Path
) -> None:
    import torch

    args = parse_args(argv)
    raw_argv = list(argv)
    started_at = time.monotonic()
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    _validate_protocol_config(config)
    model_config = dict(config["model"])  # type: ignore[arg-type]
    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    r0_reference = dict(config["r0_reference"])  # type: ignore[arg-type]
    d1_reference = dict(config["d1_reference"])  # type: ignore[arg-type]
    split_config = dict(config["split"])  # type: ignore[arg-type]
    cache_config = dict(config["cache"])  # type: ignore[arg-type]
    adapter_config = dict(config["adapter"])  # type: ignore[arg-type]
    training_config = dict(config["training"])  # type: ignore[arg-type]
    head_config = dict(config["decision_head"])  # type: ignore[arg-type]
    evaluation_config = dict(config["evaluation"])  # type: ignore[arg-type]
    controls_config = dict(config["controls"])  # type: ignore[arg-type]
    if int(training_config["batch_size"]) != int(
        training_config["local_replay_batch_size"]
    ):
        raise ValueError("Training and local replay batch shapes must be identical")
    if training_config.get("pad_evaluation_batches_to_fixed_size") is not True:
        raise ValueError("Final-MLP evaluation must pad to a fixed batch shape")
    if training_config.get("include_zero_adapter_epoch") is not True:
        raise ValueError("Final-MLP early stopping must include epoch zero")
    if str(head_config["primary_variant"]) != "adapted_fused_linear":
        raise ValueError("The frozen final-MLP primary variant changed")

    input_path = _resolve(data_config["input"])
    starter_dir = _resolve(starter_config["path"])
    model_path = _resolve(args.model_path or model_config["default_local_path"])
    r0_dir = _resolve(r0_reference["experiment_dir"])
    d1_dir = _resolve(d1_reference["experiment_dir"])
    split_path = _resolve(split_config["manifest"])
    if "path" not in cache_config or "features_sha256" not in cache_config:
        raise ValueError("Pin the completed merged final-MLP cache before OOF")
    cache_dir = _resolve(cache_config["path"])
    cache_path = cache_dir / "features.npz"
    requested_output_dir = _resolve(
        args.output_dir or f"output/experiments/{config['experiment_id']}"
    )
    if requested_output_dir != final_output_dir:
        raise RuntimeError("Final-MLP atomic output target changed during startup")
    if not output_dir.is_dir():
        raise RuntimeError("Final-MLP atomic staging directory is missing")
    _configure_logging(output_dir)

    fingerprints = _validate_static_files(config, input_path, starter_dir)
    model_audit = verify_model_snapshot(model_path, model_config)
    d1_metrics_path = d1_dir / "variants/fused_linear/metrics.json"
    artifact_hashes = {
        "r0_session_records_sha256": _check_hash(
            r0_dir / "session_records.jsonl", r0_reference["session_records_sha256"]
        ),
        "d1_predictions_sha256": _check_hash(
            d1_dir / "variants/fused_linear/predictions.jsonl",
            d1_reference["predictions_sha256"],
        ),
        "d1_metrics_sha256": _check_hash(
            d1_metrics_path,
            d1_reference["metrics_sha256"],
        ),
        "split_manifest_sha256": _check_hash(
            split_path, split_config["manifest_sha256"]
        ),
        "cache_features_sha256": _check_hash(
            cache_path, cache_config["features_sha256"]
        ),
    }
    if "records_sha256" in cache_config:
        artifact_hashes["cache_records_sha256"] = _check_hash(
            cache_dir / "records.jsonl", cache_config["records_sha256"]
        )
    if "summary_sha256" in cache_config:
        artifact_hashes["cache_summary_sha256"] = _check_hash(
            cache_dir / "summary.json", cache_config["summary_sha256"]
        )
    if "data_manifest_sha256" in cache_config:
        artifact_hashes["cache_data_manifest_sha256"] = _check_hash(
            cache_dir / "data_manifest.json", cache_config["data_manifest_sha256"]
        )
    d1_macro_f1 = _verified_d1_macro_f1(
        _load_json(d1_metrics_path), d1_reference["macro_f1"]
    )
    cache_summary = _load_json(cache_dir / "summary.json")
    if cache_summary.get("labels_read_or_stored") is not False:
        raise ValueError("Final-MLP OOF requires a label-free merged cache")
    if int(cache_summary["sessions"]) != 700 or int(cache_summary["chunks"]) != 9935:
        raise ValueError("Final-MLP cache coverage differs from 700/9935")

    torch.set_num_threads(int(head_config["torch_threads"]))
    source_rows = load_jsonl(input_path)
    r0_records = load_jsonl(r0_dir / "session_records.jsonl")
    manifest = _load_json(split_path)
    fold_by_index = validate_fold_manifest(manifest, strip_answers(source_rows))
    label_free = build_label_free_chunks(
        strip_answers(source_rows),
        r0_records,
        fold_by_index,
        max_history_turns=int(cache_config["max_history_turns"]),
        max_model_frames=int(cache_config["max_frames"]),
    )
    examples = attach_gold_labels(label_free, source_rows)
    rows = len(examples)
    labels = np.asarray([example.gold_interrupt for example in examples], dtype=np.int64)
    fold_values = np.asarray([example.feature.fold for example in examples], dtype=np.int64)
    expected_keys = np.asarray([example.key for example in examples], dtype=np.int32)
    cache = load_final_mlp_cache_arrays(
        cache_path,
        expected_sha256=str(cache_config["features_sha256"]),
        rows=rows,
        hidden_size=int(cache_config["hidden_size"]),
        tag_length=int(cache_config["tag_tokens_each"]),
    )
    actual_keys = np.stack([cache.input_index, cache.chunk_index], axis=1)
    if not np.array_equal(actual_keys, expected_keys):
        raise ValueError("Final-MLP cache does not align with D1 examples")
    domains = sorted({example.feature.domain for example in examples})
    scalar_names = feature_names("response_temporal", domains)
    scalar = np.asarray(
        [
            [example.feature.values[name] for name in scalar_names]
            for example in examples
        ],
        dtype=np.float32,
    )

    base_values, base_names = _variant_matrix(
        "adapted_fused_linear",
        scalar,
        scalar_names,
        cache.base_tag_margin,
        cache.base_hidden_state,
    )
    base_decisions, base_fold_details = cross_validate_neural_matrix(
        examples,
        base_values,
        base_names,
        folds=int(split_config["folds"]),
        calibration_fold_offset=int(split_config["calibration_fold_offset"]),
        seed=int(head_config["seed"]),
        max_iterations=int(head_config["max_iterations"]),
        l2_weights=[float(value) for value in head_config["l2_weights"]],  # type: ignore[index]
        l2_reduction="sum",
    )
    zero_predictions = prediction_rows(examples, base_decisions)
    validate_prediction_rows(source_rows, zero_predictions)
    zero_dir = output_dir / "zero_adapter_control"
    zero_dir.mkdir()
    zero_predictions_path = zero_dir / "predictions.jsonl"
    write_jsonl(zero_predictions_path, zero_predictions)
    zero_hash = sha256_file(zero_predictions_path)
    expected_zero_hash = str(controls_config["zero_adapter_must_reproduce_d1_predictions_sha256"])
    if zero_hash != expected_zero_hash:
        raise RuntimeError(f"Zero-adapter D1 prediction mismatch: {zero_hash}")
    write_json(
        zero_dir / "audit.json",
        {
            "predictions_sha256": zero_hash,
            "expected_predictions_sha256": expected_zero_hash,
            "exact_match": True,
            "fold_details": base_fold_details,
        },
    )

    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "model_path": str(model_path),
        "cache_path": str(cache_path),
        "output_dir": str(final_output_dir),
        "atomic_staging_dir": str(output_dir),
        "device": args.device,
        "audit_only": args.audit_only,
        "require_exclusive_gpu": args.require_exclusive_gpu,
    }
    write_json(output_dir / "config.json", effective)
    _write_command(output_dir / "command.sh", raw_argv)
    write_json(
        output_dir / "environment.txt",
        _add_peft_version(environment_snapshot()),
    )
    write_json(
        output_dir / "code_state.txt",
        code_snapshot(PROJECT_ROOT, _tracked_paths(config_path)),
    )
    write_json(
        output_dir / "data_manifest.json",
        {
            "source": {"path": str(input_path), "sha256": fingerprints["input_sha256"]},
            "model": {**model_audit, "path": str(model_path)},
            "starter_kit_sha256": fingerprints,
            "frozen_artifact_sha256": artifact_hashes,
            "cache_summary": cache_summary,
            "cache_labels_read_or_stored": False,
            "zero_adapter_control_predictions_sha256": zero_hash,
        },
    )
    if args.audit_only:
        write_json(
            output_dir / "runtime.json",
            {
                "status": "audit_only_zero_adapter_control_passed",
                "wall_time_seconds": round(time.monotonic() - started_at, 3),
                "gpu_used": False,
            },
        )
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_started = time.monotonic()
    extractor = InternVLDecisionFeatureExtractor(
        model_path=str(model_path),
        device=args.device,
        dtype_name=str(model_config["dtype"]),
        attention_implementation=str(model_config["attention_implementation"]),
        seed=int(training_config["seed"]),
        require_exclusive_gpu=args.require_exclusive_gpu,
        video_frame_size=int(cache_config["video_frame_size"]),
        pad_token_id=int(cache_config["pad_token_id"]),
    )
    peft_model, parameter_audit = configure_final_mlp_lora(
        extractor.model,
        layer_index=int(adapter_config["language_layer_index"]),
        hidden_size=int(cache_config["hidden_size"]),
        intermediate_size=int(model_config["intermediate_size"]),
        rank=int(adapter_config["rank"]),
        alpha=int(adapter_config["alpha"]),
        dropout=float(adapter_config["dropout"]),
    )
    outer, decoder_layer, final_norm, lm_head = internvl_lora_components(
        peft_model, int(adapter_config["language_layer_index"])
    )
    del outer
    if parameter_audit.trainable_parameters != int(adapter_config["trainable_parameters"]):
        raise RuntimeError("Final-MLP trainable parameter count differs from protocol")
    initial_state = _adapter_initial_state(peft_model)
    all_indices = np.arange(rows, dtype=np.int64)
    replay_batch_size = int(training_config["local_replay_batch_size"])
    zero_margin, zero_hidden, zero_candidate_difference = export_adapter_features(
        cache,
        all_indices,
        batch_size=replay_batch_size,
        peft_model=peft_model,
        decoder_layer=decoder_layer,
        final_norm=final_norm,
        lm_head=lm_head,
        silent_token_ids=extractor.silent_token_ids,
        interrupt_token_ids=extractor.interrupt_token_ids,
        device=device,
    )
    zero_margin_difference = float(np.max(np.abs(zero_margin - cache.base_tag_margin)))
    zero_hidden_difference = float(np.max(np.abs(zero_hidden - cache.base_hidden_state)))
    if (
        zero_margin_difference != 0.0
        or zero_hidden_difference != 0.0
        or zero_candidate_difference != 0.0
    ):
        raise RuntimeError(
            "PEFT zero-adapter replay differs from the base cache: "
            f"margin={zero_margin_difference}, hidden={zero_hidden_difference}"
        )
    zero_replay_audit = {
        "rows": rows,
        "fixed_replay_batch_size": replay_batch_size,
        "max_margin_abs_difference": zero_margin_difference,
        "max_hidden_abs_difference": zero_hidden_difference,
        "max_candidate_hidden_abs_difference": zero_candidate_difference,
        "exact": True,
    }
    write_json(zero_dir / "peft_replay_audit.json", zero_replay_audit)
    model_load_and_zero_audit_seconds = time.monotonic() - model_started

    decisions_by_variant: dict[str, dict[tuple[int, int], int]] = {
        variant: {} for variant in VARIANTS
    }
    fold_details: list[dict[str, object]] = []
    oof_margin = np.empty(rows, dtype=np.float32)
    oof_hidden_norm = np.empty(rows, dtype=np.float32)
    folds_dir = output_dir / "folds"
    folds_dir.mkdir()
    for test_fold in range(int(split_config["folds"])):
        fold_started = time.monotonic()
        calibration_fold = (
            test_fold + int(split_config["calibration_fold_offset"])
        ) % int(split_config["folds"])
        fit_indices = np.flatnonzero(
            (fold_values != test_fold) & (fold_values != calibration_fold)
        )
        calibration_indices = np.flatnonzero(fold_values == calibration_fold)
        test_indices = np.flatnonzero(fold_values == test_fold)
        fold_labels = _restricted_fold_labels(
            labels, fit_indices, calibration_indices, test_indices
        )
        LOGGER.info(
            "Fold %d start: fit=%d calibration=%d test=%d",
            test_fold,
            len(fit_indices),
            len(calibration_indices),
            len(test_indices),
        )
        training_result = train_adapter_fold(
            cache,
            fold_labels,
            fit_indices,
            calibration_indices,
            peft_model=peft_model,
            initial_state=initial_state,
            decoder_layer=decoder_layer,
            final_norm=final_norm,
            lm_head=lm_head,
            silent_token_ids=extractor.silent_token_ids,
            interrupt_token_ids=extractor.interrupt_token_ids,
            device=device,
            learning_rate=float(training_config["learning_rate"]),
            weight_decay=float(training_config["weight_decay"]),
            batch_size=int(training_config["batch_size"]),
            max_epochs=int(training_config["max_epochs"]),
            patience=int(training_config["early_stopping_patience"]),
            min_delta=float(training_config["early_stopping_min_delta"]),
            gradient_clip_norm=float(training_config["gradient_clip_norm"]),
            seed=int(training_config["seed"]) + test_fold * 1000,
        )
        _assert_restricted_fold_labels(
            fold_labels, fit_indices, calibration_indices, test_indices
        )
        restore_trainable_state(peft_model, training_result.selected_state)
        adapted_margin, adapted_hidden, candidate_difference = export_adapter_features(
            cache,
            all_indices,
            batch_size=replay_batch_size,
            peft_model=peft_model,
            decoder_layer=decoder_layer,
            final_norm=final_norm,
            lm_head=lm_head,
            silent_token_ids=extractor.silent_token_ids,
            interrupt_token_ids=extractor.interrupt_token_ids,
            device=device,
        )
        if candidate_difference != 0.0:
            raise RuntimeError(f"Fold {test_fold} adapted causal hidden depends on tag")
        oof_margin[test_indices] = adapted_margin[test_indices]
        oof_hidden_norm[test_indices] = np.linalg.norm(
            adapted_hidden[test_indices], axis=1
        )
        fold_dir = folds_dir / f"fold_{test_fold}"
        fold_dir.mkdir()
        peft_model.save_pretrained(fold_dir / "adapter", safe_serialization=True)
        np.savez_compressed(
            fold_dir / "adapted_features.npz",
            tag_margin=adapted_margin,
            hidden_state=adapted_hidden,
            input_index=cache.input_index,
            chunk_index=cache.chunk_index,
        )
        head_details: dict[str, object] = {}
        for variant in VARIANTS:
            values, names = _variant_matrix(
                variant,
                scalar,
                scalar_names,
                adapted_margin,
                adapted_hidden,
            )
            test_predictions, details = _fit_head_for_fold(
                values,
                names,
                fold_labels,
                fit_indices,
                calibration_indices,
                test_indices,
                l2_weights=[float(value) for value in head_config["l2_weights"]],  # type: ignore[index]
                seed=int(head_config["seed"]),
                max_iterations=int(head_config["max_iterations"]),
                test_fold=test_fold,
            )
            for row_index, decision in zip(test_indices, test_predictions):
                key = examples[int(row_index)].key
                if key in decisions_by_variant[variant]:
                    raise RuntimeError(f"Duplicate OOF decision for {variant}/{key}")
                decisions_by_variant[variant][key] = int(decision)
            head_details[variant] = details
        fold_payload = {
            "test_fold": test_fold,
            "calibration_fold": calibration_fold,
            "fit_folds": sorted(
                set(range(int(split_config["folds"])))
                - {test_fold, calibration_fold}
            ),
            "fit_chunks": len(fit_indices),
            "calibration_chunks": len(calibration_indices),
            "test_chunks": len(test_indices),
            "adapter": {
                "best_epoch": training_result.best_epoch,
                "epochs_run": training_result.epochs_run,
                "zero_adapter_calibration_loss": (
                    training_result.zero_adapter_calibration_loss
                ),
                "best_calibration_loss": training_result.best_calibration_loss,
                "history": list(training_result.history),
                "max_candidate_hidden_abs_difference": candidate_difference,
                "adapter_weights_sha256": sha256_file(
                    fold_dir / "adapter/adapter_model.safetensors"
                ),
                "adapted_features_sha256": sha256_file(
                    fold_dir / "adapted_features.npz"
                ),
            },
            "heads": head_details,
            "wall_time_seconds": round(time.monotonic() - fold_started, 3),
        }
        write_json(fold_dir / "fold.json", fold_payload)
        fold_details.append(fold_payload)
        LOGGER.info(
            "Fold %d complete: best_epoch=%d calibration_bce=%.6f elapsed=%.2fs",
            test_fold,
            training_result.best_epoch,
            training_result.best_calibration_loss,
            time.monotonic() - fold_started,
        )

    expected_decision_keys = {example.key for example in examples}
    if any(set(decisions) != expected_decision_keys for decisions in decisions_by_variant.values()):
        raise RuntimeError("Final-MLP OOF decisions do not cover every chunk")
    d1_predictions = load_jsonl(d1_dir / "variants/fused_linear/predictions.jsonl")
    d1_decisions = _prediction_decisions(d1_predictions)
    comparison_variants: dict[str, object] = {}
    for variant, decisions in decisions_by_variant.items():
        variant_dir = output_dir / "variants" / variant
        variant_dir.mkdir(parents=True)
        predictions = prediction_rows(examples, decisions)
        validation = validate_prediction_rows(source_rows, predictions)
        predictions_path = variant_dir / "predictions.jsonl"
        write_jsonl(predictions_path, predictions)
        metrics_path = variant_dir / "metrics.json"
        _run_official_scorer(
            starter_dir,
            input_path,
            predictions_path,
            metrics_path,
            variant_dir / "scorer.log",
        )
        metrics = _load_json(metrics_path)
        overall = dict(metrics["overall"])  # type: ignore[arg-type]
        internal = metrics_for_subset(examples, decisions, include_first=True)
        non_first = metrics_for_subset(examples, decisions, include_first=False)
        d1_non_first = metrics_for_subset(examples, d1_decisions, include_first=False)
        bootstrap = paired_session_bootstrap(
            examples,
            decisions,
            d1_decisions,
            repetitions=int(evaluation_config["bootstrap_repetitions"]),
            seed=int(evaluation_config["bootstrap_seed"]),
        )
        per_fold: list[dict[str, object]] = []
        positive_folds = 0
        for fold in range(int(split_config["folds"])):
            selected = [example for example in examples if example.feature.fold == fold]
            candidate_metrics = binary_metrics(
                [example.gold_interrupt for example in selected],
                [decisions[example.key] for example in selected],
            )
            baseline_metrics = binary_metrics(
                [example.gold_interrupt for example in selected],
                [d1_decisions[example.key] for example in selected],
            )
            delta = float(candidate_metrics["macro_f1"]) - float(
                baseline_metrics["macro_f1"]
            )
            positive_folds += int(delta > 0)
            per_fold.append(
                {
                    "fold": fold,
                    "candidate_macro_f1": candidate_metrics["macro_f1"],
                    "d1_macro_f1": baseline_metrics["macro_f1"],
                    "delta_macro_f1": delta,
                }
            )
        delta_macro = float(overall["macro_f1"]) - d1_macro_f1
        promotion_config = dict(evaluation_config["promotion"])  # type: ignore[arg-type]
        promotion_checks = _promotion_checks(
            candidate_macro_f1=float(overall["macro_f1"]),
            d1_macro_f1=d1_macro_f1,
            bootstrap_lower_bound=float(bootstrap["delta_macro_f1_p2_5"]),
            positive_folds=positive_folds,
            non_first_macro_f1=float(non_first["macro_f1"]),
            d1_non_first_macro_f1=float(d1_non_first["macro_f1"]),
            interrupt_f1=float(overall["interrupt_f1"]),
            silent_f1=float(overall["silent_f1"]),
            config=promotion_config,
        )
        diagnostics = {
            **validation,
            "variant": variant,
            "internal_oof_metrics": internal,
            "non_first_chunk_metrics": non_first,
            "d1_non_first_chunk_metrics": d1_non_first,
            "paired_session_bootstrap_vs_d1": bootstrap,
            "per_fold": per_fold,
            "positive_folds": positive_folds,
            "promotion_checks": promotion_checks,
            "promotion_gate_passed": all(promotion_checks.values()),
            "predictions_sha256": sha256_file(predictions_path),
        }
        write_json(variant_dir / "diagnostics.json", diagnostics)
        write_jsonl(
            variant_dir / "oof_records.jsonl",
            [
                {
                    "input_index": example.feature.input_index,
                    "video_path": example.feature.video_path,
                    "fold": example.feature.fold,
                    "chunk_index": example.feature.chunk_index,
                    "gold_interrupt": example.gold_interrupt,
                    "predicted_interrupt": decisions[example.key],
                    "d1_interrupt": d1_decisions[example.key],
                    "adapted_tag_margin": float(oof_margin[row_index]),
                    "adapted_hidden_l2_norm": float(oof_hidden_norm[row_index]),
                }
                for row_index, example in enumerate(examples)
            ],
        )
        comparison_variants[variant] = {
            "macro_f1": overall["macro_f1"],
            "interrupt_f1": overall["interrupt_f1"],
            "silent_f1": overall["silent_f1"],
            "delta_macro_f1_vs_d1": delta_macro,
            "non_first_chunk_macro_f1": non_first["macro_f1"],
            "d1_non_first_chunk_macro_f1": d1_non_first["macro_f1"],
            "positive_folds": positive_folds,
            "bootstrap_vs_d1": bootstrap,
            "promotion_checks": promotion_checks,
            "promotion_gate_passed": all(promotion_checks.values()),
        }
        LOGGER.info(
            "%s official Macro F1=%s delta_vs_d1=%.6f promotion=%s",
            variant,
            overall["macro_f1"],
            delta_macro,
            all(promotion_checks.values()),
        )

    comparison = {
        "classification": config["validation_policy"],
        "d1_reference_macro_f1": d1_macro_f1,
        "primary_variant": head_config["primary_variant"],
        "variants": comparison_variants,
        "fold_adapter_best_epochs": [
            details["adapter"]["best_epoch"] for details in fold_details  # type: ignore[index]
        ],
        "zero_adapter_controls": {
            "d1_predictions_sha256_exact": True,
            "peft_replay": zero_replay_audit,
        },
        "protocol_changed_after_oof_metric": False,
    }
    write_json(output_dir / "comparison.json", comparison)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    runtime = {
        "status": "complete final-MLP adapter OOF",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started_at, 3),
        "model_load_and_zero_audit_seconds": round(
            model_load_and_zero_audit_seconds, 3
        ),
        "device": args.device,
        "peak_gpu_memory_bytes": peak_memory,
        "preexisting_gpu_processes": extractor.preexisting_gpu_processes,
        "trainable_parameters": parameter_audit.trainable_parameters,
        "total_system_parameters": head_config["total_system_parameters"],
        "sessions": 700,
        "chunks": rows,
    }
    write_json(output_dir / "runtime.json", runtime)
    primary = comparison_variants[str(head_config["primary_variant"])]
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "Status: **complete session-level val-supervised OOF**",
                "",
                f"- Primary Macro F1: `{primary['macro_f1']}`",
                f"- Delta versus D1: `{primary['delta_macro_f1_vs_d1']}`",
                f"- Promotion gate passed: `{primary['promotion_gate_passed']}`",
                f"- Trainable adapter parameters: `{parameter_audit.trainable_parameters}`",
                "- Hidden test claim: `False`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(comparison, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    config = _load_json(_resolve(args.config))
    final_output_dir = _resolve(
        args.output_dir or f"output/experiments/{config['experiment_id']}"
    )
    with _atomic_output_directory(final_output_dir) as staging_dir:
        _run_main(
            raw_argv,
            output_dir=staging_dir,
            final_output_dir=final_output_dir,
        )


if __name__ == "__main__":
    main()
