"""Frozen feature construction and linear modeling for S1."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from proactive_d1.core import (
    LabeledChunk,
    TEMPORAL_FEATURES,
    build_label_free_chunks,
    feature_names,
    strip_answers,
    validate_fold_manifest,
)
from proactive_d1.neural_core import load_aligned_neural_cache, neural_matrix
from proactive_d3.core import build_causal_dynamics, d3_matrix
from proactive_r0.artifacts import sha256_file
from proactive_r0.core import load_jsonl
from proactive_state_s1.core import (
    PROGRESS_VALUES,
    STEP_IDS,
    load_json,
    validate_collection,
)


FEATURE_VARIANTS = ("temporal_only", "current_d1", "d3_dynamics")
TARGETS = ("step", "progress", "error")
TARGET_CLASSES = {
    "step": STEP_IDS,
    "progress": PROGRESS_VALUES,
    "error": (False, True),
}


@dataclass(frozen=True)
class FeatureBundle:
    keys: np.ndarray
    domains: tuple[str, ...]
    matrices: dict[str, np.ndarray]
    names: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class StateTargets:
    keys: np.ndarray
    input_index: np.ndarray
    domains: np.ndarray
    values: dict[str, np.ndarray]


def _check_hash(path: Path, expected: object) -> None:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"Frozen S1 source mismatch for {path}: {actual} != {expected}")


def build_feature_bundle(config: dict[str, object], root: Path) -> FeatureBundle:
    data = dict(config["data"])  # type: ignore[arg-type]
    r0 = dict(config["r0_reference"])  # type: ignore[arg-type]
    d1 = dict(config["d1_reference"])  # type: ignore[arg-type]
    source_path = (root / str(data["input"])).resolve()
    r0_path = (root / str(r0["records"])).resolve()
    split_path = (root / str(d1["split_manifest"])).resolve()
    cache_path = (root / str(d1["feature_cache"])).resolve()
    _check_hash(source_path, data["input_sha256"])
    _check_hash(r0_path, r0["records_sha256"])
    _check_hash(split_path, d1["split_manifest_sha256"])
    _check_hash(cache_path, d1["feature_cache_sha256"])
    source = load_jsonl(source_path)
    label_free_source = strip_answers(source)
    split = load_json(split_path)
    if not isinstance(split, dict):
        raise ValueError("D1 split manifest must be an object")
    fold_by_index = validate_fold_manifest(split, label_free_source)
    features = build_label_free_chunks(
        label_free_source,
        load_jsonl(r0_path),
        fold_by_index,
        int(d1["max_history_turns"]),
        int(d1["max_model_frames"]),
    )
    examples = [LabeledChunk(feature=value, gold_interrupt=0) for value in features]
    cache = load_aligned_neural_cache(cache_path, examples, int(d1["hidden_size"]))
    dynamics = build_causal_dynamics(cache)
    domains = tuple(sorted({value.domain for value in features}))
    scalar_names = feature_names(str(d1["scalar_variant"]), domains)  # type: ignore[arg-type]
    temporal = np.asarray(
        [[value.values[name] for name in TEMPORAL_FEATURES] for value in features],
        dtype=np.float64,
    )
    current, current_names = neural_matrix(
        examples, cache, scalar_names, "fused_linear"
    )
    dynamic, dynamic_names = d3_matrix(
        examples, cache, scalar_names, dynamics, "dynamics_fused"
    )
    matrices = {
        "temporal_only": temporal,
        "current_d1": current.astype(np.float64),
        "d3_dynamics": dynamic.astype(np.float64),
    }
    names = {
        "temporal_only": tuple(TEMPORAL_FEATURES),
        "current_d1": tuple(current_names),
        "d3_dynamics": tuple(dynamic_names),
    }
    keys = np.asarray([value.key for value in examples], dtype=np.int32)
    for variant in FEATURE_VARIANTS:
        if matrices[variant].shape != (len(keys), len(names[variant])):
            raise ValueError(f"S1 {variant} matrix/name mismatch")
        if not np.isfinite(matrices[variant]).all():
            raise ValueError(f"S1 {variant} contains non-finite values")
    return FeatureBundle(keys=keys, domains=domains, matrices=matrices, names=names)


def load_targets(
    annotations_path: Path,
    sessions_path: Path,
    split: str,
) -> StateTargets:
    sessions = load_jsonl(sessions_path)
    annotations = load_json(annotations_path)
    if not isinstance(annotations, list) or not all(
        isinstance(row, dict) for row in annotations
    ):
        raise ValueError("S1 annotations must be an array of objects")
    validate_collection(annotations, sessions, expected_split=split)
    session_by_index = {
        int(row["input_index"]): row
        for row in sessions if row.get("state_split") == split
    }
    keys: list[tuple[int, int]] = []
    input_indices: list[int] = []
    domains: list[str] = []
    target_values: dict[str, list[int]] = {name: [] for name in TARGETS}
    for row in annotations:
        input_index = int(row["input_index"])
        session = session_by_index[input_index]
        states = row["chunk_states"]
        assert isinstance(states, list)
        for state in states:
            assert isinstance(state, dict)
            chunk_index = int(state["chunk_index"])
            keys.append((input_index, chunk_index))
            input_indices.append(input_index)
            domains.append(str(session["domain"]))
            target_values["step"].append(STEP_IDS.index(str(state["current_step_id"])))
            target_values["progress"].append(
                PROGRESS_VALUES.index(str(state["progress"]))
            )
            target_values["error"].append(
                int(bool(state["incompletion_or_error_evidence"]))
            )
    return StateTargets(
        keys=np.asarray(keys, dtype=np.int32),
        input_index=np.asarray(input_indices, dtype=np.int32),
        domains=np.asarray(domains, dtype=str),
        values={name: np.asarray(value, dtype=np.int64) for name, value in target_values.items()},
    )


def align_matrix(bundle: FeatureBundle, targets: StateTargets, variant: str) -> np.ndarray:
    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"Unknown S1 feature variant {variant}")
    row_by_key = {tuple(map(int, key)): index for index, key in enumerate(bundle.keys)}
    indices: list[int] = []
    for key in targets.keys:
        normalized = tuple(map(int, key))
        if normalized not in row_by_key:
            raise ValueError(f"S1 target key is absent from frozen cache: {normalized}")
        indices.append(row_by_key[normalized])
    return bundle.matrices[variant][indices]


def classification_metrics(
    gold: np.ndarray, predicted: np.ndarray, class_count: int
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(
            f1_score(
                gold,
                predicted,
                labels=list(range(class_count)),
                average="macro",
                zero_division=0,
            )
        ),
    }


def fit_head(values: np.ndarray, labels: np.ndarray, l2: float) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression

    if l2 <= 0 or not math.isfinite(l2):
        raise ValueError("S1 L2 must be finite and positive")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    standardized = (values - mean) / scale
    model = LogisticRegression(
        C=1.0 / l2,
        solver="lbfgs",
        class_weight="balanced",
        fit_intercept=True,
        max_iter=2000,
        tol=1e-8,
        random_state=20260718,
    )
    model.fit(standardized, labels)
    if int(np.max(model.n_iter_)) >= 2000:
        raise RuntimeError("S1 LogisticRegression did not converge")
    return {
        "classes": model.classes_.astype(int).tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "n_iter": model.n_iter_.astype(int).tolist(),
    }


def predict_head(model: dict[str, object], values: np.ndarray) -> np.ndarray:
    classes = np.asarray(model["classes"], dtype=np.int64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    intercept = np.asarray(model["intercept"], dtype=np.float64)
    standardized = (values - mean) / scale
    logits = standardized @ coef.T + intercept
    if len(classes) == 2 and logits.shape[1] == 1:
        indices = (logits[:, 0] >= 0).astype(np.int64)
    else:
        indices = np.argmax(logits, axis=1)
    return classes[indices]


def composite_correctness(
    targets: StateTargets, predictions: dict[str, np.ndarray]
) -> float:
    correctness = np.stack(
        [predictions[name] == targets.values[name] for name in TARGETS], axis=1
    )
    return float(correctness.mean())


def validate_l2_grid(values: Sequence[object]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if result != (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        raise ValueError(f"S1 L2 grid differs from frozen protocol: {result}")
    return result
