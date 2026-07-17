"""Deterministic residual MLP training and serialization for D2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from proactive_d1.core import LinearModel


@dataclass(frozen=True)
class ResidualMLPModel:
    base: LinearModel
    hidden_weight: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weight: tuple[float, ...]
    output_bias: float
    fit_loss: float
    calibration_loss: float
    best_epoch: int
    epochs_run: int


@dataclass(frozen=True)
class ResidualDecisionHead:
    feature_names: tuple[str, ...]
    model: ResidualMLPModel
    threshold_logit: float


def residual_parameter_count(input_dim: int, hidden_width: int) -> int:
    if input_dim <= 0 or hidden_width <= 0:
        raise ValueError("Residual dimensions must be positive")
    return input_dim * hidden_width + hidden_width + hidden_width + 1


def _matrix(value: Sequence[Sequence[float]], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _labels(value: Sequence[int], rows: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (rows,) or set(result.tolist()) != {0.0, 1.0}:
        raise ValueError(f"{name} must align and contain both binary classes")
    return result


def _normalized(values: np.ndarray, base: LinearModel) -> np.ndarray:
    if values.shape[1] != len(base.weight):
        raise ValueError("Residual input dimension differs from the D1 base")
    mean = np.asarray(base.mean, dtype=np.float32)
    scale = np.asarray(base.scale, dtype=np.float32)
    if mean.shape != (values.shape[1],) or scale.shape != mean.shape:
        raise ValueError("D1 base normalization arrays are misaligned")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("D1 base normalization is invalid")
    return (values - mean) / scale


def _base_logits(values: np.ndarray, base: LinearModel) -> np.ndarray:
    normalized = _normalized(values, base).astype(np.float64)
    weight = np.asarray(base.weight, dtype=np.float64)
    return normalized @ weight + float(base.bias)


def fit_residual_mlp(
    fit_values: Sequence[Sequence[float]],
    fit_labels: Sequence[int],
    calibration_values: Sequence[Sequence[float]],
    calibration_labels: Sequence[int],
    base: LinearModel,
    *,
    hidden_width: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    gradient_clip_norm: float,
    seed: int,
) -> tuple[ResidualMLPModel, list[dict[str, float | int]]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    fit_x = _matrix(fit_values, "fit_values")
    calibration_x = _matrix(calibration_values, "calibration_values")
    if fit_x.shape[1] != calibration_x.shape[1]:
        raise ValueError("Fit and calibration feature dimensions differ")
    fit_y = _labels(fit_labels, fit_x.shape[0], "fit_labels")
    calibration_y = _labels(
        calibration_labels, calibration_x.shape[0], "calibration_labels"
    )
    if hidden_width <= 0 or batch_size <= 0 or max_epochs <= 0 or patience <= 0:
        raise ValueError("Residual training dimensions and durations must be positive")
    numeric = (
        learning_rate,
        weight_decay,
        min_delta,
        gradient_clip_norm,
    )
    if any(not math.isfinite(value) or value < 0 for value in numeric):
        raise ValueError("Residual optimizer values must be finite and non-negative")
    if learning_rate == 0 or gradient_clip_norm == 0:
        raise ValueError("Residual learning rate and gradient clipping must be positive")

    fit_normalized = torch.from_numpy(_normalized(fit_x, base))
    calibration_normalized = torch.from_numpy(_normalized(calibration_x, base))
    fit_base = torch.from_numpy(_base_logits(fit_x, base).astype(np.float32))
    calibration_base = torch.from_numpy(
        _base_logits(calibration_x, base).astype(np.float32)
    )
    fit_targets = torch.from_numpy(fit_y)
    calibration_targets = torch.from_numpy(calibration_y)
    positive_weight = torch.tensor(
        float((fit_y == 0).sum() / (fit_y == 1).sum()), dtype=torch.float32
    )

    class ResidualModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = nn.Linear(fit_x.shape[1], hidden_width)
            self.output = nn.Linear(hidden_width, 1)
            nn.init.xavier_uniform_(self.hidden.weight)
            nn.init.zeros_(self.hidden.bias)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.output(functional.gelu(self.hidden(inputs))).squeeze(-1)

    torch.manual_seed(seed)
    module = ResidualModule()
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)

    def balanced_loss(
        logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=positive_weight
        )

    def calibration_loss() -> float:
        module.eval()
        with torch.no_grad():
            logits = calibration_base + module(calibration_normalized)
            return float(balanced_loss(logits, calibration_targets))

    best_loss = calibration_loss()
    best_epoch = 0
    best_state = {
        name: value.detach().clone() for name, value in module.state_dict().items()
    }
    history: list[dict[str, float | int]] = [
        {"epoch": 0, "calibration_loss": best_loss}
    ]
    stale_epochs = 0
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        module.train()
        permutation = torch.randperm(fit_x.shape[0], generator=generator)
        epoch_loss_sum = 0.0
        epoch_rows = 0
        for start in range(0, fit_x.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = fit_base[indices] + module(fit_normalized[indices])
            loss = balanced_loss(logits, fit_targets[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), gradient_clip_norm)
            optimizer.step()
            batch_rows = int(indices.numel())
            epoch_loss_sum += float(loss.detach()) * batch_rows
            epoch_rows += batch_rows
        current_calibration = calibration_loss()
        history.append(
            {
                "epoch": epoch,
                "fit_loss": epoch_loss_sum / epoch_rows,
                "calibration_loss": current_calibration,
            }
        )
        epochs_run = epoch
        if current_calibration < best_loss - min_delta:
            best_loss = current_calibration
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in module.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    module.load_state_dict(best_state)
    module.eval()
    with torch.no_grad():
        best_fit_loss = float(
            balanced_loss(
                fit_base + module(fit_normalized), fit_targets
            )
        )
    hidden_weight = tuple(
        tuple(float(value) for value in row)
        for row in module.hidden.weight.detach().cpu().tolist()
    )
    hidden_bias = tuple(
        float(value) for value in module.hidden.bias.detach().cpu().tolist()
    )
    output_weight = tuple(
        float(value)
        for value in module.output.weight.detach().cpu().reshape(-1).tolist()
    )
    model = ResidualMLPModel(
        base=base,
        hidden_weight=hidden_weight,
        hidden_bias=hidden_bias,
        output_weight=output_weight,
        output_bias=float(module.output.bias.detach().cpu()),
        fit_loss=best_fit_loss,
        calibration_loss=best_loss,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
    )
    return model, history


def predict_residual_logits(
    model: ResidualMLPModel,
    values: Sequence[Sequence[float]],
) -> list[float]:
    import torch
    import torch.nn.functional as functional

    matrix = _matrix(values, "values")
    normalized = torch.from_numpy(_normalized(matrix, model.base))
    hidden_weight = torch.tensor(model.hidden_weight, dtype=torch.float32)
    hidden_bias = torch.tensor(model.hidden_bias, dtype=torch.float32)
    output_weight = torch.tensor(model.output_weight, dtype=torch.float32)
    with torch.no_grad():
        hidden = functional.gelu(normalized @ hidden_weight.T + hidden_bias)
        residual = hidden @ output_weight + float(model.output_bias)
    base = _base_logits(matrix, model.base)
    return [float(value) for value in base + residual.numpy().astype(np.float64)]


def serialize_residual_head(
    head: ResidualDecisionHead,
    metadata: dict[str, object],
) -> dict[str, object]:
    model = head.model
    input_dim = len(head.feature_names)
    hidden_width = len(model.hidden_weight)
    if input_dim == 0 or len(set(head.feature_names)) != input_dim:
        raise ValueError("Residual feature names must be non-empty and unique")
    if len(model.base.weight) != input_dim:
        raise ValueError("Residual feature names and D1 base are misaligned")
    if any(len(row) != input_dim for row in model.hidden_weight):
        raise ValueError("Residual hidden matrix is misaligned")
    if len(model.hidden_bias) != hidden_width or len(model.output_weight) != hidden_width:
        raise ValueError("Residual hidden/output arrays are misaligned")
    return {
        "schema_version": 1,
        "head_type": "standardized_linear_plus_gelu_residual_mlp",
        "feature_names": list(head.feature_names),
        "base": {
            "mean": list(model.base.mean),
            "scale": list(model.base.scale),
            "weight": list(model.base.weight),
            "bias": model.base.bias,
            "train_loss": model.base.train_loss,
        },
        "residual": {
            "hidden_weight": [list(row) for row in model.hidden_weight],
            "hidden_bias": list(model.hidden_bias),
            "output_weight": list(model.output_weight),
            "output_bias": model.output_bias,
            "fit_loss": model.fit_loss,
            "calibration_loss": model.calibration_loss,
            "best_epoch": model.best_epoch,
            "epochs_run": model.epochs_run,
        },
        "threshold_logit": head.threshold_logit,
        "metadata": metadata,
    }


def load_residual_head(payload: dict[str, object]) -> ResidualDecisionHead:
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported residual-head schema")
    if payload.get("head_type") != "standardized_linear_plus_gelu_residual_mlp":
        raise ValueError("Unsupported residual-head type")
    names_value = payload.get("feature_names")
    base_value = payload.get("base")
    residual_value = payload.get("residual")
    if not isinstance(names_value, list) or not isinstance(base_value, dict):
        raise ValueError("Residual head lacks feature names or base")
    if not isinstance(residual_value, dict):
        raise ValueError("Residual head lacks residual arrays")
    names = tuple(str(value) for value in names_value)
    base = LinearModel(
        mean=tuple(float(value) for value in base_value["mean"]),  # type: ignore[index]
        scale=tuple(float(value) for value in base_value["scale"]),  # type: ignore[index]
        weight=tuple(float(value) for value in base_value["weight"]),  # type: ignore[index]
        bias=float(base_value["bias"]),
        train_loss=float(base_value["train_loss"]),
    )
    model = ResidualMLPModel(
        base=base,
        hidden_weight=tuple(
            tuple(float(value) for value in row)
            for row in residual_value["hidden_weight"]  # type: ignore[index]
        ),
        hidden_bias=tuple(
            float(value) for value in residual_value["hidden_bias"]  # type: ignore[index]
        ),
        output_weight=tuple(
            float(value) for value in residual_value["output_weight"]  # type: ignore[index]
        ),
        output_bias=float(residual_value["output_bias"]),
        fit_loss=float(residual_value["fit_loss"]),
        calibration_loss=float(residual_value["calibration_loss"]),
        best_epoch=int(residual_value["best_epoch"]),
        epochs_run=int(residual_value["epochs_run"]),
    )
    head = ResidualDecisionHead(
        feature_names=names,
        model=model,
        threshold_logit=float(payload["threshold_logit"]),
    )
    # Reuse serializer validation and additionally reject non-finite payloads.
    serialize_residual_head(head, {})
    numeric = (
        *base.mean,
        *base.scale,
        *base.weight,
        base.bias,
        *(value for row in model.hidden_weight for value in row),
        *model.hidden_bias,
        *model.output_weight,
        model.output_bias,
        head.threshold_logit,
    )
    if not all(math.isfinite(value) for value in numeric) or any(
        value <= 0 for value in base.scale
    ):
        raise ValueError("Residual head contains invalid numeric state")
    return head
