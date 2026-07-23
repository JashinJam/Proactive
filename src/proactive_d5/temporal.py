"""Fixed causal visual GRU residual and session-level OOF training."""

from __future__ import annotations

import copy
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from proactive_d1.core import (
    LabeledChunk,
    binary_metrics,
    fit_linear_logistic,
    predict_logits,
    select_threshold,
)


class SingleBiasGRUCell(nn.Module):
    """Standard reset/update/new GRU equations with one shared gate bias."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.weight_ih = nn.Parameter(torch.empty(3 * width, width))
        self.weight_hh = nn.Parameter(torch.empty(3 * width, width))
        self.bias = nn.Parameter(torch.empty(3 * width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.width**-0.5
        nn.init.uniform_(self.weight_ih, -bound, bound)
        nn.init.uniform_(self.weight_hh, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, value: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        input_gates = nn.functional.linear(value, self.weight_ih, self.bias)
        hidden_gates = nn.functional.linear(hidden, self.weight_hh)
        input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        new = torch.tanh(input_new + reset * hidden_new)
        return (1.0 - update) * new + update * hidden


class VisualTemporalResidual(nn.Module):
    def __init__(self, input_width: int = 1024, hidden_width: int = 32) -> None:
        super().__init__()
        self.projection = nn.Linear(input_width, hidden_width)
        self.gru = SingleBiasGRUCell(hidden_width)
        self.output = nn.Linear(hidden_width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2 or sequence.shape[0] == 0:
            raise ValueError("D5 temporal input must be a non-empty [chunks, width] tensor")
        projected = self.projection(sequence)
        hidden = torch.zeros(
            projected.shape[1], dtype=projected.dtype, device=projected.device
        )
        outputs = []
        for value in projected:
            hidden = self.gru(value, hidden)
            outputs.append(self.output(hidden).squeeze(-1))
        return torch.stack(outputs)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _session_indices(examples: Sequence[LabeledChunk]) -> dict[int, np.ndarray]:
    grouped: dict[int, list[int]] = {}
    for index, example in enumerate(examples):
        grouped.setdefault(example.feature.input_index, []).append(index)
    result: dict[int, np.ndarray] = {}
    for input_index, indices in grouped.items():
        if [examples[index].feature.chunk_index for index in indices] != list(range(len(indices))):
            raise ValueError(f"D5 temporal chunks are not contiguous for session {input_index}")
        result[input_index] = np.asarray(indices, dtype=np.int64)
    return result


def _split_sequences(
    model: VisualTemporalResidual,
    vision: torch.Tensor,
    base_logits: torch.Tensor,
    sessions: Mapping[int, np.ndarray],
    selected_indices: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray]:
    selected = set(int(index) for index in selected_indices)
    parts: list[torch.Tensor] = []
    order: list[int] = []
    for input_index in sorted(sessions):
        indices = sessions[input_index]
        membership = [int(index) in selected for index in indices]
        if any(membership) and not all(membership):
            raise ValueError("D5 temporal split divides a session")
        if not any(membership):
            continue
        tensor_indices = torch.as_tensor(indices, dtype=torch.long, device=vision.device)
        parts.append(base_logits[tensor_indices] + model(vision[tensor_indices]))
        order.extend(int(index) for index in indices)
    if set(order) != selected:
        raise ValueError("D5 temporal split does not cover its expected chunks")
    return torch.cat(parts), np.asarray(order, dtype=np.int64)


def temporal_residual_oof(
    examples: Sequence[LabeledChunk],
    base_values: np.ndarray,
    vision_values: np.ndarray,
    *,
    folds: int,
    calibration_fold_offset: int,
    base_config: Mapping[str, object],
    temporal_config: Mapping[str, object],
    device: str,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], float], list[dict[str, object]]]:
    if base_values.shape[0] != len(examples) or vision_values.shape != (len(examples), 1024):
        raise ValueError("D5 temporal base/vision matrices are not aligned")
    if not np.isfinite(base_values).all() or not np.isfinite(vision_values).all():
        raise ValueError("D5 temporal matrices contain non-finite values")
    labels = np.asarray([example.gold_interrupt for example in examples], dtype=np.int64)
    fold_values = np.asarray([example.feature.fold for example in examples], dtype=np.int64)
    sessions = _session_indices(examples)
    decisions: dict[tuple[int, int], int] = {}
    oof_logits: dict[tuple[int, int], float] = {}
    details: list[dict[str, object]] = []
    torch_device = torch.device(device)
    vision = torch.as_tensor(vision_values, dtype=torch.float32, device=torch_device)
    label_tensor = torch.as_tensor(labels, dtype=torch.float32, device=torch_device)

    for test_fold in range(folds):
        calibration_fold = (test_fold + calibration_fold_offset) % folds
        fit_indices = np.flatnonzero((fold_values != test_fold) & (fold_values != calibration_fold))
        calibration_indices = np.flatnonzero(fold_values == calibration_fold)
        test_indices = np.flatnonzero(fold_values == test_fold)
        base_candidates = []
        for grid_index, l2_weight in enumerate(base_config["l2_weights"]):  # type: ignore[index]
            base_model = fit_linear_logistic(
                base_values[fit_indices],
                labels[fit_indices],
                seed=int(base_config["seed"]) + test_fold * 100 + grid_index,
                max_iterations=int(base_config["max_iterations"]),
                l2_weight=float(l2_weight),
                l2_reduction=str(base_config["l2_reduction"]),  # type: ignore[arg-type]
            )
            calibration_base = predict_logits(base_model, base_values[calibration_indices])
            base_threshold, base_metrics = select_threshold(
                calibration_base, labels[calibration_indices].tolist()
            )
            base_candidates.append(
                (float(l2_weight), base_model, base_threshold, base_metrics)
            )
        selected_l2, base_model, base_threshold, base_calibration_metrics = max(
            base_candidates,
            key=lambda item: (
                float(item[3]["macro_f1"]),
                -item[0],
                -abs(item[2]),
            ),
        )
        all_base_logits = torch.as_tensor(
            predict_logits(base_model, base_values),
            dtype=torch.float32,
            device=torch_device,
        )
        seed = int(temporal_config["seed"]) + test_fold
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = VisualTemporalResidual().to(torch_device)
        expected_parameters = int(temporal_config["parameters"])
        if parameter_count(model) != expected_parameters:
            raise ValueError("D5 temporal parameter count changed")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(temporal_config["learning_rate"]),
            weight_decay=float(temporal_config["weight_decay"]),
        )
        positives = label_tensor[torch.as_tensor(fit_indices, device=torch_device)].sum()
        negatives = len(fit_indices) - positives
        if positives <= 0 or negatives <= 0:
            raise ValueError("D5 temporal fit split requires both classes")
        positive_weight = negatives / positives
        best_loss = float("inf")
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        epochs_run = 0
        for epoch in range(int(temporal_config["max_epochs"])):
            model.train()
            optimizer.zero_grad()
            fit_logits, fit_order = _split_sequences(
                model, vision, all_base_logits, sessions, fit_indices
            )
            fit_labels = label_tensor[
                torch.as_tensor(fit_order, dtype=torch.long, device=torch_device)
            ]
            loss = nn.functional.binary_cross_entropy_with_logits(
                fit_logits, fit_labels, pos_weight=positive_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), float(temporal_config["gradient_norm_clip"])
            )
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                calibration_logits, calibration_order = _split_sequences(
                    model, vision, all_base_logits, sessions, calibration_indices
                )
                calibration_labels = label_tensor[
                    torch.as_tensor(
                        calibration_order, dtype=torch.long, device=torch_device
                    )
                ]
                calibration_loss = float(
                    nn.functional.binary_cross_entropy_with_logits(
                        calibration_logits,
                        calibration_labels,
                        pos_weight=positive_weight,
                    ).cpu()
                )
            epochs_run = epoch + 1
            if calibration_loss < best_loss - 1e-8:
                best_loss = calibration_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= int(temporal_config["calibration_loss_patience"]):
                    break
        if best_state is None:
            raise RuntimeError("D5 temporal training never produced a finite checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            calibration_logits, calibration_order = _split_sequences(
                model, vision, all_base_logits, sessions, calibration_indices
            )
            test_logits, test_order = _split_sequences(
                model, vision, all_base_logits, sessions, test_indices
            )
        threshold, calibration_metrics = select_threshold(
            calibration_logits.cpu().tolist(), labels[calibration_order].tolist()
        )
        test_values = test_logits.cpu().tolist()
        test_predictions = [int(value >= threshold) for value in test_values]
        for index, logit, decision in zip(test_order.tolist(), test_values, test_predictions):
            key = examples[index].key
            if key in decisions:
                raise ValueError(f"Duplicate D5 temporal OOF decision: {key}")
            decisions[key] = decision
            oof_logits[key] = float(logit)
        details.append(
            {
                "test_fold": test_fold,
                "calibration_fold": calibration_fold,
                "fit_folds": sorted(set(range(folds)) - {test_fold, calibration_fold}),
                "fit_chunks": len(fit_indices),
                "calibration_chunks": len(calibration_indices),
                "test_chunks": len(test_indices),
                "selected_base_l2_weight": selected_l2,
                "base_threshold_logit": base_threshold,
                "base_calibration_metrics": base_calibration_metrics,
                "base_calibration_grid": [
                    {
                        "l2_weight": value,
                        "macro_f1": metrics["macro_f1"],
                        "threshold_logit": candidate_threshold,
                    }
                    for value, _, candidate_threshold, metrics in base_candidates
                ],
                "temporal_seed": seed,
                "temporal_parameters": expected_parameters,
                "epochs_run": epochs_run,
                "best_epoch": best_epoch,
                "best_calibration_loss": best_loss,
                "threshold_logit": threshold,
                "calibration_metrics": calibration_metrics,
                "test_metrics_internal": binary_metrics(
                    labels[test_order].tolist(), test_predictions
                ),
            }
        )
    expected_keys = {example.key for example in examples}
    if set(decisions) != expected_keys or set(oof_logits) != expected_keys:
        raise ValueError("D5 temporal OOF outputs do not cover every chunk")
    return decisions, oof_logits, details
