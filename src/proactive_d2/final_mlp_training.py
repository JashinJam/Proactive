"""Frozen fixed-shape adapter training over the bit-exact final-MLP cache."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from proactive_r0.artifacts import sha256_file

from .final_mlp_cache import STATE_ARRAY_NAMES, STATE_NAMES, state_from_bit_arrays
from .final_mlp_lora import decision_margin_from_logits


@dataclass(frozen=True)
class FinalMLPCacheArrays:
    state_bits: dict[str, np.ndarray]
    base_hidden_state: np.ndarray
    base_tag_margin: np.ndarray
    prompt_tokens: np.ndarray
    input_index: np.ndarray
    chunk_index: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.base_tag_margin.shape[0])


@dataclass(frozen=True)
class AdapterTrainingResult:
    best_epoch: int
    epochs_run: int
    best_calibration_loss: float
    zero_adapter_calibration_loss: float
    history: tuple[dict[str, object], ...]
    selected_state: dict[str, object]


def load_final_mlp_cache_arrays(
    path: Path,
    *,
    expected_sha256: str,
    rows: int,
    hidden_size: int,
    tag_length: int,
) -> FinalMLPCacheArrays:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Merged final-MLP cache fingerprint mismatch")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            *STATE_ARRAY_NAMES,
            "base_hidden_state",
            "base_tag_margin",
            "base_silent_log_probability",
            "base_interrupt_log_probability",
            "prompt_tokens",
            "input_index",
            "chunk_index",
        }
        if set(archive.files) != required:
            raise ValueError("Merged final-MLP cache has unexpected arrays")
        arrays = {name: archive[name].copy() for name in archive.files}
    state_bits = {name: arrays[name] for name in STATE_ARRAY_NAMES}
    for name, value in state_bits.items():
        if value.shape != (rows, tag_length, hidden_size) or value.dtype != np.uint16:
            raise ValueError(f"Merged final-MLP state is invalid: {name}")
    if arrays["base_hidden_state"].shape != (rows, hidden_size):
        raise ValueError("Merged final-MLP base hidden shape mismatch")
    for name in ("base_tag_margin", "prompt_tokens", "input_index", "chunk_index"):
        if arrays[name].shape != (rows,):
            raise ValueError(f"Merged final-MLP diagnostic shape mismatch: {name}")
    if not np.isfinite(arrays["base_hidden_state"]).all() or not np.isfinite(
        arrays["base_tag_margin"]
    ).all():
        raise ValueError("Merged final-MLP base diagnostics contain non-finite values")
    return FinalMLPCacheArrays(
        state_bits=state_bits,
        base_hidden_state=arrays["base_hidden_state"].astype(np.float32, copy=False),
        base_tag_margin=arrays["base_tag_margin"].astype(np.float32, copy=False),
        prompt_tokens=arrays["prompt_tokens"].astype(np.int32, copy=False),
        input_index=arrays["input_index"].astype(np.int32, copy=False),
        chunk_index=arrays["chunk_index"].astype(np.int32, copy=False),
    )


def fixed_shape_batches(
    indices: Sequence[int] | np.ndarray,
    batch_size: int,
) -> list[tuple[np.ndarray, int]]:
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size == 0 or batch_size <= 0:
        raise ValueError("Fixed-shape batching requires non-empty 1D indices")
    batches: list[tuple[np.ndarray, int]] = []
    for start in range(0, len(values), batch_size):
        real = values[start : start + batch_size]
        real_count = len(real)
        if real_count < batch_size:
            padding = np.full(batch_size - real_count, int(real[-1]), dtype=np.int64)
            padded = np.concatenate([real, padding])
        else:
            padded = real
        if padded.shape != (batch_size,):
            raise RuntimeError("Fixed-shape batch construction failed")
        batches.append((padded, real_count))
    return batches


def _candidate_state(
    cache: FinalMLPCacheArrays,
    candidate: str,
    indices: np.ndarray,
    device: object,
) -> object:
    arrays = {
        state: cache.state_bits[f"{candidate}_{state}_bits"][indices]
        for state in STATE_NAMES
    }
    return state_from_bit_arrays(arrays, device=device)


def _reconstruct_adapter_delta_final_hidden(
    peft_model: object,
    decoder_layer: object,
    final_norm: object,
    state: object,
) -> object:
    """Apply only the same-shape adapter delta to the cached reference path."""
    import torch

    residual = getattr(state, "residual", None)
    normalized = getattr(state, "normalized", None)
    reference_mlp = getattr(state, "reference_mlp_output", None)
    reference_hidden = getattr(state, "reference_final_hidden", None)
    if not all(
        isinstance(value, torch.Tensor)
        for value in (residual, normalized, reference_mlp, reference_hidden)
    ):
        raise TypeError("Adapter replay requires tensor reference states")
    if not (
        residual.shape
        == normalized.shape
        == reference_mlp.shape
        == reference_hidden.shape
    ):
        raise ValueError("Adapter replay reference-state shapes must align")
    mlp = getattr(decoder_layer, "mlp", None)
    disable_adapter = getattr(peft_model, "disable_adapter", None)
    if mlp is None or not callable(final_norm) or not callable(disable_adapter):
        raise TypeError("Adapter replay requires an MLP, final norm, and PEFT context")

    # Both paths use the identical padded replay shape. This cancels BF16 GEMM
    # and RMSNorm batch-shape offsets while preserving gradients through LoRA.
    with disable_adapter(), torch.no_grad():
        replay_base_mlp = mlp(normalized)
    replay_adapted_mlp = mlp(normalized)
    adapter_delta = (replay_adapted_mlp - replay_base_mlp).to(reference_mlp.dtype)
    corrected_mlp = reference_mlp + adapter_delta
    with torch.no_grad():
        replay_base_hidden = final_norm(residual + reference_mlp)
    replay_adapted_hidden = final_norm(residual + corrected_mlp)
    return reference_hidden + (replay_adapted_hidden - replay_base_hidden)


def adapter_batch_outputs(
    cache: FinalMLPCacheArrays,
    indices: np.ndarray,
    *,
    peft_model: object,
    decoder_layer: object,
    final_norm: object,
    lm_head: object,
    silent_token_ids: Sequence[int],
    interrupt_token_ids: Sequence[int],
    device: object,
) -> tuple[object, object, float]:
    import torch

    silent_state = _candidate_state(cache, "silent", indices, device)
    interrupt_state = _candidate_state(cache, "interrupt", indices, device)
    silent_hidden = _reconstruct_adapter_delta_final_hidden(
        peft_model, decoder_layer, final_norm, silent_state
    )
    interrupt_hidden = _reconstruct_adapter_delta_final_hidden(
        peft_model, decoder_layer, final_norm, interrupt_state
    )
    if not isinstance(silent_hidden, torch.Tensor) or not isinstance(
        interrupt_hidden, torch.Tensor
    ):
        raise TypeError("Adapted final-MLP hidden states must be tensors")
    silent_logits = lm_head(silent_hidden)
    interrupt_logits = lm_head(interrupt_hidden)
    adapted_replay_margin = decision_margin_from_logits(
        silent_logits,
        interrupt_logits,
        silent_token_ids,
        interrupt_token_ids,
    )
    silent_reference = getattr(silent_state, "reference_final_hidden", None)
    interrupt_reference = getattr(interrupt_state, "reference_final_hidden", None)
    if not isinstance(silent_reference, torch.Tensor) or not isinstance(
        interrupt_reference, torch.Tensor
    ):
        raise TypeError("Adapter margin replay requires reference hidden tensors")
    with torch.no_grad():
        base_replay_margin = decision_margin_from_logits(
            lm_head(silent_reference.to(silent_hidden.dtype)),
            lm_head(interrupt_reference.to(interrupt_hidden.dtype)),
            silent_token_ids,
            interrupt_token_ids,
        )
    cached_base_margin = torch.as_tensor(
        cache.base_tag_margin[indices], dtype=torch.float32, device=device
    )
    margins = cached_base_margin + (adapted_replay_margin - base_replay_margin)
    if not isinstance(margins, torch.Tensor):
        raise TypeError("Adapted tag margins must be a tensor")
    candidate_difference = float(
        (silent_hidden[:, 0].float() - interrupt_hidden[:, 0].float())
        .abs()
        .max()
        .detach()
        .cpu()
    )
    return margins, silent_hidden[:, 0].float(), candidate_difference


def _balanced_bce(
    margins: object,
    labels: object,
    positive_weight: float,
) -> object:
    import torch
    import torch.nn.functional as functional

    if not isinstance(margins, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Class-balanced BCE requires tensors")
    return functional.binary_cross_entropy_with_logits(
        margins.float(),
        labels.float(),
        pos_weight=torch.tensor(positive_weight, device=margins.device),
    )


def _positive_weight(labels: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError("Adapter split must contain both binary classes")
    return negatives / positives


def evaluate_adapter_bce(
    cache: FinalMLPCacheArrays,
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    peft_model: object,
    decoder_layer: object,
    final_norm: object,
    lm_head: object,
    silent_token_ids: Sequence[int],
    interrupt_token_ids: Sequence[int],
    device: object,
) -> tuple[float, float]:
    import torch

    positive_weight = _positive_weight(labels[indices])
    total_loss = 0.0
    total_rows = 0
    max_candidate_difference = 0.0
    with torch.no_grad():
        for padded, real_count in fixed_shape_batches(indices, batch_size):
            margins, _, difference = adapter_batch_outputs(
                cache,
                padded,
                peft_model=peft_model,
                decoder_layer=decoder_layer,
                final_norm=final_norm,
                lm_head=lm_head,
                silent_token_ids=silent_token_ids,
                interrupt_token_ids=interrupt_token_ids,
                device=device,
            )
            real_margins = margins[:real_count]
            batch_labels = torch.tensor(
                labels[padded[:real_count]], dtype=torch.float32, device=device
            )
            loss = _balanced_bce(real_margins, batch_labels, positive_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Adapter calibration loss is non-finite")
            total_loss += float(loss.detach().cpu()) * real_count
            total_rows += real_count
            max_candidate_difference = max(max_candidate_difference, difference)
    return total_loss / total_rows, max_candidate_difference


def _trainable_state(peft_model: object) -> dict[str, object]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(peft_model: object, state: dict[str, object]) -> None:
    import torch

    trainable = {
        name: parameter
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    }
    if set(trainable) != set(state):
        raise ValueError("Adapter trainable-state tensor names differ")
    with torch.no_grad():
        for name, parameter in trainable.items():
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise ValueError(f"Adapter trainable-state shape differs: {name}")
            parameter.copy_(value.to(parameter.device, parameter.dtype))


def train_adapter_fold(
    cache: FinalMLPCacheArrays,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    *,
    peft_model: object,
    initial_state: dict[str, object],
    decoder_layer: object,
    final_norm: object,
    lm_head: object,
    silent_token_ids: Sequence[int],
    interrupt_token_ids: Sequence[int],
    device: object,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    gradient_clip_norm: float,
    seed: int,
) -> AdapterTrainingResult:
    import torch

    numeric = (
        learning_rate,
        weight_decay,
        min_delta,
        gradient_clip_norm,
    )
    if any(not math.isfinite(value) or value < 0 for value in numeric):
        raise ValueError("Adapter optimization settings must be finite and non-negative")
    if batch_size <= 0 or max_epochs <= 0 or patience <= 0:
        raise ValueError("Adapter training durations and batch size must be positive")
    restore_trainable_state(peft_model, initial_state)
    trainable = [
        parameter for parameter in peft_model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=weight_decay
    )
    fit_positive_weight = _positive_weight(labels[fit_indices])
    zero_loss, zero_difference = evaluate_adapter_bce(
        cache,
        calibration_indices,
        labels,
        batch_size=batch_size,
        peft_model=peft_model,
        decoder_layer=decoder_layer,
        final_norm=final_norm,
        lm_head=lm_head,
        silent_token_ids=silent_token_ids,
        interrupt_token_ids=interrupt_token_ids,
        device=device,
    )
    if zero_difference != 0.0:
        raise RuntimeError("Zero adapter changes causal hidden across candidates")
    best_loss = zero_loss
    best_epoch = 0
    best_state = _trainable_state(peft_model)
    history: list[dict[str, object]] = [
        {
            "epoch": 0,
            "fit_loss": None,
            "calibration_loss": zero_loss,
            "max_gradient_norm": None,
            "selected": True,
        }
    ]
    stale_epochs = 0
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        rng = np.random.default_rng(seed + epoch)
        shuffled = np.asarray(fit_indices, dtype=np.int64).copy()
        rng.shuffle(shuffled)
        total_loss = 0.0
        total_rows = 0
        max_gradient_norm = 0.0
        for padded, real_count in fixed_shape_batches(shuffled, batch_size):
            optimizer.zero_grad(set_to_none=True)
            margins, _, difference = adapter_batch_outputs(
                cache,
                padded,
                peft_model=peft_model,
                decoder_layer=decoder_layer,
                final_norm=final_norm,
                lm_head=lm_head,
                silent_token_ids=silent_token_ids,
                interrupt_token_ids=interrupt_token_ids,
                device=device,
            )
            if difference != 0.0:
                raise RuntimeError("Adapter changes causal hidden across tag candidates")
            batch_labels = torch.tensor(
                labels[padded[:real_count]], dtype=torch.float32, device=device
            )
            loss = _balanced_bce(
                margins[:real_count], batch_labels, fit_positive_weight
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Adapter fit loss is non-finite at epoch {epoch}")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, gradient_clip_norm)
                .detach()
                .cpu()
            )
            if not math.isfinite(gradient_norm):
                raise RuntimeError(f"Adapter gradient is non-finite at epoch {epoch}")
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * real_count
            total_rows += real_count
            max_gradient_norm = max(max_gradient_norm, gradient_norm)
        calibration_loss, calibration_difference = evaluate_adapter_bce(
            cache,
            calibration_indices,
            labels,
            batch_size=batch_size,
            peft_model=peft_model,
            decoder_layer=decoder_layer,
            final_norm=final_norm,
            lm_head=lm_head,
            silent_token_ids=silent_token_ids,
            interrupt_token_ids=interrupt_token_ids,
            device=device,
        )
        if calibration_difference != 0.0:
            raise RuntimeError("Adapter calibration candidates have different causal hidden")
        improved = calibration_loss < best_loss - min_delta
        if improved:
            best_loss = calibration_loss
            best_epoch = epoch
            best_state = _trainable_state(peft_model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "fit_loss": total_loss / total_rows,
                "calibration_loss": calibration_loss,
                "max_gradient_norm": max_gradient_norm,
                "selected": improved,
            }
        )
        if stale_epochs >= patience:
            break
    restore_trainable_state(peft_model, best_state)
    return AdapterTrainingResult(
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        best_calibration_loss=best_loss,
        zero_adapter_calibration_loss=zero_loss,
        history=tuple(history),
        selected_state=best_state,
    )


def export_adapter_features(
    cache: FinalMLPCacheArrays,
    indices: np.ndarray,
    *,
    batch_size: int,
    peft_model: object,
    decoder_layer: object,
    final_norm: object,
    lm_head: object,
    silent_token_ids: Sequence[int],
    interrupt_token_ids: Sequence[int],
    device: object,
) -> tuple[np.ndarray, np.ndarray, float]:
    import torch

    margins = np.empty(len(indices), dtype=np.float32)
    hidden_size = cache.base_hidden_state.shape[1]
    hidden = np.empty((len(indices), hidden_size), dtype=np.float32)
    max_candidate_difference = 0.0
    output_position = 0
    with torch.no_grad():
        for padded, real_count in fixed_shape_batches(indices, batch_size):
            batch_margin, batch_hidden, difference = adapter_batch_outputs(
                cache,
                padded,
                peft_model=peft_model,
                decoder_layer=decoder_layer,
                final_norm=final_norm,
                lm_head=lm_head,
                silent_token_ids=silent_token_ids,
                interrupt_token_ids=interrupt_token_ids,
                device=device,
            )
            margins[output_position : output_position + real_count] = (
                batch_margin[:real_count].detach().cpu().numpy().astype(np.float32)
            )
            hidden[output_position : output_position + real_count] = (
                batch_hidden[:real_count].detach().cpu().numpy().astype(np.float32)
            )
            output_position += real_count
            max_candidate_difference = max(max_candidate_difference, difference)
    if output_position != len(indices) or not np.isfinite(margins).all() or not np.isfinite(
        hidden
    ).all():
        raise RuntimeError("Adapted feature export is incomplete or non-finite")
    return margins, hidden, max_candidate_difference
