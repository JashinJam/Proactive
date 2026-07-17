"""Exact final-MLP LoRA reconstruction primitives for InternVL D2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


FINAL_MLP_PROJECTIONS: tuple[str, ...] = (
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class FinalMLPScoringState:
    """Inputs needed to replay one decoder layer's position-local MLP."""

    residual: object
    normalized: object
    reference_mlp_output: object | None = None
    local_base_mlp_output: object | None = None
    reference_final_hidden: object | None = None
    local_base_final_hidden: object | None = None


@dataclass(frozen=True)
class CandidateForward:
    state: FinalMLPScoringState
    hidden: object
    logits: object
    prompt_tokens: int


@dataclass(frozen=True)
class LoRAParameterAudit:
    target_regex: str
    trainable_parameters: int
    expected_trainable_parameters: int
    total_parameters_with_adapter: int
    trainable_tensor_names: tuple[str, ...]
    matched_projection_suffixes: tuple[str, ...]


def final_mlp_target_regex(layer_index: int) -> str:
    if layer_index < 0:
        raise ValueError("Final MLP layer index must be non-negative")
    return (
        rf"model\.language_model\.layers\.{layer_index}\.mlp\."
        r"(gate_proj|up_proj|down_proj)"
    )


def final_mlp_lora_parameter_count(
    hidden_size: int,
    intermediate_size: int,
    rank: int,
) -> int:
    if hidden_size <= 0 or intermediate_size <= 0 or rank <= 0:
        raise ValueError("LoRA dimensions and rank must be positive")
    return 3 * rank * (hidden_size + intermediate_size)


class FinalMLPStateCapture:
    """Capture final-layer residual and normalized MLP input at tag positions."""

    def __init__(self, decoder_layer: object, tag_length: int) -> None:
        if tag_length <= 0:
            raise ValueError("Tag length must be positive")
        layer_norm = getattr(decoder_layer, "post_attention_layernorm", None)
        if layer_norm is None:
            raise TypeError("Decoder layer has no post_attention_layernorm")
        self.layer_norm = layer_norm
        self.mlp = getattr(decoder_layer, "mlp", None)
        if self.mlp is None:
            raise TypeError("Decoder layer has no MLP")
        self.tag_length = tag_length
        self._residual: object | None = None
        self._normalized: object | None = None
        self._mlp_output: object | None = None
        self._handles: list[object] = []

    def _slice(self, value: object, name: str) -> object:
        import torch

        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise RuntimeError(f"Captured {name} must have shape [batch, tokens, hidden]")
        selected = value[:, -(self.tag_length + 1) : -1, :]
        if selected.shape[1] != self.tag_length:
            raise RuntimeError(f"Captured {name} does not contain every tag position")
        return selected.detach().clone()

    def _capture_residual(self, _module: object, inputs: tuple[object, ...]) -> None:
        if self._residual is not None:
            raise RuntimeError("Final MLP residual was captured more than once")
        if not inputs:
            raise RuntimeError("Final MLP layer norm received no input")
        self._residual = self._slice(inputs[0], "residual")

    def _capture_normalized(
        self,
        _module: object,
        _inputs: tuple[object, ...],
        output: object,
    ) -> None:
        if self._normalized is not None:
            raise RuntimeError("Final MLP normalized input was captured more than once")
        self._normalized = self._slice(output, "normalized input")

    def _capture_mlp_output(
        self,
        _module: object,
        _inputs: tuple[object, ...],
        output: object,
    ) -> None:
        if self._mlp_output is not None:
            raise RuntimeError("Final MLP output was captured more than once")
        self._mlp_output = self._slice(output, "MLP output")

    def __enter__(self) -> "FinalMLPStateCapture":
        self._handles = [
            self.layer_norm.register_forward_pre_hook(self._capture_residual),
            self.layer_norm.register_forward_hook(self._capture_normalized),
            self.mlp.register_forward_hook(self._capture_mlp_output),
        ]
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        for handle in self._handles:
            handle.remove()  # type: ignore[union-attr]
        self._handles = []

    def state(self) -> FinalMLPScoringState:
        if (
            self._residual is None
            or self._normalized is None
            or self._mlp_output is None
        ):
            raise RuntimeError("Final MLP scoring state is incomplete")
        return FinalMLPScoringState(
            residual=self._residual,
            normalized=self._normalized,
        )

    def mlp_output(self) -> object:
        if self._mlp_output is None:
            raise RuntimeError("Final MLP output was not captured")
        return self._mlp_output


def reconstruct_final_hidden(
    decoder_layer: object,
    final_norm: object,
    state: FinalMLPScoringState,
) -> object:
    """Replay exactly the final position-local MLP, residual, and model norm."""
    import torch

    residual = state.residual
    normalized = state.normalized
    if not isinstance(residual, torch.Tensor) or not isinstance(normalized, torch.Tensor):
        raise TypeError("Final MLP scoring state must contain tensors")
    if residual.shape != normalized.shape or residual.ndim != 3:
        raise ValueError("Final MLP residual and normalized input shapes must align")
    mlp = getattr(decoder_layer, "mlp", None)
    if mlp is None or not callable(final_norm):
        raise TypeError("Decoder layer MLP and final norm must be callable")
    local_output = mlp(normalized)
    reference = state.reference_mlp_output
    local_base = state.local_base_mlp_output
    if (reference is None) != (local_base is None):
        raise ValueError("Reference and local-base MLP outputs must appear together")
    if reference is not None and local_base is not None:
        if not isinstance(reference, torch.Tensor) or not isinstance(
            local_base, torch.Tensor
        ):
            raise TypeError("MLP correction cache must contain tensors")
        if reference.shape != residual.shape or local_base.shape != residual.shape:
            raise ValueError("MLP correction cache shapes must align with the residual")
        local_output = reference + (local_output - local_base)
    local_hidden = final_norm(residual + local_output)
    reference_hidden = state.reference_final_hidden
    local_base_hidden = state.local_base_final_hidden
    if (reference_hidden is None) != (local_base_hidden is None):
        raise ValueError("Reference and local-base final hidden must appear together")
    if reference_hidden is not None and local_base_hidden is not None:
        if not isinstance(reference_hidden, torch.Tensor) or not isinstance(
            local_base_hidden, torch.Tensor
        ):
            raise TypeError("Final-hidden correction cache must contain tensors")
        if (
            reference_hidden.shape != residual.shape
            or local_base_hidden.shape != residual.shape
        ):
            raise ValueError("Final-hidden correction shapes must align with the residual")
        local_hidden = reference_hidden + (local_hidden - local_base_hidden)
    return local_hidden


def tag_sequence_log_probability_tensor(
    logits: object,
    token_ids: Sequence[int],
) -> object:
    """Return differentiable fixed-tag log probability for one batch."""
    import torch
    import torch.nn.functional as functional

    if not isinstance(logits, torch.Tensor) or logits.ndim not in (2, 3):
        raise ValueError("Tag logits must have shape [tokens, vocab] or [batch, tokens, vocab]")
    if not token_ids or logits.shape[-2] != len(token_ids):
        raise ValueError("Tag token IDs and scoring positions must align")
    ids = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    if int(ids.min()) < 0 or int(ids.max()) >= logits.shape[-1]:
        raise ValueError("Tag token ID is outside the model vocabulary")
    log_probabilities = functional.log_softmax(logits.float(), dim=-1)
    if logits.ndim == 2:
        return log_probabilities.gather(1, ids[:, None]).squeeze(1).sum()
    expanded = ids[None, :, None].expand(logits.shape[0], -1, -1)
    return log_probabilities.gather(2, expanded).squeeze(2).sum(dim=1)


def decision_margin_from_logits(
    silent_logits: object,
    interrupt_logits: object,
    silent_token_ids: Sequence[int],
    interrupt_token_ids: Sequence[int],
) -> object:
    silent_logp = tag_sequence_log_probability_tensor(
        silent_logits, silent_token_ids
    )
    interrupt_logp = tag_sequence_log_probability_tensor(
        interrupt_logits, interrupt_token_ids
    )
    return interrupt_logp - silent_logp


def configure_final_mlp_lora(
    model: object,
    *,
    layer_index: int,
    hidden_size: int,
    intermediate_size: int,
    rank: int,
    alpha: int,
    dropout: float,
) -> tuple[object, LoRAParameterAudit]:
    """Inject LoRA into exactly gate/up/down projections of one language MLP."""
    from peft import LoraConfig, get_peft_model

    if alpha <= 0 or dropout < 0 or dropout >= 1:
        raise ValueError("LoRA alpha/dropout configuration is invalid")
    target_regex = final_mlp_target_regex(layer_index)
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=target_regex,
        ),
    )
    trainable = tuple(
        name for name, parameter in peft_model.named_parameters() if parameter.requires_grad
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in peft_model.parameters()
        if parameter.requires_grad
    )
    expected = final_mlp_lora_parameter_count(
        hidden_size, intermediate_size, rank
    )
    if trainable_parameters != expected:
        raise RuntimeError(
            f"Unexpected trainable LoRA parameters: {trainable_parameters} != {expected}"
        )
    if len(trainable) != 2 * len(FINAL_MLP_PROJECTIONS):
        raise RuntimeError(f"Unexpected trainable LoRA tensor count: {len(trainable)}")
    matched = tuple(
        projection
        for projection in FINAL_MLP_PROJECTIONS
        if any(f".mlp.{projection}.lora_" in name for name in trainable)
    )
    if matched != FINAL_MLP_PROJECTIONS:
        raise RuntimeError(f"LoRA projection audit failed: {matched}")
    forbidden = [
        name
        for name in trainable
        if f".layers.{layer_index}.mlp." not in name
        or not any(f".{projection}.lora_" in name for projection in FINAL_MLP_PROJECTIONS)
    ]
    if forbidden:
        raise RuntimeError(f"LoRA escaped the final language MLP: {forbidden}")
    audit = LoRAParameterAudit(
        target_regex=target_regex,
        trainable_parameters=trainable_parameters,
        expected_trainable_parameters=expected,
        total_parameters_with_adapter=sum(
            parameter.numel() for parameter in peft_model.parameters()
        ),
        trainable_tensor_names=trainable,
        matched_projection_suffixes=matched,
    )
    return peft_model, audit


def internvl_lora_components(
    peft_model: object,
    layer_index: int,
) -> tuple[object, object, object, object]:
    get_base_model = getattr(peft_model, "get_base_model", None)
    if not callable(get_base_model):
        raise TypeError("Expected a PEFT model with get_base_model()")
    outer = get_base_model()
    internvl = getattr(outer, "model", None)
    language_model = getattr(internvl, "language_model", None)
    layers = getattr(language_model, "layers", None)
    final_norm = getattr(language_model, "norm", None)
    lm_head = getattr(outer, "lm_head", None)
    if layers is None or final_norm is None or lm_head is None:
        raise TypeError("Loaded model does not expose the expected InternVL/Qwen components")
    if layer_index >= len(layers):
        raise ValueError(f"Language layer index {layer_index} is out of range")
    return outer, layers[layer_index], final_norm, lm_head


def final_mlp_cache_bytes_per_chunk(
    *,
    candidates: int,
    tag_length: int,
    hidden_size: int,
    bytes_per_value: int,
    stored_tensors_per_candidate: int = 2,
) -> int:
    values = (
        candidates,
        tag_length,
        hidden_size,
        bytes_per_value,
        stored_tensors_per_candidate,
    )
    if any(value <= 0 for value in values):
        raise ValueError("Cache dimensions must be positive")
    return (
        candidates
        * tag_length
        * stored_tensors_per_candidate
        * hidden_size
        * bytes_per_value
    )
