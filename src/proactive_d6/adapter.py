"""Local LoRA and query-conditioned causal memory for the frozen D6 model."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional

from proactive_d1.internvl_features import (
    InternVLDecisionFeatureExtractor,
    NeuralDecisionFeatures,
    validate_tag_suffix,
)
from proactive_r0.core import INTERRUPT_TAG, SILENT_TAG


MEMORY_PARAMETERS = 627_072
LORA_PARAMETERS = 327_680
LORA_LAYERS = (24, 25, 26, 27)
LORA_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def differentiable_tag_log_probability(
    logits: torch.Tensor, token_ids: Sequence[int]
) -> torch.Tensor:
    """Match the frozen D4 gather-minus-logsumexp reduction exactly."""
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(token_ids):
        raise ValueError("D6 tag logits must have shape [1, tag_tokens, vocabulary]")
    ids = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    values = logits.float()[0]
    selected = values.gather(1, ids[:, None]).squeeze(1)
    normalizer = torch.logsumexp(values, dim=-1)
    return (selected - normalizer).sum()


@dataclass(frozen=True)
class MemoryUpdate:
    state: torch.Tensor
    residual: torch.Tensor
    attention_entropy: torch.Tensor
    normalized_attention_entropy: torch.Tensor


@dataclass(frozen=True)
class D6Forward:
    hidden_state: torch.Tensor
    silent_log_probability: torch.Tensor
    interrupt_log_probability: torch.Tensor
    tag_margin: torch.Tensor
    prompt_tokens: int
    new_memory_state: torch.Tensor
    residual_norm: torch.Tensor
    attention_entropy: torch.Tensor
    normalized_attention_entropy: torch.Tensor
    candidate_update_max_abs_difference: float
    candidate_hidden_max_abs_difference: float
    candidate_hidden_cosine_similarity: float
    current_interval_frames: int
    current_interval_patch_tokens: int

    def detached_features(self, mode: str) -> NeuralDecisionFeatures:
        return NeuralDecisionFeatures(
            hidden_state=self.hidden_state.detach().float().cpu(),
            silent_log_probability=float(self.silent_log_probability.detach().cpu()),
            interrupt_log_probability=float(self.interrupt_log_probability.detach().cpu()),
            tag_margin=float(self.tag_margin.detach().cpu()),
            prompt_tokens=self.prompt_tokens,
            hidden_max_abs_difference=self.candidate_hidden_max_abs_difference,
            hidden_cosine_similarity=self.candidate_hidden_cosine_similarity,
            extraction_mode=mode,
            candidate_forward_passes=2,
        )


class LoRALinear(nn.Module):
    """Dependency-free rank-r residual around one frozen Linear projection."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("D6 LoRA rank and alpha must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        device = self.base.weight.device
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=device, dtype=torch.float32)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.enabled = True

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        if not self.enabled:
            return base
        residual = functional.linear(
            functional.linear(value.float(), self.lora_a), self.lora_b
        )
        return base + (residual * self.scaling).to(base.dtype)


class CausalVisualMemory(nn.Module):
    """Frozen-width manual multi-head attention followed by a recurrent state."""

    def __init__(self, input_size: int = 1024, memory_size: int = 128, heads: int = 4) -> None:
        super().__init__()
        if memory_size % heads:
            raise ValueError("D6 memory width must divide evenly across heads")
        self.input_size = input_size
        self.memory_size = memory_size
        self.heads = heads
        self.head_size = memory_size // heads
        self.input_norm = nn.LayerNorm(input_size)
        self.q_proj = nn.Linear(input_size, memory_size, bias=True)
        self.k_proj = nn.Linear(input_size, memory_size, bias=True)
        self.v_proj = nn.Linear(input_size, memory_size, bias=True)
        self.gru = nn.GRUCell(memory_size, memory_size)
        self.state_norm = nn.LayerNorm(memory_size)
        self.injection = nn.Linear(memory_size, input_size, bias=True)
        nn.init.zeros_(self.injection.weight)
        nn.init.zeros_(self.injection.bias)

    def initial_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros((1, self.memory_size), device=device, dtype=torch.float32)

    def forward(
        self,
        query_hidden: torch.Tensor,
        visual_tokens: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> MemoryUpdate:
        if query_hidden.shape != (1, self.input_size):
            raise ValueError(f"D6 query shape changed: {tuple(query_hidden.shape)}")
        if visual_tokens.ndim != 2 or visual_tokens.shape[1] != self.input_size:
            raise ValueError("D6 visual patch token shape changed")
        if visual_tokens.shape[0] == 0:
            raise ValueError("D6 current interval has no projected patch tokens")
        if previous_state.shape != (1, self.memory_size):
            raise ValueError("D6 memory state shape changed")
        query = self.input_norm(query_hidden.float())
        visual = self.input_norm(visual_tokens.float())
        q = self.q_proj(query).view(1, self.heads, self.head_size)
        k = self.k_proj(visual).view(-1, self.heads, self.head_size).transpose(0, 1)
        v = self.v_proj(visual).view(-1, self.heads, self.head_size).transpose(0, 1)
        scores = torch.einsum("bhd,htd->bht", q, k) / math.sqrt(self.head_size)
        probabilities = scores.softmax(dim=-1)
        context = torch.einsum("bht,htd->bhd", probabilities, v).reshape(
            1, self.memory_size
        )
        state = self.state_norm(self.gru(context, previous_state.float()))
        residual = self.injection(state)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean()
        denominator = math.log(probabilities.shape[-1]) if probabilities.shape[-1] > 1 else 1.0
        return MemoryUpdate(
            state=state,
            residual=residual,
            attention_entropy=entropy,
            normalized_attention_entropy=entropy / denominator,
        )


def memory_parameter_count(module: CausalVisualMemory) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class D6DecisionModel(InternVLDecisionFeatureExtractor):
    """InternVL decision extractor with the one frozen D6 adapter architecture."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        require_exclusive_gpu = bool(kwargs.get("require_exclusive_gpu", True))
        kwargs["decision_feature_mode"] = "shared_vision"
        super().__init__(*args, **kwargs)
        self.d6_require_exclusive_gpu = require_exclusive_gpu
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        torch.manual_seed(20260722)
        torch.cuda.manual_seed_all(20260722)
        self.memory = CausalVisualMemory(self.hidden_size, 128, 4).to(
            device=self.device, dtype=torch.float32
        )
        self.lora_modules = self._inject_lora(rank=8, alpha=16)
        if memory_parameter_count(self.memory) != MEMORY_PARAMETERS:
            raise RuntimeError("D6 memory parameter count changed")
        if self.lora_parameter_count() != LORA_PARAMETERS:
            raise RuntimeError("D6 LoRA parameter count changed")
        self._audit_trainable_tensors()
        self.model.eval()

    def _language_layers(self) -> Sequence[object]:
        layers = getattr(self.model.model.language_model, "layers", None)
        if layers is None or len(layers) != 28:
            raise TypeError("D6 expected 28 InternVL/Qwen language layers")
        return layers

    def _inject_lora(self, rank: int, alpha: int) -> dict[str, LoRALinear]:
        result: dict[str, LoRALinear] = {}
        layers = self._language_layers()
        for layer_index in LORA_LAYERS:
            attention = getattr(layers[layer_index], "self_attn", None)
            if attention is None:
                raise TypeError("D6 target layer has no self attention")
            for projection_name in LORA_PROJECTIONS:
                base = getattr(attention, projection_name, None)
                if not isinstance(base, nn.Linear):
                    raise TypeError(f"D6 target is not Linear: {layer_index}.{projection_name}")
                wrapper = LoRALinear(base, rank=rank, alpha=alpha)
                setattr(attention, projection_name, wrapper)
                result[f"layers.{layer_index}.self_attn.{projection_name}"] = wrapper
        return result

    def lora_parameter_count(self) -> int:
        return sum(
            module.lora_a.numel() + module.lora_b.numel()
            for module in self.lora_modules.values()
        )

    def memory_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(f"memory.{name}", parameter) for name, parameter in self.memory.named_parameters()]

    def lora_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        result: list[tuple[str, nn.Parameter]] = []
        for prefix, module in sorted(self.lora_modules.items()):
            result.extend(
                [
                    (f"lora.{prefix}.lora_a", module.lora_a),
                    (f"lora.{prefix}.lora_b", module.lora_b),
                ]
            )
        return result

    def trainable_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [*self.memory_named_parameters(), *self.lora_named_parameters()]

    def _audit_trainable_tensors(self) -> None:
        expected = {name for name, _ in self.trainable_named_parameters()}
        actual = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }
        model_expected = {
            f"model.language_model.{name.removeprefix('lora.')}"
            for name in expected
            if name.startswith("lora.")
        }
        if actual != model_expected:
            raise RuntimeError(f"D6 unexpected trainable base-model tensors: {sorted(actual ^ model_expected)}")
        if len(self.lora_named_parameters()) != 32:
            raise RuntimeError("D6 LoRA trainable tensor count changed")

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.trainable_named_parameters()
        }

    def load_adapter_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        parameters = dict(self.trainable_named_parameters())
        if set(state) != set(parameters):
            raise ValueError("D6 adapter checkpoint tensor names changed")
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = state[name]
                if value.shape != parameter.shape:
                    raise ValueError(f"D6 adapter tensor shape changed: {name}")
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    @contextlib.contextmanager
    def lora_enabled(self, enabled: bool) -> Iterator[None]:
        previous = [module.enabled for module in self.lora_modules.values()]
        for module in self.lora_modules.values():
            module.enabled = enabled
        try:
            yield
        finally:
            for module, value in zip(self.lora_modules.values(), previous):
                module.enabled = value

    def initial_memory_state(self) -> torch.Tensor:
        return self.memory.initial_state(self.device)

    def _prepare_shared_vision(
        self,
        frames: Sequence[object],
        current_interval_mask: Sequence[bool],
        messages: list[dict[str, str]],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if len(frames) != len(current_interval_mask) or not frames:
            raise ValueError("D6 frame provenance does not align with model frames")
        prompt = self._causal_prompt(list(frames), messages)
        processor_kwargs: dict[str, object] = {
            "text": [prompt + SILENT_TAG],
            "padding": True,
            "return_tensors": "pt",
            "videos": [list(frames)],
        }
        inputs = self.processor(**processor_kwargs)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        prompt_tokens = validate_tag_suffix(inputs["input_ids"], self.silent_token_ids)
        interrupt_ids = inputs["input_ids"].clone()
        interrupt_ids[0, -len(self.interrupt_token_ids) :] = torch.tensor(
            self.interrupt_token_ids, device=self.device, dtype=torch.long
        )
        if validate_tag_suffix(interrupt_ids, self.interrupt_token_ids) != prompt_tokens:
            raise RuntimeError("D6 candidate prompt lengths changed")
        with torch.no_grad():
            image_features = self.model.model.get_image_features(
                pixel_values=inputs["pixel_values"], return_dict=True
            ).pooler_output
        if image_features.ndim != 3 or image_features.shape[0] != len(frames):
            raise RuntimeError(f"D6 projected image shape changed: {tuple(image_features.shape)}")
        mask = torch.tensor(current_interval_mask, device=image_features.device, dtype=torch.bool)
        current_tokens = image_features[mask].reshape(-1, self.hidden_size).detach()
        return inputs, interrupt_ids, image_features.detach(), current_tokens, prompt_tokens

    def _embeddings(
        self, input_ids: torch.Tensor, inputs: Mapping[str, torch.Tensor], image_features: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.model.model.get_input_embeddings()(input_ids)
            projected = image_features.to(embeddings.device, embeddings.dtype)
            image_mask = self.model.model.get_placeholder_mask(
                input_ids, inputs_embeds=embeddings, image_features=projected
            )
            return embeddings.masked_scatter(image_mask, projected)

    def forward_decision(
        self,
        frames: Sequence[object],
        current_interval_mask: Sequence[bool],
        messages: list[dict[str, str]],
        previous_state: torch.Tensor,
        *,
        memory_enabled: bool = True,
        lora_enabled: bool = True,
    ) -> D6Forward:
        inputs, interrupt_ids, image_features, current_tokens, prompt_tokens = (
            self._prepare_shared_vision(frames, current_interval_mask, messages)
        )
        silent_ids = inputs["input_ids"]
        shared: dict[str, object] = {}

        def run_candidate(input_ids: torch.Tensor, candidate: str) -> torch.Tensor:
            embeddings = self._embeddings(input_ids, inputs, image_features)

            def inject(_module: object, args: tuple[object, ...]) -> tuple[object, ...]:
                hidden = args[0]
                if not isinstance(hidden, torch.Tensor):
                    raise TypeError("D6 layer-24 input is not a tensor")
                query = hidden[:, prompt_tokens - 1, :]
                if not memory_enabled:
                    update = MemoryUpdate(
                        state=previous_state,
                        residual=torch.zeros((1, self.hidden_size), device=self.device),
                        attention_entropy=torch.zeros((), device=self.device),
                        normalized_attention_entropy=torch.zeros((), device=self.device),
                    )
                elif candidate == "silent":
                    update = self.memory(query, current_tokens, previous_state)
                    shared["update"] = update
                else:
                    existing = shared.get("update")
                    if not isinstance(existing, MemoryUpdate):
                        raise RuntimeError("D6 interrupt candidate ran before silent update")
                    with torch.no_grad():
                        check = self.memory(query, current_tokens, previous_state)
                    difference = float((check.state - existing.state.detach()).abs().max().cpu())
                    shared["update_difference"] = difference
                    if difference != 0.0:
                        raise RuntimeError(f"D6 candidate memory updates differ: {difference}")
                    update = existing
                modified = hidden.clone()
                modified[:, prompt_tokens - 1, :] = (
                    modified[:, prompt_tokens - 1, :]
                    + update.residual.to(modified.dtype)
                )
                return (modified, *args[1:])

            handle = self._language_layers()[24].register_forward_pre_hook(inject)
            try:
                return self.model.model.language_model(
                    attention_mask=inputs.get("attention_mask"),
                    inputs_embeds=embeddings,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                ).last_hidden_state
            finally:
                handle.remove()

        with self.lora_enabled(lora_enabled):
            silent_hidden_all = run_candidate(silent_ids, "silent")
            interrupt_hidden_all = run_candidate(interrupt_ids, "interrupt")
        update = shared.get("update")
        if not isinstance(update, MemoryUpdate):
            update = MemoryUpdate(
                state=previous_state,
                residual=torch.zeros((1, self.hidden_size), device=self.device),
                attention_entropy=torch.zeros((), device=self.device),
                normalized_attention_entropy=torch.zeros((), device=self.device),
            )
        tag_length = len(self.silent_token_ids)
        silent_logits = self.model.lm_head(silent_hidden_all[:, -(tag_length + 1) : -1, :])
        interrupt_logits = self.model.lm_head(
            interrupt_hidden_all[:, -(tag_length + 1) : -1, :]
        )
        silent_logp = differentiable_tag_log_probability(
            silent_logits, self.silent_token_ids
        ).reshape(())
        interrupt_logp = differentiable_tag_log_probability(
            interrupt_logits, self.interrupt_token_ids
        ).reshape(())
        silent_hidden = silent_hidden_all[0, prompt_tokens - 1].float()
        interrupt_hidden = interrupt_hidden_all[0, prompt_tokens - 1].float()
        difference = (silent_hidden - interrupt_hidden).abs()
        cosine = functional.cosine_similarity(
            silent_hidden[None], interrupt_hidden[None], dim=-1
        )
        return D6Forward(
            hidden_state=silent_hidden,
            silent_log_probability=silent_logp,
            interrupt_log_probability=interrupt_logp,
            tag_margin=interrupt_logp - silent_logp,
            prompt_tokens=prompt_tokens,
            new_memory_state=update.state,
            residual_norm=update.residual.norm(dim=-1).mean(),
            attention_entropy=update.attention_entropy,
            normalized_attention_entropy=update.normalized_attention_entropy,
            candidate_update_max_abs_difference=float(shared.get("update_difference", 0.0)),
            candidate_hidden_max_abs_difference=float(difference.max().detach().cpu()),
            candidate_hidden_cosine_similarity=float(cosine.detach().cpu()),
            current_interval_frames=sum(bool(value) for value in current_interval_mask),
            current_interval_patch_tokens=int(current_tokens.shape[0]),
        )
