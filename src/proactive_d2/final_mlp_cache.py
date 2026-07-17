"""Label-free, bit-exact final-MLP cache extraction primitives for D2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from proactive_d1.internvl_features import (
    InternVLDecisionFeatureExtractor,
    tag_sequence_log_probability,
    validate_tag_suffix,
)
from proactive_r0.core import INTERRUPT_TAG, SILENT_TAG

from .final_mlp_lora import (
    FinalMLPScoringState,
    FinalMLPStateCapture,
    reconstruct_final_hidden,
)


CANDIDATE_NAMES: tuple[str, ...] = ("silent", "interrupt")
STATE_NAMES: tuple[str, ...] = (
    "residual",
    "normalized",
    "reference_mlp_output",
    "local_base_mlp_output",
    "reference_final_hidden",
    "local_base_final_hidden",
)
STATE_ARRAY_NAMES: tuple[str, ...] = tuple(
    f"{candidate}_{state}_bits"
    for candidate in CANDIDATE_NAMES
    for state in STATE_NAMES
)


@dataclass(frozen=True)
class CachedCandidate:
    state: FinalMLPScoringState
    hidden: object
    log_probability: float
    hidden_max_abs_difference: float
    logit_max_abs_difference: float


@dataclass(frozen=True)
class FinalMLPDecisionCache:
    silent: CachedCandidate
    interrupt: CachedCandidate
    hidden_state: object
    silent_log_probability: float
    interrupt_log_probability: float
    tag_margin: float
    prompt_tokens: int
    candidate_hidden_max_abs_difference: float


def bfloat16_tensor_to_uint16(tensor: object) -> np.ndarray:
    """Preserve a BF16 tensor's exact bits in a NumPy-compatible dtype."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("BF16 cache serialization requires a torch tensor")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16 tensor, received {tensor.dtype}")
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.view(torch.uint16).numpy().copy()


def uint16_to_bfloat16_tensor(array: np.ndarray, device: object | None = None) -> object:
    """Restore exact BF16 values from their uint16 bit representation."""
    import torch

    value = np.asarray(array)
    if value.dtype != np.uint16:
        raise ValueError(f"Expected uint16 BF16 bits, received {value.dtype}")
    if not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    tensor = torch.from_numpy(value.copy()).view(torch.bfloat16)
    return tensor.to(device) if device is not None else tensor


def state_to_bit_arrays(
    state: FinalMLPScoringState,
    *,
    remove_batch_dimension: bool,
) -> dict[str, np.ndarray]:
    import torch

    values = {
        "residual": state.residual,
        "normalized": state.normalized,
        "reference_mlp_output": state.reference_mlp_output,
        "local_base_mlp_output": state.local_base_mlp_output,
        "reference_final_hidden": state.reference_final_hidden,
        "local_base_final_hidden": state.local_base_final_hidden,
    }
    arrays: dict[str, np.ndarray] = {}
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Final-MLP cache state {name} is not a tensor")
        selected = value
        if remove_batch_dimension:
            if selected.ndim != 3 or selected.shape[0] != 1:
                raise ValueError(f"Final-MLP cache state {name} has invalid batch shape")
            selected = selected[0]
        arrays[name] = bfloat16_tensor_to_uint16(selected)
    return arrays


def state_from_bit_arrays(
    arrays: dict[str, np.ndarray],
    *,
    device: object | None = None,
) -> FinalMLPScoringState:
    if set(arrays) != set(STATE_NAMES):
        raise ValueError("Final-MLP state bit arrays have unexpected keys")
    tensors = {
        name: uint16_to_bfloat16_tensor(arrays[name], device=device)
        for name in STATE_NAMES
    }
    return FinalMLPScoringState(
        residual=tensors["residual"],
        normalized=tensors["normalized"],
        reference_mlp_output=tensors["reference_mlp_output"],
        local_base_mlp_output=tensors["local_base_mlp_output"],
        reference_final_hidden=tensors["reference_final_hidden"],
        local_base_final_hidden=tensors["local_base_final_hidden"],
    )


def state_batch_from_archive(
    archive: object,
    candidate: str,
    indices: np.ndarray,
    device: object,
) -> FinalMLPScoringState:
    if candidate not in CANDIDATE_NAMES:
        raise ValueError(f"Unknown final-MLP candidate: {candidate}")
    arrays = {
        state: archive[f"{candidate}_{state}_bits"][indices]  # type: ignore[index]
        for state in STATE_NAMES
    }
    return state_from_bit_arrays(arrays, device=device)


class InternVLFinalMLPCacheExtractor(InternVLDecisionFeatureExtractor):
    """Extract corrected final-MLP states without reading C1 labels."""

    def __init__(self, *args: object, language_layer_index: int, **kwargs: object) -> None:
        super().__init__(*args, decision_feature_mode="shared_vision", **kwargs)
        language_model = self.model.model.language_model
        if language_layer_index < 0 or language_layer_index >= len(language_model.layers):
            raise ValueError("Final-MLP cache language layer index is out of range")
        if language_layer_index != len(language_model.layers) - 1:
            raise ValueError("Corrected cache currently supports only the final language layer")
        self.language_layer_index = language_layer_index
        self.decoder_layer = language_model.layers[language_layer_index]
        self.final_norm = language_model.norm

    def _shared_inputs(
        self,
        frames: list[object],
        prompt: str,
    ) -> dict[str, object]:
        import torch

        processor_kwargs: dict[str, object] = {
            "text": [prompt + SILENT_TAG],
            "padding": True,
            "return_tensors": "pt",
        }
        if frames:
            processor_kwargs["videos"] = [frames]
        inputs = self.processor(**processor_kwargs)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        silent_ids = inputs["input_ids"]
        prompt_tokens = validate_tag_suffix(silent_ids, self.silent_token_ids)
        interrupt_ids = silent_ids.clone()
        interrupt_ids[0, -len(self.interrupt_token_ids) :] = torch.tensor(
            self.interrupt_token_ids,
            dtype=torch.long,
            device=self.device,
        )
        if validate_tag_suffix(interrupt_ids, self.interrupt_token_ids) != prompt_tokens:
            raise RuntimeError("Final-MLP cache candidates have different prompt lengths")

        image_features = None
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            image_features = self.model.model.get_image_features(
                pixel_values=pixel_values,
                return_dict=True,
            ).pooler_output

        def embeddings(candidate_ids: object) -> object:
            if not isinstance(candidate_ids, torch.Tensor):
                raise TypeError("Final-MLP candidate IDs must be a tensor")
            result = self.model.model.get_input_embeddings()(candidate_ids)
            if image_features is not None:
                projected = image_features.to(result.device, result.dtype)
                image_mask = self.model.model.get_placeholder_mask(
                    candidate_ids,
                    inputs_embeds=result,
                    image_features=projected,
                )
                result = result.masked_scatter(image_mask, projected)
            return result

        return {
            "attention_mask": inputs.get("attention_mask"),
            "silent_embeddings": embeddings(silent_ids),
            "interrupt_embeddings": embeddings(interrupt_ids),
            "prompt_tokens": prompt_tokens,
        }

    def _candidate(
        self,
        embeddings: object,
        attention_mask: object,
        token_ids: Sequence[int],
    ) -> CachedCandidate:
        import torch

        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("Final-MLP candidate embeddings must be a tensor")
        with FinalMLPStateCapture(self.decoder_layer, len(token_ids)) as capture:
            outputs = self.model.model.language_model(
                attention_mask=attention_mask,
                inputs_embeds=embeddings,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        hidden = outputs.last_hidden_state[:, -(len(token_ids) + 1) : -1, :]
        raw_state = capture.state()
        reference_mlp_output = capture.mlp_output()
        local_base_output = self.decoder_layer.mlp(raw_state.normalized).detach().clone()
        local_base_final_hidden = self.final_norm(
            raw_state.residual + reference_mlp_output
        ).detach().clone()
        state = FinalMLPScoringState(
            residual=raw_state.residual,
            normalized=raw_state.normalized,
            reference_mlp_output=reference_mlp_output,
            local_base_mlp_output=local_base_output,
            reference_final_hidden=hidden.detach().clone(),
            local_base_final_hidden=local_base_final_hidden,
        )
        logits = self.model.lm_head(hidden)
        reconstructed_hidden = reconstruct_final_hidden(
            self.decoder_layer, self.final_norm, state
        )
        reconstructed_logits = self.model.lm_head(reconstructed_hidden)
        hidden_difference = float(
            (hidden.float() - reconstructed_hidden.float()).abs().max().cpu()
        )
        logit_difference = float(
            (logits.float() - reconstructed_logits.float()).abs().max().cpu()
        )
        return CachedCandidate(
            state=state,
            hidden=hidden.detach().cpu().float(),
            log_probability=tag_sequence_log_probability(logits[0], list(token_ids)),
            hidden_max_abs_difference=hidden_difference,
            logit_max_abs_difference=logit_difference,
        )

    def extract_final_mlp_cache(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
    ) -> FinalMLPDecisionCache:
        import torch

        prompt = self._causal_prompt(frames, messages)
        with torch.no_grad():
            shared = self._shared_inputs(frames, prompt)
            silent = self._candidate(
                shared["silent_embeddings"],
                shared["attention_mask"],
                self.silent_token_ids,
            )
            interrupt = self._candidate(
                shared["interrupt_embeddings"],
                shared["attention_mask"],
                self.interrupt_token_ids,
            )
        silent_hidden = silent.hidden
        interrupt_hidden = interrupt.hidden
        if not isinstance(silent_hidden, torch.Tensor) or not isinstance(
            interrupt_hidden, torch.Tensor
        ):
            raise TypeError("Final-MLP cached hidden states must be tensors")
        causal_difference = float(
            (silent_hidden[:, 0] - interrupt_hidden[:, 0]).abs().max()
        )
        hidden_state = silent_hidden[0, 0].clone()
        if hidden_state.shape != (self.hidden_size,):
            raise RuntimeError("Final-MLP causal hidden width differs from the model")
        margin = interrupt.log_probability - silent.log_probability
        return FinalMLPDecisionCache(
            silent=silent,
            interrupt=interrupt,
            hidden_state=hidden_state,
            silent_log_probability=silent.log_probability,
            interrupt_log_probability=interrupt.log_probability,
            tag_margin=margin,
            prompt_tokens=int(shared["prompt_tokens"]),
            candidate_hidden_max_abs_difference=causal_difference,
        )
