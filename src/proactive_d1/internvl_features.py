"""Frozen InternVL tag-margin and causal hidden-state extraction for D1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from proactive_r0.core import INTERRUPT_TAG, SILENT_TAG
from proactive_r0.internvl import InternVLProactiveModel

DecisionFeatureMode = Literal[
    "sequential",
    "batched",
    "prefix_cache",
    "shared_vision",
]
DECISION_FEATURE_MODES: tuple[DecisionFeatureMode, ...] = (
    "sequential",
    "batched",
    "prefix_cache",
    "shared_vision",
)


@dataclass(frozen=True)
class NeuralDecisionFeatures:
    hidden_state: object
    silent_log_probability: float
    interrupt_log_probability: float
    tag_margin: float
    prompt_tokens: int
    hidden_max_abs_difference: float
    hidden_cosine_similarity: float
    extraction_mode: str = "sequential"
    candidate_forward_passes: int = 2


def tag_sequence_log_probability(logits: object, token_ids: list[int]) -> float:
    """Sum autoregressive log probabilities for one fixed tag sequence."""
    import torch

    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("Tag logits must have shape [tag_tokens, vocabulary]")
    if logits.shape[0] != len(token_ids) or not token_ids:
        raise ValueError("Tag logits and token sequence must be non-empty and aligned")
    ids = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    if int(ids.min()) < 0 or int(ids.max()) >= logits.shape[1]:
        raise ValueError("Tag token ID is outside the model vocabulary")
    selected = logits.float().gather(1, ids[:, None]).squeeze(1)
    normalizer = torch.logsumexp(logits.float(), dim=-1)
    return float((selected - normalizer).sum().detach().cpu())


def validate_tag_suffix(input_ids: object, tag_token_ids: list[int]) -> int:
    """Return the causal prompt length after validating the fixed tag suffix."""
    import torch

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("Candidate input_ids must have shape [1, sequence]")
    if input_ids.shape[0] != 1:
        raise ValueError("D1 candidate scoring requires one non-empty sequence")
    return validate_batched_tag_suffixes(input_ids, [tag_token_ids])


def validate_batched_tag_suffixes(
    input_ids: object,
    candidate_token_ids: list[list[int]],
) -> int:
    """Validate a candidate batch and return its shared causal prompt length."""
    import torch

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("Batched candidate input_ids must have shape [batch, sequence]")
    if input_ids.shape[0] != len(candidate_token_ids) or not candidate_token_ids:
        raise ValueError("D1 candidate batch and token sequences must align")
    prompt_lengths: list[int] = []
    for row_index, token_ids in enumerate(candidate_token_ids):
        if not token_ids or input_ids.shape[1] <= len(token_ids):
            raise ValueError("Candidate sequence has no causal prompt before its tag")
        actual = input_ids[row_index, -len(token_ids) :].detach().cpu().tolist()
        if actual != token_ids:
            raise ValueError(
                f"Candidate tag tokenization changed at row {row_index}: "
                f"{actual} != {token_ids}"
            )
        prompt_lengths.append(int(input_ids.shape[1] - len(token_ids)))
    if len(set(prompt_lengths)) != 1:
        raise ValueError("Batched candidates have different causal prompt lengths")
    return prompt_lengths[0]


class InternVLDecisionFeatureExtractor(InternVLProactiveModel):
    """Extract label-free neural decision features from the frozen R0 context."""

    def __init__(
        self,
        *args: object,
        decision_feature_mode: DecisionFeatureMode = "sequential",
        **kwargs: object,
    ) -> None:
        if decision_feature_mode not in DECISION_FEATURE_MODES:
            raise ValueError(
                f"Unsupported D1 decision feature mode: {decision_feature_mode}"
            )
        super().__init__(*args, **kwargs)
        tokenizer = self.processor.tokenizer
        self.silent_token_ids = tokenizer.encode(SILENT_TAG, add_special_tokens=False)
        self.interrupt_token_ids = tokenizer.encode(
            INTERRUPT_TAG, add_special_tokens=False
        )
        if tokenizer.decode(
            self.silent_token_ids, skip_special_tokens=False
        ) != SILENT_TAG:
            raise ValueError("Silent tag does not round-trip through the tokenizer")
        if tokenizer.decode(
            self.interrupt_token_ids, skip_special_tokens=False
        ) != INTERRUPT_TAG:
            raise ValueError("Interrupt tag does not round-trip through the tokenizer")
        if len(self.silent_token_ids) != len(self.interrupt_token_ids):
            raise ValueError("D1 tag margin requires equal-length tag encodings")
        self.hidden_size = int(self.model.config.text_config.hidden_size)
        self.decision_feature_mode = decision_feature_mode

    def _score_candidate(
        self,
        frames: list[object],
        prompt: str,
        tag: str,
        token_ids: list[int],
    ) -> tuple[object, float, int]:
        import torch

        processor_kwargs: dict[str, object] = {
            "text": [prompt + tag],
            "padding": True,
            "return_tensors": "pt",
        }
        if frames:
            processor_kwargs["videos"] = [frames]
        inputs = self.processor(**processor_kwargs)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        prompt_tokens = validate_tag_suffix(inputs["input_ids"], token_ids)
        tag_length = len(token_ids)
        with torch.inference_mode():
            outputs = self.model.model(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError(f"Unexpected InternVL hidden shape: {hidden.shape}")
            if hidden.shape[1] != inputs["input_ids"].shape[1]:
                raise RuntimeError("InternVL hidden/input token lengths differ")
            if hidden.shape[2] != self.hidden_size:
                raise RuntimeError("InternVL hidden width differs from model config")
            causal_hidden = hidden[0, prompt_tokens - 1].float()
            scoring_hidden = hidden[:, -(tag_length + 1) : -1, :]
            tag_logits = self.model.lm_head(scoring_hidden)[0]
            log_probability = tag_sequence_log_probability(tag_logits, token_ids)
        return causal_hidden.detach().cpu(), log_probability, prompt_tokens

    @staticmethod
    def _finalize_features(
        silent_hidden: object,
        interrupt_hidden: object,
        silent_logp: float,
        interrupt_logp: float,
        prompt_tokens: int,
        extraction_mode: DecisionFeatureMode,
        candidate_forward_passes: int,
    ) -> NeuralDecisionFeatures:
        import torch
        import torch.nn.functional as functional

        if not isinstance(silent_hidden, torch.Tensor) or not isinstance(
            interrupt_hidden, torch.Tensor
        ):
            raise TypeError("D1 causal hidden states must be tensors")
        difference = (silent_hidden - interrupt_hidden).abs()
        cosine = functional.cosine_similarity(
            silent_hidden[None], interrupt_hidden[None], dim=-1
        )
        if not torch.isfinite(silent_hidden).all() or not torch.isfinite(
            interrupt_hidden
        ).all():
            raise RuntimeError("D1 causal hidden state contains non-finite values")
        if not all(
            torch.isfinite(torch.tensor(value))
            for value in (silent_logp, interrupt_logp)
        ):
            raise RuntimeError("D1 tag score contains non-finite values")
        return NeuralDecisionFeatures(
            hidden_state=silent_hidden,
            silent_log_probability=silent_logp,
            interrupt_log_probability=interrupt_logp,
            tag_margin=interrupt_logp - silent_logp,
            prompt_tokens=prompt_tokens,
            hidden_max_abs_difference=float(difference.max()),
            hidden_cosine_similarity=float(cosine.item()),
            extraction_mode=extraction_mode,
            candidate_forward_passes=candidate_forward_passes,
        )

    def _causal_prompt(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
    ) -> str:
        multimodal = self._to_multimodal_messages(frames, messages)
        return self.processor.apply_chat_template(
            multimodal,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _extract_decision_features_sequential(
        self,
        frames: list[object],
        prompt: str,
    ) -> NeuralDecisionFeatures:
        silent_hidden, silent_logp, silent_prompt_tokens = self._score_candidate(
            frames, prompt, SILENT_TAG, self.silent_token_ids
        )
        interrupt_hidden, interrupt_logp, interrupt_prompt_tokens = self._score_candidate(
            frames, prompt, INTERRUPT_TAG, self.interrupt_token_ids
        )
        if silent_prompt_tokens != interrupt_prompt_tokens:
            raise RuntimeError("Silent and interrupt candidates have different prompt lengths")
        return self._finalize_features(
            silent_hidden=silent_hidden,
            interrupt_hidden=interrupt_hidden,
            silent_logp=silent_logp,
            interrupt_logp=interrupt_logp,
            prompt_tokens=silent_prompt_tokens,
            extraction_mode="sequential",
            candidate_forward_passes=2,
        )

    def _extract_decision_features_batched(
        self,
        frames: list[object],
        prompt: str,
    ) -> NeuralDecisionFeatures:
        import torch

        candidate_ids = [self.silent_token_ids, self.interrupt_token_ids]
        processor_kwargs: dict[str, object] = {
            "text": [prompt + SILENT_TAG, prompt + INTERRUPT_TAG],
            "padding": True,
            "return_tensors": "pt",
        }
        if frames:
            processor_kwargs["videos"] = [list(frames), list(frames)]
        inputs = self.processor(**processor_kwargs)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        prompt_tokens = validate_batched_tag_suffixes(
            inputs["input_ids"], candidate_ids
        )
        tag_length = len(self.silent_token_ids)
        with torch.inference_mode():
            outputs = self.model.model(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            if hidden.ndim != 3 or hidden.shape[0] != 2:
                raise RuntimeError(f"Unexpected batched InternVL hidden shape: {hidden.shape}")
            if hidden.shape[1] != inputs["input_ids"].shape[1]:
                raise RuntimeError("Batched InternVL hidden/input token lengths differ")
            if hidden.shape[2] != self.hidden_size:
                raise RuntimeError("Batched InternVL hidden width differs from model config")
            causal_hidden = hidden[:, prompt_tokens - 1].float()
            scoring_hidden = hidden[:, -(tag_length + 1) : -1, :]
            tag_logits = self.model.lm_head(scoring_hidden)
            silent_logp = tag_sequence_log_probability(
                tag_logits[0], self.silent_token_ids
            )
            interrupt_logp = tag_sequence_log_probability(
                tag_logits[1], self.interrupt_token_ids
            )
        return self._finalize_features(
            silent_hidden=causal_hidden[0].detach().cpu(),
            interrupt_hidden=causal_hidden[1].detach().cpu(),
            silent_logp=silent_logp,
            interrupt_logp=interrupt_logp,
            prompt_tokens=prompt_tokens,
            extraction_mode="batched",
            candidate_forward_passes=1,
        )

    def _extract_decision_features_prefix_cache(
        self,
        frames: list[object],
        prompt: str,
    ) -> NeuralDecisionFeatures:
        """Reuse one full silent-candidate prefill for interrupt tag scoring."""
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
        prompt_tokens = validate_tag_suffix(
            inputs["input_ids"], self.silent_token_ids
        )
        tag_length = len(self.silent_token_ids)
        with torch.inference_mode():
            outputs = self.model.model(
                **inputs,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError(f"Unexpected cached InternVL hidden shape: {hidden.shape}")
            if hidden.shape[1] != inputs["input_ids"].shape[1]:
                raise RuntimeError("Cached InternVL hidden/input token lengths differ")
            if hidden.shape[2] != self.hidden_size:
                raise RuntimeError("Cached InternVL hidden width differs from model config")
            causal_hidden = hidden[0, prompt_tokens - 1].float()
            silent_scoring_hidden = hidden[:, -(tag_length + 1) : -1, :]
            silent_logits = self.model.lm_head(silent_scoring_hidden)[0]
            silent_logp = tag_sequence_log_probability(
                silent_logits, self.silent_token_ids
            )

            cache = outputs.past_key_values
            if cache is None:
                raise RuntimeError("InternVL prefix prefill did not return a cache")
            cache.crop(prompt_tokens)
            if cache.get_seq_length() != prompt_tokens:
                raise RuntimeError("InternVL prefix cache crop length mismatch")

            interrupt_ids = torch.tensor(
                self.interrupt_token_ids,
                dtype=torch.long,
                device=self.device,
            )[None]
            if tag_length > 1:
                continuation_ids = interrupt_ids[:, :-1]
                prefix_mask = inputs.get("attention_mask")
                if prefix_mask is None:
                    prefix_mask = torch.ones(
                        (1, prompt_tokens), dtype=torch.long, device=self.device
                    )
                else:
                    prefix_mask = prefix_mask[:, :prompt_tokens]
                continuation_mask = torch.cat(
                    [
                        prefix_mask,
                        torch.ones_like(continuation_ids, dtype=prefix_mask.dtype),
                    ],
                    dim=1,
                )
                continuation_outputs = self.model.model(
                    input_ids=continuation_ids,
                    attention_mask=continuation_mask,
                    past_key_values=cache,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                continuation_logits = self.model.lm_head(
                    continuation_outputs.last_hidden_state
                )[0]
                interrupt_logits = torch.cat(
                    [silent_logits[0:1], continuation_logits], dim=0
                )
            else:
                interrupt_logits = silent_logits[0:1]
            interrupt_logp = tag_sequence_log_probability(
                interrupt_logits, self.interrupt_token_ids
            )
        causal_hidden = causal_hidden.detach().cpu()
        return self._finalize_features(
            silent_hidden=causal_hidden,
            interrupt_hidden=causal_hidden,
            silent_logp=silent_logp,
            interrupt_logp=interrupt_logp,
            prompt_tokens=prompt_tokens,
            extraction_mode="prefix_cache",
            candidate_forward_passes=2,
        )

    def _extract_decision_features_shared_vision(
        self,
        frames: list[object],
        prompt: str,
    ) -> NeuralDecisionFeatures:
        """Compute vision features once, then preserve two batch-one language passes."""
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
        prompt_tokens = validate_tag_suffix(
            inputs["input_ids"], self.silent_token_ids
        )
        interrupt_input_ids = inputs["input_ids"].clone()
        interrupt_input_ids[0, -len(self.interrupt_token_ids) :] = torch.tensor(
            self.interrupt_token_ids,
            dtype=torch.long,
            device=self.device,
        )
        interrupt_prompt_tokens = validate_tag_suffix(
            interrupt_input_ids, self.interrupt_token_ids
        )
        if interrupt_prompt_tokens != prompt_tokens:
            raise RuntimeError("Shared-vision candidates have different prompt lengths")

        tag_length = len(self.silent_token_ids)
        with torch.inference_mode():
            image_features = None
            pixel_values = inputs.get("pixel_values")
            if pixel_values is not None:
                image_features = self.model.model.get_image_features(
                    pixel_values=pixel_values,
                    return_dict=True,
                ).pooler_output

            def language_hidden(input_ids: object) -> object:
                if not isinstance(input_ids, torch.Tensor):
                    raise TypeError("Shared-vision input IDs must be a tensor")
                embeddings = self.model.model.get_input_embeddings()(input_ids)
                if image_features is not None:
                    projected = image_features.to(
                        embeddings.device, embeddings.dtype
                    )
                    image_mask = self.model.model.get_placeholder_mask(
                        input_ids,
                        inputs_embeds=embeddings,
                        image_features=projected,
                    )
                    embeddings = embeddings.masked_scatter(image_mask, projected)
                outputs = self.model.model.language_model(
                    attention_mask=inputs.get("attention_mask"),
                    inputs_embeds=embeddings,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                return outputs.last_hidden_state

            silent_hidden_all = language_hidden(inputs["input_ids"])
            interrupt_hidden_all = language_hidden(interrupt_input_ids)
            if not isinstance(silent_hidden_all, torch.Tensor) or not isinstance(
                interrupt_hidden_all, torch.Tensor
            ):
                raise TypeError("Shared-vision language hidden states must be tensors")
            expected_shape = inputs["input_ids"].shape
            if silent_hidden_all.shape[:2] != expected_shape:
                raise RuntimeError("Shared-vision silent hidden/input lengths differ")
            if interrupt_hidden_all.shape[:2] != expected_shape:
                raise RuntimeError("Shared-vision interrupt hidden/input lengths differ")
            if silent_hidden_all.shape[2] != self.hidden_size:
                raise RuntimeError("Shared-vision hidden width differs from model config")
            silent_hidden = silent_hidden_all[0, prompt_tokens - 1].float()
            interrupt_hidden = interrupt_hidden_all[0, prompt_tokens - 1].float()
            silent_logits = self.model.lm_head(
                silent_hidden_all[:, -(tag_length + 1) : -1, :]
            )[0]
            interrupt_logits = self.model.lm_head(
                interrupt_hidden_all[:, -(tag_length + 1) : -1, :]
            )[0]
            silent_logp = tag_sequence_log_probability(
                silent_logits, self.silent_token_ids
            )
            interrupt_logp = tag_sequence_log_probability(
                interrupt_logits, self.interrupt_token_ids
            )
        return self._finalize_features(
            silent_hidden=silent_hidden.detach().cpu(),
            interrupt_hidden=interrupt_hidden.detach().cpu(),
            silent_logp=silent_logp,
            interrupt_logp=interrupt_logp,
            prompt_tokens=prompt_tokens,
            extraction_mode="shared_vision",
            candidate_forward_passes=2,
        )

    def extract_decision_features(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        mode: DecisionFeatureMode | None = None,
    ) -> NeuralDecisionFeatures:
        selected = self.decision_feature_mode if mode is None else mode
        if selected not in DECISION_FEATURE_MODES:
            raise ValueError(f"Unsupported D1 decision feature mode: {selected}")
        prompt = self._causal_prompt(frames, messages)
        if selected == "batched":
            return self._extract_decision_features_batched(frames, prompt)
        if selected == "prefix_cache":
            return self._extract_decision_features_prefix_cache(frames, prompt)
        if selected == "shared_vision":
            return self._extract_decision_features_shared_vision(frames, prompt)
        return self._extract_decision_features_sequential(frames, prompt)
