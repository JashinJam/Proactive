"""Equal-length fixed-candidate state scoring with one shared vision pass."""

from __future__ import annotations

from typing import Sequence

from proactive_d1.internvl_features import (
    tag_sequence_log_probability,
    validate_tag_suffix,
)
from proactive_r0.internvl import InternVLProactiveModel
from proactive_state_s0.core import CANDIDATE_TEXT, STATE_TARGETS


class InternVLStateCandidateScorer(InternVLProactiveModel):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        tokenizer = self.processor.tokenizer
        self.candidate_token_ids: dict[str, list[list[int]]] = {}
        for target in STATE_TARGETS:
            encoded = [
                tokenizer.encode(value, add_special_tokens=False)
                for value in CANDIDATE_TEXT[target]
            ]
            if any(not ids for ids in encoded) or len({len(ids) for ids in encoded}) != 1:
                raise ValueError(f"S0 {target} candidates are not non-empty equal length")
            if len({tuple(ids) for ids in encoded}) != len(encoded):
                raise ValueError(f"S0 {target} candidates do not tokenize uniquely")
            for value, ids in zip(CANDIDATE_TEXT[target], encoded):
                if tokenizer.decode(ids, skip_special_tokens=False) != value:
                    raise ValueError(f"S0 candidate does not round-trip: {value!r}")
            self.candidate_token_ids[target] = encoded

    def score_state(
        self,
        frames: list[object],
        messages_by_target: dict[str, Sequence[dict[str, str]]],
    ) -> dict[str, object]:
        import torch

        image_features = None
        scores: dict[str, list[float]] = {}
        prompt_tokens: dict[str, int] = {}
        vision_forward_passes = 0
        language_forward_passes = 0
        with torch.inference_mode():
            for target in STATE_TARGETS:
                messages = list(messages_by_target[target])
                multimodal = self._to_multimodal_messages(frames, messages)
                prompt = self.processor.apply_chat_template(
                    multimodal,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                target_scores: list[float] = []
                candidates = CANDIDATE_TEXT[target]
                token_sequences = self.candidate_token_ids[target]
                kwargs: dict[str, object] = {
                    "text": [prompt + candidates[0]],
                    "padding": True,
                    "return_tensors": "pt",
                }
                if frames:
                    kwargs["videos"] = [frames]
                batch = self.processor(**kwargs)
                inputs = {name: value.to(self.device) for name, value in batch.items()}
                target_prompt_tokens = validate_tag_suffix(
                    inputs["input_ids"], token_sequences[0]
                )
                if image_features is None and inputs.get("pixel_values") is not None:
                    image_features = self.model.model.get_image_features(
                        pixel_values=inputs["pixel_values"],
                        return_dict=True,
                    ).pooler_output
                    vision_forward_passes += 1
                for token_ids in token_sequences:
                    candidate_input_ids = inputs["input_ids"].clone()
                    candidate_input_ids[0, -len(token_ids) :] = torch.tensor(
                        token_ids, dtype=torch.long, device=self.device
                    )
                    current_prompt_tokens = validate_tag_suffix(
                        candidate_input_ids, token_ids
                    )
                    if target_prompt_tokens != current_prompt_tokens:
                        raise RuntimeError("S0 candidate prompt lengths differ")
                    embeddings = self.model.model.get_input_embeddings()(
                        candidate_input_ids
                    )
                    if image_features is not None:
                        projected = image_features.to(embeddings.device, embeddings.dtype)
                        image_mask = self.model.model.get_placeholder_mask(
                            candidate_input_ids,
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
                    hidden = outputs.last_hidden_state
                    length = len(token_ids)
                    logits = self.model.lm_head(
                        hidden[:, -(length + 1) : -1, :]
                    )[0]
                    target_scores.append(
                        tag_sequence_log_probability(logits, token_ids)
                    )
                    language_forward_passes += 1
                scores[target] = target_scores
                prompt_tokens[target] = target_prompt_tokens
        if frames and vision_forward_passes != 1:
            raise RuntimeError("S0 must perform exactly one shared vision pass per state")
        return {
            "scores": scores,
            "prompt_tokens": prompt_tokens,
            "vision_forward_passes": vision_forward_passes,
            "language_forward_passes": language_forward_passes,
            "candidate_token_ids": self.candidate_token_ids,
        }

