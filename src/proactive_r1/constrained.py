"""Label-independent grammar constraint for exact proactive response tags."""

from __future__ import annotations

from collections.abc import Sequence

from proactive_r0.core import INTERRUPT_TAG, SILENT_TAG
from proactive_r0.internvl import InternVLProactiveModel


class TagPrefixConstraint:
    """Finite-state prefix grammar for `$silent$` or `$interrupt$`."""

    def __init__(
        self,
        silent_token_ids: Sequence[int],
        interrupt_token_ids: Sequence[int],
        eos_token_id: int,
    ) -> None:
        self.silent = tuple(int(token_id) for token_id in silent_token_ids)
        self.interrupt = tuple(int(token_id) for token_id in interrupt_token_ids)
        self.eos_token_id = int(eos_token_id)
        if not self.silent or not self.interrupt:
            raise ValueError("Tag token sequences must be non-empty")
        if self.silent == self.interrupt:
            raise ValueError("Silent and interrupt token sequences must differ")

    def allowed_next(self, generated_token_ids: Sequence[int]) -> tuple[int, ...] | None:
        """Return allowed next tokens, or None after a complete interrupt tag."""
        prefix = tuple(int(token_id) for token_id in generated_token_ids)
        if prefix == self.silent:
            return (self.eos_token_id,)
        if len(prefix) >= len(self.interrupt) and prefix[: len(self.interrupt)] == self.interrupt:
            return None
        candidates = [
            sequence
            for sequence in (self.silent, self.interrupt)
            if len(prefix) < len(sequence) and sequence[: len(prefix)] == prefix
        ]
        if not candidates:
            raise RuntimeError(f"Generated tokens escaped the tag grammar: {prefix}")
        return tuple(sorted({sequence[len(prefix)] for sequence in candidates}))


class TagPrefixLogitsProcessor:
    """Transformers-compatible logits processor backed by TagPrefixConstraint."""

    def __init__(self, prompt_length: int, constraint: TagPrefixConstraint) -> None:
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        self.prompt_length = prompt_length
        self.constraint = constraint

    def __call__(self, input_ids: object, scores: object) -> object:
        import torch

        if not isinstance(input_ids, torch.Tensor) or not isinstance(scores, torch.Tensor):
            raise TypeError("TagPrefixLogitsProcessor expects torch tensors")
        if input_ids.ndim != 2 or scores.ndim != 2:
            raise ValueError("Expected [batch, sequence] IDs and [batch, vocab] scores")
        constrained_scores = scores.clone()
        for batch_index in range(input_ids.shape[0]):
            generated = input_ids[batch_index, self.prompt_length :].tolist()
            allowed = self.constraint.allowed_next(generated)
            if allowed is None:
                continue
            row = torch.full_like(constrained_scores[batch_index], float("-inf"))
            indices = torch.tensor(allowed, device=scores.device, dtype=torch.long)
            row[indices] = scores[batch_index, indices]
            constrained_scores[batch_index] = row
        return constrained_scores


class ForcedPrefixConstraint:
    """Force one finite token prefix, then release all subsequent tokens."""

    def __init__(self, token_ids: Sequence[int]) -> None:
        self.prefix = tuple(int(token_id) for token_id in token_ids)
        if not self.prefix:
            raise ValueError("Forced token prefix must be non-empty")

    def allowed_next(self, generated_token_ids: Sequence[int]) -> tuple[int, ...] | None:
        generated = tuple(int(token_id) for token_id in generated_token_ids)
        if len(generated) >= len(self.prefix):
            if generated[: len(self.prefix)] != self.prefix:
                raise RuntimeError(f"Generated tokens escaped forced prefix: {generated}")
            return None
        if self.prefix[: len(generated)] != generated:
            raise RuntimeError(f"Generated tokens escaped forced prefix: {generated}")
        return (self.prefix[len(generated)],)


class ForcedPrefixLogitsProcessor:
    def __init__(self, prompt_length: int, constraint: ForcedPrefixConstraint) -> None:
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        self.prompt_length = prompt_length
        self.constraint = constraint

    def __call__(self, input_ids: object, scores: object) -> object:
        import torch

        if not isinstance(input_ids, torch.Tensor) or not isinstance(scores, torch.Tensor):
            raise TypeError("ForcedPrefixLogitsProcessor expects torch tensors")
        constrained_scores = scores.clone()
        for batch_index in range(input_ids.shape[0]):
            generated = input_ids[batch_index, self.prompt_length :].tolist()
            allowed = self.constraint.allowed_next(generated)
            if allowed is None:
                continue
            row = torch.full_like(constrained_scores[batch_index], float("-inf"))
            token_id = allowed[0]
            row[token_id] = scores[batch_index, token_id]
            constrained_scores[batch_index] = row
        return constrained_scores


class ConstrainedInternVLProactiveModel(InternVLProactiveModel):
    """Frozen InternVL adapter with a required proactive-tag prefix grammar."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        tokenizer = self.processor.tokenizer
        silent_token_ids = tokenizer.encode(SILENT_TAG, add_special_tokens=False)
        interrupt_token_ids = tokenizer.encode(INTERRUPT_TAG, add_special_tokens=False)
        if tokenizer.decode(silent_token_ids, skip_special_tokens=False) != SILENT_TAG:
            raise ValueError("Silent tag does not round-trip through the tokenizer")
        if tokenizer.decode(interrupt_token_ids, skip_special_tokens=False) != INTERRUPT_TAG:
            raise ValueError("Interrupt tag does not round-trip through the tokenizer")
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no EOS token for constrained silent output")
        self.tag_constraint = TagPrefixConstraint(
            silent_token_ids,
            interrupt_token_ids,
            tokenizer.eos_token_id,
        )
        self.tag_token_ids = {
            "silent": silent_token_ids,
            "interrupt": interrupt_token_ids,
            "eos": tokenizer.eos_token_id,
        }

    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> str:
        import torch
        from transformers import LogitsProcessorList

        multimodal = self._to_multimodal_messages(frames, messages)
        prompt = self.processor.apply_chat_template(
            multimodal,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs: dict[str, object] = {
            "text": [prompt],
            "padding": True,
            "return_tensors": "pt",
        }
        if frames:
            processor_kwargs["videos"] = [frames]
        inputs = self.processor(**processor_kwargs)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        logits_processor = LogitsProcessorList(
            [TagPrefixLogitsProcessor(prompt_length, self.tag_constraint)]
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.pad_token_id,
                logits_processor=logits_processor,
            )
        generated_ids = output_ids[0, prompt_length:]
        decoded = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        if not (decoded == SILENT_TAG or decoded.startswith(INTERRUPT_TAG)):
            raise RuntimeError(f"Tag grammar produced an invalid response: {decoded!r}")
        return decoded


class SequenceChoiceConstrainedInternVLProactiveModel(ConstrainedInternVLProactiveModel):
    """Choose the complete tag by beam score, then generate interrupt text."""

    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> str:
        import torch
        from transformers import LogitsProcessorList

        multimodal = self._to_multimodal_messages(frames, messages)
        prompt = self.processor.apply_chat_template(
            multimodal,
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs: dict[str, object] = {
            "text": [prompt],
            "padding": True,
            "return_tensors": "pt",
        }
        if frames:
            processor_kwargs["videos"] = [frames]
        inputs = self.processor(**processor_kwargs)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        choice_processor = LogitsProcessorList(
            [TagPrefixLogitsProcessor(prompt_length, self.tag_constraint)]
        )
        tag_length = max(
            len(self.tag_token_ids["silent"]),
            len(self.tag_token_ids["interrupt"]),
        )
        if len(self.tag_token_ids["silent"]) != len(self.tag_token_ids["interrupt"]):
            raise ValueError("Sequence-choice grammar requires equal-length tag encodings")
        with torch.inference_mode():
            choice_ids = self.model.generate(
                **inputs,
                max_new_tokens=tag_length,
                do_sample=False,
                num_beams=2,
                num_return_sequences=1,
                length_penalty=0.0,
                use_cache=True,
                pad_token_id=self.pad_token_id,
                logits_processor=choice_processor,
            )
        selected_ids = choice_ids[0, prompt_length:].tolist()
        selected_tag = self.processor.decode(
            selected_ids,
            skip_special_tokens=True,
        ).strip()
        if selected_tag == SILENT_TAG:
            return SILENT_TAG
        if selected_tag != INTERRUPT_TAG:
            raise RuntimeError(f"Sequence-choice grammar selected invalid tag: {selected_tag!r}")
        if max_new_tokens <= tag_length:
            return INTERRUPT_TAG

        forced_processor = LogitsProcessorList(
            [
                ForcedPrefixLogitsProcessor(
                    prompt_length,
                    ForcedPrefixConstraint(self.tag_token_ids["interrupt"]),
                )
            ]
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.pad_token_id,
                logits_processor=forced_processor,
            )
        generated_ids = output_ids[0, prompt_length:]
        decoded = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        if not decoded.startswith(INTERRUPT_TAG):
            raise RuntimeError(f"Forced interrupt prefix produced invalid text: {decoded!r}")
        return decoded
