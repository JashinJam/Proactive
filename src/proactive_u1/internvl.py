"""InternVL adapter that continues an assistant-side decision prefix."""

from __future__ import annotations

from typing import Sequence

from proactive_r0.internvl import InternVLProactiveModel


def append_text_prefix(
    inputs: dict[str, object], prefix_ids: object
) -> tuple[dict[str, object], int, int]:
    """Append token IDs to supported sequence tensors without touching vision inputs."""
    import torch

    if "input_ids" not in inputs or "attention_mask" not in inputs:
        raise ValueError("Processor inputs lack input_ids/attention_mask")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if not isinstance(input_ids, torch.Tensor) or not isinstance(
        attention_mask, torch.Tensor
    ):
        raise TypeError("Processor sequence inputs must be torch tensors")
    if not isinstance(prefix_ids, torch.Tensor) or prefix_ids.ndim != 2:
        raise TypeError("Assistant prefix IDs must be a rank-2 torch tensor")
    if input_ids.shape[0] != prefix_ids.shape[0]:
        raise ValueError("Assistant prefix batch size differs from prompt batch")
    original_length = int(input_ids.shape[1])
    prefix_length = int(prefix_ids.shape[1])
    if prefix_length <= 0:
        raise ValueError("Assistant prefix must contain at least one token")
    result = dict(inputs)
    result["input_ids"] = torch.cat([input_ids, prefix_ids], dim=1)
    suffix_mask = torch.ones(
        (attention_mask.shape[0], prefix_length),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    result["attention_mask"] = torch.cat([attention_mask, suffix_mask], dim=1)
    if "token_type_ids" in result:
        token_types = result["token_type_ids"]
        if not isinstance(token_types, torch.Tensor):
            raise TypeError("token_type_ids must be a torch tensor")
        suffix_types = token_types[:, -1:].expand(-1, prefix_length)
        result["token_type_ids"] = torch.cat([token_types, suffix_types], dim=1)
    if "position_ids" in result:
        positions = result["position_ids"]
        if not isinstance(positions, torch.Tensor):
            raise TypeError("position_ids must be a torch tensor")
        increments = torch.arange(
            1,
            prefix_length + 1,
            dtype=positions.dtype,
            device=positions.device,
        ).unsqueeze(0)
        result["position_ids"] = torch.cat(
            [positions, positions[:, -1:] + increments], dim=1
        )
    supported = {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
    for name, value in result.items():
        if (
            name not in supported
            and isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and value.shape[-1] == original_length
        ):
            raise ValueError(f"Unhandled prompt-length processor tensor: {name}")
    return result, original_length, prefix_length


class PrefillInternVLProactiveModel(InternVLProactiveModel):
    def generate_prefilled(
        self,
        frames: Sequence[object],
        messages: list[dict[str, str]],
        assistant_prefix: str,
        max_new_tokens: int,
    ) -> dict[str, object]:
        import torch

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
            processor_kwargs["videos"] = [list(frames)]
        batch = self.processor(**processor_kwargs)
        inputs = {name: value.to(self.device) for name, value in batch.items()}
        prefix = self.processor.tokenizer(
            assistant_prefix,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)
        inputs, prompt_tokens, prefix_tokens = append_text_prefix(inputs, prefix)
        input_length = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.pad_token_id,
            )
        generated_ids = output_ids[0, input_length:]
        continuation = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        return {
            "continuation": continuation,
            "prompt_tokens": prompt_tokens,
            "assistant_prefix_tokens": prefix_tokens,
            "generated_tokens": int(generated_ids.numel()),
        }
