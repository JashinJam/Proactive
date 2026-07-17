"""InternVL3.5 HF adapter for causal frame sequences."""

from __future__ import annotations

import os
import warnings
from typing import Sequence


def resolve_physical_cuda_identifier(
    device: object,
    visible_devices: str | None,
) -> int | str:
    """Map a process-local CUDA ordinal to its physical NVML identifier."""
    import torch

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {torch_device}")
    logical_index = torch_device.index if torch_device.index is not None else 0
    if visible_devices is None:
        return logical_index
    entries = [entry.strip() for entry in visible_devices.split(",") if entry.strip()]
    if logical_index >= len(entries):
        raise RuntimeError(
            f"Logical CUDA device {logical_index} is absent from "
            f"CUDA_VISIBLE_DEVICES={visible_devices!r}"
        )
    entry = entries[logical_index]
    return int(entry) if entry.isdigit() else entry


def check_cuda_device_occupancy(
    device: object,
    require_exclusive: bool = False,
) -> list[dict[str, int]]:
    """Inspect existing compute processes and optionally require exclusivity."""
    import pynvml
    import torch

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return []
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical_identifier = resolve_physical_cuda_identifier(
        torch_device, visible_devices
    )

    pynvml.nvmlInit()
    try:
        if isinstance(physical_identifier, int):
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_identifier)
        else:
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(physical_identifier)
            except TypeError:
                handle = pynvml.nvmlDeviceGetHandleByUUID(
                    physical_identifier.encode("ascii")
                )
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        foreign = [process for process in processes if process.pid != os.getpid()]
        occupancy = [
            {
                "pid": int(process.pid),
                "used_memory_bytes": int(getattr(process, "usedGpuMemory", 0)),
            }
            for process in foreign
        ]
        if foreign:
            details = ", ".join(
                f"pid={process.pid}, used_memory={getattr(process, 'usedGpuMemory', 'unknown')}"
                for process in foreign
            )
            message = (
                "Existing compute processes on physical GPU "
                f"{physical_identifier}: {details}"
            )
            if require_exclusive:
                raise RuntimeError(f"Exclusive GPU required. {message}")
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return occupancy
    finally:
        pynvml.nvmlShutdown()


class InternVLProactiveModel:
    def __init__(
        self,
        model_path: str,
        device: str,
        dtype_name: str = "bfloat16",
        attention_implementation: str = "sdpa",
        seed: int = 20260713,
        require_exclusive_gpu: bool = False,
        video_frame_size: int = 448,
        pad_token_id: int = 151643,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_by_name:
            raise ValueError(f"Unsupported dtype: {dtype_name}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.device = torch.device(device)
        self.preexisting_gpu_processes = check_cuda_device_occupancy(
            self.device,
            require_exclusive=require_exclusive_gpu,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        from transformers.image_utils import SizeDict

        self.processor.video_processor.size = SizeDict(
            height=video_frame_size,
            width=video_frame_size,
        )
        self.processor.tokenizer.padding_side = "left"
        tokenizer_pad_token_id = self.processor.tokenizer.pad_token_id
        if tokenizer_pad_token_id != pad_token_id:
            raise ValueError(
                "Configured pad token does not match the checkpoint tokenizer: "
                f"{pad_token_id} != {tokenizer_pad_token_id}"
            )
        self.pad_token_id = pad_token_id
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype_by_name[dtype_name],
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation=attention_implementation,
        ).eval()
        self.model.to(self.device)
        vision_size = tuple(self.model.config.vision_config.image_size)
        if vision_size != (video_frame_size, video_frame_size):
            raise ValueError(
                f"Video frame size {video_frame_size} does not match vision tower "
                f"size {vision_size}"
            )
        self.video_frame_size = video_frame_size
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    @staticmethod
    def _to_multimodal_messages(
        frames: Sequence[object],
        messages: list[dict[str, str]],
    ) -> list[dict[str, object]]:
        multimodal: list[dict[str, object]] = []
        inserted_video = False
        for message in messages:
            role = message["role"]
            text = message["content"]
            if role == "user" and frames and not inserted_video:
                multimodal.append(
                    {
                        "role": role,
                        "content": [
                            {"type": "video"},
                            {"type": "text", "text": text},
                        ],
                    }
                )
                inserted_video = True
            else:
                multimodal.append({"role": role, "content": text})
        if frames and not inserted_video:
            raise ValueError("Video frames are present but no user message can hold them")
        return multimodal

    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> str:
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
            processor_kwargs["videos"] = [frames]
        inputs = self.processor(**processor_kwargs)
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.pad_token_id,
            )
        generated_ids = output_ids[0, prompt_length:]
        return self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

    def peak_memory_bytes(self) -> int | None:
        import torch

        if self.device.type != "cuda":
            return None
        return int(torch.cuda.max_memory_allocated(self.device))
