"""Verify exact fixed-shape zero-adapter replay on one extracted cache session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from proactive_d1.internvl_features import InternVLDecisionFeatureExtractor
from proactive_r0.artifacts import sha256_file, write_json

from .final_mlp_cache import STATE_ARRAY_NAMES
from .final_mlp_lora import configure_final_mlp_lora, internvl_lora_components
from .final_mlp_training import (
    FinalMLPCacheArrays,
    adapter_batch_outputs,
    export_adapter_features,
    fixed_shape_batches,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/d2_internvl35_1b_final_mlp_lora_oof.json"


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_session(path: Path) -> FinalMLPCacheArrays:
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
            raise ValueError("Replay smoke cache has unexpected arrays")
        arrays = {name: archive[name].copy() for name in archive.files}
    rows = int(arrays["base_tag_margin"].shape[0])
    input_index = arrays["input_index"]
    if input_index.shape != () or rows <= 1:
        raise ValueError("Replay smoke requires one multi-chunk session cache")
    return FinalMLPCacheArrays(
        state_bits={name: arrays[name] for name in STATE_ARRAY_NAMES},
        base_hidden_state=arrays["base_hidden_state"].astype(np.float32, copy=False),
        base_tag_margin=arrays["base_tag_margin"].astype(np.float32, copy=False),
        prompt_tokens=arrays["prompt_tokens"].astype(np.int32, copy=False),
        input_index=np.full(rows, int(input_index), dtype=np.int32),
        chunk_index=arrays["chunk_index"].astype(np.int32, copy=False),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--session-cache", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = dict(config["model"])
    cache_config = dict(config["cache"])
    adapter_config = dict(config["adapter"])
    training_config = dict(config["training"])
    batch_size = int(training_config["local_replay_batch_size"])
    cache_path = _resolve(args.session_cache)
    cache = _load_session(cache_path)
    if cache.rows >= batch_size:
        raise ValueError("Replay smoke must exercise fixed-shape tail padding")

    model_path = _resolve(args.model_path or model_config["default_local_path"])
    extractor = InternVLDecisionFeatureExtractor(
        model_path=str(model_path),
        device=args.device,
        dtype_name=str(model_config["dtype"]),
        attention_implementation=str(model_config["attention_implementation"]),
        seed=int(training_config["seed"]),
        require_exclusive_gpu=False,
        video_frame_size=int(cache_config["video_frame_size"]),
        pad_token_id=int(cache_config["pad_token_id"]),
    )
    peft_model, parameter_audit = configure_final_mlp_lora(
        extractor.model,
        layer_index=int(adapter_config["language_layer_index"]),
        hidden_size=int(cache_config["hidden_size"]),
        intermediate_size=int(model_config["intermediate_size"]),
        rank=int(adapter_config["rank"]),
        alpha=int(adapter_config["alpha"]),
        dropout=float(adapter_config["dropout"]),
    )
    _, decoder_layer, final_norm, lm_head = internvl_lora_components(
        peft_model, int(adapter_config["language_layer_index"])
    )
    margin, hidden, candidate_difference = export_adapter_features(
        cache,
        np.arange(cache.rows, dtype=np.int64),
        batch_size=batch_size,
        peft_model=peft_model,
        decoder_layer=decoder_layer,
        final_norm=final_norm,
        lm_head=lm_head,
        silent_token_ids=extractor.silent_token_ids,
        interrupt_token_ids=extractor.interrupt_token_ids,
        device=extractor.device,
    )
    import torch
    import torch.nn.functional as functional

    padded, real_count = fixed_shape_batches(
        np.arange(cache.rows, dtype=np.int64), batch_size
    )[0]
    peft_model.zero_grad(set_to_none=True)
    gradient_margin, _, gradient_candidate_difference = adapter_batch_outputs(
        cache,
        padded,
        peft_model=peft_model,
        decoder_layer=decoder_layer,
        final_norm=final_norm,
        lm_head=lm_head,
        silent_token_ids=extractor.silent_token_ids,
        interrupt_token_ids=extractor.interrupt_token_ids,
        device=extractor.device,
    )
    synthetic_labels = torch.tensor(
        np.arange(real_count) % 2,
        dtype=torch.float32,
        device=extractor.device,
    )
    gradient_loss = functional.binary_cross_entropy_with_logits(
        gradient_margin[:real_count].float(), synthetic_labels
    )
    gradient_loss.backward()
    gradients = [
        parameter.grad.detach().float()
        for parameter in peft_model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or any(not torch.isfinite(value).all() for value in gradients):
        raise RuntimeError("Fixed-shape adapter gradient is missing or non-finite")
    gradient_abs_max = max(float(value.abs().max().cpu()) for value in gradients)
    gradient_nonzero = sum(int(torch.count_nonzero(value).cpu()) for value in gradients)
    if gradient_abs_max == 0.0 or gradient_nonzero == 0:
        raise RuntimeError("Fixed-shape adapter gradient is identically zero")
    result = {
        "status": "exact zero-adapter fixed-shape replay passed",
        "session_cache": str(cache_path),
        "session_cache_sha256": sha256_file(cache_path),
        "rows": cache.rows,
        "fixed_replay_batch_size": batch_size,
        "tail_padding_rows": batch_size - cache.rows,
        "max_margin_abs_difference": float(
            np.max(np.abs(margin - cache.base_tag_margin))
        ),
        "max_hidden_abs_difference": float(
            np.max(np.abs(hidden - cache.base_hidden_state))
        ),
        "max_candidate_hidden_abs_difference": candidate_difference,
        "gradient_smoke_loss": float(gradient_loss.detach().cpu()),
        "gradient_abs_max": gradient_abs_max,
        "gradient_nonzero_elements": gradient_nonzero,
        "gradient_candidate_hidden_abs_difference": gradient_candidate_difference,
        "trainable_parameters": parameter_audit.trainable_parameters,
        "device": args.device,
    }
    if any(
        result[name] != 0.0
        for name in (
            "max_margin_abs_difference",
            "max_hidden_abs_difference",
            "max_candidate_hidden_abs_difference",
            "gradient_candidate_hidden_abs_difference",
        )
    ):
        raise RuntimeError(f"Fixed-shape zero-adapter replay differs: {result}")
    if args.output:
        write_json(_resolve(args.output), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
