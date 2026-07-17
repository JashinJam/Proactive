"""Run a bounded exactness and backward smoke for final-language-MLP LoRA."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as functional

from proactive_d1.internvl_features import (
    InternVLDecisionFeatureExtractor,
    validate_tag_suffix,
)
from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    verify_model_snapshot,
    write_json,
)
from proactive_r0.core import (
    CausalInferenceConfig,
    INTERRUPT_TAG,
    SILENT_TAG,
    build_messages,
    load_jsonl,
    load_starter_kit,
    subsample_frames,
)

from .final_mlp_lora import (
    CandidateForward,
    FinalMLPScoringState,
    FinalMLPStateCapture,
    configure_final_mlp_lora,
    decision_margin_from_logits,
    final_mlp_cache_bytes_per_chunk,
    internvl_lora_components,
    reconstruct_final_hidden,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/d2_internvl35_1b_final_mlp_lora_smoke_v2.json"
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _max_abs(first: object, second: object) -> float:
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError("Exactness comparison requires tensors")
    if first.shape != second.shape:
        raise ValueError(f"Exactness shapes differ: {first.shape} != {second.shape}")
    return float((first.float() - second.float()).abs().max().detach().cpu())


def _state_to_device(state: FinalMLPScoringState, device: torch.device) -> FinalMLPScoringState:
    if not isinstance(state.residual, torch.Tensor) or not isinstance(
        state.normalized, torch.Tensor
    ):
        raise TypeError("Cached final MLP state must contain tensors")
    return FinalMLPScoringState(
        residual=state.residual.to(device),
        normalized=state.normalized.to(device),
        reference_mlp_output=(
            state.reference_mlp_output.to(device)
            if isinstance(state.reference_mlp_output, torch.Tensor)
            else None
        ),
        local_base_mlp_output=(
            state.local_base_mlp_output.to(device)
            if isinstance(state.local_base_mlp_output, torch.Tensor)
            else None
        ),
        reference_final_hidden=(
            state.reference_final_hidden.to(device)
            if isinstance(state.reference_final_hidden, torch.Tensor)
            else None
        ),
        local_base_final_hidden=(
            state.local_base_final_hidden.to(device)
            if isinstance(state.local_base_final_hidden, torch.Tensor)
            else None
        ),
    )


def _state_to_cpu(state: FinalMLPScoringState) -> FinalMLPScoringState:
    if not isinstance(state.residual, torch.Tensor) or not isinstance(
        state.normalized, torch.Tensor
    ):
        raise TypeError("Cached final MLP state must contain tensors")
    return FinalMLPScoringState(
        residual=state.residual.detach().cpu().clone(),
        normalized=state.normalized.detach().cpu().clone(),
        reference_mlp_output=(
            state.reference_mlp_output.detach().cpu().clone()
            if isinstance(state.reference_mlp_output, torch.Tensor)
            else None
        ),
        local_base_mlp_output=(
            state.local_base_mlp_output.detach().cpu().clone()
            if isinstance(state.local_base_mlp_output, torch.Tensor)
            else None
        ),
        reference_final_hidden=(
            state.reference_final_hidden.detach().cpu().clone()
            if isinstance(state.reference_final_hidden, torch.Tensor)
            else None
        ),
        local_base_final_hidden=(
            state.local_base_final_hidden.detach().cpu().clone()
            if isinstance(state.local_base_final_hidden, torch.Tensor)
            else None
        ),
    )


def _stack_states(states: Sequence[FinalMLPScoringState], device: torch.device) -> FinalMLPScoringState:
    if not states:
        raise ValueError("Cannot stack an empty final MLP cache")
    on_device = [_state_to_device(state, device) for state in states]
    if any(
        state.reference_mlp_output is None or state.local_base_mlp_output is None
        for state in on_device
    ):
        raise ValueError("Cached LoRA training states require base MLP correction tensors")
    return FinalMLPScoringState(
        residual=torch.cat(
            [state.residual for state in on_device], dim=0  # type: ignore[list-item]
        ),
        normalized=torch.cat(
            [state.normalized for state in on_device], dim=0  # type: ignore[list-item]
        ),
        reference_mlp_output=torch.cat(
            [state.reference_mlp_output for state in on_device], dim=0  # type: ignore[list-item]
        ),
        local_base_mlp_output=torch.cat(
            [state.local_base_mlp_output for state in on_device], dim=0  # type: ignore[list-item]
        ),
        reference_final_hidden=(
            torch.cat(
                [state.reference_final_hidden for state in on_device], dim=0  # type: ignore[list-item]
            )
            if all(state.reference_final_hidden is not None for state in on_device)
            else None
        ),
        local_base_final_hidden=(
            torch.cat(
                [state.local_base_final_hidden for state in on_device], dim=0  # type: ignore[list-item]
            )
            if all(state.local_base_final_hidden is not None for state in on_device)
            else None
        ),
    )


def _prepare_shared_vision_inputs(
    extractor: InternVLDecisionFeatureExtractor,
    outer_model: object,
    frames: list[object],
    prompt: str,
) -> dict[str, object]:
    processor_kwargs: dict[str, object] = {
        "text": [prompt + SILENT_TAG],
        "padding": True,
        "return_tensors": "pt",
    }
    if frames:
        processor_kwargs["videos"] = [frames]
    inputs = extractor.processor(**processor_kwargs)
    inputs = {name: value.to(extractor.device) for name, value in inputs.items()}
    input_ids = inputs["input_ids"]
    prompt_tokens = validate_tag_suffix(input_ids, extractor.silent_token_ids)
    interrupt_ids = input_ids.clone()
    interrupt_ids[0, -len(extractor.interrupt_token_ids) :] = torch.tensor(
        extractor.interrupt_token_ids,
        dtype=torch.long,
        device=extractor.device,
    )
    if validate_tag_suffix(interrupt_ids, extractor.interrupt_token_ids) != prompt_tokens:
        raise RuntimeError("Silent and interrupt candidate prompt lengths differ")

    internvl = getattr(outer_model, "model")
    with torch.no_grad():
        image_features = None
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            image_features = internvl.get_image_features(
                pixel_values=pixel_values,
                return_dict=True,
            ).pooler_output

        def embeddings(candidate_ids: torch.Tensor) -> torch.Tensor:
            result = internvl.get_input_embeddings()(candidate_ids)
            if image_features is not None:
                projected = image_features.to(result.device, result.dtype)
                image_mask = internvl.get_placeholder_mask(
                    candidate_ids,
                    inputs_embeds=result,
                    image_features=projected,
                )
                result = result.masked_scatter(image_mask, projected)
            return result.detach()

        silent_embeddings = embeddings(input_ids)
        interrupt_embeddings = embeddings(interrupt_ids)
    return {
        "attention_mask": inputs.get("attention_mask"),
        "silent_embeddings": silent_embeddings,
        "interrupt_embeddings": interrupt_embeddings,
        "prompt_tokens": prompt_tokens,
    }


def _forward_candidate(
    outer_model: object,
    decoder_layer: object,
    lm_head: object,
    *,
    embeddings: torch.Tensor,
    attention_mask: object,
    token_ids: Sequence[int],
    prompt_tokens: int,
    calibrate_base_correction: bool = False,
) -> CandidateForward:
    language_model = getattr(getattr(outer_model, "model"), "language_model")
    with FinalMLPStateCapture(decoder_layer, len(token_ids)) as capture:
        outputs = language_model(
            attention_mask=attention_mask,
            inputs_embeds=embeddings,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    state = capture.state()
    if calibrate_base_correction:
        local_base_output = decoder_layer.mlp(state.normalized).detach().clone()
        state = FinalMLPScoringState(
            residual=state.residual,
            normalized=state.normalized,
            reference_mlp_output=capture.mlp_output(),
            local_base_mlp_output=local_base_output,
        )
    hidden_all = outputs.last_hidden_state
    hidden = hidden_all[:, -(len(token_ids) + 1) : -1, :].detach().clone()
    logits = lm_head(hidden).detach().clone()
    return CandidateForward(
        state=state,
        hidden=hidden,
        logits=logits,
        prompt_tokens=prompt_tokens,
    )


def _reconstruct_candidate(
    decoder_layer: object,
    final_norm: object,
    lm_head: object,
    state: FinalMLPScoringState,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = reconstruct_final_hidden(decoder_layer, final_norm, state)
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("Reconstructed final hidden state must be a tensor")
    logits = lm_head(hidden)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Reconstructed tag logits must be a tensor")
    return hidden, logits


def _margin(
    silent_logits: object,
    interrupt_logits: object,
    extractor: InternVLDecisionFeatureExtractor,
) -> torch.Tensor:
    value = decision_margin_from_logits(
        silent_logits,
        interrupt_logits,
        extractor.silent_token_ids,
        extractor.interrupt_token_ids,
    )
    if not isinstance(value, torch.Tensor):
        raise TypeError("Decision margin must be a tensor")
    return value


def _generate_disabled(
    peft_model: object,
    outer_model: object,
    extractor: InternVLDecisionFeatureExtractor,
    *,
    frames: list[object],
    prompt: str,
    max_new_tokens: int,
) -> str:
    processor_kwargs: dict[str, object] = {
        "text": [prompt],
        "padding": True,
        "return_tensors": "pt",
    }
    if frames:
        processor_kwargs["videos"] = [frames]
    inputs = extractor.processor(**processor_kwargs)
    inputs = {name: value.to(extractor.device) for name, value in inputs.items()}
    prompt_length = int(inputs["input_ids"].shape[1])
    with peft_model.disable_adapter(), torch.inference_mode():
        output_ids = outer_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=extractor.pad_token_id,
        )
    return extractor.processor.decode(
        output_ids[0, prompt_length:], skip_special_tokens=True
    ).strip()


def _set_nonzero_adapter_b(peft_model: object, seed: int, standard_deviation: float) -> None:
    torch.manual_seed(seed)
    changed = 0
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if parameter.requires_grad and ".lora_B." in name:
                parameter.normal_(mean=0.0, std=standard_deviation)
                changed += 1
    if changed != 3:
        raise RuntimeError(f"Expected three LoRA B tensors, changed {changed}")


def _zero_adapter_b(peft_model: object) -> None:
    changed = 0
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if parameter.requires_grad and ".lora_B." in name:
                parameter.zero_()
                changed += 1
    if changed != 3:
        raise RuntimeError(f"Expected three LoRA B tensors, reset {changed}")


def _train_cached_smoke(
    peft_model: object,
    decoder_layer: object,
    final_norm: object,
    lm_head: object,
    extractor: InternVLDecisionFeatureExtractor,
    silent_states: Sequence[FinalMLPScoringState],
    interrupt_states: Sequence[FinalMLPScoringState],
    labels: Sequence[int],
    training_config: dict[str, object],
) -> dict[str, object]:
    if set(labels) != {0, 1}:
        raise ValueError("Backward smoke requires both binary classes")
    device = extractor.device
    silent_batch = _stack_states(silent_states, device)
    interrupt_batch = _stack_states(interrupt_states, device)
    label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
    positives = int(label_tensor.sum().item())
    negatives = len(labels) - positives
    pos_weight = torch.tensor([negatives / positives], device=device)
    trainable = [
        parameter for parameter in peft_model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    steps = int(training_config["steps"])
    gradient_clip = float(training_config["gradient_clip_norm"])
    losses: list[float] = []
    margin_history: list[list[float]] = []
    gradient_norms: list[float] = []
    zero_gradient_convergence = False

    def forward_loss() -> tuple[torch.Tensor, torch.Tensor]:
        silent_hidden = reconstruct_final_hidden(
            decoder_layer, final_norm, silent_batch
        )
        interrupt_hidden = reconstruct_final_hidden(
            decoder_layer, final_norm, interrupt_batch
        )
        silent_logits = lm_head(silent_hidden)
        interrupt_logits = lm_head(interrupt_hidden)
        margins = _margin(silent_logits, interrupt_logits, extractor)
        loss = functional.binary_cross_entropy_with_logits(
            margins,
            label_tensor,
            pos_weight=pos_weight,
        )
        return loss, margins

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, margins = forward_loss()
        if not torch.isfinite(loss) or not torch.isfinite(margins).all():
            raise RuntimeError(f"Non-finite cached training value at step {step}")
        losses.append(float(loss.detach().cpu()))
        margin_history.append([float(value) for value in margins.detach().cpu()])
        if step == steps:
            break
        loss.backward()
        squared = torch.zeros((), dtype=torch.float32, device=device)
        for parameter in trainable:
            if parameter.grad is not None:
                squared += parameter.grad.float().pow(2).sum()
        gradient_norm = float(squared.sqrt().detach().cpu())
        if not math.isfinite(gradient_norm) or gradient_norm < 0:
            raise RuntimeError(f"Invalid LoRA gradient norm at step {step}")
        gradient_norms.append(gradient_norm)
        if gradient_norm == 0:
            if not any(value > 0 for value in gradient_norms[:-1]):
                raise RuntimeError("LoRA never produced a nonzero gradient")
            zero_gradient_convergence = True
            break
        torch.nn.utils.clip_grad_norm_(trainable, gradient_clip)
        optimizer.step()
    return {
        "requested_steps": steps,
        "optimizer_steps": sum(value > 0 for value in gradient_norms),
        "zero_gradient_convergence": zero_gradient_convergence,
        "labels": list(labels),
        "positive_weight": float(pos_weight.item()),
        "losses": losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_delta": losses[-1] - losses[0],
        "margin_history": margin_history,
        "gradient_norms": gradient_norms,
        "minimum_gradient_norm": min(gradient_norms),
        "maximum_gradient_norm": max(gradient_norms),
    }


def _static_fingerprints(
    config: dict[str, object],
    input_path: Path,
    starter_dir: Path,
    r0_records_path: Path,
) -> dict[str, str]:
    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    r0_config = dict(config["r0_reference"])  # type: ignore[arg-type]
    actual = {
        "input_sha256": sha256_file(input_path),
        "starter_model_py_sha256": sha256_file(starter_dir / "model.py"),
        "starter_proactive_py_sha256": sha256_file(
            starter_dir / "run_generate_proactive.py"
        ),
        "starter_scorer_py_sha256": sha256_file(starter_dir / "run_evaluation.py"),
        "r0_session_records_sha256": sha256_file(r0_records_path),
    }
    expected = {
        "input_sha256": str(data_config["input_sha256"]),
        "starter_model_py_sha256": str(starter_config["model_py_sha256"]),
        "starter_proactive_py_sha256": str(starter_config["proactive_py_sha256"]),
        "starter_scorer_py_sha256": str(starter_config["scorer_py_sha256"]),
        "r0_session_records_sha256": str(r0_config["session_records_sha256"]),
    }
    if actual != expected:
        raise ValueError(f"Static fingerprint mismatch: {actual} != {expected}")
    return actual


def _run(
    config: dict[str, object],
    config_path: Path,
    output_dir: Path,
    device_name: str,
    require_exclusive_gpu: bool,
) -> dict[str, object]:
    started = time.monotonic()
    model_config = dict(config["model"])  # type: ignore[arg-type]
    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    inference_config = dict(config["inference"])  # type: ignore[arg-type]
    adapter_config = dict(config["adapter"])  # type: ignore[arg-type]
    cache_config = dict(config["cache"])  # type: ignore[arg-type]
    selection_config = dict(config["smoke_selection"])  # type: ignore[arg-type]
    training_config = dict(config["training_smoke"])  # type: ignore[arg-type]
    gates_config = dict(config["acceptance_gates"])  # type: ignore[arg-type]

    input_path = _resolve(str(data_config["input"]))
    video_folder = _resolve(str(data_config["video_folder"]))
    starter_dir = _resolve(str(starter_config["path"]))
    model_path = _resolve(str(model_config["default_local_path"]))
    r0_dir = _resolve(str(dict(config["r0_reference"])["experiment_dir"]))  # type: ignore[arg-type]
    r0_records_path = r0_dir / "session_records.jsonl"
    fingerprints = _static_fingerprints(
        config, input_path, starter_dir, r0_records_path
    )
    model_snapshot = verify_model_snapshot(model_path, model_config)

    source_rows = load_jsonl(input_path)
    input_index = int(selection_config["input_index"])
    if input_index < 0 or input_index >= len(source_rows):
        raise ValueError("Smoke input index is out of range")
    source_row = dict(source_rows[input_index])
    gold_answers = source_row.pop("answers", None)
    if not isinstance(gold_answers, list):
        raise ValueError("Selected smoke session has no gold answers")
    generation_row = source_row
    if "answers" in generation_row:
        raise RuntimeError("Gold answers escaped into the generation row")
    chunk_indices = [int(value) for value in selection_config["chunk_indices"]]  # type: ignore[index]
    if sorted(set(chunk_indices)) != chunk_indices:
        raise ValueError("Smoke chunk indices must be unique and sorted")
    intervals = generation_row.get("video_intervals")
    if not isinstance(intervals, list) or max(chunk_indices) >= len(intervals):
        raise ValueError("Smoke chunk index is outside the selected session")

    r0_rows = load_jsonl(r0_records_path)
    r0_record = next(
        (row for row in r0_rows if int(row["input_index"]) == input_index), None
    )
    if r0_record is None:
        raise ValueError("Selected session is absent from frozen R0 records")

    environment = environment_snapshot()
    import peft

    packages = dict(environment["packages"])  # type: ignore[arg-type]
    packages["peft"] = peft.__version__
    environment["packages"] = packages
    write_json(output_dir / "environment.txt", environment)
    write_json(
        output_dir / "code_state.txt",
        code_snapshot(
            PROJECT_ROOT,
            [
                config_path,
                Path(__file__),
                PROJECT_ROOT / "src/proactive_d2/final_mlp_lora.py",
                PROJECT_ROOT / "src/proactive_d1/internvl_features.py",
                PROJECT_ROOT / "src/proactive_r0/internvl.py",
            ],
        ),
    )
    write_json(
        output_dir / "data_manifest.json",
        {
            "source": {
                "path": str(input_path),
                "sha256": fingerprints["input_sha256"],
                "input_index": input_index,
                "chunk_indices": chunk_indices,
                "answers_removed_before_cache_construction": True,
            },
            "model": {**model_snapshot, "path": str(model_path)},
            "starter_kit_sha256": fingerprints,
            "supervision": config["validation_policy"],
        },
    )

    torch.manual_seed(int(inference_config["seed"]))
    torch.cuda.manual_seed_all(int(inference_config["seed"]))
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_started = time.monotonic()
    extractor = InternVLDecisionFeatureExtractor(
        model_path=str(model_path),
        device=device_name,
        dtype_name=str(model_config["dtype"]),
        attention_implementation=str(model_config["attention_implementation"]),
        seed=int(inference_config["seed"]),
        require_exclusive_gpu=require_exclusive_gpu,
        video_frame_size=int(inference_config["video_frame_size"]),
        pad_token_id=int(inference_config["pad_token_id"]),
        decision_feature_mode="shared_vision",
    )
    if extractor.parameter_count != int(model_config["total_parameters"]):
        raise RuntimeError("Loaded base parameter count differs from the config")
    if len(extractor.silent_token_ids) != int(cache_config["tag_tokens_each"]):
        raise RuntimeError("Silent tag token count differs from the cache contract")
    if len(extractor.interrupt_token_ids) != int(cache_config["tag_tokens_each"]):
        raise RuntimeError("Interrupt tag token count differs from the cache contract")

    peft_model, parameter_audit = configure_final_mlp_lora(
        extractor.model,
        layer_index=int(adapter_config["language_layer_index"]),
        hidden_size=int(model_config["hidden_size"]),
        intermediate_size=int(model_config["intermediate_size"]),
        rank=int(adapter_config["rank"]),
        alpha=int(adapter_config["alpha"]),
        dropout=float(adapter_config["dropout"]),
    )
    if parameter_audit.target_regex != str(adapter_config["target_modules_regex"]):
        raise RuntimeError("Effective LoRA target regex differs from the config")
    if parameter_audit.total_parameters_with_adapter != int(
        adapter_config["total_parameters_with_adapter"]
    ):
        raise RuntimeError("Total parameter count with LoRA differs from the config")
    outer, decoder_layer, final_norm, lm_head = internvl_lora_components(
        peft_model, int(adapter_config["language_layer_index"])
    )
    model_load_seconds = time.monotonic() - model_started

    starter = load_starter_kit(starter_dir)
    causal_config = CausalInferenceConfig(
        frames_per_interval=int(inference_config["frames_per_interval"]),
        max_frames=int(inference_config["max_frames"]),
        max_history_turns=int(inference_config["max_history_turns"]),
        max_new_tokens=int(inference_config["max_new_tokens"]),
    )
    cumulative_frames: list[object] = []
    prepared: list[dict[str, object]] = []
    cache_started = time.monotonic()
    for chunk_index, interval_value in enumerate(intervals[: max(chunk_indices) + 1]):
        if not isinstance(interval_value, list) or len(interval_value) != 2:
            raise ValueError("Selected smoke interval is malformed")
        interval = (float(interval_value[0]), float(interval_value[1]))
        current_frames = starter.extract_frames(
            str(video_folder / str(generation_row["video_path"])),
            intervals=[interval],
            frames_per_interval=causal_config.frames_per_interval,
        )
        cumulative_frames.extend(current_frames)
        if chunk_index not in chunk_indices:
            continue
        model_frames = subsample_frames(cumulative_frames, causal_config.max_frames)
        messages = build_messages(
            row=generation_row,
            chunk_index=chunk_index,
            system_prompt=starter.system_prompt,
            normalize_dialog_turns=starter.normalize_dialog_turns,
            max_history_turns=causal_config.max_history_turns,
        )
        prompt = extractor._causal_prompt(model_frames, messages)
        shared_inputs = _prepare_shared_vision_inputs(
            extractor, outer, model_frames, prompt
        )
        with peft_model.disable_adapter(), torch.no_grad():
            silent_forward = _forward_candidate(
                outer,
                decoder_layer,
                lm_head,
                embeddings=shared_inputs["silent_embeddings"],  # type: ignore[arg-type]
                attention_mask=shared_inputs["attention_mask"],
                token_ids=extractor.silent_token_ids,
                prompt_tokens=int(shared_inputs["prompt_tokens"]),
                calibrate_base_correction=True,
            )
            interrupt_forward = _forward_candidate(
                outer,
                decoder_layer,
                lm_head,
                embeddings=shared_inputs["interrupt_embeddings"],  # type: ignore[arg-type]
                attention_mask=shared_inputs["attention_mask"],
                token_ids=extractor.interrupt_token_ids,
                prompt_tokens=int(shared_inputs["prompt_tokens"]),
                calibrate_base_correction=True,
            )
            reconstructed_silent_hidden, reconstructed_silent_logits = (
                _reconstruct_candidate(
                    decoder_layer,
                    final_norm,
                    lm_head,
                    silent_forward.state,
                )
            )
            reconstructed_interrupt_hidden, reconstructed_interrupt_logits = (
                _reconstruct_candidate(
                    decoder_layer,
                    final_norm,
                    lm_head,
                    interrupt_forward.state,
                )
            )
            naive_silent_hidden, naive_silent_logits = _reconstruct_candidate(
                decoder_layer,
                final_norm,
                lm_head,
                FinalMLPScoringState(
                    residual=silent_forward.state.residual,
                    normalized=silent_forward.state.normalized,
                ),
            )
            naive_interrupt_hidden, naive_interrupt_logits = _reconstruct_candidate(
                decoder_layer,
                final_norm,
                lm_head,
                FinalMLPScoringState(
                    residual=interrupt_forward.state.residual,
                    normalized=interrupt_forward.state.normalized,
                ),
            )
        full_margin = _margin(
            silent_forward.logits, interrupt_forward.logits, extractor
        )
        reconstructed_margin = _margin(
            reconstructed_silent_logits,
            reconstructed_interrupt_logits,
            extractor,
        )
        naive_margin = _margin(
            naive_silent_logits,
            naive_interrupt_logits,
            extractor,
        )
        prepared.append(
            {
                "chunk_index": chunk_index,
                "interval": list(interval),
                "current_interval_frames": len(current_frames),
                "model_input_frames": len(model_frames),
                "frames": model_frames,
                "prompt": prompt,
                "shared_inputs": shared_inputs,
                "silent": silent_forward,
                "interrupt": interrupt_forward,
                "base_exactness": {
                    "corrected_silent_hidden_max_abs_difference": _max_abs(
                        silent_forward.hidden, reconstructed_silent_hidden
                    ),
                    "corrected_interrupt_hidden_max_abs_difference": _max_abs(
                        interrupt_forward.hidden, reconstructed_interrupt_hidden
                    ),
                    "corrected_silent_logit_max_abs_difference": _max_abs(
                        silent_forward.logits, reconstructed_silent_logits
                    ),
                    "corrected_interrupt_logit_max_abs_difference": _max_abs(
                        interrupt_forward.logits, reconstructed_interrupt_logits
                    ),
                    "naive_silent_hidden_max_abs_difference": _max_abs(
                        silent_forward.hidden, naive_silent_hidden
                    ),
                    "naive_interrupt_hidden_max_abs_difference": _max_abs(
                        interrupt_forward.hidden, naive_interrupt_hidden
                    ),
                    "naive_silent_logit_max_abs_difference": _max_abs(
                        silent_forward.logits, naive_silent_logits
                    ),
                    "naive_interrupt_logit_max_abs_difference": _max_abs(
                        interrupt_forward.logits, naive_interrupt_logits
                    ),
                    "full_margin": float(full_margin.item()),
                    "corrected_margin": float(reconstructed_margin.item()),
                    "corrected_margin_abs_difference": float(
                        (full_margin - reconstructed_margin).abs().item()
                    ),
                    "naive_margin": float(naive_margin.item()),
                    "naive_margin_abs_difference": float(
                        (full_margin - naive_margin).abs().item()
                    ),
                },
            }
        )
    cache_seconds = time.monotonic() - cache_started
    if [int(item["chunk_index"]) for item in prepared] != chunk_indices:
        raise RuntimeError("Did not construct every selected smoke cache")

    labels = [
        int(str(gold_answers[index]).lstrip().startswith(INTERRUPT_TAG))
        for index in chunk_indices
    ]
    if set(labels) != {0, 1}:
        raise RuntimeError("Predeclared smoke chunks do not contain both classes")

    _set_nonzero_adapter_b(
        peft_model,
        seed=int(training_config["seed"]) + 1,
        standard_deviation=0.02,
    )
    adapted_effects: list[float] = []
    for item in prepared:
        shared_inputs = item["shared_inputs"]  # type: ignore[assignment]
        with torch.no_grad():
            adapted_silent = _forward_candidate(
                outer,
                decoder_layer,
                lm_head,
                embeddings=shared_inputs["silent_embeddings"],  # type: ignore[index,arg-type]
                attention_mask=shared_inputs["attention_mask"],  # type: ignore[index]
                token_ids=extractor.silent_token_ids,
                prompt_tokens=int(shared_inputs["prompt_tokens"]),  # type: ignore[index]
            )
            adapted_interrupt = _forward_candidate(
                outer,
                decoder_layer,
                lm_head,
                embeddings=shared_inputs["interrupt_embeddings"],  # type: ignore[index,arg-type]
                attention_mask=shared_inputs["attention_mask"],  # type: ignore[index]
                token_ids=extractor.interrupt_token_ids,
                prompt_tokens=int(shared_inputs["prompt_tokens"]),  # type: ignore[index]
            )
            cached_silent_hidden, cached_silent_logits = _reconstruct_candidate(
                decoder_layer,
                final_norm,
                lm_head,
                _state_to_device(item["silent"].state, device),  # type: ignore[union-attr]
            )
            cached_interrupt_hidden, cached_interrupt_logits = _reconstruct_candidate(
                decoder_layer,
                final_norm,
                lm_head,
                _state_to_device(item["interrupt"].state, device),  # type: ignore[union-attr]
            )
        adapted_full_margin = _margin(
            adapted_silent.logits, adapted_interrupt.logits, extractor
        )
        adapted_cached_margin = _margin(
            cached_silent_logits, cached_interrupt_logits, extractor
        )
        base_silent = item["silent"]  # type: ignore[assignment]
        base_interrupt = item["interrupt"]  # type: ignore[assignment]
        state_invariance = {
            "silent_residual_max_abs_difference": _max_abs(
                adapted_silent.state.residual, base_silent.state.residual
            ),
            "silent_normalized_max_abs_difference": _max_abs(
                adapted_silent.state.normalized, base_silent.state.normalized
            ),
            "interrupt_residual_max_abs_difference": _max_abs(
                adapted_interrupt.state.residual, base_interrupt.state.residual
            ),
            "interrupt_normalized_max_abs_difference": _max_abs(
                adapted_interrupt.state.normalized, base_interrupt.state.normalized
            ),
        }
        adapted_effects.extend(
            [
                _max_abs(cached_silent_hidden, base_silent.hidden),
                _max_abs(cached_interrupt_hidden, base_interrupt.hidden),
            ]
        )
        item["adapted_exactness"] = {
            "local_vs_full_silent_hidden_max_abs_difference": _max_abs(
                adapted_silent.hidden, cached_silent_hidden
            ),
            "local_vs_full_interrupt_hidden_max_abs_difference": _max_abs(
                adapted_interrupt.hidden, cached_interrupt_hidden
            ),
            "local_vs_full_silent_logit_max_abs_difference": _max_abs(
                adapted_silent.logits, cached_silent_logits
            ),
            "local_vs_full_interrupt_logit_max_abs_difference": _max_abs(
                adapted_interrupt.logits, cached_interrupt_logits
            ),
            "full_margin": float(adapted_full_margin.item()),
            "local_margin": float(adapted_cached_margin.item()),
            "local_vs_full_margin_abs_difference": float(
                (adapted_full_margin - adapted_cached_margin).abs().item()
            ),
            "state_invariance": state_invariance,
        }

    _zero_adapter_b(peft_model)
    silent_states = [
        _state_to_cpu(item["silent"].state) for item in prepared  # type: ignore[union-attr]
    ]
    interrupt_states = [
        _state_to_cpu(item["interrupt"].state) for item in prepared  # type: ignore[union-attr]
    ]
    training_started = time.monotonic()
    training_audit = _train_cached_smoke(
        peft_model,
        decoder_layer,
        final_norm,
        lm_head,
        extractor,
        silent_states,
        interrupt_states,
        labels,
        training_config,
    )
    training_seconds = time.monotonic() - training_started

    peft_model.eval()
    first = prepared[0]
    disabled_generation = _generate_disabled(
        peft_model,
        outer,
        extractor,
        frames=first["frames"],  # type: ignore[arg-type]
        prompt=str(first["prompt"]),
        max_new_tokens=int(inference_config["max_new_tokens"]),
    )
    r0_chunks = r0_record.get("chunks")
    if not isinstance(r0_chunks, list):
        raise RuntimeError("Frozen R0 session record has no chunks")
    expected_generation = str(r0_chunks[chunk_indices[0]]["raw_response"])
    disabled_generation_exact = disabled_generation == expected_generation

    adapter_dir = output_dir / "adapter_smoke_only"
    peft_model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise RuntimeError("PEFT smoke adapter was not serialized")

    corrected_base_hidden_max = max(
        max(
            float(item["base_exactness"][key])  # type: ignore[index]
            for key in (
                "corrected_silent_hidden_max_abs_difference",
                "corrected_interrupt_hidden_max_abs_difference",
            )
        )
        for item in prepared
    )
    corrected_base_logit_max = max(
        max(
            float(item["base_exactness"][key])  # type: ignore[index]
            for key in (
                "corrected_silent_logit_max_abs_difference",
                "corrected_interrupt_logit_max_abs_difference",
            )
        )
        for item in prepared
    )
    naive_base_hidden_max = max(
        max(
            float(item["base_exactness"][key])  # type: ignore[index]
            for key in (
                "naive_silent_hidden_max_abs_difference",
                "naive_interrupt_hidden_max_abs_difference",
            )
        )
        for item in prepared
    )
    naive_base_logit_max = max(
        max(
            float(item["base_exactness"][key])  # type: ignore[index]
            for key in (
                "naive_silent_logit_max_abs_difference",
                "naive_interrupt_logit_max_abs_difference",
            )
        )
        for item in prepared
    )
    local_vs_full_adapted_hidden_max = max(
        max(
            float(item["adapted_exactness"][key])  # type: ignore[index]
            for key in (
                "local_vs_full_silent_hidden_max_abs_difference",
                "local_vs_full_interrupt_hidden_max_abs_difference",
            )
        )
        for item in prepared
    )
    local_vs_full_adapted_logit_max = max(
        max(
            float(item["adapted_exactness"][key])  # type: ignore[index]
            for key in (
                "local_vs_full_silent_logit_max_abs_difference",
                "local_vs_full_interrupt_logit_max_abs_difference",
            )
        )
        for item in prepared
    )
    corrected_base_margin_max = max(
        float(item["base_exactness"]["corrected_margin_abs_difference"])  # type: ignore[index]
        for item in prepared
    )
    naive_base_margin_max = max(
        float(item["base_exactness"]["naive_margin_abs_difference"])  # type: ignore[index]
        for item in prepared
    )
    local_vs_full_adapted_margin_max = max(
        float(
            item["adapted_exactness"]["local_vs_full_margin_abs_difference"]  # type: ignore[index]
        )
        for item in prepared
    )
    state_invariance_max = max(
        float(value)
        for item in prepared
        for value in item["adapted_exactness"]["state_invariance"].values()  # type: ignore[index,union-attr]
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    gates = {
        "trainable_parameter_count": parameter_audit.trainable_parameters
        == int(gates_config["exact_trainable_parameters"]),
        "trainable_tensor_count": len(parameter_audit.trainable_tensor_names)
        == int(gates_config["expected_trainable_tensors"]),
        "corrected_base_hidden_reconstruction": corrected_base_hidden_max
        <= float(gates_config["max_corrected_base_hidden_abs_difference"]),
        "corrected_base_logit_reconstruction": corrected_base_logit_max
        <= float(gates_config["max_corrected_base_logit_abs_difference"]),
        "corrected_base_margin_reconstruction": corrected_base_margin_max
        <= float(gates_config["max_corrected_base_margin_abs_difference"]),
        "local_vs_full_adapted_margin": local_vs_full_adapted_margin_max
        <= float(gates_config["max_local_vs_full_adapted_margin_abs_difference"]),
        "adapter_state_invariance": state_invariance_max == 0.0,
        "adapter_has_nonzero_effect": max(adapted_effects) > 0.0,
        "disabled_generation_exact": disabled_generation_exact,
        "finite_nonzero_gradients": bool(training_audit["gradient_norms"])
        and any(
            math.isfinite(float(value)) and float(value) > 0
            for value in training_audit["gradient_norms"]  # type: ignore[union-attr]
        )
        and all(
            math.isfinite(float(value)) and float(value) >= 0
            for value in training_audit["gradient_norms"]  # type: ignore[union-attr]
        ),
        "training_loss_decreases": float(training_audit["final_loss"])
        < float(training_audit["initial_loss"]),
        "peak_gpu_memory": peak_memory
        <= int(gates_config["max_peak_gpu_memory_bytes"]),
    }
    gate_passed = all(gates.values())

    public_records: list[dict[str, object]] = []
    for item, label in zip(prepared, labels):
        public_records.append(
            {
                "input_index": input_index,
                "chunk_index": item["chunk_index"],
                "interval": item["interval"],
                "label_used_only_after_cache_construction": label,
                "current_interval_frames": item["current_interval_frames"],
                "model_input_frames": item["model_input_frames"],
                "prompt_tokens": item["silent"].prompt_tokens,  # type: ignore[union-attr]
                "base_exactness": item["base_exactness"],
                "adapted_exactness": item["adapted_exactness"],
            }
        )
    _write_jsonl(output_dir / "smoke_records.jsonl", public_records)
    audit = {
        "status": "passed" if gate_passed else "failed",
        "gate_passed": gate_passed,
        "gates": gates,
        "parameter_audit": asdict(parameter_audit),
        "tag_token_ids": {
            "silent": extractor.silent_token_ids,
            "interrupt": extractor.interrupt_token_ids,
        },
        "exactness": {
            "corrected_base_hidden_max_abs_difference": corrected_base_hidden_max,
            "corrected_base_logit_max_abs_difference": corrected_base_logit_max,
            "corrected_base_margin_max_abs_difference": corrected_base_margin_max,
            "naive_base_hidden_max_abs_difference": naive_base_hidden_max,
            "naive_base_logit_max_abs_difference": naive_base_logit_max,
            "naive_base_margin_max_abs_difference": naive_base_margin_max,
            "local_vs_full_adapted_hidden_max_abs_difference": local_vs_full_adapted_hidden_max,
            "local_vs_full_adapted_logit_max_abs_difference": local_vs_full_adapted_logit_max,
            "local_vs_full_adapted_margin_max_abs_difference": local_vs_full_adapted_margin_max,
            "adapter_state_invariance_max_abs_difference": state_invariance_max,
            "nonzero_adapter_effect_max_abs_difference": max(adapted_effects),
        },
        "training_smoke": training_audit,
        "generation_switch": {
            "input_index": input_index,
            "chunk_index": chunk_indices[0],
            "expected_frozen_r0_raw_response": expected_generation,
            "disabled_adapter_raw_response": disabled_generation,
            "exact_match": disabled_generation_exact,
        },
        "cache_accounting": {
            "bytes_per_chunk": final_mlp_cache_bytes_per_chunk(
                candidates=2,
                tag_length=len(extractor.silent_token_ids),
                hidden_size=int(model_config["hidden_size"]),
                bytes_per_value=2,
                stored_tensors_per_candidate=4,
            ),
            "projected_bytes_for_9935_chunks": int(
                cache_config["projected_full_cache_bytes_for_9935_chunks"]
            ),
        },
        "scope": "engineering feasibility only; no OOF or leaderboard metric",
    }
    write_json(output_dir / "audit.json", audit)
    runtime = {
        "status": "complete" if gate_passed else "complete_with_failed_gate",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "model_load_seconds": round(model_load_seconds, 3),
        "cache_construction_seconds": round(cache_seconds, 3),
        "cached_training_seconds": round(training_seconds, 3),
        "device": device_name,
        "peak_gpu_memory_bytes": peak_memory,
        "preexisting_gpu_processes": extractor.preexisting_gpu_processes,
        "adapter_weights_sha256": sha256_file(adapter_weights),
    }
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                f"状态：**{'通过' if gate_passed else '未通过'} final-MLP LoRA 工程可行性门**。",
                "",
                "- 范围：2 个 chunk 的精确性与反向传播 smoke，不产生 OOF 效果结论。",
                f"- LoRA 可训练参数：`{parameter_audit.trainable_parameters}`。",
                f"- 校正后基座重建最大 hidden/logit 误差：`{corrected_base_hidden_max}` / `{corrected_base_logit_max}`。",
                f"- 局部/完整适配器最大 margin 误差：`{local_vs_full_adapted_margin_max}`。",
                f"- 缓存训练损失：`{training_audit['initial_loss']}` -> `{training_audit['final_loss']}`。",
                f"- 禁用适配器后 R0 生成精确一致：`{disabled_generation_exact}`。",
                f"- 峰值 GPU 显存：`{peak_memory}` bytes。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"audit": audit, "runtime": runtime}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    output_dir = _resolve(
        args.output_dir or f"output/experiments/{config['experiment_id']}"
    )
    if output_dir.exists():
        raise FileExistsError(f"Smoke output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "device": args.device,
        "require_exclusive_gpu": args.require_exclusive_gpu,
    }
    write_json(output_dir / "config.json", effective)
    (output_dir / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        + f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        + shlex.join([sys.executable, "-m", "proactive_d2.smoke_final_mlp_lora", *sys.argv[1:]])
        + "\n",
        encoding="utf-8",
    )
    try:
        result = _run(
            config,
            config_path,
            output_dir,
            args.device,
            args.require_exclusive_gpu,
        )
    except Exception as error:
        write_json(
            output_dir / "runtime.json",
            {
                "status": "failed",
                "completed_at": datetime.now().astimezone().isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
