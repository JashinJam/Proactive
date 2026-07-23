"""Run the frozen D6 trainability smoke or one resumable formal OOF fold."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch

from proactive_d1.core import serialize_decision_head
from proactive_d4_1.core import atomic_append_jsonl
from proactive_r0.artifacts import environment_snapshot, sha256_file, write_json
from proactive_r0.core import load_jsonl, load_starter_kit, write_jsonl

from .adapter import D6DecisionModel, LORA_PARAMETERS, MEMORY_PARAMETERS
from .contract import (
    ExperimentInputs,
    SHARED_MINIMUM_FREE_GIB,
    gpu_resource_audit,
    inference_config,
    labels_for_allowed_sessions,
    load_experiment,
)
from .data import rotation_indices
from .head import (
    apply_head_to_test,
    build_label_free_matrix,
    fit_rotation_head,
    metrics_after_unseal,
)
from .runtime import process_session
from .training import FoldTrainer


DEFAULT_CONFIG = Path("configs/d6_internvl35_1b_query_memory_lora_oof_v1.json")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"D6 config section is not an object: {name}")
    return value


def _code_state() -> dict[str, object]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            arguments,
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

    project_root = Path(__file__).resolve().parents[2]
    relevant_files = sorted(
        [
            project_root / "annotations/d6_query_memory_lora_oof_v1/PROTOCOL.md",
            project_root
            / "annotations/d6_query_memory_lora_oof_v1/RESOURCE_AMENDMENT_20260722.md",
            project_root / "configs/d6_internvl35_1b_query_memory_lora_oof_v1.json",
            *(
                path
                for path in (project_root / "src/proactive_d6").rglob("*.py")
                if "__pycache__" not in path.parts
            ),
        ],
        key=lambda path: str(path.relative_to(project_root)),
    )
    return {
        "git_commit": run("git", "rev-parse", "HEAD"),
        "git_status_porcelain": run("git", "status", "--porcelain"),
        "python_executable": sys.executable,
        "command": sys.argv,
        "relevant_file_sha256": {
            str(path.relative_to(project_root)): sha256_file(path)
            for path in relevant_files
        },
    }


def _write_contract(output_dir: Path, inputs: ExperimentInputs, resource: object) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    code_state_path = output_dir / "code_state.json"
    initial_code_state_path = output_dir / "initial_execution_code_state.json"
    if code_state_path.exists() and not initial_code_state_path.exists():
        previous = json.loads(code_state_path.read_text(encoding="utf-8"))
        if not isinstance(previous, dict):
            raise ValueError("D6 previous code state is malformed")
        write_json(initial_code_state_path, previous)
    write_json(output_dir / "environment.json", environment_snapshot())
    write_json(code_state_path, _code_state())
    write_json(
        output_dir / "experiment_contract.json",
        {
            "schema_version": 1,
            "experiment_id": inputs.config["experiment_id"],
            "config_sha256": inputs.config_sha256,
            "config_file_sha256": inputs.config_file_sha256,
            "protocol_sha256": inputs.config["protocol"]["sha256"],  # type: ignore[index]
            "input_sha256": inputs.config["data"]["input_sha256"],  # type: ignore[index]
            "manifest_sha256": inputs.config["d4_2_reference"]["fold_manifest_sha256"],  # type: ignore[index]
            "generation_records_sha256": inputs.config["d4_2_reference"]["history8_raw_generation_records_sha256"],  # type: ignore[index]
            "model_weights_sha256": inputs.config["model"]["weights_sha256"],  # type: ignore[index]
            "scorer_sha256": inputs.config["starter_kit"]["scorer_sha256"],  # type: ignore[index]
            "resource_amendment_sha256": sha256_file(
                Path(__file__).resolve().parents[2]
                / "annotations/d6_query_memory_lora_oof_v1/RESOURCE_AMENDMENT_20260722.md"
            ),
            "gpu_preload_resource_audit": resource,
        },
    )


def _load_gate(path: Path, inputs: ExperimentInputs, kind: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != kind:
        raise ValueError(f"D6 required gate has the wrong kind: {path}")
    if value.get("config_sha256") != inputs.config_sha256:
        raise ValueError(f"D6 required gate used a different config: {path}")
    gates = value.get("gates")
    if not isinstance(gates, Mapping) or not gates or not all(gates.values()):
        raise ValueError(f"D6 required gate did not pass: {path}")
    return value


def _build_model(
    inputs: ExperimentInputs,
    device: str,
    *,
    require_exclusive_gpu: bool = True,
) -> D6DecisionModel:
    model = _mapping(inputs.config["model"], "model")
    frozen = _mapping(inputs.config["frozen_inference"], "frozen_inference")
    loaded = D6DecisionModel(
        model_path=str(inputs.model_path),
        device=device,
        dtype_name=str(model["dtype"]),
        attention_implementation=str(model["attention_implementation"]),
        seed=int(frozen["seed"]),
        require_exclusive_gpu=require_exclusive_gpu,
        video_frame_size=int(frozen["video_frame_size"]),
        pad_token_id=int(frozen["pad_token_id"]),
    )
    if loaded.parameter_count != int(model["base_parameters"]):
        raise ValueError("D6 loaded base parameter count changed")
    if loaded.hidden_size != int(model["hidden_size"]):
        raise ValueError("D6 loaded hidden width changed")
    if loaded.lora_parameter_count() != LORA_PARAMETERS:
        raise ValueError("D6 loaded LoRA parameter count changed")
    if sum(parameter.numel() for parameter in loaded.memory.parameters()) != MEMORY_PARAMETERS:
        raise ValueError("D6 loaded memory parameter count changed")
    return loaded


def _validate_record_prefix(
    records: Sequence[Mapping[str, object]],
    expected_indices: Sequence[int],
    rows: Sequence[Mapping[str, object]],
) -> None:
    if len(records) > len(expected_indices):
        raise ValueError("D6 extraction resume contains extra sessions")
    for position, record in enumerate(records):
        input_index = expected_indices[position]
        if int(record.get("input_index", -1)) != input_index:
            raise ValueError("D6 extraction resume order changed")
        if record.get("video_path") != rows[input_index].get("video_path"):
            raise ValueError("D6 extraction resume video changed")
        chunks = record.get("chunks")
        if not isinstance(chunks, list) or len(chunks) != len(
            rows[input_index]["video_intervals"]  # type: ignore[arg-type,index]
        ):
            raise ValueError("D6 extraction resume chunk coverage changed")
        for chunk_index, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping) or int(chunk.get("chunk_index", -1)) != chunk_index:
                raise ValueError("D6 extraction resume chunk order changed")
            if "gold_interrupt" in chunk or "answers" in chunk:
                raise ValueError("D6 extraction artifact contains labels")


def _extract_records(
    *,
    path: Path,
    session_indices: Sequence[int],
    inputs: ExperimentInputs,
    model: D6DecisionModel,
    memory_enabled: bool,
    lora_enabled: bool,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    records = load_jsonl(path) if path.exists() else []
    _validate_record_prefix(records, session_indices, inputs.answer_free_rows)
    starter = load_starter_kit(inputs.starter_dir)
    inference = inference_config(inputs.config)
    maximum_model_seconds = max(
        (
            float(record["timing"]["model_inference_seconds"])  # type: ignore[index]
            for record in records
        ),
        default=0.0,
    )
    started = time.monotonic()
    for position in range(len(records), len(session_indices)):
        input_index = session_indices[position]
        with torch.no_grad():
            record = process_session(
                row=inputs.answer_free_rows[input_index],
                input_index=input_index,
                video_folder=inputs.video_folder,
                model=model,
                starter=starter,
                inference=inference,
                reference=inputs.references[input_index],
                memory_enabled=memory_enabled,
                lora_enabled=lora_enabled,
                record_hidden_state=True,
                record_chunks=True,
            )
        maximum_model_seconds = max(
            maximum_model_seconds,
            float(record["timing"]["model_inference_seconds"]),  # type: ignore[index]
        )
        atomic_append_jsonl(path, record)
        records.append(record)
    _validate_record_prefix(records, session_indices, inputs.answer_free_rows)
    session_model_seconds = [
        float(record["timing"]["model_inference_seconds"])  # type: ignore[index]
        for record in records
    ]
    return records, {
        "sessions": float(len(records)),
        "chunks": float(sum(len(record["chunks"]) for record in records)),  # type: ignore[arg-type,index]
        "maximum_session_model_seconds": maximum_model_seconds,
        "total_session_model_seconds": sum(session_model_seconds),
        "resume_wall_seconds": time.monotonic() - started,
    }


def _prediction_records(
    keys: Sequence[tuple[int, int]],
    logits: Sequence[float],
    decisions: Sequence[int],
    threshold: float,
) -> list[dict[str, object]]:
    if not (len(keys) == len(logits) == len(decisions)):
        raise ValueError("D6 prediction arrays are not aligned")
    return [
        {
            "input_index": key[0],
            "chunk_index": key[1],
            "logit": float(logit),
            "threshold_logit": float(threshold),
            "predicted_interrupt": int(decision),
            "gold_interrupt": "SENTINEL_UNSEALED",
        }
        for key, logit, decision in zip(keys, logits, decisions)
    ]


def _train(
    *,
    inputs: ExperimentInputs,
    model: D6DecisionModel,
    fold: int,
    output_dir: Path,
    maximum_epochs: int,
) -> tuple[FoldTrainer, object, dict[str, object]]:
    rotation = rotation_indices(inputs.manifest, inputs.answer_free_rows, fold)
    fit = list(rotation["fit"])  # type: ignore[arg-type]
    calibration = list(rotation["calibration"])  # type: ignore[arg-type]
    labels = labels_for_allowed_sessions(inputs.source_rows, [*fit, *calibration])
    test = set(rotation["test"])  # type: ignore[arg-type]
    if any(key[0] in test for key in labels):
        raise RuntimeError("D6 test labels leaked into adapter training")
    training = _mapping(inputs.config["adapter_training"], "adapter_training")
    trainer = FoldTrainer(
        model=model,
        rows=inputs.answer_free_rows,
        references=inputs.references,
        labels=labels,
        fit_sessions=fit,
        calibration_sessions=calibration,
        video_folder=inputs.video_folder,
        starter=load_starter_kit(inputs.starter_dir),
        inference=inference_config(inputs.config),
        fold=fold,
        output_dir=output_dir / "training",
        config_sha256=inputs.config_sha256,
        model_weights_sha256=str(inputs.config["model"]["weights_sha256"]),  # type: ignore[index]
        memory_learning_rate=float(training["memory_learning_rate"]),
        lora_learning_rate=float(training["lora_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        accumulation_target_chunks=int(training["gradient_accumulation_target_chunks"]),
        maximum_epochs=maximum_epochs,
        patience=int(training["early_stopping_patience"]),
        seed=int(training["seed"]),
    )
    started = time.monotonic()
    result = trainer.train()
    audit = {
        "wall_seconds": time.monotonic() - started,
        "fit_sessions": len(fit),
        "calibration_sessions": len(calibration),
        "test_sessions": len(test),
        "fit_chunks": sum(len(inputs.answer_free_rows[index]["video_intervals"]) for index in fit),  # type: ignore[arg-type,index]
        "calibration_chunks": sum(len(inputs.answer_free_rows[index]["video_intervals"]) for index in calibration),  # type: ignore[arg-type,index]
        "test_chunks": sum(len(inputs.answer_free_rows[index]["video_intervals"]) for index in test),  # type: ignore[arg-type,index]
    }
    return trainer, result, audit


def _maximum_session_seconds(history: Sequence[Mapping[str, object]]) -> float:
    return max(
        (
            max(
                float(row.get("maximum_session_model_seconds", 0.0)),
                float(row.get("maximum_fit_session_model_seconds", 0.0)),
            )
            for row in history
        ),
        default=0.0,
    )


def run_trainability_smoke(
    inputs: ExperimentInputs,
    model: D6DecisionModel,
    output_dir: Path,
) -> dict[str, object]:
    previous_summary: dict[str, object] | None = None
    previous_path = output_dir / "summary.json"
    if previous_path.exists():
        value = json.loads(previous_path.read_text(encoding="utf-8"))
        if (
            isinstance(value, dict)
            and value.get("kind") == "d6_rotation0_trainability_smoke"
            and value.get("config_sha256") == inputs.config_sha256
        ):
            previous_summary = value
    initial_adapter = model.adapter_state_dict()
    trainer, result, audit = _train(
        inputs=inputs,
        model=model,
        fold=0,
        output_dir=output_dir,
        maximum_epochs=1,
    )
    del trainer
    previous_audit: Mapping[str, object] | None = None
    if previous_summary is not None:
        value = previous_summary.get("training_audit")
        if isinstance(value, Mapping):
            previous_audit = value
            audit["wall_seconds"] = float(previous_audit["wall_seconds"])
    segments_path = output_dir / "training" / "execution_segments.json"
    prior_segment_seconds = 0.0
    if segments_path.exists():
        segments_payload = json.loads(segments_path.read_text(encoding="utf-8"))
        if (
            not isinstance(segments_payload, Mapping)
            or segments_payload.get("kind") != "d6_trainability_execution_segments"
            or not isinstance(segments_payload.get("segments"), list)
        ):
            raise ValueError("D6 trainability execution segment audit is malformed")
        prior_segment_seconds = sum(
            float(segment["wall_seconds"])
            for segment in segments_payload["segments"]
            if isinstance(segment, Mapping)
        )
    previously_included = (
        float(previous_audit.get("prior_completed_segment_wall_seconds", 0.0))
        if previous_audit is not None
        else 0.0
    )
    if prior_segment_seconds > previously_included:
        audit["wall_seconds"] = float(audit["wall_seconds"]) + (
            prior_segment_seconds - previously_included
        )
    audit["prior_completed_segment_wall_seconds"] = prior_segment_seconds
    final_adapter = model.adapter_state_dict()
    adapter_delta_squared = 0.0
    adapter_delta_max = 0.0
    changed_adapter_tensors = 0
    for name in initial_adapter:
        difference = final_adapter[name].float() - initial_adapter[name].float()
        adapter_delta_squared += float(difference.square().sum())
        adapter_delta_max = max(adapter_delta_max, float(difference.abs().max()))
        changed_adapter_tensors += int(bool(torch.count_nonzero(difference)))
    training_checkpoint = torch.load(
        output_dir / "training" / "training_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    optimizer_state = training_checkpoint["optimizer_state"]["state"]
    moment_tensors = [
        value["exp_avg"].float()
        for value in optimizer_state.values()
        if isinstance(value, Mapping) and "exp_avg" in value
    ]
    optimizer_moment_audit = {
        "state_entries": len(optimizer_state),
        "moment_tensors": len(moment_tensors),
        "nonzero_moment_tensors": sum(
            int(bool(torch.count_nonzero(value))) for value in moment_tensors
        ),
        "moment_l2_norm": math.sqrt(
            sum(float(value.square().sum()) for value in moment_tensors)
        ),
        "maximum_optimizer_step": max(
            (
                float(value["step"])
                for value in optimizer_state.values()
                if isinstance(value, Mapping) and "step" in value
            ),
            default=0.0,
        ),
    }
    resources = _mapping(inputs.config["resources"], "resources")
    history = list(result.history)  # type: ignore[attr-defined]
    measured_chunks = audit["fit_chunks"] + 2 * audit["calibration_chunks"]
    formal_chunks = (
        audit["calibration_chunks"]
        + int(inputs.config["adapter_training"]["maximum_epochs"])  # type: ignore[index]
        * (audit["fit_chunks"] + audit["calibration_chunks"])
        + int(inputs.config["data"]["chunks"])  # type: ignore[index]
        + 2 * audit["test_chunks"]
    )
    estimated_hours = float(audit["wall_seconds"]) * formal_chunks / measured_chunks / 3600
    peak_gib = torch.cuda.max_memory_allocated(model.device) / 2**30
    if previous_summary is not None:
        peak_gib = max(peak_gib, float(previous_summary["peak_allocated_gib"]))
    maximum_session_seconds = _maximum_session_seconds(history)
    gates = {
        "peak_allocated_at_most_70_gib": peak_gib
        <= float(resources["maximum_peak_allocated_gib"]),
        "estimated_formal_fold_at_most_48_hours": estimated_hours
        <= float(resources["maximum_estimated_single_fold_hours"]),
        "maximum_session_model_at_most_240_seconds": maximum_session_seconds
        <= float(resources["maximum_smoke_session_model_seconds"]),
        "adapter_checkpoint_excludes_base_weights": bool(
            torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)[
                "base_model_tensors_saved"
            ]
            is False
        ),
        "optimizer_performed_nonzero_updates": bool(
            optimizer_moment_audit["nonzero_moment_tensors"] > 0
            and optimizer_moment_audit["maximum_optimizer_step"] > 0
        ),
    }
    summary = {
        "schema_version": 1,
        "kind": "d6_rotation0_trainability_smoke",
        "status": "complete_no_efficacy_conclusion" if all(gates.values()) else "resource_gate_failed",
        "config_sha256": inputs.config_sha256,
        "fold": 0,
        "maximum_epochs": 1,
        "best_epoch": result.best_epoch,
        "best_calibration_bce": result.best_calibration_bce,
        "history": history,
        "training_audit": audit,
        "peak_allocated_gib": peak_gib,
        "maximum_session_model_seconds": maximum_session_seconds,
        "estimated_formal_fold_hours": estimated_hours,
        "adapter_sha256": result.checkpoint_sha256,
        "best_adapter_change_from_initial": {
            "changed_tensors": changed_adapter_tensors,
            "l2_norm": math.sqrt(adapter_delta_squared),
            "max_abs": adapter_delta_max,
        },
        "optimizer_moment_audit": optimizer_moment_audit,
        "gates": gates,
    }
    write_json(output_dir / "summary.json", summary)
    if not all(gates.values()):
        raise RuntimeError(f"D6 trainability smoke gate failed: {gates}")
    return summary


def run_formal_fold(
    inputs: ExperimentInputs,
    model: D6DecisionModel,
    fold: int,
    output_dir: Path,
) -> dict[str, object]:
    training = _mapping(inputs.config["adapter_training"], "adapter_training")
    trainer, result, training_audit = _train(
        inputs=inputs,
        model=model,
        fold=fold,
        output_dir=output_dir,
        maximum_epochs=int(training["maximum_epochs"]),
    )
    trainer.load_best()
    del trainer
    primary_records, primary_timing = _extract_records(
        path=output_dir / "primary_session_records.jsonl",
        session_indices=list(range(len(inputs.answer_free_rows))),
        inputs=inputs,
        model=model,
        memory_enabled=True,
        lora_enabled=True,
    )
    matrix = build_label_free_matrix(
        answer_free_rows=inputs.answer_free_rows,
        records=primary_records,
        fold_by_index=inputs.fold_by_index,
    )
    rotation = rotation_indices(inputs.manifest, inputs.answer_free_rows, fold)
    test_session_set = set(rotation["test"])  # type: ignore[arg-type]
    oof_test_chunks = [
        chunk
        for record in primary_records
        if int(record["input_index"]) in test_session_set
        for chunk in record["chunks"]  # type: ignore[index]
    ]
    representation_audit = {
        "chunks": len(oof_test_chunks),
        "memory_residual_norm_mean": sum(
            float(chunk["memory_residual_norm"]) for chunk in oof_test_chunks
        )
        / len(oof_test_chunks),
        "memory_residual_norm_max": max(
            float(chunk["memory_residual_norm"]) for chunk in oof_test_chunks
        ),
        "attention_entropy_mean": sum(
            float(chunk["attention_entropy"]) for chunk in oof_test_chunks
        )
        / len(oof_test_chunks),
        "normalized_attention_entropy_mean": sum(
            float(chunk["normalized_attention_entropy"]) for chunk in oof_test_chunks
        )
        / len(oof_test_chunks),
        "candidate_update_max_abs_difference": max(
            float(chunk["candidate_memory_update_max_abs_difference"])
            for chunk in oof_test_chunks
        ),
    }
    fit_cal_sessions = [*rotation["fit"], *rotation["calibration"]]  # type: ignore[list-item]
    fit_cal_labels = labels_for_allowed_sessions(inputs.source_rows, fit_cal_sessions)
    head_config = _mapping(inputs.config["head_training"], "head_training")
    head_result = fit_rotation_head(
        matrix=matrix,
        labels=fit_cal_labels,
        test_fold=fold,
        l2_weights=[float(value) for value in head_config["l2_weights"]],  # type: ignore[arg-type]
        seed=int(head_config["seed"]),
        max_iterations=int(head_config["max_iterations"]),
    )
    head_path = output_dir / "decision_head.json"
    write_json(
        head_path,
        serialize_decision_head(
            head_result.head,
            {
                "experiment_id": inputs.config["experiment_id"],
                "config_sha256": inputs.config_sha256,
                "fold": fold,
                "fit_chunks": head_result.fit_chunks,
                "calibration_chunks": head_result.calibration_chunks,
                "test_chunks": head_result.test_chunks,
                "selected_l2_weight": head_result.selected_l2_weight,
                "calibration_grid": list(head_result.calibration_grid),
                "test_labels_used": False,
            },
        ),
    )
    frozen_path = output_dir / "primary_test_predictions_frozen.jsonl"
    write_jsonl(
        frozen_path,
        _prediction_records(
            head_result.test_keys,
            head_result.test_logits,
            head_result.test_decisions,
            head_result.head.threshold_logit,
        ),
    )
    frozen_sha = sha256_file(frozen_path)

    frozen_diagnostics: dict[
        str,
        tuple[
            tuple[tuple[int, int], ...],
            tuple[int, ...],
            Path,
            dict[str, float],
        ],
    ] = {}
    maximum_session_seconds = float(primary_timing["maximum_session_model_seconds"])
    test_sessions = list(rotation["test"])  # type: ignore[arg-type]
    primary_by_index = {int(record["input_index"]): record for record in primary_records}
    for name, memory_enabled, lora_enabled in (
        ("lora_disabled", True, False),
        ("memory_disabled", False, True),
    ):
        diagnostic_records, timing = _extract_records(
            path=output_dir / f"{name}_test_session_records.jsonl",
            session_indices=test_sessions,
            inputs=inputs,
            model=model,
            memory_enabled=memory_enabled,
            lora_enabled=lora_enabled,
        )
        combined = [
            diagnostic_records[test_sessions.index(index)]
            if index in set(test_sessions)
            else primary_by_index[index]
            for index in range(len(inputs.answer_free_rows))
        ]
        diagnostic_matrix = build_label_free_matrix(
            answer_free_rows=inputs.answer_free_rows,
            records=combined,
            fold_by_index=inputs.fold_by_index,
        )
        keys, logits, decisions = apply_head_to_test(
            matrix=diagnostic_matrix,
            head=head_result.head,
            test_fold=fold,
        )
        path = output_dir / f"{name}_test_predictions_frozen.jsonl"
        write_jsonl(
            path,
            _prediction_records(keys, logits, decisions, head_result.head.threshold_logit),
        )
        frozen_diagnostics[name] = (keys, decisions, path, timing)
        maximum_session_seconds = max(
            maximum_session_seconds, float(timing["maximum_session_model_seconds"])
        )

    # Test supervision is unsealed only after primary and both fixed diagnostics
    # have immutable, hashed prediction artifacts on disk.
    test_labels = labels_for_allowed_sessions(inputs.source_rows, rotation["test"])  # type: ignore[arg-type]
    primary_metrics = metrics_after_unseal(
        head_result.test_keys, head_result.test_decisions, test_labels
    )
    diagnostic_summaries: dict[str, object] = {}
    for name, (keys, decisions, path, timing) in frozen_diagnostics.items():
        diagnostic_summaries[name] = {
            "selection_eligible": False,
            "head_refit": False,
            "metrics": metrics_after_unseal(keys, decisions, test_labels),
            "predictions_sha256": sha256_file(path),
            "timing": timing,
        }

    peak_gib = torch.cuda.max_memory_allocated(model.device) / 2**30
    resources = _mapping(inputs.config["resources"], "resources")
    resource_gates = {
        "peak_allocated_at_most_70_gib": peak_gib
        <= float(resources["maximum_peak_allocated_gib"]),
        "maximum_session_model_at_most_300_seconds": maximum_session_seconds
        <= float(resources["official_session_timeout_seconds"]),
    }
    summary = {
        "schema_version": 1,
        "kind": "d6_formal_oof_fold",
        "status": "complete" if all(resource_gates.values()) else "resource_gate_failed",
        "config_sha256": inputs.config_sha256,
        "fold": fold,
        "rotation": {
            "fit_folds": rotation["fit_folds"],
            "calibration_fold": rotation["calibration_fold"],
            "test_fold": fold,
        },
        "adapter": {
            "best_epoch": result.best_epoch,
            "best_calibration_bce": result.best_calibration_bce,
            "epochs_completed": result.epochs_completed,
            "stopped_early": result.stopped_early,
            "checkpoint_sha256": result.checkpoint_sha256,
            "history": list(result.history),
            "training_audit": training_audit,
        },
        "head": {
            "parameters": 1052,
            "selected_l2_weight": head_result.selected_l2_weight,
            "threshold_logit": head_result.head.threshold_logit,
            "calibration_metrics": head_result.calibration_metrics,
            "calibration_grid": list(head_result.calibration_grid),
            "sha256": sha256_file(head_path),
        },
        "primary": {
            "test_metrics": primary_metrics,
            "test_predictions_sha256": frozen_sha,
            "features_timing": primary_timing,
            "representation_audit": representation_audit,
        },
        "fixed_diagnostics": diagnostic_summaries,
        "peak_allocated_gib": peak_gib,
        "maximum_session_model_seconds": maximum_session_seconds,
        "resource_gates": resource_gates,
    }
    write_json(output_dir / "fold_summary.json", summary)
    if not all(resource_gates.values()):
        raise RuntimeError(f"D6 formal fold resource gate failed: {resource_gates}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--fold", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trainability-smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--zero-init-summary")
    parser.add_argument("--trainability-summary")
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="Use the user-authorized shared-GPU execution amendment.",
    )
    args = parser.parse_args(argv)

    inputs = load_experiment(
        Path(args.config),
        model_path_override=Path(args.model_path) if args.model_path else None,
    )
    if not 0 <= args.fold < 5:
        raise ValueError("D6 fold must be in [0, 4]")
    if args.trainability_smoke and args.fold != 0:
        raise ValueError("D6 trainability smoke is frozen to rotation 0")
    if args.formal:
        if not args.zero_init_summary or not args.trainability_summary:
            raise ValueError("D6 formal execution requires both smoke summaries")
        _load_gate(Path(args.zero_init_summary), inputs, "d6_zero_init_causality_smoke")
        _load_gate(
            Path(args.trainability_summary),
            inputs,
            "d6_rotation0_trainability_smoke",
        )
    resources = _mapping(inputs.config["resources"], "resources")
    require_exclusive_gpu = not args.allow_shared_gpu
    minimum_free_gib = (
        SHARED_MINIMUM_FREE_GIB
        if args.allow_shared_gpu
        else float(resources["minimum_free_memory_gib"])
    )
    resource = gpu_resource_audit(
        args.device,
        minimum_free_gib,
        require_exclusive=require_exclusive_gpu,
    )
    output_dir = Path(args.output_dir).resolve()
    _write_contract(output_dir, inputs, resource)
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    model = _build_model(
        inputs,
        args.device,
        require_exclusive_gpu=require_exclusive_gpu,
    )
    if args.trainability_smoke:
        result = run_trainability_smoke(inputs, model, output_dir)
    else:
        result = run_formal_fold(inputs, model, args.fold, output_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
