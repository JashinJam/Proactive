"""Run the gate-conditional frozen D6 all-development refit and online audit."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as functional

from proactive_d1.core import (
    LinearDecisionHead,
    decision_answer,
    fit_linear_logistic,
    predict_logits,
    serialize_decision_head,
)
from proactive_d4_1.core import atomic_append_jsonl
from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl, load_starter_kit, validate_prediction_rows, write_jsonl
from proactive_r0.run import _run_official_scorer

from .adapter import D6DecisionModel
from .contract import (
    SHARED_MINIMUM_FREE_GIB,
    gpu_resource_audit,
    inference_config,
    labels_for_allowed_sessions,
    load_experiment,
)
from .head import build_label_free_matrix
from .run_fold import _build_model, _extract_records, _mapping, _write_contract
from .runtime import process_session


DEFAULT_CONFIG = Path("configs/d6_internvl35_1b_query_memory_lora_oof_v1.json")


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class FullDevelopmentTrainer:
    """Fixed-epoch all-development training with the frozen session boundaries."""

    def __init__(
        self,
        *,
        model: D6DecisionModel,
        inputs: object,
        labels: Mapping[tuple[int, int], int],
        output_dir: Path,
        epochs: int,
    ) -> None:
        self.model = model
        self.inputs = inputs
        self.labels = labels
        self.output_dir = output_dir
        self.epochs = epochs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        training = _mapping(inputs.config["adapter_training"], "adapter_training")
        positives = sum(labels.values())
        negatives = len(labels) - positives
        if positives <= 0 or negatives <= 0:
            raise ValueError("D6 full-development labels require both classes")
        self.positive_weight = negatives / positives
        self.seed = int(training["seed"])
        self.accumulation_target = int(training["gradient_accumulation_target_chunks"])
        self.gradient_clip = float(training["gradient_clip_norm"])
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for _, p in model.memory_named_parameters()],
                    "lr": float(training["memory_learning_rate"]),
                    "weight_decay": float(training["weight_decay"]),
                },
                {
                    "params": [p for _, p in model.lora_named_parameters()],
                    "lr": float(training["lora_learning_rate"]),
                    "weight_decay": float(training["weight_decay"]),
                },
            ]
        )
        self.checkpoint = output_dir / "training_checkpoint.pt"
        self.adapter = output_dir / "adapter.pt"

    def _loss(self, margin: torch.Tensor, label: int) -> torch.Tensor:
        return functional.binary_cross_entropy_with_logits(
            margin.float(),
            torch.tensor(float(label), device=margin.device),
            pos_weight=torch.tensor(self.positive_weight, device=margin.device),
        )

    def _save(
        self,
        *,
        epoch: int,
        next_position: int,
        history: Sequence[Mapping[str, object]],
        complete: bool,
    ) -> None:
        _atomic_torch_save(
            self.checkpoint,
            {
                "schema_version": 1,
                "kind": "d6_full_development_training_state",
                "config_sha256": self.inputs.config_sha256,
                "model_weights_sha256": self.inputs.config["model"]["weights_sha256"],
                "fixed_epochs": self.epochs,
                "epoch": epoch,
                "next_session_position": next_position,
                "history": [dict(row) for row in history],
                "adapter_state": self.model.adapter_state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "python_random_state": random.getstate(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": torch.cuda.get_rng_state_all(),
                "complete": complete,
                "base_model_tensors_saved": False,
            },
        )

    def _load(self) -> dict[str, object] | None:
        if not self.checkpoint.exists():
            return None
        value = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(value, dict) or value.get("kind") != "d6_full_development_training_state":
            raise ValueError("D6 full-development checkpoint schema changed")
        if value.get("config_sha256") != self.inputs.config_sha256:
            raise ValueError("D6 full-development checkpoint config changed")
        if value.get("fixed_epochs") != self.epochs:
            raise ValueError("D6 full-development fixed epoch changed")
        if value.get("base_model_tensors_saved") is not False:
            raise ValueError("D6 full-development checkpoint contains base weights")
        self.model.load_adapter_state_dict(value["adapter_state"])
        self.optimizer.load_state_dict(value["optimizer_state"])
        random.setstate(value["python_random_state"])
        torch.set_rng_state(value["torch_random_state"])
        torch.cuda.set_rng_state_all(value["cuda_random_state"])
        return value

    def _step(self, chunks: int) -> float:
        parameters = [parameter for _, parameter in self.model.trainable_named_parameters()]
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(chunks)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip).cpu())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return norm

    def train(self) -> dict[str, object]:
        checkpoint = self._load()
        if checkpoint is not None and checkpoint.get("complete") is True:
            return {
                "history": checkpoint["history"],
                "adapter_sha256": sha256_file(self.adapter),
            }
        history = [] if checkpoint is None else list(checkpoint["history"])
        start_epoch = 1 if checkpoint is None else int(checkpoint["epoch"])
        next_position = 0 if checkpoint is None else int(checkpoint["next_session_position"])
        starter = load_starter_kit(self.inputs.starter_dir)
        inference = inference_config(self.inputs.config)
        for epoch in range(start_epoch, self.epochs + 1):
            order = list(range(len(self.inputs.answer_free_rows)))
            random.Random(self.seed + epoch).shuffle(order)
            position = next_position if epoch == start_epoch else 0
            accumulated = 0
            loss_sum = 0.0
            steps = 0
            norms: list[float] = []
            maximum_session_seconds = 0.0
            started = time.monotonic()
            self.optimizer.zero_grad(set_to_none=True)
            while position < len(order):
                input_index = order[position]
                session_chunks = 0

                def consume(chunk_index: int, output: object) -> None:
                    nonlocal accumulated, loss_sum, session_chunks
                    loss = self._loss(
                        output.tag_margin, self.labels[(input_index, chunk_index)]
                    )
                    loss.backward()
                    loss_sum += float(loss.detach().cpu())
                    accumulated += 1
                    session_chunks += 1

                record = process_session(
                    row=self.inputs.answer_free_rows[input_index],
                    input_index=input_index,
                    video_folder=self.inputs.video_folder,
                    model=self.model,
                    starter=starter,
                    inference=inference,
                    reference=self.inputs.references[input_index],
                    callback=consume,
                    record_hidden_state=False,
                    record_chunks=False,
                )
                maximum_session_seconds = max(
                    maximum_session_seconds,
                    float(record["timing"]["model_inference_seconds"]),
                )
                if session_chunks == 0:
                    raise RuntimeError("D6 full-development session has no chunks")
                position += 1
                if accumulated >= self.accumulation_target:
                    norms.append(self._step(accumulated))
                    accumulated = 0
                    steps += 1
                    self._save(
                        epoch=epoch,
                        next_position=position,
                        history=history,
                        complete=False,
                    )
            if accumulated:
                norms.append(self._step(accumulated))
                steps += 1
            history.append(
                {
                    "epoch": epoch,
                    "fit_chunks": len(self.labels),
                    "fit_loss_sum": loss_sum,
                    "optimizer_steps": steps,
                    "gradient_norm_max_before_clip": max(norms, default=0.0),
                    "maximum_session_model_seconds": maximum_session_seconds,
                    "wall_seconds": time.monotonic() - started,
                }
            )
            next_position = 0
            self._save(
                epoch=epoch + 1,
                next_position=0,
                history=history,
                complete=False,
            )
        _atomic_torch_save(
            self.adapter,
            {
                "schema_version": 1,
                "kind": "d6_full_development_adapter_only",
                "config_sha256": self.inputs.config_sha256,
                "fixed_epochs": self.epochs,
                "adapter_state": self.model.adapter_state_dict(),
                "base_model_tensors_saved": False,
            },
        )
        self._save(
            epoch=self.epochs + 1,
            next_position=0,
            history=history,
            complete=True,
        )
        return {"history": history, "adapter_sha256": sha256_file(self.adapter)}


def _online_replay_audit(
    *,
    inputs: object,
    model: D6DecisionModel,
    primary_records: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> dict[str, object]:
    indices = [int(value) for value in inputs.config["smoke"]["source_indices"]]
    replay, timing = _extract_records(
        path=output_dir / "online_audit_records.jsonl",
        session_indices=indices,
        inputs=inputs,
        model=model,
        memory_enabled=True,
        lora_enabled=True,
    )
    primary = {int(record["input_index"]): record for record in primary_records}
    maximum = {name: 0.0 for name in ("tag_margin", "silent_log_probability", "interrupt_log_probability", "hidden_state")}
    chunks = 0
    for record in replay:
        reference = primary[int(record["input_index"])]
        for actual, expected in zip(record["chunks"], reference["chunks"]):
            for name in ("tag_margin", "silent_log_probability", "interrupt_log_probability"):
                maximum[name] = max(maximum[name], abs(float(actual[name]) - float(expected[name])))
            maximum["hidden_state"] = max(
                maximum["hidden_state"],
                max(abs(float(a) - float(b)) for a, b in zip(actual["hidden_state"], expected["hidden_state"])),
            )
            chunks += 1
    return {
        "chunks": chunks,
        "maximum_abs_differences": maximum,
        "all_exact": chunks == int(inputs.config["smoke"]["chunks"])
        and all(value == 0.0 for value in maximum.values()),
        "timing": timing,
        "records_sha256": sha256_file(output_dir / "online_audit_records.jsonl"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args(argv)
    inputs = load_experiment(
        Path(args.config),
        model_path_override=Path(args.model_path) if args.model_path else None,
    )
    evaluation_path = Path(args.experiment_dir).resolve() / "evaluation" / "summary.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not isinstance(evaluation, dict) or evaluation.get("config_sha256") != inputs.config_sha256:
        raise ValueError("D6 refit evaluation identity changed")
    if evaluation.get("promotion_passed") is not True:
        raise ValueError("D6 full-development refit is forbidden before all gates pass")
    folds = evaluation["fold_summaries"]
    epochs = int(statistics.median(int(row["adapter"]["best_epoch"]) for row in folds))
    l2_weight = float(statistics.median(float(row["head"]["selected_l2_weight"]) for row in folds))
    threshold = float(statistics.median(float(row["head"]["threshold_logit"]) for row in folds))
    if epochs < 0:
        raise ValueError("D6 refit median best epoch changed")

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
    labels = labels_for_allowed_sessions(
        inputs.source_rows, list(range(len(inputs.source_rows)))
    )
    trainer = FullDevelopmentTrainer(
        model=model,
        inputs=inputs,
        labels=labels,
        output_dir=output_dir / "training",
        epochs=epochs,
    )
    training_result = trainer.train()
    records, timing = _extract_records(
        path=output_dir / "session_records.jsonl",
        session_indices=list(range(len(inputs.answer_free_rows))),
        inputs=inputs,
        model=model,
        memory_enabled=True,
        lora_enabled=True,
    )
    matrix = build_label_free_matrix(
        answer_free_rows=inputs.answer_free_rows,
        records=records,
        fold_by_index=inputs.fold_by_index,
    )
    aligned_labels = [labels[(chunk.input_index, chunk.chunk_index)] for chunk in matrix.chunks]
    head_config = _mapping(inputs.config["head_training"], "head_training")
    linear = fit_linear_logistic(
        matrix.values,
        aligned_labels,
        seed=int(head_config["seed"]),
        max_iterations=int(head_config["max_iterations"]),
        l2_weight=l2_weight,
        l2_reduction="sum",
    )
    head = LinearDecisionHead(matrix.names, linear, threshold)
    head_path = output_dir / "decision_head.json"
    write_json(
        head_path,
        serialize_decision_head(
            head,
            {
                "experiment_id": inputs.config["experiment_id"],
                "config_sha256": inputs.config_sha256,
                "classification": "all-public-development train-fit deployment artifact",
                "fixed_epochs_median": epochs,
                "fixed_l2_median": l2_weight,
                "fixed_threshold_median": threshold,
                "selection_after_oof": False,
            },
        ),
    )
    logits = predict_logits(linear, matrix.values)
    decisions = [int(value >= threshold) for value in logits]
    predictions = []
    cursor = 0
    for input_index, row in enumerate(inputs.answer_free_rows):
        answers = []
        for chunk_index in range(len(row["video_intervals"])):
            answers.append(
                decision_answer(
                    matrix.chunks[cursor].raw_response,
                    decisions[cursor],
                )
            )
            cursor += 1
        predictions.append({"video_path": row["video_path"], "answers": answers})
    validate_prediction_rows(inputs.source_rows, predictions)
    predictions_path = output_dir / "train_fit_predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    metrics_path = output_dir / "train_fit_metrics.json"
    _run_official_scorer(
        inputs.starter_dir,
        inputs.input_path,
        predictions_path,
        metrics_path,
        output_dir / "scorer.log",
    )
    online = _online_replay_audit(
        inputs=inputs,
        model=model,
        primary_records=records,
        output_dir=output_dir,
    )
    peak_gib = torch.cuda.max_memory_allocated(model.device) / 2**30
    gates = {
        "online_102_chunk_replay_exact": bool(online["all_exact"]),
        "peak_allocated_at_most_70_gib": peak_gib
        <= float(resources["maximum_peak_allocated_gib"]),
        "maximum_session_model_at_most_300_seconds": float(
            online["timing"]["maximum_session_model_seconds"]
        )
        <= float(resources["official_session_timeout_seconds"]),
    }
    summary = {
        "schema_version": 1,
        "kind": "d6_all_development_refit",
        "status": "complete" if all(gates.values()) else "audit_failed",
        "classification": "train-fit deployment artifact; not held-out performance",
        "config_sha256": inputs.config_sha256,
        "fixed_epochs_median": epochs,
        "fixed_l2_median": l2_weight,
        "fixed_threshold_median": threshold,
        "training": training_result,
        "feature_timing": timing,
        "online_audit": online,
        "peak_allocated_gib": peak_gib,
        "head_sha256": sha256_file(head_path),
        "predictions_sha256": sha256_file(predictions_path),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "gates": gates,
        "external_upload_authorized": False,
    }
    write_json(output_dir / "summary.json", summary)
    if not all(gates.values()):
        raise RuntimeError(f"D6 full-development online audit failed: {gates}")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
