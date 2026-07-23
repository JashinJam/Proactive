"""Session-boundary resumable D6 adapter training."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as functional

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import CausalInferenceConfig, StarterKitSymbols

from .adapter import D6DecisionModel
from .runtime import process_session


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_calibration_bce: float
    epochs_completed: int
    stopped_early: bool
    checkpoint_path: str
    checkpoint_sha256: str
    history: tuple[dict[str, object], ...]


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _json_safe_history(history: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(value) for value in history]


class FoldTrainer:
    def __init__(
        self,
        *,
        model: D6DecisionModel,
        rows: Sequence[dict[str, object]],
        references: Mapping[int, dict[str, object]],
        labels: Mapping[tuple[int, int], int],
        fit_sessions: Sequence[int],
        calibration_sessions: Sequence[int],
        video_folder: Path,
        starter: StarterKitSymbols,
        inference: CausalInferenceConfig,
        fold: int,
        output_dir: Path,
        config_sha256: str,
        model_weights_sha256: str,
        memory_learning_rate: float,
        lora_learning_rate: float,
        weight_decay: float,
        gradient_clip_norm: float,
        accumulation_target_chunks: int,
        maximum_epochs: int,
        patience: int,
        seed: int,
    ) -> None:
        if any("answers" in row for row in rows):
            raise ValueError("D6 trainer received answers in model-facing rows")
        self.model = model
        self.rows = rows
        self.references = references
        self.labels = labels
        self.fit_sessions = tuple(fit_sessions)
        self.calibration_sessions = tuple(calibration_sessions)
        self.video_folder = video_folder
        self.starter = starter
        self.inference = inference
        self.fold = fold
        self.output_dir = output_dir
        self.config_sha256 = config_sha256
        self.model_weights_sha256 = model_weights_sha256
        self.gradient_clip_norm = gradient_clip_norm
        self.accumulation_target_chunks = accumulation_target_chunks
        self.maximum_epochs = maximum_epochs
        self.patience_limit = patience
        self.seed = seed
        fit_labels = [
            label
            for (input_index, _), label in labels.items()
            if input_index in set(self.fit_sessions)
        ]
        positives = sum(fit_labels)
        negatives = len(fit_labels) - positives
        if positives <= 0 or negatives <= 0:
            raise ValueError("D6 fit folds require both classes")
        self.positive_weight = negatives / positives
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": [parameter for _, parameter in model.memory_named_parameters()],
                    "lr": memory_learning_rate,
                    "weight_decay": weight_decay,
                    "name": "memory",
                },
                {
                    "params": [parameter for _, parameter in model.lora_named_parameters()],
                    "lr": lora_learning_rate,
                    "weight_decay": weight_decay,
                    "name": "lora",
                },
            ]
        )
        self.checkpoint_path = output_dir / "training_checkpoint.pt"
        self.best_path = output_dir / "best_adapter.pt"
        self.status_path = output_dir / "training_status.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _loss(self, margin: torch.Tensor, label: int) -> torch.Tensor:
        target = torch.tensor(float(label), device=margin.device, dtype=torch.float32)
        weight = torch.tensor(
            self.positive_weight, device=margin.device, dtype=torch.float32
        )
        return functional.binary_cross_entropy_with_logits(
            margin.float(), target, pos_weight=weight
        )

    def _checkpoint(
        self,
        *,
        epoch: int,
        next_session_position: int,
        best_epoch: int,
        best_calibration_bce: float,
        no_improvement_epochs: int,
        history: Sequence[Mapping[str, object]],
        complete: bool,
        stopped_early: bool = False,
    ) -> None:
        payload = {
            "schema_version": 1,
            "kind": "d6_adapter_training_state",
            "fold": self.fold,
            "config_sha256": self.config_sha256,
            "model_weights_sha256": self.model_weights_sha256,
            "epoch": epoch,
            "next_session_position": next_session_position,
            "best_epoch": best_epoch,
            "best_calibration_bce": best_calibration_bce,
            "no_improvement_epochs": no_improvement_epochs,
            "history": _json_safe_history(history),
            "adapter_state": self.model.adapter_state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all(),
            "complete": complete,
            "stopped_early": stopped_early,
            "base_model_tensors_saved": False,
        }
        _atomic_torch_save(self.checkpoint_path, payload)

    def _load_checkpoint(self) -> dict[str, object] | None:
        if not self.checkpoint_path.exists():
            return None
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("kind") != "d6_adapter_training_state":
            raise ValueError("D6 training checkpoint schema changed")
        if payload.get("fold") != self.fold or payload.get("config_sha256") != self.config_sha256:
            raise ValueError("D6 training checkpoint identity changed")
        if payload.get("model_weights_sha256") != self.model_weights_sha256:
            raise ValueError("D6 training checkpoint base-model hash changed")
        if payload.get("base_model_tensors_saved") is not False:
            raise ValueError("D6 checkpoint unexpectedly contains base weights")
        self.model.load_adapter_state_dict(payload["adapter_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        random.setstate(payload["python_random_state"])
        torch.set_rng_state(payload["torch_random_state"])
        torch.cuda.set_rng_state_all(payload["cuda_random_state"])
        return payload

    def _save_best(self, epoch: int, calibration_bce: float) -> None:
        _atomic_torch_save(
            self.best_path,
            {
                "schema_version": 1,
                "kind": "d6_adapter_only",
                "fold": self.fold,
                "epoch": epoch,
                "calibration_bce": calibration_bce,
                "config_sha256": self.config_sha256,
                "model_weights_sha256": self.model_weights_sha256,
                "adapter_state": self.model.adapter_state_dict(),
                "base_model_tensors_saved": False,
            },
        )

    def load_best(self) -> dict[str, object]:
        payload = torch.load(self.best_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("kind") != "d6_adapter_only":
            raise ValueError("D6 best adapter schema changed")
        if payload.get("config_sha256") != self.config_sha256:
            raise ValueError("D6 best adapter config changed")
        self.model.load_adapter_state_dict(payload["adapter_state"])
        return payload

    def calibration_bce(self) -> tuple[float, dict[str, float]]:
        total = 0.0
        chunks = 0
        residual_sum = 0.0
        entropy_sum = 0.0
        maximum_session_model_seconds = 0.0
        started = time.monotonic()
        with torch.no_grad():
            for input_index in self.calibration_sessions:
                def consume(chunk_index: int, output: object) -> None:
                    nonlocal total, chunks, residual_sum, entropy_sum
                    forward = output
                    label = self.labels[(input_index, chunk_index)]
                    total += float(self._loss(forward.tag_margin, label).cpu())
                    residual_sum += float(forward.residual_norm.cpu())
                    entropy_sum += float(forward.normalized_attention_entropy.cpu())
                    chunks += 1

                record = process_session(
                    row=self.rows[input_index],
                    input_index=input_index,
                    video_folder=self.video_folder,
                    model=self.model,
                    starter=self.starter,
                    inference=self.inference,
                    reference=self.references[input_index],
                    record_hidden_state=False,
                    record_chunks=False,
                    callback=consume,
                )
                maximum_session_model_seconds = max(
                    maximum_session_model_seconds,
                    float(record["timing"]["model_inference_seconds"]),  # type: ignore[index]
                )
        if chunks == 0:
            raise ValueError("D6 calibration fold has no chunks")
        return total / chunks, {
            "chunks": float(chunks),
            "mean_residual_norm": residual_sum / chunks,
            "mean_normalized_attention_entropy": entropy_sum / chunks,
            "wall_time_seconds": time.monotonic() - started,
            "maximum_session_model_seconds": maximum_session_model_seconds,
        }

    def _optimizer_step(self, accumulated_chunks: int) -> float:
        if accumulated_chunks <= 0:
            raise ValueError("D6 optimizer step has no accumulated chunks")
        parameters = [parameter for _, parameter in self.model.trainable_named_parameters()]
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(accumulated_chunks)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_norm).detach().cpu()
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return gradient_norm

    def train(self) -> TrainingResult:
        checkpoint = self._load_checkpoint()
        if checkpoint is not None and checkpoint.get("complete") is True:
            self.load_best()
            history = tuple(dict(value) for value in checkpoint["history"])
            return TrainingResult(
                best_epoch=int(checkpoint["best_epoch"]),
                best_calibration_bce=float(checkpoint["best_calibration_bce"]),
                epochs_completed=max(int(value["epoch"]) for value in history),
                stopped_early=bool(checkpoint.get("stopped_early", False)),
                checkpoint_path=str(self.best_path),
                checkpoint_sha256=sha256_file(self.best_path),
                history=history,
            )
        history = [] if checkpoint is None else list(checkpoint["history"])
        best_epoch = -1 if checkpoint is None else int(checkpoint["best_epoch"])
        best_bce = float("inf") if checkpoint is None else float(
            checkpoint["best_calibration_bce"]
        )
        no_improvement = 0 if checkpoint is None else int(
            checkpoint["no_improvement_epochs"]
        )
        start_epoch = 0 if checkpoint is None else int(checkpoint["epoch"])
        next_position = 0 if checkpoint is None else int(
            checkpoint["next_session_position"]
        )
        if checkpoint is None:
            epoch_zero_bce, audit = self.calibration_bce()
            best_epoch = 0
            best_bce = epoch_zero_bce
            self._save_best(0, epoch_zero_bce)
            history.append({"epoch": 0, "calibration_bce": epoch_zero_bce, **audit})
            self._checkpoint(
                epoch=1,
                next_session_position=0,
                best_epoch=best_epoch,
                best_calibration_bce=best_bce,
                no_improvement_epochs=no_improvement,
                history=history,
                complete=False,
            )
            start_epoch = 1
        stopped_early = False
        epochs_completed = 0
        for epoch in range(start_epoch, self.maximum_epochs + 1):
            order = list(self.fit_sessions)
            random.Random(self.seed + 1000 * self.fold + epoch).shuffle(order)
            position = next_position if epoch == start_epoch else 0
            accumulated_chunks = 0
            train_loss_sum = 0.0
            optimizer_steps = 0
            gradient_norms: list[float] = []
            maximum_fit_session_model_seconds = 0.0
            epoch_started = time.monotonic()
            self.optimizer.zero_grad(set_to_none=True)
            while position < len(order):
                input_index = order[position]
                session_chunks = 0

                def consume(chunk_index: int, output: object) -> None:
                    nonlocal accumulated_chunks, train_loss_sum, session_chunks
                    forward = output
                    label = self.labels[(input_index, chunk_index)]
                    loss = self._loss(forward.tag_margin, label)
                    loss.backward()
                    train_loss_sum += float(loss.detach().cpu())
                    accumulated_chunks += 1
                    session_chunks += 1

                record = process_session(
                    row=self.rows[input_index],
                    input_index=input_index,
                    video_folder=self.video_folder,
                    model=self.model,
                    starter=self.starter,
                    inference=self.inference,
                    reference=self.references[input_index],
                    record_hidden_state=False,
                    record_chunks=False,
                    callback=consume,
                )
                maximum_fit_session_model_seconds = max(
                    maximum_fit_session_model_seconds,
                    float(record["timing"]["model_inference_seconds"]),  # type: ignore[index]
                )
                if session_chunks == 0:
                    raise RuntimeError("D6 fit session produced no chunks")
                position += 1
                if accumulated_chunks >= self.accumulation_target_chunks:
                    gradient_norms.append(self._optimizer_step(accumulated_chunks))
                    optimizer_steps += 1
                    accumulated_chunks = 0
                    self._checkpoint(
                        epoch=epoch,
                        next_session_position=position,
                        best_epoch=best_epoch,
                        best_calibration_bce=best_bce,
                        no_improvement_epochs=no_improvement,
                        history=history,
                        complete=False,
                    )
            if accumulated_chunks:
                gradient_norms.append(self._optimizer_step(accumulated_chunks))
                optimizer_steps += 1
            calibration_bce, audit = self.calibration_bce()
            improved = calibration_bce < best_bce
            if improved:
                best_bce = calibration_bce
                best_epoch = epoch
                no_improvement = 0
                self._save_best(epoch, calibration_bce)
            else:
                no_improvement += 1
            history.append(
                {
                    "epoch": epoch,
                    "fit_loss_sum": train_loss_sum,
                    "fit_chunks": sum(
                        len(self.rows[index]["video_intervals"])  # type: ignore[arg-type]
                        for index in self.fit_sessions
                    ),
                    "optimizer_steps": optimizer_steps,
                    "gradient_norm_max_before_clip": max(gradient_norms, default=0.0),
                    "maximum_fit_session_model_seconds": maximum_fit_session_model_seconds,
                    "calibration_bce": calibration_bce,
                    "is_best": improved,
                    "epoch_wall_time_seconds": time.monotonic() - epoch_started,
                    **audit,
                }
            )
            epochs_completed = epoch
            next_position = 0
            if no_improvement >= self.patience_limit:
                stopped_early = True
                break
            self._checkpoint(
                epoch=epoch + 1,
                next_session_position=0,
                best_epoch=best_epoch,
                best_calibration_bce=best_bce,
                no_improvement_epochs=no_improvement,
                history=history,
                complete=False,
            )
        self.load_best()
        self._checkpoint(
            epoch=epochs_completed + 1,
            next_session_position=0,
            best_epoch=best_epoch,
            best_calibration_bce=best_bce,
            no_improvement_epochs=no_improvement,
            history=history,
            complete=True,
            stopped_early=stopped_early,
        )
        status = {
            "status": "complete",
            "fold": self.fold,
            "best_epoch": best_epoch,
            "best_calibration_bce": best_bce,
            "epochs_completed": epochs_completed,
            "stopped_early": stopped_early,
            "best_adapter": str(self.best_path),
            "best_adapter_sha256": sha256_file(self.best_path),
            "base_model_tensors_saved": False,
        }
        write_json(self.status_path, status)
        return TrainingResult(
            best_epoch=best_epoch,
            best_calibration_bce=best_bce,
            epochs_completed=epochs_completed,
            stopped_early=stopped_early,
            checkpoint_path=str(self.best_path),
            checkpoint_sha256=sha256_file(self.best_path),
            history=tuple(dict(value) for value in history),
        )
