"""Extract answer-free current-interval InternVL vision states."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

from proactive_d1.core import strip_answers
from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    verify_model_snapshot,
    write_json,
)
from proactive_r0.core import (
    CAUSAL_MULTISCALE_FRAME_SAMPLING,
    CausalInferenceConfig,
    load_jsonl,
    load_starter_kit,
    select_causal_frames,
    validate_source_rows,
)
from proactive_r0.internvl import InternVLProactiveModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("proactive_d5.visual_features")


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def current_interval_vision_state(image_features: object, current_frames: int) -> object:
    """Mean current-frame/spatial pooler tokens and return an L2-normalized vector."""
    import torch
    import torch.nn.functional as functional

    if not isinstance(image_features, torch.Tensor) or image_features.ndim != 3:
        raise ValueError("D5 image features must have [frames, spatial_tokens, width]")
    if current_frames <= 0 or current_frames > image_features.shape[0]:
        raise ValueError("D5 current-frame count is outside the selected vision batch")
    state = image_features[-current_frames:].float().mean(dim=(0, 1))
    if state.ndim != 1 or not torch.isfinite(state).all():
        raise ValueError("D5 pooled current-interval vision state is invalid")
    return functional.normalize(state, dim=0).detach().cpu()


def _configure(output_dir: Path, append: bool) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "run.log", mode="a" if append else "w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def _write_command(path: Path, argv: Sequence[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d5.run_visual_features", *argv])
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--session-indices", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    args = parser.parse_args(raw_argv)
    started = time.monotonic()
    config_path = _resolve(args.config)
    config = _load_object(config_path)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "session_records.jsonl"
    if records_path.exists() and not args.resume:
        raise FileExistsError(f"Existing visual records require --resume: {records_path}")
    _configure(output_dir, args.resume)
    try:
        indices = [int(value) for value in args.session_indices.split(",") if value]
    except ValueError as error:
        parser.error(f"Invalid --session-indices: {error}")
    if not indices or indices != sorted(set(indices)):
        parser.error("--session-indices must be non-empty, unique, and sorted")

    data = config["data"]
    model_config = config["model"]
    starter_config = config["starter_kit"]
    inference = config["inference"]
    if not all(isinstance(value, dict) for value in (data, model_config, starter_config, inference)):
        raise ValueError("D5 visual feature config sections are malformed")
    input_path = _resolve(data["input"])
    video_folder = _resolve(data["video_folder"])
    starter_dir = _resolve(starter_config["path"])
    model_path = _resolve(model_config["default_local_path"])
    if sha256_file(input_path) != data["input_sha256"]:
        raise ValueError("D5 visual source fingerprint changed")
    if sha256_file(starter_dir / "model.py") != starter_config["model_py_sha256"]:
        raise ValueError("D5 visual starter extractor fingerprint changed")
    protocol = config["protocol"]
    if not isinstance(protocol, dict) or sha256_file(_resolve(protocol["path"])) != protocol["sha256"]:
        raise ValueError("D5 visual protocol fingerprint changed")
    model_audit = verify_model_snapshot(model_path, model_config)
    all_rows = load_jsonl(input_path)
    if indices[-1] >= len(all_rows):
        parser.error("--session-indices contains an out-of-range value")
    rows = [strip_answers([all_rows[index]])[0] for index in indices]
    validation = validate_source_rows(rows, video_folder)
    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "device": args.device,
        "session_indices": indices,
        "require_exclusive_gpu": args.require_exclusive_gpu,
    }
    config_output = output_dir / "config.json"
    if args.resume and config_output.exists():
        if _load_object(config_output) != effective:
            raise ValueError("D5 visual effective config changed on resume")
    else:
        write_json(config_output, effective)
        _write_command(output_dir / "command.sh", raw_argv)
        write_json(output_dir / "environment.txt", environment_snapshot())
        tracked = [
            *sorted((PROJECT_ROOT / "src" / "proactive_r0").glob("*.py")),
            *sorted((PROJECT_ROOT / "src" / "proactive_d5").glob("*.py")),
            config_path,
            PROJECT_ROOT / "Agent.md",
            PROJECT_ROOT / "CURRENT_ROUTE.md",
        ]
        write_json(output_dir / "code_state.txt", code_snapshot(PROJECT_ROOT, tracked))
    write_json(
        output_dir / "data_manifest.json",
        {
            "source": {"path": str(input_path), "sha256": sha256_file(input_path), "input_indices": indices},
            "model": {"path": str(model_path), **model_audit},
            "starter_model_sha256": sha256_file(starter_dir / "model.py"),
            "protocol_sha256": protocol["sha256"],
            "feature_rows_contain_answers": False,
            "labels_used_for_features": False,
            "future_intervals_read": False,
        },
    )
    records = load_jsonl(records_path) if records_path.exists() else []
    if len(records) > len(rows):
        raise ValueError("D5 visual records exceed selected sessions")
    for position, record in enumerate(records):
        if record.get("input_index") != indices[position] or record.get("video_path") != rows[position].get("video_path"):
            raise ValueError("D5 visual resume identity changed")

    causal_config = CausalInferenceConfig(
        frames_per_interval=int(inference["frames_per_interval"]),
        max_frames=int(inference["max_frames"]),
        max_history_turns=8,
        max_new_tokens=64,
        frame_sampling=str(inference["frame_sampling"]),
    )
    if causal_config.frame_sampling != CAUSAL_MULTISCALE_FRAME_SAMPLING:
        raise ValueError("D5 visual extraction requires frozen multiscale sampling")
    starter = load_starter_kit(starter_dir)
    model = InternVLProactiveModel(
        model_path=str(model_path),
        device=args.device,
        dtype_name=str(model_config["dtype"]),
        attention_implementation=str(model_config["attention_implementation"]),
        seed=int(inference["seed"]),
        require_exclusive_gpu=args.require_exclusive_gpu,
        video_frame_size=int(inference["video_frame_size"]),
        pad_token_id=int(inference["pad_token_id"]),
    )
    if model.parameter_count != int(model_config["total_parameters"]):
        raise ValueError("D5 visual model parameter count changed")
    import torch

    with records_path.open("a", encoding="utf-8") as handle:
        for position in range(len(records), len(rows)):
            session_started = time.monotonic()
            row = rows[position]
            intervals = [tuple(float(item) for item in interval) for interval in row["video_intervals"]]  # type: ignore[index]
            groups: list[list[object]] = []
            chunks: list[dict[str, object]] = []
            for chunk_index, interval in enumerate(intervals):
                current = starter.extract_frames(
                    str(video_folder / str(row["video_path"])),
                    intervals=[interval],
                    frames_per_interval=causal_config.frames_per_interval,
                )
                groups.append(current)
                selected = select_causal_frames(groups, intervals[: chunk_index + 1], causal_config)
                if len(current) > len(selected) or any(
                    selected[len(selected) - len(current) + index] is not frame
                    for index, frame in enumerate(current)
                ):
                    raise RuntimeError("D5 multiscale current frames are not the selected chronological tail")
                video_inputs = model.processor.video_processor(
                    videos=[selected], return_tensors="pt"
                )
                pixel_values = video_inputs["pixel_values_videos"].flatten(0, 1).to(model.device)
                with torch.inference_mode():
                    image_features = model.model.model.get_image_features(
                        pixel_values=pixel_values, return_dict=True
                    ).pooler_output
                if image_features.shape[0] != len(selected) or image_features.shape[2] != int(inference["vision_width"]):
                    raise RuntimeError("D5 vision pooler shape changed")
                state = current_interval_vision_state(image_features, len(current))
                vector = [float(value) for value in state.tolist()]
                if len(vector) != int(inference["vision_width"]) or not all(math.isfinite(value) for value in vector):
                    raise RuntimeError("D5 visual state width or finiteness changed")
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "interval": list(interval),
                        "current_interval_frames": len(current),
                        "model_input_frames": len(selected),
                        "frame_sampling": causal_config.frame_sampling,
                        "vision_state": vector,
                    }
                )
            record = {
                "input_index": indices[position],
                "video_path": row["video_path"],
                "chunks": chunks,
                "session_wall_time_seconds": time.monotonic() - session_started,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            records.append(record)
            LOGGER.info(
                "Session %d/%d complete: chunks=%d elapsed=%.2fs",
                position + 1,
                len(rows),
                len(chunks),
                record["session_wall_time_seconds"],
            )
    write_json(
        output_dir / "runtime.json",
        {
            "status": "complete answer-free visual feature shard",
            "completed_at": datetime.now().astimezone().isoformat(),
            "wall_time_seconds": time.monotonic() - started,
            "sessions": validation["sessions"],
            "chunks": validation["chunks"],
            "peak_gpu_memory_bytes": model.peak_memory_bytes(),
            "preexisting_gpu_processes": model.preexisting_gpu_processes,
            "total_parameters": model.parameter_count,
        },
    )


if __name__ == "__main__":
    main()
