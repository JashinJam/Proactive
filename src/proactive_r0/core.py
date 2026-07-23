"""Causal, model-agnostic EgoProactive inference flow."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Protocol, Sequence

INTERRUPT_TAG = "$interrupt$"
SILENT_TAG = "$silent$"
EMPTY_INTERRUPT_UTTERANCE = "Please continue with the next step."
UNIFORM_CUMULATIVE_FRAME_SAMPLING = "uniform_cumulative_v1"
CAUSAL_MULTISCALE_FRAME_SAMPLING = "causal_multiscale_16_8_8_v1"
DETERMINISTIC_HALF_STRIDE_JITTER_FRAME_SAMPLING = (
    "deterministic_half_stride_jitter_v1"
)
FRAME_SAMPLING_POLICIES = (
    UNIFORM_CUMULATIVE_FRAME_SAMPLING,
    CAUSAL_MULTISCALE_FRAME_SAMPLING,
    DETERMINISTIC_HALF_STRIDE_JITTER_FRAME_SAMPLING,
)


class ProactiveModel(Protocol):
    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> str: ...


FrameExtractor = Callable[..., list[object]]
DialogNormalizer = Callable[[list[dict[str, object]]], list[dict[str, str]]]


@dataclass(frozen=True)
class CausalInferenceConfig:
    frames_per_interval: int
    max_frames: int
    max_history_turns: int
    max_new_tokens: int
    frame_sampling: str = UNIFORM_CUMULATIVE_FRAME_SAMPLING

    def __post_init__(self) -> None:
        if self.frames_per_interval <= 0:
            raise ValueError("frames_per_interval must be positive")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.max_history_turns < -1:
            raise ValueError("max_history_turns must be -1 or non-negative")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.frame_sampling not in FRAME_SAMPLING_POLICIES:
            raise ValueError(f"Unsupported frame sampling policy: {self.frame_sampling}")
        if self.frame_sampling == CAUSAL_MULTISCALE_FRAME_SAMPLING and (
            self.frames_per_interval != 16 or self.max_frames != 32
        ):
            raise ValueError(
                "causal_multiscale_16_8_8_v1 requires frames_per_interval=16 "
                "and max_frames=32"
            )


@dataclass(frozen=True)
class StarterKitSymbols:
    system_prompt: str
    normalize_dialog_turns: DialogNormalizer
    extract_frames: FrameExtractor


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_starter_kit(starter_kit_dir: Path) -> StarterKitSymbols:
    """Load the official prompt, dialog normalizer, and frame extractor in place."""
    starter_kit_dir = starter_kit_dir.resolve()
    model_module = _load_module(
        "_wearable_ai_starter_model", starter_kit_dir / "model.py"
    )
    proactive_module = _load_module(
        "_wearable_ai_starter_proactive",
        starter_kit_dir / "run_generate_proactive.py",
    )
    return StarterKitSymbols(
        system_prompt=proactive_module.SYSTEM_PROMPT,
        normalize_dialog_turns=proactive_module._normalize_dialog_turns,
        extract_frames=model_module.extract_frames,
    )


def canonicalize_response(raw_response: str) -> tuple[str, str | None]:
    """Return a submission-safe answer and any normalization reason.

    The official scorer treats every malformed response as silent. This function
    preserves that behavior while ensuring every emitted value satisfies the
    stricter submission schema.
    """
    stripped = str(raw_response).lstrip()
    if stripped.startswith(INTERRUPT_TAG):
        utterance = stripped[len(INTERRUPT_TAG) :].strip()
        if utterance:
            return f"{INTERRUPT_TAG}{utterance}", None
        return (
            f"{INTERRUPT_TAG}{EMPTY_INTERRUPT_UTTERANCE}",
            "empty_interrupt_utterance",
        )
    if stripped.startswith(SILENT_TAG):
        reason = None if stripped == SILENT_TAG else "trimmed_silent_suffix"
        return SILENT_TAG, reason
    return SILENT_TAG, "malformed_response_scored_as_silent"


def subsample_frames(frames: Sequence[object], max_frames: int) -> list[object]:
    """Apply the starter kit's deterministic uniform-stride frame cap."""
    selected = list(frames)
    if len(selected) <= max_frames:
        return selected
    stride = len(selected) / max_frames
    return [selected[int(index * stride)] for index in range(max_frames)]


def half_stride_jitter_frames(
    frames: Sequence[object], max_frames: int
) -> list[object]:
    """Select the midpoint of each uniform stride bin deterministically."""
    selected = list(frames)
    if len(selected) <= max_frames:
        return selected
    stride = len(selected) / max_frames
    return [
        selected[min(len(selected) - 1, int((index + 0.5) * stride))]
        for index in range(max_frames)
    ]


def _inclusive_uniform_indices(size: int, count: int) -> list[int]:
    """Select ordered unique indices spanning both ends of a non-empty sequence."""
    if size <= 0 or count <= 0:
        return []
    if count >= size:
        return list(range(size))
    if count == 1:
        return [size - 1]
    return [round(index * (size - 1) / (count - 1)) for index in range(count)]


def causal_multiscale_frames(
    interval_frames: Sequence[Sequence[object]],
    intervals: Sequence[tuple[float, float]],
    *,
    max_frames: int = 32,
) -> list[object]:
    """Keep current detail while preserving recent and long-range causal context."""
    if len(interval_frames) != len(intervals) or not interval_frames:
        raise ValueError("Frame groups and observed intervals must align and be non-empty")
    if max_frames != 32:
        raise ValueError("causal_multiscale_16_8_8_v1 has a frozen 32-frame budget")

    references: list[tuple[int, int, float, object]] = []
    for interval_index, (frames, interval) in enumerate(zip(interval_frames, intervals)):
        start, end = interval
        if start < 0 or end <= start:
            raise ValueError("Observed intervals must satisfy 0 <= start < end")
        count = len(frames)
        for frame_index, frame in enumerate(frames):
            # This matches the official extractor: the interval endpoint itself is
            # not sampled, so the final item is the available interval tail.
            timestamp = start + (end - start) * frame_index / max(count, 1)
            references.append((interval_index, frame_index, timestamp, frame))
    if len(references) <= max_frames:
        return [reference[3] for reference in references]

    selected: dict[tuple[int, int], tuple[int, int, float, object]] = {}

    def add(reference: tuple[int, int, float, object]) -> None:
        selected[(reference[0], reference[1])] = reference

    current = [reference for reference in references if reference[0] == len(intervals) - 1]
    for index in _inclusive_uniform_indices(len(current), 16):
        add(current[index])

    if len(intervals) >= 2:
        previous = [
            reference for reference in references if reference[0] == len(intervals) - 2
        ]
        for index in _inclusive_uniform_indices(len(previous), 8):
            add(previous[index])

    older = [reference for reference in references if reference[0] < len(intervals) - 2]
    if older:
        anchor_count = min(8, len(older))
        start_time, end_time = older[0][2], older[-1][2]
        targets = (
            [end_time]
            if anchor_count == 1
            else [
                start_time + (end_time - start_time) * index / (anchor_count - 1)
                for index in range(anchor_count)
            ]
        )
        available = list(older)
        for target in targets:
            reference = min(
                available,
                key=lambda value: (abs(value[2] - target), -value[2], value[0], value[1]),
            )
            add(reference)
            available.remove(reference)

    if len(selected) < max_frames:
        for reference in reversed(references):
            add(reference)
            if len(selected) == max_frames:
                break

    ordered = sorted(selected.values(), key=lambda value: (value[0], value[1]))
    if len(ordered) > max_frames:
        raise RuntimeError("Causal multiscale sampling exceeded its frame budget")
    return [reference[3] for reference in ordered]


def select_causal_frames(
    interval_frames: Sequence[Sequence[object]],
    intervals: Sequence[tuple[float, float]],
    config: CausalInferenceConfig,
) -> list[object]:
    """Apply the configured policy to frames from observed intervals only."""
    if len(interval_frames) != len(intervals):
        raise ValueError("Frame groups and observed intervals must align")
    if config.frame_sampling == UNIFORM_CUMULATIVE_FRAME_SAMPLING:
        return subsample_frames(
            [frame for group in interval_frames for frame in group], config.max_frames
        )
    if config.frame_sampling == CAUSAL_MULTISCALE_FRAME_SAMPLING:
        return causal_multiscale_frames(
            interval_frames, intervals, max_frames=config.max_frames
        )
    if config.frame_sampling == DETERMINISTIC_HALF_STRIDE_JITTER_FRAME_SAMPLING:
        return half_stride_jitter_frames(
            [frame for group in interval_frames for frame in group], config.max_frames
        )
    raise ValueError(f"Unsupported frame sampling policy: {config.frame_sampling}")


def build_messages(
    row: dict[str, object],
    chunk_index: int,
    system_prompt: str,
    normalize_dialog_turns: DialogNormalizer,
    max_history_turns: int,
) -> list[dict[str, str]]:
    """Build exactly the causal text context used by the official baseline."""
    query = str(row.get("query", ""))
    dialog = row.get("dialog", [])
    if not isinstance(dialog, list):
        raise ValueError("dialog must be a list")

    turns_after_query: list[dict[str, object]] = []
    if chunk_index < len(dialog):
        dialog_at_chunk = dialog[chunk_index]
        if not isinstance(dialog_at_chunk, list):
            raise ValueError(f"dialog[{chunk_index}] must be a list")
        if dialog_at_chunk:
            turns_after_query = dialog_at_chunk[1:]

    if max_history_turns == 0:
        turns_after_query = []
    elif max_history_turns > 0:
        turns_after_query = turns_after_query[-max_history_turns:]

    messages = [{"role": "system", "content": system_prompt}]
    if query:
        messages.append({"role": "user", "content": query})
    messages.extend(normalize_dialog_turns(turns_after_query))
    return messages


def validate_source_rows(
    rows: list[dict[str, object]],
    video_folder: Path,
) -> dict[str, object]:
    """Validate only inference-visible structure; do not inspect gold answers."""
    total_chunks = 0
    missing_videos: list[str] = []
    for row_index, row in enumerate(rows):
        video_path = row.get("video_path")
        intervals = row.get("video_intervals")
        dialog = row.get("dialog")
        if not isinstance(video_path, str) or not video_path:
            raise ValueError(f"row {row_index}: invalid video_path")
        if not isinstance(intervals, list) or not intervals:
            raise ValueError(f"row {row_index}: video_intervals must be non-empty")
        if not isinstance(dialog, list) or len(dialog) != len(intervals):
            raise ValueError(
                f"row {row_index}: dialog length must match video_intervals"
            )
        previous_start = -1.0
        for chunk_index, interval in enumerate(intervals):
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(
                    f"row {row_index} chunk {chunk_index}: invalid interval"
                )
            start, end = float(interval[0]), float(interval[1])
            if start < 0 or end <= start:
                raise ValueError(
                    f"row {row_index} chunk {chunk_index}: interval must satisfy "
                    "0 <= start < end"
                )
            if start < previous_start:
                raise ValueError(
                    f"row {row_index} chunk {chunk_index}: intervals are not ordered"
                )
            previous_start = start
        if not (video_folder / video_path).is_file():
            missing_videos.append(video_path)
        total_chunks += len(intervals)

    if missing_videos:
        preview = ", ".join(missing_videos[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_videos)} videos under {video_folder}: {preview}"
        )
    return {
        "sessions": len(rows),
        "chunks": total_chunks,
        "missing_videos": 0,
    }


def process_session(
    row: dict[str, object],
    input_index: int,
    video_folder: Path,
    model: ProactiveModel,
    starter: StarterKitSymbols,
    config: CausalInferenceConfig,
) -> dict[str, object]:
    """Generate all decisions for one session without accessing gold answers."""
    video_name = str(row["video_path"])
    video_path = video_folder / video_name
    intervals = [
        (float(interval[0]), float(interval[1]))
        for interval in row["video_intervals"]  # type: ignore[index]
    ]

    observed_frame_groups: list[list[object]] = []
    answers: list[str] = []
    chunks: list[dict[str, object]] = []
    for chunk_index, interval in enumerate(intervals):
        current_frames = starter.extract_frames(
            str(video_path),
            intervals=[interval],
            frames_per_interval=config.frames_per_interval,
        )
        observed_frame_groups.append(current_frames)
        model_frames = select_causal_frames(
            observed_frame_groups, intervals[: chunk_index + 1], config
        )
        messages = build_messages(
            row=row,
            chunk_index=chunk_index,
            system_prompt=starter.system_prompt,
            normalize_dialog_turns=starter.normalize_dialog_turns,
            max_history_turns=config.max_history_turns,
        )
        raw_response = model.generate(
            model_frames,
            messages,
            max_new_tokens=config.max_new_tokens,
        )
        answer, normalization = canonicalize_response(raw_response)
        answers.append(answer)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "interval": [interval[0], interval[1]],
                "current_interval_frames": len(current_frames),
                "model_input_frames": len(model_frames),
                "raw_response": raw_response,
                "answer": answer,
                "normalization": normalization,
            }
        )

    return {
        "input_index": input_index,
        "video_path": video_name,
        "prediction": {"video_path": video_name, "answers": answers},
        "chunks": chunks,
    }


def validate_prediction_rows(
    source_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
) -> dict[str, int]:
    if len(source_rows) != len(prediction_rows):
        raise ValueError(
            f"Prediction rows ({len(prediction_rows)}) do not match source rows "
            f"({len(source_rows)})"
        )

    chunks = 0
    interrupts = 0
    for row_index, (source, prediction) in enumerate(
        zip(source_rows, prediction_rows)
    ):
        if prediction.get("video_path") != source.get("video_path"):
            raise ValueError(f"row {row_index}: video_path or source order mismatch")
        intervals = source["video_intervals"]
        answers = prediction.get("answers")
        if not isinstance(answers, list) or len(answers) != len(intervals):  # type: ignore[arg-type]
            raise ValueError(f"row {row_index}: answer count mismatch")
        for chunk_index, answer in enumerate(answers):
            if not isinstance(answer, str):
                raise ValueError(
                    f"row {row_index} chunk {chunk_index}: answer must be text"
                )
            if answer == SILENT_TAG:
                pass
            elif answer.startswith(INTERRUPT_TAG) and answer != INTERRUPT_TAG:
                interrupts += 1
            else:
                raise ValueError(
                    f"row {row_index} chunk {chunk_index}: malformed answer {answer!r}"
                )
            chunks += 1
    return {"sessions": len(prediction_rows), "chunks": chunks, "interrupts": interrupts}


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    temporary.replace(path)
