"""Prepare label-free, current-chunk-only annotation packets for the R1 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proactive_r0.core import load_jsonl, write_jsonl

from .state import load_json, text_sha256, validate_and_select_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "annotations" / "r1_oracle_pilot_v1" / "manifest.json"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "egoproactive" / "wearable_ai_2026_egoproactive_val_700.jsonl"
DEFAULT_VIDEO_FOLDER = PROJECT_ROOT / "data" / "egoproactive" / "val"


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sanitized_chunk_contexts(
    selected: list[tuple[int, dict[str, object]]]
) -> list[dict[str, object]]:
    """Expose exactly dialog[i] and interval i, never answers or future dialog."""
    contexts: list[dict[str, object]] = []
    for input_index, row in selected:
        intervals = row["video_intervals"]
        dialogs = row["dialog"]
        assert isinstance(intervals, list) and isinstance(dialogs, list)
        for chunk_index, (interval, dialog_at_chunk) in enumerate(zip(intervals, dialogs)):
            contexts.append(
                {
                    "input_index": input_index,
                    "video_path": row["video_path"],
                    "domain": row["domain"],
                    "task": row["task"],
                    "query": row["query"],
                    "chunk_index": chunk_index,
                    "interval": interval,
                    "dialog_at_chunk": dialog_at_chunk,
                }
            )
    return contexts


def annotation_template(
    selected: list[tuple[int, dict[str, object]]]
) -> list[dict[str, object]]:
    templates: list[dict[str, object]] = []
    for input_index, row in selected:
        states = [
            {
                "chunk_index": chunk_index,
                "observed_through_sec": interval[1],
                "current_step_id": "TODO",
                "progress": "TODO",
                "completion_evidence": [],
                "incompletion_or_error_evidence": [],
                "next_step_id": None,
                "confidence": 0.0,
                "last_update_chunk": chunk_index,
            }
            for chunk_index, interval in enumerate(row["video_intervals"])  # type: ignore[index]
        ]
        templates.append(
            {
                "schema_version": 1,
                "status": "incomplete",
                "input_index": input_index,
                "video_path": row["video_path"],
                "query_sha256": text_sha256(str(row["query"])),
                "provenance": {
                    "plan_inputs": ["task", "query"],
                    "chunk_inputs": [
                        "task",
                        "query",
                        "dialog_at_chunk",
                        "video_through_interval_end",
                    ],
                    "excluded_inputs": ["answers", "future_dialog", "future_video"],
                    "annotation_type": "evaluation_only_oracle_non_deployable",
                },
                "goal": row["query"],
                "steps": [],
                "chunk_states": states,
            }
        )
    return templates


def _sample_interval_frames(
    video_path: Path, start: float, end: float, count: int
) -> list[tuple[float, object]]:
    import cv2
    import numpy as np
    from PIL import Image

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    timestamps = np.linspace(start, end, count + 2, dtype=float)[1:-1]
    frames: list[tuple[float, object]] = []
    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode {video_path} at {timestamp:.3f}s")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((float(timestamp), Image.fromarray(rgb)))
    finally:
        capture.release()
    return frames


def _write_contact_image(
    destination: Path,
    video_path: Path,
    input_index: int,
    chunk_index: int,
    interval: list[object],
    frames_per_chunk: int,
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    start, end = float(interval[0]), float(interval[1])
    sampled = _sample_interval_frames(video_path, start, end, frames_per_chunk)
    thumb_width, thumb_height = 448, 252
    margin, header = 12, 44
    canvas = Image.new(
        "RGB",
        (frames_per_chunk * thumb_width + (frames_per_chunk + 1) * margin, thumb_height + header + 2 * margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, margin),
        f"source={input_index} chunk={chunk_index} visible interval=[{start:.1f}, {end:.1f}]",
        fill="black",
    )
    for column, (timestamp, image) in enumerate(sampled):
        thumb = ImageOps.fit(image, (thumb_width, thumb_height))
        x = margin + column * (thumb_width + margin)
        y = margin + header
        canvas.paste(thumb, (x, y))
        draw.rectangle((x, y, x + 92, y + 22), fill="black")
        draw.text((x + 4, y + 4), f"t={timestamp:.2f}s", fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--video-folder", default=str(DEFAULT_VIDEO_FOLDER))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames-per-chunk", type=int, default=4)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args(argv)
    if args.frames_per_chunk <= 0:
        parser.error("--frames-per-chunk must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = _resolve(args.manifest)
    input_path = _resolve(args.input)
    video_folder = _resolve(args.video_folder)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(input_path)
    selected = validate_and_select_manifest(load_json(manifest_path), rows)
    contexts = sanitized_chunk_contexts(selected)
    assert all("answers" not in context for context in contexts)
    write_jsonl(output_dir / "annotation_context.jsonl", contexts)
    write_jsonl(output_dir / "states_template.jsonl", annotation_template(selected))

    if not args.no_images:
        for context in contexts:
            input_index = int(context["input_index"])
            chunk_index = int(context["chunk_index"])
            _write_contact_image(
                output_dir / "frames" / str(input_index) / f"chunk_{chunk_index:03d}.jpg",
                video_folder / str(context["video_path"]),
                input_index,
                chunk_index,
                context["interval"],  # type: ignore[arg-type]
                args.frames_per_chunk,
            )

    summary = {
        "pilot_sessions": len(selected),
        "pilot_chunks": len(contexts),
        "labels_exported": False,
        "future_dialog_exported_per_context": False,
        "frames_per_current_interval": 0 if args.no_images else args.frames_per_chunk,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
