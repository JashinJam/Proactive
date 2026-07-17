"""Create timestamped contact sheets for causal oracle-state annotation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def make_contact_sheet(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    frames: int,
    columns: int,
    cell_width: int,
) -> None:
    import cv2
    from PIL import Image, ImageDraw

    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError("Contact-sheet range must satisfy 0 <= start < end")
    if frames <= 0 or columns <= 0 or cell_width <= 0:
        raise ValueError("Contact-sheet dimensions must be positive")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    timestamps = [
        start_sec + (end_sec - start_sec) * (index + 0.5) / frames
        for index in range(frames)
    ]
    images: list[Image.Image] = []
    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"Cannot read {video_path} at {timestamp:.3f}s")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            height = max(1, round(image.height * cell_width / image.width))
            image = image.resize((cell_width, height), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(image)
            label = f"{timestamp:.1f}s"
            draw.rectangle((0, 0, 76, 20), fill="black")
            draw.text((4, 3), label, fill="white")
            images.append(image)
    finally:
        capture.release()
    cell_height = max(image.height for image in images)
    rows = math.ceil(len(images) / columns)
    margin = 4
    canvas = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * margin,
            rows * cell_height + (rows + 1) * margin,
        ),
        "white",
    )
    for index, image in enumerate(images):
        x = margin + (index % columns) * (cell_width + margin)
        y = margin + (index // columns) * (cell_height + margin)
        canvas.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=320)
    args = parser.parse_args()
    make_contact_sheet(
        Path(args.video).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        args.start,
        args.end,
        args.frames,
        args.columns,
        args.cell_width,
    )


if __name__ == "__main__":
    main()
