#!/usr/bin/env python3
"""Extract the three-frame recovery sequence used by the presentation."""

from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO = PROJECT_ROOT / "data/egoproactive/val/359ed9ce38fdf4dc.mp4"
OUTPUT = Path(__file__).resolve().parent / "assets/session143_recovery_sequence.jpg"
TIMESTAMPS = (84.0, 88.0, 96.0)


def main() -> None:
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO}")

    frames = []
    for timestamp in TIMESTAMPS:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode {VIDEO} at {timestamp:.1f}s")
        frame = cv2.resize(frame, (360, 479), interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (0, 0), (102, 43), (20, 25, 28), thickness=-1)
        cv2.putText(
            frame,
            f"{timestamp:.0f} s",
            (13, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(frame)
    capture.release()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    montage = cv2.hconcat(frames)
    if not cv2.imwrite(str(OUTPUT), montage, [cv2.IMWRITE_JPEG_QUALITY, 91]):
        raise RuntimeError(f"Cannot write {OUTPUT}")
    print(OUTPUT)


if __name__ == "__main__":
    main()

