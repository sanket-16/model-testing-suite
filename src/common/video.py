"""Video iteration — every frame sequentially (no stride sampling)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def iter_video_frames(
    video_path: Path,
    max_duration_sec: float | None = None,
) -> Iterator[tuple[int, np.ndarray, float]]:
    """Yield (frame_index, BGR frame, fps) for each frame in order."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    max_frame = None
    if max_duration_sec is not None:
        max_frame = int(max_duration_sec * fps)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frame is not None and idx >= max_frame:
            break
        yield idx, frame, fps
        idx += 1

    cap.release()
    if idx == 0:
        raise RuntimeError(f"No frames read from {video_path}")


def video_metadata(video_path: Path) -> tuple[int, int, float]:
    """Return (width, height, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    cap.release()
    return w, h, fps
