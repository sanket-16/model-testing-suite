"""Draw detections on video frames."""

from __future__ import annotations

import cv2
import numpy as np

from src.common.types import Detection

_COLORS = [
    (56, 56, 255),
    (56, 255, 56),
    (255, 56, 56),
    (255, 255, 56),
    (255, 56, 255),
    (56, 255, 255),
]


def draw_detections(frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame_bgr.copy()
    for det in detections:
        color = _COLORS[det.class_id % len(_COLORS)]
        x1, y1 = int(det.x1), int(det.y1)
        x2, y2 = int(det.x2), int(det.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out
