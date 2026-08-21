"""VisDrone-VID ground truth loading and COCO class mapping."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.common.types import Detection

# VisDrone object_category → name (category 0 = ignored regions)
VISDRONE_NAMES: dict[int, str] = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

# Map VisDrone categories onto COCO-80 ids used by YOLO / RT-DETR.
# Unlisted VisDrone categories (tricycle, awning-tricycle, others) are skipped.
VISDRONE_TO_COCO: dict[int, int] = {
    1: 0,  # pedestrian → person
    2: 0,  # people → person
    3: 1,  # bicycle
    4: 2,  # car
    5: 2,  # van → car (closest COCO class for aerial)
    6: 7,  # truck
    9: 5,  # bus
    10: 3,  # motor → motorcycle
}

COCO_EVAL_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

EVAL_CLASS_IDS: tuple[int, ...] = tuple(sorted(COCO_EVAL_NAMES))

# Fixture MP4 stem → VisDrone sequence folder (must match download_fixtures.SEQUENCES)
FIXTURE_SEQUENCES: dict[str, str] = {
    "uav_seq_a": "uav0000086_00000_v",
    "uav_seq_b": "uav0000117_02622_v",
    "uav_seq_c": "uav0000137_00458_v",
    "uav_seq_d": "uav0000182_00000_v",
}


@dataclass(frozen=True)
class GtBox:
    class_id: int  # COCO id
    x1: float
    y1: float
    x2: float
    y2: float

    def as_detection(self) -> Detection:
        return Detection(
            class_id=self.class_id,
            class_name=COCO_EVAL_NAMES.get(self.class_id, f"class_{self.class_id}"),
            confidence=1.0,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
        )


def parse_visdrone_annotation(
    text: str,
    *,
    max_frame_index_0based: int | None = None,
) -> dict[int, list[GtBox]]:
    """Parse a VisDrone-VID .txt annotation file.

    VisDrone frame indices are 1-based and match ``0000001.jpg``.
    Returned keys are **0-based** frame indices to match suite JSON results.
    """
    by_frame: dict[int, list[GtBox]] = defaultdict(list)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        frame_1based = int(parts[0])
        left, top, width, height = map(float, parts[2:6])
        score = int(float(parts[6]))
        category = int(parts[7])
        if score == 0 or category == 0:
            continue  # ignored / not considered
        coco_id = VISDRONE_TO_COCO.get(category)
        if coco_id is None:
            continue
        frame_0 = frame_1based - 1
        if max_frame_index_0based is not None and frame_0 > max_frame_index_0based:
            continue
        if width <= 0 or height <= 0:
            continue
        by_frame[frame_0].append(
            GtBox(
                class_id=coco_id,
                x1=left,
                y1=top,
                x2=left + width,
                y2=top + height,
            )
        )
    return dict(by_frame)


def load_sequence_gt(
    annotation_path: Path,
    *,
    max_frames: int | None = None,
) -> dict[int, list[GtBox]]:
    text = annotation_path.read_text(encoding="utf-8")
    max_idx = (max_frames - 1) if max_frames is not None else None
    return parse_visdrone_annotation(text, max_frame_index_0based=max_idx)


def filter_preds_for_eval(detections: list[Detection]) -> list[Detection]:
    """Keep only predictions whose class is in the VisDrone↔COCO eval set."""
    return [d for d in detections if d.class_id in COCO_EVAL_NAMES]
