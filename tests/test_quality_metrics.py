"""Unit tests for VisDrone GT parsing and mAP metrics."""

from __future__ import annotations

from src.common.metrics import average_precision, compute_class_metrics, mean_ap
from src.common.types import Detection
from src.common.visdrone import GtBox, parse_visdrone_annotation


def test_parse_visdrone_maps_person_and_frame_index() -> None:
    text = "\n".join(
        [
            "1,0,10,20,30,40,1,1,0,0",  # pedestrian → person, frame 0
            "1,1,50,60,10,10,1,2,0,0",  # people → person
            "2,2,1,1,5,5,1,4,0,0",  # car, frame 1
            "2,3,1,1,5,5,0,4,0,0",  # score=0 ignored
            "2,4,1,1,5,5,1,0,0,0",  # category 0 ignored
            "2,5,1,1,5,5,1,7,0,0",  # tricycle unmapped
        ]
    )
    gt = parse_visdrone_annotation(text)
    assert set(gt) == {0, 1}
    assert len(gt[0]) == 2
    assert all(b.class_id == 0 for b in gt[0])
    assert gt[0][0].x1 == 10 and gt[0][0].y2 == 60
    assert len(gt[1]) == 1 and gt[1][0].class_id == 2


def test_average_precision_perfect() -> None:
    assert average_precision([0.5, 1.0], [1.0, 1.0]) == 1.0


def test_compute_class_metrics_perfect_match() -> None:
    gts = {0: [GtBox(class_id=0, x1=0, y1=0, x2=10, y2=10)]}
    preds = {
        0: [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.9,
                x1=0,
                y1=0,
                x2=10,
                y2=10,
            )
        ]
    }
    m = compute_class_metrics(preds, gts, 0, "person", 0.5)
    assert m.tp == 1 and m.fp == 0 and m.fn == 0
    assert m.precision == 1.0 and m.recall == 1.0
    assert m.ap == 1.0
    assert mean_ap([m]) == 1.0


def test_compute_class_metrics_false_positive() -> None:
    gts = {0: [GtBox(class_id=0, x1=0, y1=0, x2=10, y2=10)]}
    preds = {
        0: [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.9,
                x1=100,
                y1=100,
                x2=110,
                y2=110,
            )
        ]
    }
    m = compute_class_metrics(preds, gts, 0, "person", 0.5)
    assert m.tp == 0 and m.fp == 1 and m.fn == 1
    assert m.precision == 0.0 and m.recall == 0.0
