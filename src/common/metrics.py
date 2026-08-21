"""Detection quality metrics: precision, recall, AP / mAP at fixed IoU."""

from __future__ import annotations

from dataclasses import dataclass

from src.common.types import Detection, box_iou
from src.common.visdrone import GtBox


@dataclass
class ClassMetrics:
    class_id: int
    class_name: str
    ap: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    n_gt: int
    n_pred: int


def average_precision(recalls: list[float], precisions: list[float]) -> float:
    """All-point interpolated AP (COCO-style area under PR curve)."""
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] > mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def compute_class_metrics(
    frame_preds: dict[int, list[Detection]],
    frame_gts: dict[int, list[GtBox]],
    class_id: int,
    class_name: str,
    iou_threshold: float,
) -> ClassMetrics:
    """Compute AP / P / R for one class across frames.

    Predictions should already be confidence-filtered (as in suite JSON).
    Operating-point P/R use all preds at that threshold; AP ranks by confidence.
    """
    scored: list[tuple[float, int, Detection]] = []  # conf, frame, det
    n_gt = 0
    for frame_idx, gts in frame_gts.items():
        class_gts = [g for g in gts if g.class_id == class_id]
        n_gt += len(class_gts)
        preds = [d for d in frame_preds.get(frame_idx, []) if d.class_id == class_id]
        for d in preds:
            scored.append((d.confidence, frame_idx, d))

    # Frames with preds but no GT entry still count as FP candidates
    for frame_idx, preds_all in frame_preds.items():
        if frame_idx in frame_gts:
            continue
        for d in preds_all:
            if d.class_id == class_id:
                scored.append((d.confidence, frame_idx, d))

    scored.sort(key=lambda t: t[0], reverse=True)
    n_pred = len(scored)

    if n_gt == 0 and n_pred == 0:
        return ClassMetrics(
            class_id=class_id,
            class_name=class_name,
            ap=0.0,
            precision=0.0,
            recall=0.0,
            tp=0,
            fp=0,
            fn=0,
            n_gt=0,
            n_pred=0,
        )

    # Per-frame GT bookkeeping for greedy matching
    remaining: dict[int, list[GtBox]] = {
        fi: [g for g in gts if g.class_id == class_id] for fi, gts in frame_gts.items()
    }
    used_per_frame: dict[int, set[int]] = {fi: set() for fi in remaining}

    tp_flags: list[bool] = []
    for _conf, frame_idx, pred in scored:
        gts = remaining.get(frame_idx, [])
        used = used_per_frame.setdefault(frame_idx, set())
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gts):
            if j in used:
                continue
            iou = box_iou(pred, gt.as_detection())
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            used.add(best_j)
            tp_flags.append(True)
        else:
            tp_flags.append(False)

    tp_cum = 0
    fp_cum = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for is_tp in tp_flags:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        recalls.append(tp_cum / n_gt if n_gt else 0.0)
        precisions.append(tp_cum / (tp_cum + fp_cum))

    ap = average_precision(recalls, precisions) if n_gt else 0.0
    tp = tp_cum
    fp = fp_cum
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_gt if n_gt else 0.0

    return ClassMetrics(
        class_id=class_id,
        class_name=class_name,
        ap=ap,
        precision=precision,
        recall=recall,
        tp=tp,
        fp=fp,
        fn=fn,
        n_gt=n_gt,
        n_pred=n_pred,
    )


def mean_ap(class_metrics: list[ClassMetrics], *, only_with_gt: bool = True) -> float:
    scores = [m.ap for m in class_metrics if (m.n_gt > 0 if only_with_gt else True)]
    return sum(scores) / len(scores) if scores else 0.0
