"""Evaluate saved detections against VisDrone-VID ground truth.

Usage:
  python scripts/download_fixtures.py          # extracts annotations too
  python scripts/evaluate_visdrone.py
  python scripts/evaluate_visdrone.py --runtime server_cpu --results-dir results
  python scripts/evaluate_visdrone.py --results-dir "json files"
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_results import load_result  # noqa: E402
from scripts.download_fixtures import ZIP_PATH, download_vid_zip  # noqa: E402
from src.common.metrics import ClassMetrics, compute_class_metrics, mean_ap  # noqa: E402
from src.common.types import Detection, InferenceResult, load_baseline_config, model_keys  # noqa: E402
from src.common.visdrone import (  # noqa: E402
    COCO_EVAL_NAMES,
    EVAL_CLASS_IDS,
    FIXTURE_SEQUENCES,
    filter_preds_for_eval,
    load_sequence_gt,
)

ANNOTATIONS_DIR = ROOT / "fixtures" / "annotations"
EXTRACT_DIR = ROOT / "fixtures" / "cache" / "VisDrone2019-VID-val"


def ensure_annotations() -> dict[str, Path]:
    """Return fixture_stem → annotation .txt path, extracting from zip if needed."""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    missing: list[tuple[str, str]] = []

    for stem, seq in FIXTURE_SEQUENCES.items():
        dest = ANNOTATIONS_DIR / f"{seq}.txt"
        if not dest.exists():
            missing.append((stem, seq))
        paths[stem] = dest

    if not missing:
        return paths

    if not ZIP_PATH.exists():
        download_vid_zip()

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for stem, seq in missing:
            member = f"VisDrone2019-VID-val/annotations/{seq}.txt"
            try:
                data = zf.read(member)
            except KeyError as exc:
                raise FileNotFoundError(
                    f"No VisDrone annotation for sequence {seq} ({stem}). "
                    f"Update FIXTURE_SEQUENCES / SEQUENCES to match the zip."
                ) from exc
            dest = ANNOTATIONS_DIR / f"{seq}.txt"
            dest.write_bytes(data)
            print(f"Extracted {dest.relative_to(ROOT)}")
            # Keep EXTRACT_DIR copy in sync with download_fixtures layout
            alt = EXTRACT_DIR / "annotations" / f"{seq}.txt"
            alt.parent.mkdir(parents=True, exist_ok=True)
            if not alt.exists():
                alt.write_bytes(data)
            paths[stem] = dest

    return paths


def result_path(results_dir: Path, runtime: str, model_key: str, stem: str) -> Path:
    return results_dir / runtime / f"{model_key}_{stem}.json"


def preds_by_frame(result: InferenceResult) -> dict[int, list[Detection]]:
    return {
        f.frame_index: filter_preds_for_eval(f.detections)
        for f in result.frames
    }


def evaluate_result(
    result: InferenceResult,
    frame_gts: dict,
    iou_threshold: float,
) -> dict:
    preds = preds_by_frame(result)
    # Restrict GT to frames present in the result
    max_frame = max((f.frame_index for f in result.frames), default=-1)
    gts = {i: boxes for i, boxes in frame_gts.items() if i <= max_frame}

    per_class: list[ClassMetrics] = []
    for class_id in EVAL_CLASS_IDS:
        per_class.append(
            compute_class_metrics(
                preds,
                gts,
                class_id,
                COCO_EVAL_NAMES[class_id],
                iou_threshold,
            )
        )

    with_gt = [m for m in per_class if m.n_gt > 0]
    tp = sum(m.tp for m in with_gt)
    fp = sum(m.fp for m in with_gt)
    fn = sum(m.fn for m in with_gt)
    return {
        "model": result.model,
        "runtime": result.runtime,
        "source": result.source,
        "frames": len(result.frames),
        "iou_threshold": iou_threshold,
        "map50": round(mean_ap(per_class, only_with_gt=True), 4),
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_class": [
            {
                "class_id": m.class_id,
                "class_name": m.class_name,
                "ap50": round(m.ap, 4),
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "tp": m.tp,
                "fp": m.fp,
                "fn": m.fn,
                "n_gt": m.n_gt,
                "n_pred": m.n_pred,
            }
            for m in per_class
        ],
    }


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No results found to evaluate.")
        return
    headers = ("model", "runtime", "seq", "mAP@50", "P", "R", "TP", "FP", "FN")
    table = []
    for r in rows:
        stem = Path(r["source"]).stem if r.get("source") else "?"
        table.append(
            (
                r["model"],
                r["runtime"],
                stem,
                f"{r['map50']:.3f}",
                f"{r['precision']:.3f}",
                f"{r['recall']:.3f}",
                str(r["tp"]),
                str(r["fp"]),
                str(r["fn"]),
            )
        )
    widths = [max(len(h), max(len(row[i]) for row in table)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in table:
        print(fmt.format(*row))

    # Mean across sequences per model+runtime
    from collections import defaultdict

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["runtime"])].append(r["map50"])
    print("\nMean mAP@50 across sequences:")
    for (model, runtime), maps in sorted(groups.items()):
        mean = sum(maps) / len(maps)
        print(f"  {model:12} {runtime:16} {mean:.3f}  (n={len(maps)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Root containing <runtime>/<model>_<seq>.json",
    )
    parser.add_argument(
        "--runtime",
        action="append",
        dest="runtimes",
        help="Runtime subdir to score (repeatable). Default: all present.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model key to score (repeatable). Default: benchmark_models.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="Match IoU (default: defaults.match_iou from baseline.yaml)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write full JSON report (default: <results-dir>/quality/visdrone_map.json)",
    )
    args = parser.parse_args()

    cfg = load_baseline_config()
    iou = args.iou if args.iou is not None else float(cfg["defaults"]["match_iou"])
    models = args.models or model_keys(cfg)
    max_frames = None
    max_dur = cfg["defaults"].get("max_duration_sec")
    if max_dur is not None:
        max_frames = int(float(max_dur) * 30)  # fixtures encoded at 30 fps

    ann_paths = ensure_annotations()
    gt_cache: dict[str, dict] = {}
    for stem, path in ann_paths.items():
        gt_cache[stem] = load_sequence_gt(path, max_frames=max_frames)

    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir

    runtimes = args.runtimes
    if not runtimes:
        runtimes = [
            p.name
            for p in results_dir.iterdir()
            if p.is_dir() and p.name not in ("output", "comparisons", "quality")
        ]

    rows: list[dict] = []
    for runtime in sorted(runtimes):
        for model_key in models:
            for stem in FIXTURE_SEQUENCES:
                path = result_path(results_dir, runtime, model_key, stem)
                if not path.exists():
                    continue
                result = load_result(path)
                report = evaluate_result(result, gt_cache[stem], iou)
                report["model_key"] = model_key
                report["sequence"] = stem
                report["result_path"] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
                rows.append(report)

    print_table(rows)

    out = args.out
    if out is None:
        out = results_dir / "quality" / "visdrone_map.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iou_threshold": iou,
        "max_frames": max_frames,
        "class_mapping": {
            "visdrone_to_coco": {
                "1_pedestrian": 0,
                "2_people": 0,
                "3_bicycle": 1,
                "4_car": 2,
                "5_van": 2,
                "6_truck": 7,
                "9_bus": 5,
                "10_motor": 3,
            },
            "eval_classes": {str(k): v for k, v in COCO_EVAL_NAMES.items()},
        },
        "results": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
