"""Compare CPU vs GPU vs browser inference results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.types import (  # noqa: E402
    Detection,
    FrameResult,
    InferenceResult,
    load_baseline_config,
    match_detections,
)


def load_result(path: Path) -> InferenceResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = [
        FrameResult(
            frame_index=f["frame_index"],
            latency_ms=f["latency_ms"],
            detections=[Detection(**d) for d in f["detections"]],
        )
        for f in data.get("frames", [])
    ]
    return InferenceResult(
        model=data["model"],
        runtime=data["runtime"],
        source=data.get("source", data.get("image", "")),
        frames=frames,
        latency_ms=data["latency_ms"],
        imgsz=data.get("imgsz", 640),
        output_video=data.get("output_video"),
    )


def compare_pair(a: InferenceResult, b: InferenceResult, match_iou: float) -> dict:
    by_idx = {f.frame_index: f for f in b.frames}
    matched = server_only = browser_only = 0
    frames_compared = 0
    for af in a.frames:
        bf = by_idx.get(af.frame_index)
        if bf is None:
            continue
        frames_compared += 1
        m, so, bo = match_detections(af.detections, bf.detections, match_iou)
        matched += m
        server_only += so
        browser_only += bo
    return {
        "left_runtime": a.runtime,
        "right_runtime": b.runtime,
        "left_latency_ms": round(a.latency_ms, 2),
        "right_latency_ms": round(b.latency_ms, 2),
        "speedup_left_over_right": round(b.latency_ms / a.latency_ms, 2) if a.latency_ms else None,
        "frames_compared": frames_compared,
        "matched": matched,
        "left_only": server_only,
        "right_only": browser_only,
    }


def compare_three(
    cpu: InferenceResult | None,
    gpu: InferenceResult | None,
    browser: InferenceResult | None,
) -> dict:
    cfg = load_baseline_config()
    match_iou = cfg["defaults"]["match_iou"]
    model = (cpu or gpu or browser).model  # type: ignore[union-attr]
    source = (cpu or gpu or browser).source  # type: ignore[union-attr]

    report: dict = {
        "model": model,
        "source": source,
        "latencies_ms": {},
        "output_videos": {},
        "pairs": {},
    }
    if cpu:
        report["latencies_ms"]["server_cpu"] = round(cpu.latency_ms, 2)
        report["output_videos"]["server_cpu"] = cpu.output_video
    if gpu:
        report["latencies_ms"]["server_gpu"] = round(gpu.latency_ms, 2)
        report["output_videos"]["server_gpu"] = gpu.output_video
    if browser:
        report["latencies_ms"]["browser"] = round(browser.latency_ms, 2)
        report["output_videos"]["browser"] = browser.output_video

    if cpu and gpu:
        report["pairs"]["cpu_vs_gpu"] = compare_pair(cpu, gpu, match_iou)
    if cpu and browser:
        report["pairs"]["cpu_vs_browser"] = compare_pair(cpu, browser, match_iou)
    if gpu and browser:
        report["pairs"]["gpu_vs_browser"] = compare_pair(gpu, browser, match_iou)

    return report


def print_summary() -> None:
    """Print mean latency table from results/comparisons/*.json."""
    comp = ROOT / "results" / "comparisons"
    rows: list[tuple[str, str, float | None, float | None, float | None]] = []
    for path in sorted(comp.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        lat = data.get("latencies_ms", {})
        src = data.get("source", path.stem)
        stem = Path(src).stem if src else path.stem
        rows.append(
            (
                data["model"],
                stem,
                lat.get("server_cpu"),
                lat.get("server_gpu"),
                lat.get("browser"),
            )
        )

    if not rows:
        raise SystemExit(f"No comparison JSONs in {comp}. Run tests first.")

    print("MODEL      VIDEO         CPU_ms   GPU_ms   BROWSER_ms")
    print("-" * 60)
    by: dict[str, list] = defaultdict(list)
    for model, vid, cpu, gpu, br in rows:
        by[model].append((cpu, gpu, br))
        cpu_s = f"{cpu:.1f}" if cpu is not None else "-"
        gpu_s = f"{gpu:.1f}" if gpu is not None else "-"
        br_s = f"{br:.1f}" if br is not None else "-"
        print(f"{model:10} {vid:12} {cpu_s:>8} {gpu_s:>8} {br_s:>11}")

    print("\nAverages (ms/frame):")
    for model, vals in by.items():
        cpus = [v[0] for v in vals if v[0] is not None]
        gpus = [v[1] for v in vals if v[1] is not None]
        brs = [v[2] for v in vals if v[2] is not None]
        parts = [f"  {model}:"]
        if cpus:
            parts.append(f"CPU={sum(cpus) / len(cpus):.1f}")
        if gpus:
            parts.append(f"GPU={sum(gpus) / len(gpus):.1f}")
        if brs:
            parts.append(f"Browser={sum(brs) / len(brs):.1f}")
        print(" ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CPU / GPU / browser results")
    parser.add_argument("--summary", action="store_true", help="Print table from results/comparisons")
    parser.add_argument("--cpu", type=Path, default=None)
    parser.add_argument("--gpu", type=Path, default=None)
    parser.add_argument("--browser", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    cpu = load_result(args.cpu) if args.cpu else None
    gpu = load_result(args.gpu) if args.gpu else None
    browser = load_result(args.browser) if args.browser else None
    if not any([cpu, gpu, browser]):
        raise SystemExit("Provide --summary or at least one of --cpu/--gpu/--browser")

    result = compare_three(cpu, gpu, browser)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
