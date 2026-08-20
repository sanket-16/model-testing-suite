"""Shared types and utilities for the test suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "baseline.yaml"


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameResult:
    frame_index: int
    latency_ms: float
    detections: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "latency_ms": self.latency_ms,
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass
class InferenceResult:
    model: str
    runtime: str  # "server_cpu" | "server_gpu" | "browser"
    source: str
    frames: list[FrameResult] = field(default_factory=list)
    latency_ms: float = 0.0  # mean per-frame latency
    imgsz: int = 640
    output_video: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "runtime": self.runtime,
            "source": self.source,
            "latency_ms": self.latency_ms,
            "imgsz": self.imgsz,
            "output_video": self.output_video,
            "frames": [f.to_dict() for f in self.frames],
        }


def load_baseline_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_keys(config: dict[str, Any] | None = None) -> list[str]:
    """Return model keys to benchmark.

    If ``benchmark_models`` is set in config, only those keys are returned
    (in that order). Otherwise every key under ``models`` is used.
    """
    cfg = config or load_baseline_config()
    all_keys = list(cfg["models"].keys())
    selected = cfg.get("benchmark_models")
    if not selected:
        return all_keys
    unknown = [k for k in selected if k not in cfg["models"]]
    if unknown:
        raise ValueError(f"Unknown benchmark_models entries: {unknown}")
    return list(selected)


def get_max_latency_ms(config: dict[str, Any], model_key: str, runtime: str) -> float:
    """Return per-model latency budget, falling back to defaults."""
    default = config["defaults"]["max_latency_ms"]
    model_cfg = config["models"].get(model_key, {})
    limit = model_cfg.get("max_latency_ms", default)
    if isinstance(limit, dict):
        return limit.get(runtime, default)
    return limit


def box_iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    server: list[Detection],
    browser: list[Detection],
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Return (matched, server_only, browser_only) detection counts."""
    matched = 0
    used_browser: set[int] = set()
    for s in server:
        best_iou = 0.0
        best_j = -1
        for j, b in enumerate(browser):
            if j in used_browser or s.class_id != b.class_id:
                continue
            iou = box_iou(s, b)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            matched += 1
            used_browser.add(best_j)
    server_only = len(server) - matched
    browser_only = len(browser) - matched
    return matched, server_only, browser_only
