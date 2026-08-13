"""Cross-runtime comparison: server CPU vs GPU vs browser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_results import compare_three, load_result
from src.common.types import load_baseline_config, model_keys

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def baseline() -> dict:
    return load_baseline_config()


@pytest.mark.parametrize("model_key", model_keys())
def test_runtime_comparison(model_key: str, baseline: dict) -> None:
    for rel in baseline["fixtures"]["videos"]:
        stem = Path(rel).stem
        cpu_path = ROOT / "results" / "server_cpu" / f"{model_key}_{stem}.json"
        gpu_path = ROOT / "results" / "server_gpu" / f"{model_key}_{stem}.json"
        browser_path = ROOT / "results" / "browser" / f"{model_key}_{stem}.json"

        cpu = load_result(cpu_path) if cpu_path.exists() else None
        gpu = load_result(gpu_path) if gpu_path.exists() else None
        browser = load_result(browser_path) if browser_path.exists() else None

        if cpu is None and gpu is None:
            pytest.skip(
                f"Need at least server_cpu or server_gpu results for {model_key}/{stem}"
            )

        report = compare_three(cpu, gpu, browser)
        out = ROOT / "results" / "comparisons" / f"{model_key}_{stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"\n  Comparison: {out}")
        print(f"  Latencies:  {report['latencies_ms']}")

        # Latency sanity: GPU should beat CPU when both exist
        if cpu and gpu:
            assert gpu.latency_ms < cpu.latency_ms * 1.1, (
                f"GPU not faster than CPU for {model_key}/{stem}: "
                f"gpu={gpu.latency_ms:.1f} cpu={cpu.latency_ms:.1f}"
            )
