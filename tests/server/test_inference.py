"""Server-side video inference tests for CPU and GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.types import get_max_latency_ms, load_baseline_config, model_keys
from src.server.inference import cuda_available, output_video_path, run_server_inference

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def baseline() -> dict:
    return load_baseline_config()


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures(baseline: dict) -> None:
    for rel in baseline["fixtures"]["videos"]:
        path = ROOT / rel
        if not path.exists():
            pytest.skip(f"Video fixture missing: {path}. Run: python scripts/download_fixtures.py")


def _devices() -> list[str]:
    devices = ["cpu"]
    if cuda_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("model_key", model_keys())
def test_server_video_inference(model_key: str, device: str, baseline: dict) -> None:
    runtime = "server_gpu" if device == "cuda" else "server_cpu"
    if device == "cuda" and not cuda_available():
        pytest.skip("CUDA not available")

    defaults = baseline["defaults"]
    warmup = defaults.get("warmup_runs", 0)
    max_latency = get_max_latency_ms(baseline, model_key, runtime)

    results_dir = ROOT / "results" / runtime
    results_dir.mkdir(parents=True, exist_ok=True)

    for rel in baseline["fixtures"]["videos"]:
        video_path = ROOT / rel
        result = run_server_inference(
            model_key, video_path, device=device, warmup_runs=warmup
        )

        assert result.runtime == runtime
        assert result.frames, f"No frames processed for {model_key}/{runtime} on {rel}"
        assert any(len(f.detections) > 0 for f in result.frames), (
            f"No detections in any frame for {model_key}/{runtime} on {rel}"
        )
        assert result.latency_ms < max_latency, (
            f"Mean latency too high ({runtime}): {result.latency_ms:.1f}ms"
        )

        out_video = output_video_path(model_key, video_path, runtime)
        assert out_video.exists(), f"Annotated video not written: {out_video}"

        json_path = results_dir / f"{model_key}_{video_path.stem}.json"
        json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

        print(f"\n  [{runtime}] Annotated video: {out_video}")
        print(f"  [{runtime}] Results JSON:    {json_path}")
        print(f"  [{runtime}] Mean latency:    {result.latency_ms:.1f} ms/frame")
