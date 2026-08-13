"""Export / fetch ONNX models for browser (and ONNX-backed server) inference."""

from __future__ import annotations

import sys
from pathlib import Path

import requests
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.types import load_baseline_config  # noqa: E402


def _copy_to_public(onnx_path: Path) -> Path:
    public_path = ROOT / "public" / "models" / onnx_path.name
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_bytes(onnx_path.read_bytes())
    return public_path


def export_ultralytics(model_key: str) -> Path:
    cfg = load_baseline_config()
    model_cfg = cfg["models"][model_key]
    weights = ROOT / model_cfg["weights"]
    onnx_path = ROOT / model_cfg["onnx"]
    imgsz = cfg["defaults"]["imgsz"]

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if model_key == "yolov8":
        model = YOLO(str(weights))
    elif model_key.startswith("rt_detr"):
        model = RTDETR(str(weights))
    else:
        raise ValueError(f"Unknown ultralytics model key: {model_key}")

    exported = model.export(format="onnx", imgsz=imgsz, simplify=True)
    exported_path = Path(exported)
    if exported_path.resolve() != onnx_path.resolve():
        exported_path.replace(onnx_path)

    public_path = _copy_to_public(onnx_path)
    print(f"Exported {model_key} -> {onnx_path} (also {public_path})")
    return onnx_path


def download_onnx(model_key: str) -> Path:
    cfg = load_baseline_config()
    model_cfg = cfg["models"][model_key]
    onnx_path = ROOT / model_cfg["onnx"]
    url = model_cfg["onnx_url"]
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if onnx_path.exists():
        print(f"Already exists: {onnx_path}")
        _copy_to_public(onnx_path)
        return onnx_path

    print(f"Downloading {url} -> {onnx_path}")
    with requests.get(url, timeout=300, stream=True) as resp:
        resp.raise_for_status()
        onnx_path.write_bytes(resp.content)

    public_path = _copy_to_public(onnx_path)
    print(f"Downloaded {model_key} -> {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB, also {public_path})")
    return onnx_path


def export_model(model_key: str) -> Path:
    cfg = load_baseline_config()
    model_cfg = cfg["models"][model_key]
    backend = model_cfg.get("backend", "ultralytics")
    if backend == "onnx":
        return download_onnx(model_key)
    return export_ultralytics(model_key)


def main() -> None:
    cfg = load_baseline_config()
    for key in cfg["models"]:
        export_model(key)


if __name__ == "__main__":
    main()
