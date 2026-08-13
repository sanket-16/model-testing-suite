# TAS Model Test Suite

Baseline suite comparing **server** (CPU / GPU) vs **browser** (ONNX Runtime Web) object detection on **VisDrone-VID UAV video**.

Models:
- **YOLOv8n** (Ultralytics)
- **RT-DETR-L** (Ultralytics)
- **RT-DETR-S** (R18vd COCO ONNX — Ultralytics has no pretrained S weights)

Fixtures are consecutive VisDrone-VID frames encoded to H.264 MP4 (browser-playable). Every frame in the first `max_duration_sec` is processed (no stride).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python scripts/download_fixtures.py
python scripts/export_onnx.py

npm install
npx playwright install chromium
```

### GPU (Windows)

Default pip torch is CPU-only. For CUDA:

```bash
pip uninstall -y torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install onnxruntime-gpu
```

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Note: RT-DETR-S uses ONNX Runtime. If `onnxruntime-gpu` cannot load the CUDA EP (e.g. CUDA 13 mismatch), it falls back to CPU.

## Run tests

```bash
# Server CPU + GPU (GPU skipped if CUDA unavailable)
pytest tests/server -v -s

# Browser (ONNX in Chromium)
$env:PLAYWRIGHT_BROWSERS_PATH = "$env:LOCALAPPDATA\ms-playwright"
npx playwright test

# Cross-runtime comparison
pytest tests/test_parity.py -v -s

# Print latency summary
python scripts/compare_results.py --summary
```

## Outputs

| Path | Contents |
|------|----------|
| `results/server_cpu/` | Per-run JSON |
| `results/server_gpu/` | Per-run JSON |
| `results/browser/` | Per-run JSON |
| `results/comparisons/` | CPU/GPU/browser reports |
| `results/output/server_cpu/` | Annotated `.mp4` |
| `results/output/server_gpu/` | Annotated `.mp4` |
| `results/output/browser/` | Annotated `.webm` |

## Config

`config/baseline.yaml`:

| Setting | Default | Notes |
|---------|---------|-------|
| `max_duration_sec` | `5` | First N seconds of each clip |
| `conf_threshold` | `0.25` | |
| `iou_threshold` | `0.45` | |
| `max_latency_ms` | per runtime | Per-model budgets under `models.*.max_latency_ms` |

Set `max_duration_sec: null` to process entire clips.
