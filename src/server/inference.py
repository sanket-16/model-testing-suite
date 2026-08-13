"""Server-side video inference with CPU or GPU device selection."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import RTDETR, YOLO

from src.common.annotate import draw_detections
from src.common.types import Detection, FrameResult, InferenceResult, ROOT, load_baseline_config
from src.common.video import iter_video_frames, video_metadata

DeviceName = Literal["cpu", "cuda"]

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def resolve_device(device: DeviceName | str) -> DeviceName:
    if device == "cuda":
        if not cuda_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch (see README)."
            )
        return "cuda"
    return "cpu"


def runtime_label(device: DeviceName) -> str:
    return "server_gpu" if device == "cuda" else "server_cpu"


class ServerModel(ABC):
    model_key: str
    device: DeviceName

    @abstractmethod
    def predict_frame(
        self, frame_bgr: np.ndarray, imgsz: int, conf: float, iou: float
    ) -> list[Detection]:
        pass


class YoloV8Model(ServerModel):
    model_key = "yolov8"

    def __init__(self, weights: str | Path, device: DeviceName) -> None:
        self.device = device
        self.model = YOLO(str(weights))
        # Move weights to target device once
        self.model.to("cuda:0" if device == "cuda" else "cpu")

    def predict_frame(
        self, frame_bgr: np.ndarray, imgsz: int, conf: float, iou: float
    ) -> list[Detection]:
        return _ultralytics_to_detections(
            self.model.predict(
                source=frame_bgr,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=0 if self.device == "cuda" else "cpu",
                verbose=False,
            )
        )


class RtDetrUltralyticsModel(ServerModel):
    def __init__(self, model_key: str, weights: str | Path, device: DeviceName) -> None:
        self.model_key = model_key
        self.device = device
        self.model = RTDETR(str(weights))
        self.model.to("cuda:0" if device == "cuda" else "cpu")

    def predict_frame(
        self, frame_bgr: np.ndarray, imgsz: int, conf: float, iou: float
    ) -> list[Detection]:
        return _ultralytics_to_detections(
            self.model.predict(
                source=frame_bgr,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=0 if self.device == "cuda" else "cpu",
                verbose=False,
            )
        )


class RtDetrOnnxModel(ServerModel):
    def __init__(self, model_key: str, onnx_path: str | Path, device: DeviceName) -> None:
        self.model_key = model_key
        self.device = device
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        available = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDA ONNX Runtime requested but CUDAExecutionProvider is unavailable. "
                "Install onnxruntime-gpu."
            )
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict_frame(
        self, frame_bgr: np.ndarray, imgsz: int, conf: float, iou: float
    ) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        tensor, scale, pad_x, pad_y = letterbox_bgr(frame_bgr, imgsz)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return decode_rtdetr_hf_outputs(outputs, self.output_names, conf, scale, pad_x, pad_y, w, h)


def _ultralytics_to_detections(results) -> list[Detection]:
    detections: list[Detection] = []
    for r in results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=names[cls_id],
                    confidence=float(box.conf[0]),
                    x1=xyxy[0],
                    y1=xyxy[1],
                    x2=xyxy[2],
                    y2=xyxy[3],
                )
            )
    return detections


def letterbox_bgr(
    frame_bgr: np.ndarray, size: int
) -> tuple[np.ndarray, float, float, float]:
    h, w = frame_bgr.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    x0, y0 = int(round(pad_x)), int(round(pad_y))
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
    return tensor, scale, pad_x, pad_y


def decode_rtdetr_hf_outputs(
    outputs: list,
    output_names: list[str],
    conf_threshold: float,
    scale: float,
    pad_x: float,
    pad_y: float,
    orig_w: int,
    orig_h: int,
) -> list[Detection]:
    by_name = {name: arr for name, arr in zip(output_names, outputs)}
    logits = boxes = None
    for name, arr in by_name.items():
        lower = name.lower()
        if "logit" in lower:
            logits = arr
        elif "box" in lower:
            boxes = arr
    if logits is None or boxes is None:
        for arr in outputs:
            if arr.ndim == 3 and arr.shape[-1] == 4:
                boxes = arr
            elif arr.ndim == 3 and arr.shape[-1] > 4:
                logits = arr
    if logits is None or boxes is None:
        raise RuntimeError(f"Unexpected RT-DETR ONNX outputs: {[a.shape for a in outputs]}")

    logits = logits[0]
    boxes = boxes[0]
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    class_ids = probs.argmax(axis=-1)
    scores = probs.max(axis=-1)

    detections: list[Detection] = []
    for i in range(len(scores)):
        score = float(scores[i])
        if score < conf_threshold:
            continue
        cls_id = int(class_ids[i])
        if cls_id >= len(COCO_NAMES):
            continue
        cx, cy, bw, bh = boxes[i].tolist()
        x1 = (cx - bw / 2) * 640
        y1 = (cy - bh / 2) * 640
        x2 = (cx + bw / 2) * 640
        y2 = (cy + bh / 2) * 640
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        detections.append(
            Detection(
                class_id=cls_id,
                class_name=COCO_NAMES[cls_id],
                confidence=score,
                x1=max(0.0, x1),
                y1=max(0.0, y1),
                x2=min(float(orig_w), x2),
                y2=min(float(orig_h), y2),
            )
        )
    return detections


def load_server_model(model_key: str, device: DeviceName = "cpu") -> ServerModel:
    device = resolve_device(device)
    cfg = load_baseline_config()
    model_cfg = cfg["models"][model_key]
    backend = model_cfg.get("backend", "ultralytics")

    if backend == "onnx":
        onnx_path = ROOT / model_cfg["onnx"]
        return RtDetrOnnxModel(model_key, onnx_path, device)

    weights = ROOT / model_cfg["weights"]
    if model_key == "yolov8":
        return YoloV8Model(weights, device)
    if model_key.startswith("rt_detr"):
        return RtDetrUltralyticsModel(model_key, weights, device)
    raise ValueError(f"Unknown model key: {model_key}")


def output_video_path(model_key: str, video_path: Path, runtime: str) -> Path:
    cfg = load_baseline_config()
    out_cfg = cfg["output"]
    key = f"{runtime}_subdir"
    sub = out_cfg.get(key, runtime)
    ext = "webm" if runtime == "browser" else "mp4"
    return ROOT / out_cfg["dir"] / sub / f"{model_key}_{video_path.stem}.{ext}"


def run_server_inference(
    model_key: str,
    video_path: Path,
    *,
    device: DeviceName | str = "cpu",
    warmup_runs: int = 0,
    write_output: bool = True,
) -> InferenceResult:
    device = resolve_device(device)
    runtime = runtime_label(device)
    cfg = load_baseline_config()
    defaults = cfg["defaults"]
    model = load_server_model(model_key, device)

    imgsz = defaults["imgsz"]
    conf = defaults["conf_threshold"]
    iou = defaults["iou_threshold"]
    max_duration = defaults.get("max_duration_sec")

    width, height, fps = video_metadata(video_path)
    writer: cv2.VideoWriter | None = None
    out_path: Path | None = None

    if write_output:
        out_path = output_video_path(model_key, video_path, runtime)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_iter = list(iter_video_frames(video_path, max_duration))
    if warmup_runs > 0 and frame_iter:
        for _ in range(warmup_runs):
            model.predict_frame(frame_iter[0][1], imgsz, conf, iou)
        if device == "cuda":
            torch.cuda.synchronize()

    frame_results: list[FrameResult] = []
    for frame_index, frame_bgr, _fps in frame_iter:
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        detections = model.predict_frame(frame_bgr, imgsz, conf, iou)
        if device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000
        frame_results.append(
            FrameResult(frame_index=frame_index, latency_ms=latency_ms, detections=detections)
        )
        if writer is not None:
            annotated = draw_detections(frame_bgr, detections)
            writer.write(annotated)

    if writer is not None:
        writer.release()

    mean_latency = (
        sum(f.latency_ms for f in frame_results) / len(frame_results) if frame_results else 0.0
    )
    return InferenceResult(
        model=model_key,
        runtime=runtime,
        source=str(video_path),
        frames=frame_results,
        latency_ms=mean_latency,
        imgsz=imgsz,
        output_video=str(out_path) if out_path else None,
    )
