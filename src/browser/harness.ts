// webgpu entry includes JSEP wasm needed for WebGPU; wasm EP still works.
import * as ort from "onnxruntime-web/webgpu";
import type {
  BrowserExecutionProvider,
  Detection,
  FrameResult,
  HarnessResult,
  ModelKey,
} from "./types";
import { runtimeForProvider } from "./types";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "/ort/";

const MODEL_PATHS: Record<ModelKey, string> = {
  yolov8: "/models/yolov8n.onnx",
  yolov10: "/models/yolov10n.onnx",
  rt_detr_l: "/models/rtdetr-l.onnx",
  rt_detr_s: "/models/rtdetr-s.onnx",
};

const HF_MODELS = new Set<ModelKey>(["rt_detr_s"]);

const COCO_NAMES: Record<number, string> = {
  0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
  5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
};

function getClassName(id: number): string {
  return COCO_NAMES[id] ?? `class_${id}`;
}

const COLORS = ["#ff3838", "#38ff38", "#3838ff", "#ffff38", "#ff38ff", "#38ffff"];

const sessions = new Map<string, ort.InferenceSession>();

async function ensureWebGpuAvailable(): Promise<void> {
  if (!("gpu" in navigator) || !navigator.gpu) {
    throw new Error("WebGPU is not available in this browser");
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    throw new Error("No WebGPU adapter found (GPU blocked or unsupported)");
  }
}

async function getSession(
  modelKey: ModelKey,
  executionProvider: BrowserExecutionProvider,
): Promise<ort.InferenceSession> {
  const cacheKey = `${modelKey}:${executionProvider}`;
  if (sessions.has(cacheKey)) return sessions.get(cacheKey)!;

  if (executionProvider === "webgpu") {
    await ensureWebGpuAvailable();
  }

  const session = await ort.InferenceSession.create(MODEL_PATHS[modelKey], {
    executionProviders: [executionProvider],
  });
  sessions.set(cacheKey, session);
  return session;
}

function letterboxCanvas(
  source: CanvasImageSource,
  sw: number,
  sh: number,
  size: number,
): { tensor: Float32Array; scale: number; padX: number; padY: number } {
  const canvas = document.getElementById("infer-canvas") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;
  canvas.width = size;
  canvas.height = size;

  const scale = Math.min(size / sw, size / sh);
  const newW = Math.round(sw * scale);
  const newH = Math.round(sh * scale);
  const padX = (size - newW) / 2;
  const padY = (size - newH) / 2;

  ctx.fillStyle = "#727272";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(source, padX, padY, newW, newH);

  const { data } = ctx.getImageData(0, 0, size, size);
  const tensor = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    tensor[i] = data[i * 4] / 255;
    tensor[size * size + i] = data[i * 4 + 1] / 255;
    tensor[2 * size * size + i] = data[i * 4 + 2] / 255;
  }
  return { tensor, scale, padX, padY };
}

function nms(boxes: Detection[], iouThreshold: number): Detection[] {
  const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
  const kept: Detection[] = [];
  const suppressed = new Set<number>();
  for (let i = 0; i < sorted.length; i++) {
    if (suppressed.has(i)) continue;
    kept.push(sorted[i]);
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed.has(j)) continue;
      const a = sorted[i];
      const b = sorted[j];
      const x1 = Math.max(a.x1, b.x1);
      const y1 = Math.max(a.y1, b.y1);
      const x2 = Math.min(a.x2, b.x2);
      const y2 = Math.min(a.y2, b.y2);
      const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
      const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
      const iou = inter / (areaA + areaB - inter);
      if (iou >= iouThreshold) suppressed.add(j);
    }
  }
  return kept;
}

function decodeUltralytics(
  output: ort.Tensor,
  confThreshold: number,
  scale: number,
  padX: number,
  padY: number,
  origW: number,
  origH: number,
): Detection[] {
  const data = output.data as Float32Array;
  const dims = output.dims;
  const detections: Detection[] = [];

  if (dims.length === 3 && dims[1] === 84) {
    const numAnchors = dims[2];
    for (let i = 0; i < numAnchors; i++) {
      let bestConf = 0;
      let bestCls = 0;
      for (let c = 0; c < 80; c++) {
        const conf = data[(4 + c) * numAnchors + i];
        if (conf > bestConf) {
          bestConf = conf;
          bestCls = c;
        }
      }
      if (bestConf < confThreshold) continue;
      const cx = data[0 * numAnchors + i];
      const cy = data[1 * numAnchors + i];
      const w = data[2 * numAnchors + i];
      const h = data[3 * numAnchors + i];
      detections.push({
        class_id: bestCls,
        class_name: getClassName(bestCls),
        confidence: bestConf,
        x1: Math.max(0, (cx - w / 2 - padX) / scale),
        y1: Math.max(0, (cy - h / 2 - padY) / scale),
        x2: Math.min(origW, (cx + w / 2 - padX) / scale),
        y2: Math.min(origH, (cy + h / 2 - padY) / scale),
      });
    }
  } else if (dims.length === 3 && dims[2] === 6) {
    const numDet = dims[1];
    for (let i = 0; i < numDet; i++) {
      const base = i * 6;
      const conf = data[base + 4];
      if (conf < confThreshold) continue;
      const cls = Math.round(data[base + 5]);
      detections.push({
        class_id: cls,
        class_name: getClassName(cls),
        confidence: conf,
        x1: Math.max(0, (data[base] - padX) / scale),
        y1: Math.max(0, (data[base + 1] - padY) / scale),
        x2: Math.min(origW, (data[base + 2] - padX) / scale),
        y2: Math.min(origH, (data[base + 3] - padY) / scale),
      });
    }
  }
  return detections;
}

function softmax2d(logits: Float32Array, rows: number, cols: number): Float32Array {
  const probs = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    let max = -Infinity;
    for (let c = 0; c < cols; c++) max = Math.max(max, logits[r * cols + c]);
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      const e = Math.exp(logits[r * cols + c] - max);
      probs[r * cols + c] = e;
      sum += e;
    }
    for (let c = 0; c < cols; c++) probs[r * cols + c] /= sum;
  }
  return probs;
}

function decodeHfRtDetr(
  outputs: ort.InferenceSession.OnnxValueMapType,
  outputNames: string[],
  confThreshold: number,
  scale: number,
  padX: number,
  padY: number,
  origW: number,
  origH: number,
  imgsz: number,
): Detection[] {
  let logits: ort.Tensor | undefined;
  let boxes: ort.Tensor | undefined;
  for (const name of outputNames) {
    const t = outputs[name] as ort.Tensor;
    const lower = name.toLowerCase();
    if (lower.includes("logit")) logits = t;
    else if (lower.includes("box")) boxes = t;
  }
  if (!logits || !boxes) {
    for (const name of outputNames) {
      const t = outputs[name] as ort.Tensor;
      if (t.dims.length === 3 && t.dims[2] === 4) boxes = t;
      else if (t.dims.length === 3 && t.dims[2]! > 4) logits = t;
    }
  }
  if (!logits || !boxes) return [];

  const n = logits.dims[1]!;
  const classes = logits.dims[2]!;
  const probs = softmax2d(logits.data as Float32Array, n, classes);
  const boxData = boxes.data as Float32Array;
  const detections: Detection[] = [];

  for (let i = 0; i < n; i++) {
    let bestCls = 0;
    let bestConf = 0;
    for (let c = 0; c < classes; c++) {
      const p = probs[i * classes + c];
      if (p > bestConf) {
        bestConf = p;
        bestCls = c;
      }
    }
    if (bestConf < confThreshold || bestCls >= 80) continue;
    const cx = boxData[i * 4];
    const cy = boxData[i * 4 + 1];
    const bw = boxData[i * 4 + 2];
    const bh = boxData[i * 4 + 3];
    detections.push({
      class_id: bestCls,
      class_name: getClassName(bestCls),
      confidence: bestConf,
      x1: Math.max(0, ((cx - bw / 2) * imgsz - padX) / scale),
      y1: Math.max(0, ((cy - bh / 2) * imgsz - padY) / scale),
      x2: Math.min(origW, ((cx + bw / 2) * imgsz - padX) / scale),
      y2: Math.min(origH, ((cy + bh / 2) * imgsz - padY) / scale),
    });
  }
  return detections;
}

function drawDetectionsOnCanvas(
  ctx: CanvasRenderingContext2D,
  detections: Detection[],
) {
  for (const det of detections) {
    const color = COLORS[det.class_id % COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
    const label = `${det.class_name} ${det.confidence.toFixed(2)}`;
    ctx.font = "14px system-ui";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(det.x1, Math.max(0, det.y1 - 18), tw + 6, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, det.x1 + 3, Math.max(12, det.y1 - 5));
  }
}

async function loadVideo(url: string): Promise<HTMLVideoElement> {
  const video = document.getElementById("video") as HTMLVideoElement;
  return new Promise((resolve, reject) => {
    video.onloadedmetadata = () => resolve(video);
    video.onerror = () => reject(new Error(`Failed to load video: ${url}`));
    video.src = url;
    video.load();
  });
}

function seekVideo(video: HTMLVideoElement, timeSec: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      video.removeEventListener("seeked", onSeeked);
      reject(new Error("Video seek timeout"));
    }, 5000);
    const onSeeked = () => {
      clearTimeout(timeout);
      video.removeEventListener("seeked", onSeeked);
      resolve();
    };
    video.addEventListener("seeked", onSeeked);
    video.currentTime = timeSec;
  });
}

async function blobToBase64(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

async function runBrowserVideoInference(opts: {
  modelKey: ModelKey;
  videoUrl: string;
  imgsz: number;
  confThreshold: number;
  iouThreshold: number;
  maxDurationSec: number | null;
  executionProvider?: BrowserExecutionProvider;
}): Promise<HarnessResult> {
  const status = document.getElementById("status")!;
  const executionProvider: BrowserExecutionProvider = opts.executionProvider ?? "wasm";
  try {
    status.textContent = `Loading ${opts.modelKey} (${executionProvider})...`;
    const session = await getSession(opts.modelKey, executionProvider);
    const video = await loadVideo(opts.videoUrl);
    const fps = 30; // fixture metadata; stepped by frame index
    const maxFrames =
      opts.maxDurationSec != null
        ? Math.ceil(opts.maxDurationSec * fps)
        : Math.ceil((video.duration || 5) * fps);

    const display = document.getElementById("canvas") as HTMLCanvasElement;
    const dctx = display.getContext("2d")!;
    display.width = video.videoWidth;
    display.height = video.videoHeight;

    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : "video/webm";
    const stream = display.captureStream(fps);
    const recorder = new MediaRecorder(stream, { mimeType });
    const chunks: Blob[] = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };
    const recorderDone = new Promise<Blob>((resolve) => {
      recorder.onstop = () => resolve(new Blob(chunks, { type: mimeType }));
    });
    recorder.start(100);

    const frameResults: FrameResult[] = [];
    const isHf = HF_MODELS.has(opts.modelKey);

    for (let frameIndex = 0; frameIndex < maxFrames; frameIndex++) {
      const t = frameIndex / fps;
      if (t > (video.duration || 0)) break;

      await seekVideo(video, t);
      dctx.drawImage(video, 0, 0, display.width, display.height);

      const { tensor, scale, padX, padY } = letterboxCanvas(
        video,
        video.videoWidth,
        video.videoHeight,
        opts.imgsz,
      );

      const inputName = session.inputNames[0];
      const feeds: Record<string, ort.Tensor> = {
        [inputName]: new ort.Tensor("float32", tensor, [1, 3, opts.imgsz, opts.imgsz]),
      };

      const start = performance.now();
      const outputs = await session.run(feeds);
      const latency_ms = performance.now() - start;

      let raw: Detection[];
      if (isHf) {
        raw = decodeHfRtDetr(
          outputs,
          session.outputNames,
          opts.confThreshold,
          scale,
          padX,
          padY,
          video.videoWidth,
          video.videoHeight,
          opts.imgsz,
        );
      } else {
        const outputName = session.outputNames[0];
        raw = decodeUltralytics(
          outputs[outputName] as ort.Tensor,
          opts.confThreshold,
          scale,
          padX,
          padY,
          video.videoWidth,
          video.videoHeight,
        );
      }
      const detections = nms(raw, opts.iouThreshold);

      dctx.drawImage(video, 0, 0, display.width, display.height);
      drawDetectionsOnCanvas(dctx, detections);

      frameResults.push({ frame_index: frameIndex, latency_ms, detections });
      status.textContent =
        `Frame ${frameIndex + 1}/${maxFrames} [${executionProvider}] — ` +
        `${detections.length} dets, ${latency_ms.toFixed(0)}ms`;
    }

    recorder.stop();
    const webmBlob = await recorderDone;
    const output_video_base64 = await blobToBase64(webmBlob);

    const meanLatency =
      frameResults.reduce((s, f) => s + f.latency_ms, 0) / (frameResults.length || 1);

    status.textContent =
      `Done [${executionProvider}]: ${frameResults.length} frames, ` +
      `avg ${meanLatency.toFixed(0)}ms/frame`;

    return {
      ok: true,
      result: {
        model: opts.modelKey,
        runtime: runtimeForProvider(executionProvider),
        execution_provider: executionProvider,
        source: opts.videoUrl,
        latency_ms: meanLatency,
        imgsz: opts.imgsz,
        frames: frameResults,
        output_video_base64,
      },
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    status.textContent = `Error: ${message}`;
    return { ok: false, error: message };
  }
}

window.runBrowserVideoInference = runBrowserVideoInference;

export { runBrowserVideoInference };
