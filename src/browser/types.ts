export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface FrameResult {
  frame_index: number;
  latency_ms: number;
  detections: Detection[];
}

export type BrowserExecutionProvider = "wasm" | "webgpu";
export type BrowserRuntime = "browser" | "browser_webgpu";

export interface VideoInferencePayload {
  model: string;
  runtime: BrowserRuntime;
  execution_provider: BrowserExecutionProvider;
  source: string;
  latency_ms: number;
  imgsz: number;
  frames: FrameResult[];
  output_video_base64: string | null;
}

export interface HarnessResult {
  ok: boolean;
  error?: string;
  result?: VideoInferencePayload;
}

export type ModelKey = "yolov8" | "yolov10" | "rt_detr_l" | "rt_detr_s";

export function runtimeForProvider(ep: BrowserExecutionProvider): BrowserRuntime {
  return ep === "webgpu" ? "browser_webgpu" : "browser";
}

declare global {
  interface Window {
    runBrowserVideoInference: (opts: {
      modelKey: ModelKey;
      videoUrl: string;
      imgsz: number;
      confThreshold: number;
      iouThreshold: number;
      maxDurationSec: number | null;
      executionProvider?: BrowserExecutionProvider;
    }) => Promise<HarnessResult>;
  }
}

export {};
