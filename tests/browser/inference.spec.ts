import { test, expect } from "@playwright/test";
import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";
import { parse as parseYaml } from "yaml";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const baseline = parseYaml(readFileSync(resolve(ROOT, "config/baseline.yaml"), "utf-8"));

const MODELS: string[] =
  Array.isArray(baseline.benchmark_models) && baseline.benchmark_models.length > 0
    ? baseline.benchmark_models
    : Object.keys(baseline.models);

type ExecutionProvider = "wasm" | "webgpu";

const PROVIDERS: ExecutionProvider[] = (
  Array.isArray(baseline.browser_execution_providers) &&
  baseline.browser_execution_providers.length > 0
    ? baseline.browser_execution_providers
    : ["wasm"]
).filter((p: string): p is ExecutionProvider => p === "wasm" || p === "webgpu");

const { imgsz, conf_threshold, iou_threshold, max_duration_sec } = baseline.defaults;
const defaultMaxLatency = baseline.defaults.max_latency_ms;

function runtimeKey(ep: ExecutionProvider): "browser" | "browser_webgpu" {
  return ep === "webgpu" ? "browser_webgpu" : "browser";
}

function resultsSubdir(ep: ExecutionProvider): string {
  if (ep === "webgpu") {
    return baseline.output.browser_webgpu_subdir ?? "browser_webgpu";
  }
  return baseline.output.browser_subdir ?? "browser";
}

function getMaxLatencyMs(modelKey: string, ep: ExecutionProvider): number {
  const modelCfg = baseline.models[modelKey];
  const limit = modelCfg?.max_latency_ms ?? defaultMaxLatency;
  if (typeof limit === "object" && limit !== null) {
    const runtime = runtimeKey(ep);
    return limit[runtime] ?? limit.browser ?? defaultMaxLatency;
  }
  return limit;
}

for (const executionProvider of PROVIDERS) {
  for (const modelKey of MODELS) {
    for (const videoRel of baseline.fixtures.videos) {
      const videoName = videoRel.split("/").pop()!;
      const stem = videoName.replace(/\.[^.]+$/, "");
      const runtime = runtimeKey(executionProvider);
      const outputDir = resolve(ROOT, baseline.output.dir, resultsSubdir(executionProvider));
      const resultsDir = resolve(ROOT, "results", resultsSubdir(executionProvider));

      test(`${modelKey} ${executionProvider} video inference on ${videoName}`, async ({
        page,
      }) => {
        // WASM RT-DETR-L is ~3s/frame; 5s clips need well over the old 8–10m guards.
        const testTimeoutMs = executionProvider === "webgpu" ? 600_000 : 1_200_000;
        test.setTimeout(testTimeoutMs);

        const onnxPath = resolve(ROOT, baseline.models[modelKey].onnx);
        try {
          readFileSync(onnxPath);
        } catch {
          test.skip(true, `ONNX model missing: ${onnxPath}. Run: npm run export:onnx`);
        }

        await page.goto("/");
        await page.waitForFunction(() => typeof window.runBrowserVideoInference === "function");

        if (executionProvider === "webgpu") {
          const webgpuInfo = await page.evaluate(async () => {
            if (!("gpu" in navigator) || !navigator.gpu) {
              return { ok: false, detail: "navigator.gpu missing" };
            }
            const adapter = await navigator.gpu.requestAdapter({
              powerPreference: "high-performance",
            });
            if (!adapter) {
              return { ok: false, detail: "requestAdapter() returned null" };
            }
            const info =
              "info" in adapter && adapter.info
                ? // @ts-expect-error GPUAdapter.info is still settling in TS DOM libs
                  `${adapter.info.vendor || "?"} / ${adapter.info.architecture || "?"} / ${adapter.info.description || "?"}`
                : "adapter ok (no info API)";
            return { ok: true, detail: info };
          });
          console.log(`  [webgpu] adapter: ${webgpuInfo.detail}`);
          test.skip(!webgpuInfo.ok, `WebGPU unavailable: ${webgpuInfo.detail}`);
        }

        const videoUrl = `/fixtures/${videoName}`;
        // Fail faster than the test timeout if ORT/WebGPU hangs on create/run.
        const evaluateTimeoutMs = executionProvider === "webgpu" ? 240_000 : 900_000;
        const result = await Promise.race([
          page.evaluate(
            async ({
              modelKey,
              videoUrl,
              imgsz,
              conf_threshold,
              iou_threshold,
              max_duration_sec,
              executionProvider,
            }) => {
              return window.runBrowserVideoInference({
                modelKey,
                videoUrl,
                imgsz,
                confThreshold: conf_threshold,
                iouThreshold: iou_threshold,
                maxDurationSec: max_duration_sec ?? null,
                executionProvider,
              });
            },
            {
              modelKey,
              videoUrl,
              imgsz,
              conf_threshold,
              iou_threshold,
              max_duration_sec,
              executionProvider,
            },
          ),
          new Promise<never>((_, reject) => {
            setTimeout(() => {
              reject(
                new Error(
                  `Browser ${executionProvider} inference hung for ${modelKey}/${videoName} ` +
                    `(>${evaluateTimeoutMs / 1000}s). Often WebGPU session create on dual-GPU laptops.`,
                ),
              );
            }, evaluateTimeoutMs);
          }),
        ]);

        expect(result.ok, result.error).toBe(true);
        expect(result.result!.frames.length).toBeGreaterThan(0);
        expect(result.result!.execution_provider).toBe(executionProvider);
        expect(result.result!.runtime).toBe(runtime);
        expect(result.result!.latency_ms).toBeLessThan(getMaxLatencyMs(modelKey, executionProvider));
        expect(result.result!.frames.some((f) => f.detections.length > 0)).toBe(true);

        mkdirSync(outputDir, { recursive: true });
        const outVideo = resolve(outputDir, `${modelKey}_${stem}.webm`);
        const outJson = resolve(resultsDir, `${modelKey}_${stem}.json`);

        if (result.result!.output_video_base64) {
          writeFileSync(outVideo, Buffer.from(result.result!.output_video_base64, "base64"));
        }

        const { output_video_base64: _, ...jsonPayload } = result.result!;
        jsonPayload.output_video = outVideo;
        mkdirSync(dirname(outJson), { recursive: true });
        writeFileSync(outJson, JSON.stringify(jsonPayload, null, 2));

        console.log(`\n  [${executionProvider}] Annotated video: ${outVideo}`);
        console.log(`  [${executionProvider}] Results JSON:    ${outJson}`);
        console.log(
          `  [${executionProvider}] Mean latency:    ${result.result!.latency_ms.toFixed(1)} ms/frame`,
        );
      });
    }
  }
}
