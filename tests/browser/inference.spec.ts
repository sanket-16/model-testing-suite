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
const { imgsz, conf_threshold, iou_threshold, max_duration_sec } = baseline.defaults;
const defaultMaxLatency = baseline.defaults.max_latency_ms;
const outputDir = resolve(ROOT, baseline.output.dir, baseline.output.browser_subdir);

function getMaxLatencyMs(modelKey: string): number {
  const modelCfg = baseline.models[modelKey];
  const limit = modelCfg?.max_latency_ms ?? defaultMaxLatency;
  if (typeof limit === "object" && limit !== null) {
    return limit.browser ?? defaultMaxLatency;
  }
  return limit;
}

for (const modelKey of MODELS) {
  for (const videoRel of baseline.fixtures.videos) {
    const videoName = videoRel.split("/").pop()!;
    const stem = videoName.replace(/\.[^.]+$/, "");

    test(`${modelKey} browser video inference on ${videoName}`, async ({ page }) => {
      test.setTimeout(600_000);

      const onnxPath = resolve(ROOT, baseline.models[modelKey].onnx);
      try {
        readFileSync(onnxPath);
      } catch {
        test.skip(true, `ONNX model missing: ${onnxPath}. Run: npm run export:onnx`);
      }

      await page.goto("/");
      await page.waitForFunction(() => typeof window.runBrowserVideoInference === "function");

      const videoUrl = `/fixtures/${videoName}`;
      const result = await page.evaluate(
        async ({ modelKey, videoUrl, imgsz, conf_threshold, iou_threshold, max_duration_sec }) => {
          return window.runBrowserVideoInference({
            modelKey,
            videoUrl,
            imgsz,
            confThreshold: conf_threshold,
            iouThreshold: iou_threshold,
            maxDurationSec: max_duration_sec ?? null,
          });
        },
        { modelKey, videoUrl, imgsz, conf_threshold, iou_threshold, max_duration_sec },
      );

      expect(result.ok, result.error).toBe(true);
      expect(result.result!.frames.length).toBeGreaterThan(0);
      expect(result.result!.latency_ms).toBeLessThan(getMaxLatencyMs(modelKey));
      expect(result.result!.frames.some((f) => f.detections.length > 0)).toBe(true);

      mkdirSync(outputDir, { recursive: true });
      const outVideo = resolve(outputDir, `${modelKey}_${stem}.webm`);
      const outJson = resolve(ROOT, "results/browser", `${modelKey}_${stem}.json`);

      if (result.result!.output_video_base64) {
        writeFileSync(outVideo, Buffer.from(result.result!.output_video_base64, "base64"));
      }

      const { output_video_base64: _, ...jsonPayload } = result.result!;
      jsonPayload.output_video = outVideo;
      mkdirSync(dirname(outJson), { recursive: true });
      writeFileSync(outJson, JSON.stringify(jsonPayload, null, 2));

      console.log(`\n  Annotated video: ${outVideo}`);
      console.log(`  Results JSON:    ${outJson}`);
    });
  }
}
