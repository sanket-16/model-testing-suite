import { defineConfig, devices } from "@playwright/test";
import { platform } from "os";

// WebGPU is unavailable in Playwright's default headless Chromium on this host.
// Use headed system Chrome so navigator.gpu / adapters work for browser_webgpu.
const useHeadedChrome = process.env.BROWSER_HEADLESS !== "1";

// On Windows, Chrome WebGPU uses D3D12. Forcing Vulkan has caused hangs on
// some dual-GPU LEGION laptops (iGPU + NVIDIA).
const launchArgs =
  platform() === "win32"
    ? [
        "--enable-unsafe-webgpu",
        "--enable-webgpu-developer-features",
        "--ignore-gpu-blocklist",
        "--use-angle=d3d11",
        "--enable-features=WebGPUService",
      ]
    : [
        "--enable-unsafe-webgpu",
        "--enable-webgpu-developer-features",
        "--ignore-gpu-blocklist",
        "--enable-features=Vulkan,WebGPUService,UseSkiaRenderer",
      ];

export default defineConfig({
  testDir: "tests/browser",
  fullyParallel: false,
  retries: 0,
  reporter: "list",
  timeout: 1_200_000,
  use: {
    baseURL: "http://localhost:5173",
    trace: "on-first-retry",
    headless: !useHeadedChrome,
    ...(useHeadedChrome ? { channel: "chrome" as const } : {}),
    launchOptions: {
      args: launchArgs,
    },
  },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
