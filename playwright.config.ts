import { defineConfig, devices } from "@playwright/test";

// WebGPU is unavailable in Playwright's default headless Chromium on this host.
// Use headed system Chrome so navigator.gpu / adapters work for browser_webgpu.
const useHeadedChrome = process.env.BROWSER_HEADLESS !== "1";

export default defineConfig({
  testDir: "tests/browser",
  fullyParallel: false,
  retries: 0,
  reporter: "list",
  timeout: 900_000,
  use: {
    baseURL: "http://localhost:5173",
    trace: "on-first-retry",
    headless: !useHeadedChrome,
    ...(useHeadedChrome ? { channel: "chrome" as const } : {}),
    launchOptions: {
      args: [
        "--enable-unsafe-webgpu",
        "--enable-webgpu-developer-features",
        "--ignore-gpu-blocklist",
        "--enable-features=Vulkan,WebGPUService,UseSkiaRenderer",
      ],
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
