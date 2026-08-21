import { chromium } from "@playwright/test";
import { spawn } from "child_process";

const vite = spawn("npm", ["run", "dev"], { shell: true, stdio: "pipe" });
await new Promise((r) => setTimeout(r, 4000));

const browser = await chromium.launch({
  headless: false,
  channel: "chrome",
  args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-webgpu-developer-features"],
});
const page = await browser.newPage();
await page.goto("http://localhost:5173/");
const info = await page.evaluate(async () => {
  const hasGpu = !!navigator.gpu;
  let adapter = false;
  if (hasGpu) {
    const a = await navigator.gpu.requestAdapter();
    adapter = !!a;
  }
  return { hasGpu, adapter };
});
console.log(JSON.stringify(info));
await browser.close();
vite.kill();
