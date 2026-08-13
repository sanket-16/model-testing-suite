import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  root: "browser",
  publicDir: resolve(__dirname, "public"),
  resolve: {
    alias: { "@": resolve(__dirname, "src/browser") },
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  worker: {
    format: "es",
  },
  server: {
    port: 5173,
    fs: { allow: [resolve(__dirname)] },
  },
  build: { outDir: "../dist/browser" },
});
