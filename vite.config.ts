import { defineConfig } from "vite";
import { createReadStream, existsSync, statSync } from "fs";
import { resolve } from "path";
import type { Plugin } from "vite";

/** Serve onnxruntime-web dist (wasm + jsep) at /ort/ for WASM and WebGPU. */
function serveOrtAssets(): Plugin {
  const ortDist = resolve(__dirname, "node_modules/onnxruntime-web/dist");

  return {
    name: "serve-ort-assets",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith("/ort/")) {
          next();
          return;
        }

        const relative = decodeURIComponent(req.url.slice("/ort/".length).split("?")[0]);
        const file = resolve(ortDist, relative);

        if (!file.startsWith(ortDist) || !existsSync(file) || !statSync(file).isFile()) {
          res.statusCode = 404;
          res.end("Not found");
          return;
        }

        if (file.endsWith(".wasm")) {
          res.setHeader("Content-Type", "application/wasm");
        } else if (file.endsWith(".mjs") || file.endsWith(".js")) {
          res.setHeader("Content-Type", "text/javascript");
        }

        createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  root: "browser",
  publicDir: resolve(__dirname, "public"),
  plugins: [serveOrtAssets()],
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
