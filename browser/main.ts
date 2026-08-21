import "../src/browser/harness.ts";

const WHEP_BASE = "http://127.0.0.1:8090";

const modelEl = document.getElementById("model") as HTMLSelectElement;
const clipEl = document.getElementById("clip") as HTMLSelectElement;
const browserEpEl = document.getElementById("browser-ep") as HTMLSelectElement;
const startBrowser = document.getElementById("start-browser") as HTMLButtonElement;
const startServer = document.getElementById("start-server") as HTMLButtonElement;
const serverVideo = document.getElementById("server-video") as HTMLVideoElement;
const serverStatus = document.getElementById("server-status")!;

let whepPc: RTCPeerConnection | null = null;
let whepResource: string | null = null;

startBrowser.addEventListener("click", async () => {
  startBrowser.disabled = true;
  try {
    await window.runBrowserVideoInference({
      modelKey: modelEl.value as "yolov8" | "yolov10" | "rt_detr_l" | "rt_detr_s",
      videoUrl: `/fixtures/${clipEl.value}.mp4`,
      imgsz: 640,
      confThreshold: 0.25,
      iouThreshold: 0.45,
      maxDurationSec: 5,
      executionProvider: browserEpEl.value as "wasm" | "webgpu",
    });
  } finally {
    startBrowser.disabled = false;
  }
});

async function waitIceComplete(pc: RTCPeerConnection): Promise<void> {
  if (pc.iceGatheringState === "complete") return;
  await new Promise<void>((resolve) => {
    pc.onicegatheringstatechange = () => {
      if (pc.iceGatheringState === "complete") resolve();
    };
  });
}

async function stopWhep(): Promise<void> {
  if (whepResource) {
    try {
      await fetch(whepResource, { method: "DELETE" });
    } catch {
      /* session may already be gone */
    }
    whepResource = null;
  }
  if (whepPc) {
    whepPc.close();
    whepPc = null;
  }
  serverVideo.srcObject = null;
}

startServer.addEventListener("click", async () => {
  startServer.disabled = true;
  await stopWhep();
  const model = modelEl.value;
  const clip = clipEl.value;
  const whepUrl = `${WHEP_BASE}/whep?model=${encodeURIComponent(model)}&clip=${encodeURIComponent(clip)}`;
  serverStatus.textContent = `WHEP POST ${whepUrl}`;

  try {
    const pc = new RTCPeerConnection();
    whepPc = pc;
    pc.ontrack = (event) => {
      serverVideo.srcObject = event.streams[0] ?? new MediaStream([event.track]);
      void serverVideo.play();
      serverStatus.textContent = "WHEP connected — receiving annotated WebRTC video.";
    };
    pc.addTransceiver("video", { direction: "recvonly" });

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitIceComplete(pc);

    const res = await fetch(whepUrl, {
      method: "POST",
      headers: { "Content-Type": "application/sdp" },
      body: pc.localDescription?.sdp ?? "",
    });
    if (!res.ok) {
      throw new Error(`WHEP ${res.status}: ${await res.text()}`);
    }
    const location = res.headers.get("Location");
    whepResource = location ? new URL(location, WHEP_BASE).toString() : null;
    await pc.setRemoteDescription({ type: "answer", sdp: await res.text() });
  } catch (err) {
    await stopWhep();
    const message = err instanceof Error ? err.message : String(err);
    serverStatus.textContent =
      `${message}\nStart the WHEP server: python scripts/live_server.py`;
  } finally {
    startServer.disabled = false;
  }
});
