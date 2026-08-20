"""Live WHEP (WebRTC HTTP egress) stream of server-side inference."""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse

print("Starting WHEP server…", flush=True)

import aiohttp.web
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.codecs import h264 as h264_codec
from aiortc.rtcrtpsender import RTCRtpSender
from av import VideoFrame

# aiortc defaults: 1 Mbps Baseline H.264 (looks muddy on HD UAV frames).
h264_codec.DEFAULT_BITRATE = 8_000_000
h264_codec.MIN_BITRATE = 4_000_000
h264_codec.MAX_BITRATE = 16_000_000


def _hq_h264_encode(self, frame, force_keyframe: bool):  # noqa: ANN001
    import fractions as _fractions

    import av as _av

    if self.codec and (
        frame.width != self.codec.width
        or frame.height != self.codec.height
        or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate > 0.1
    ):
        self.buffer_data = b""
        self.buffer_pts = None
        self.codec = None

    if force_keyframe:
        frame.pict_type = _av.video.frame.PictureType.I
    else:
        frame.pict_type = _av.video.frame.PictureType.NONE

    if self.codec is None:
        self.codec = _av.CodecContext.create("libx264", "w")
        self.codec.width = frame.width
        self.codec.height = frame.height
        self.codec.bit_rate = self.target_bitrate
        self.codec.pix_fmt = "yuv420p"
        self.codec.framerate = _fractions.Fraction(h264_codec.MAX_FRAME_RATE, 1)
        self.codec.time_base = _fractions.Fraction(1, h264_codec.MAX_FRAME_RATE)
        self.codec.profile = "High"
        self.codec.options = {
            "preset": "veryfast",
            "tune": "zerolatency",
        }

    data_to_send = b""
    for package in self.codec.encode(frame):
        data_to_send += bytes(package)
    if data_to_send:
        yield from h264_codec.H264Encoder._split_bitstream(data_to_send)


h264_codec.H264Encoder._encode_frame = _hq_h264_encode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.annotate import draw_detections  # noqa: E402
from src.common.types import load_baseline_config  # noqa: E402
from src.common.video import iter_video_frames  # noqa: E402
from src.server.inference import cuda_available, load_server_model  # noqa: E402

HOST = "127.0.0.1"
PORT = 8090
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PATCH, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Expose-Headers": "Location",
}

_pcs: dict[str, RTCPeerConnection] = {}
_model_cache: dict[str, object] = {}


def _load_model(model_key: str):
    if model_key in _model_cache:
        return _model_cache[model_key]
    device = "cuda" if cuda_available() else "cpu"
    model = load_server_model(model_key, device)
    _model_cache[model_key] = model
    print(f"Loaded {model_key} on {device}")
    return model


class InferenceTrack(VideoStreamTrack):
    """Annotated UAV frames as a WebRTC video track."""

    kind = "video"

    def __init__(self, model_key: str, clip: str) -> None:
        super().__init__()
        cfg = load_baseline_config()
        if model_key not in cfg["models"]:
            raise ValueError(f"Unknown model {model_key}")
        video = ROOT / "fixtures" / "videos" / f"{clip}.mp4"
        if not video.exists():
            raise FileNotFoundError(video)
        self._video = video
        self._defaults = cfg["defaults"]
        self._model = _load_model(model_key)
        self._frames: list | None = None
        self._i = 0

    def _ensure_frames(self) -> None:
        if self._frames is not None:
            return
        self._frames = [
            frame
            for _idx, frame, _fps in iter_video_frames(
                self._video, self._defaults.get("max_duration_sec")
            )
        ]
        if not self._frames:
            raise RuntimeError(f"No frames in {self._video}")

    def _next_annotated(self):
        self._ensure_frames()
        assert self._frames is not None
        frame = self._frames[self._i % len(self._frames)]
        self._i += 1
        dets = self._model.predict_frame(
            frame,
            self._defaults["imgsz"],
            self._defaults["conf_threshold"],
            self._defaults["iou_threshold"],
        )
        annotated = draw_detections(frame, dets)
        h, w = annotated.shape[:2]
        if w % 2 or h % 2:
            annotated = annotated[: h - (h % 2), : w - (w % 2)]
        return annotated

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        loop = asyncio.get_running_loop()
        annotated = await loop.run_in_executor(None, self._next_annotated)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


def _force_h264(pc: RTCPeerConnection) -> None:
    capabilities = RTCRtpSender.getCapabilities("video")
    h264 = [c for c in capabilities.codecs if c.mimeType.lower() == "video/h264"]
    if not h264:
        return
    for transceiver in pc.getTransceivers():
        if transceiver.kind == "video":
            transceiver.setCodecPreferences(h264)


async def handle_options(request: aiohttp.web.Request) -> aiohttp.web.Response:
    return aiohttp.web.Response(status=204, headers=CORS_HEADERS)


async def handle_index(_request: aiohttp.web.Request) -> aiohttp.web.Response:
    return aiohttp.web.Response(text=_PAGE, content_type="text/html", headers=CORS_HEADERS)


async def handle_whep(request: aiohttp.web.Request) -> aiohttp.web.StreamResponse:
    if request.method == "OPTIONS":
        return await handle_options(request)

    qs = parse_qs(urlparse(str(request.url)).query)
    model = qs.get("model", ["yolov8"])[0]
    clip = qs.get("clip", ["uav_seq_a"])[0]
    offer_sdp = await request.text()
    if not offer_sdp.strip():
        return aiohttp.web.Response(status=400, text="Empty SDP offer", headers=CORS_HEADERS)

    pc = RTCPeerConnection()
    session_id = uuid.uuid4().hex
    _pcs[session_id] = pc

    @pc.on("connectionstatechange")
    async def _on_state() -> None:
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            _pcs.pop(session_id, None)

    try:
        pc.addTrack(InferenceTrack(model, clip))
        _force_h264(pc)
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
    except Exception as exc:  # noqa: BLE001
        await pc.close()
        _pcs.pop(session_id, None)
        return aiohttp.web.Response(status=500, text=str(exc), headers=CORS_HEADERS)

    headers = {
        **CORS_HEADERS,
        "Content-Type": "application/sdp",
        "Location": f"/whep/{session_id}",
    }
    return aiohttp.web.Response(status=201, text=pc.localDescription.sdp, headers=headers)


async def handle_whep_session(request: aiohttp.web.Request) -> aiohttp.web.Response:
    session_id = request.match_info["session_id"]
    pc = _pcs.get(session_id)
    if pc is None:
        return aiohttp.web.Response(status=404, headers=CORS_HEADERS)
    if request.method == "DELETE":
        await pc.close()
        _pcs.pop(session_id, None)
        return aiohttp.web.Response(status=204, headers=CORS_HEADERS)
    return aiohttp.web.Response(status=204, headers=CORS_HEADERS)


_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Server WHEP stream</title>
<style>body{font-family:system-ui;background:#111;color:#eee;padding:1rem}video{max-width:100%;background:#000}</style>
</head><body>
<h1>Server live inference (WHEP)</h1>
<p>Chromium. GPU if CUDA is available. Default: YOLOv8n / uav_seq_a.</p>
<video id="v" autoplay playsinline muted></video>
<script>
async function play() {
  const pc = new RTCPeerConnection();
  pc.ontrack = (e) => { document.getElementById("v").srcObject = e.streams[0]; };
  pc.addTransceiver("video", { direction: "recvonly" });
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await new Promise((resolve) => {
    if (pc.iceGatheringState === "complete") return resolve();
    pc.onicegatheringstatechange = () => {
      if (pc.iceGatheringState === "complete") resolve();
    };
  });
  const res = await fetch("/whep?model=yolov8&clip=uav_seq_a", {
    method: "POST",
    headers: { "Content-Type": "application/sdp" },
    body: pc.localDescription.sdp,
  });
  if (!res.ok) throw new Error(await res.text());
  await pc.setRemoteDescription({ type: "answer", sdp: await res.text() });
}
play().catch((err) => { document.body.append(String(err)); });
</script>
</body></html>
"""


async def main() -> None:
    device = "GPU" if cuda_available() else "CPU"
    app = aiohttp.web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_route("OPTIONS", "/whep", handle_options)
    app.router.add_post("/whep", handle_whep)
    app.router.add_route("OPTIONS", "/whep/{session_id}", handle_options)
    app.router.add_route("PATCH", "/whep/{session_id}", handle_whep_session)
    app.router.add_route("DELETE", "/whep/{session_id}", handle_whep_session)
    print(f"Server WHEP stream on {device}: http://{HOST}:{PORT}/")
    print("POST application/sdp offer to /whep?model=yolov8&clip=uav_seq_a")
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, HOST, PORT)
    await site.start()
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
