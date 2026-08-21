"""Download real VisDrone-VID sequences and encode them as MP4 fixtures.

These are authentic consecutive UAV frames (not stills with pan/zoom).
Source: VisDrone2019-VID-val via Hugging Face mirror (~1.5 GB).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import cv2
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.types import load_baseline_config  # noqa: E402

VID_URL = (
    "https://huggingface.co/datasets/AndriiDemk/visDrone_copy/resolve/main/"
    "VisDrone2019-VID-val.zip"
)
CACHE_DIR = ROOT / "fixtures" / "cache"
ZIP_PATH = CACHE_DIR / "VisDrone2019-VID-val.zip"
EXTRACT_DIR = CACHE_DIR / "VisDrone2019-VID-val"

# Friendly fixture name -> VisDrone sequence folder
# Must match sequences present in VisDrone2019-VID-val.zip (and FIXTURE_SEQUENCES).
SEQUENCES = {
    "uav_seq_a.mp4": "uav0000086_00000_v",
    "uav_seq_b.mp4": "uav0000117_02622_v",
    "uav_seq_c.mp4": "uav0000137_00458_v",
    "uav_seq_d.mp4": "uav0000182_00000_v",
}


def download_vid_zip() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 100_000_000:
        print(f"Cache hit: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e9:.2f} GB)")
        return ZIP_PATH

    print(f"Downloading VisDrone-VID-val (~1.5 GB) from Hugging Face...")
    print(f"  {VID_URL}")
    with requests.get(VID_URL, stream=True, timeout=600, headers={"User-Agent": "tas-model-test-suite/0.1"}) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        with ZIP_PATH.open("wb") as f:
            for chunk in resp.iter_content(1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total and done % (50 << 20) < (1 << 20):
                    print(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB")
    print(f"  saved {ZIP_PATH.stat().st_size / 1e9:.2f} GB")
    return ZIP_PATH


def list_sequences(zf: zipfile.ZipFile) -> list[str]:
    seqs = set()
    for name in zf.namelist():
        # .../sequences/<seq>/0000001.jpg
        parts = Path(name).parts
        if "sequences" in parts:
            i = parts.index("sequences")
            if i + 1 < len(parts):
                seqs.add(parts[i + 1])
    return sorted(seqs)


def extract_sequence(zf: zipfile.ZipFile, seq: str, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    prefix_parts = None
    for name in zf.namelist():
        parts = Path(name).parts
        if "sequences" not in parts:
            continue
        i = parts.index("sequences")
        if i + 1 >= len(parts) or parts[i + 1] != seq:
            continue
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        out = dest / Path(name).name
        if not out.exists():
            out.write_bytes(zf.read(name))
        frames.append(out)
    frames.sort()
    return frames


def frames_to_mp4(frames: list[Path], out: Path, fps: float, max_frames: int) -> None:
    """Encode as H.264 when possible so Chromium can play fixtures in browser tests."""
    if not frames:
        raise RuntimeError(f"No frames for {out.name}")
    sample = cv2.imread(str(frames[0]))
    if sample is None:
        raise RuntimeError(f"Could not read {frames[0]}")
    h, w = sample.shape[:2]
    out.parent.mkdir(parents=True, exist_ok=True)
    used = frames[:max_frames]

    raw = out.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fp in used:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()

    try:
        import subprocess

        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.check_call(
            [
                ffmpeg,
                "-y",
                "-i",
                str(raw),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(out),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        raw.unlink(missing_ok=True)
        print(f"  wrote {out} ({len(used)} frames @ {fps} fps, H.264)")
    except Exception as exc:  # noqa: BLE001
        raw.replace(out)
        print(f"  wrote {out} ({len(used)} frames @ {fps} fps, mp4v fallback: {exc})")


def download_fixtures() -> None:
    cfg = load_baseline_config()
    defaults = cfg["defaults"]
    max_duration = defaults.get("max_duration_sec") or 8
    fps = 30.0
    max_frames = int(max_duration * fps)

    zip_path = download_vid_zip()
    with zipfile.ZipFile(zip_path) as zf:
        available = list_sequences(zf)
        print(f"Found {len(available)} VisDrone-VID sequences")
        if available:
            print("  e.g.", ", ".join(available[:8]))

        # Map configured outputs to real sequences (fallback to first N)
        configured = list(SEQUENCES.items())
        for idx, rel in enumerate(cfg["fixtures"]["videos"]):
            out = ROOT / rel
            name = Path(rel).name
            seq = SEQUENCES.get(name)
            if seq not in available:
                seq = available[idx] if idx < len(available) else available[0]
            print(f"Encoding {name} from sequence {seq}")
            frame_dir = EXTRACT_DIR / "sequences" / seq
            frames = extract_sequence(zf, seq, frame_dir)
            frames_to_mp4(frames, out, fps=fps, max_frames=max_frames)
            public = ROOT / "public" / "fixtures" / name
            public.parent.mkdir(parents=True, exist_ok=True)
            public.write_bytes(out.read_bytes())

            # Extract matching VisDrone-VID annotations for quality eval
            ann_member = f"VisDrone2019-VID-val/annotations/{seq}.txt"
            ann_dest = ROOT / "fixtures" / "annotations" / f"{seq}.txt"
            try:
                ann_bytes = zf.read(ann_member)
            except KeyError:
                print(f"  warning: no annotation {ann_member}")
            else:
                ann_dest.parent.mkdir(parents=True, exist_ok=True)
                ann_dest.write_bytes(ann_bytes)
                print(f"  annotations -> {ann_dest.relative_to(ROOT)}")


if __name__ == "__main__":
    download_fixtures()
