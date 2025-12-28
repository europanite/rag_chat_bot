#!/usr/bin/env python3
"""
Capture a snapshot.

Requirements (in runner):
- ffmpeg (CLI)

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def resolve_stream_url(url: str, fmt: str) -> str:
    """
    Resolve a playable (often time-limited) stream URL via yt-dlp.
    yt-dlp -g may return multiple lines (e.g., video+audio). We take the first URL,
    which is usually the video stream (or combined HLS).
    """
    cmd = ["yt-dlp", "--no-warnings", "-g", "-f", fmt, url]
    p = run(cmd, check=False)

    if p.returncode != 0:
        raise RuntimeError(
            "yt-dlp failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )

    urls = [line.strip() for line in p.stdout.splitlines() if line.strip()]
    if not urls:
        raise RuntimeError(
            "yt-dlp returned no URLs.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )

    return urls[0]


def capture_frame(stream_url: str, out_path: Path, timeout_sec: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -rw_timeout is in microseconds; keep small to avoid hangs on CI.
    rw_timeout_us = str(timeout_sec * 1_000_000)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-rw_timeout",
        rw_timeout_us,
        "-i",
        stream_url,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out_path),
    ]
    p = run(cmd, check=False)

    if p.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            "ffmpeg failed to capture a frame.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}\n"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="url")
    ap.add_argument("--out", required=True, help="Output image path (.jpg/.png)")
    ap.add_argument(
        "--format",
        default=os.getenv("YTDLP_FORMAT", "bestvideo+bestaudio/best"),
        help='yt-dlp format selector (default: "bestvideo+bestaudio/best")',
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("CAPTURE_TIMEOUT_SEC", "15")),
        help="Network read timeout in seconds (default: 15)",
    )
    args = ap.parse_args()

    out_path = Path(args.out)

    try:
        stream_url = resolve_stream_url(args.url, args.format)
        capture_frame(stream_url, out_path, args.timeout)
        print(f"OK: saved snapshot -> {out_path} (bytes={out_path.stat().st_size})")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
