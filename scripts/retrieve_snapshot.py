import os
import time
from pathlib import Path

import requests

URL = os.environ["SNAP_URL"].strip()
OUT = Path(os.environ.get("SNAP_OUT", "snapshot.jpg"))
RETRIES = int(os.environ.get("SNAP_RETRIES", "3"))
TIMEOUT = float(os.environ.get("SNAP_TIMEOUT", "30"))

HEADERS = {
    "User-Agent": "snapshot-safe-http/1.1 (+github-actions)",
    "Accept": "image/*",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def _cache_bust(u: str) -> str:
    sep = "&" if "?" in u else "?"
    # ms to reduce accidental caching collisions
    return f"{u}{sep}t={int(time.time() * 1000)}"


def _download_once(session: requests.Session, url: str, out_path: Path) -> tuple[int, str]:
    tmp = out_path.parent / (out_path.name + ".part")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass

    u = _cache_bust(url)
    n = 0
    ctype = ""

    try:
        with session.get(u, headers=HEADERS, timeout=TIMEOUT, stream=True) as r:
            r.raise_for_status()
            ctype = (r.headers.get("Content-Type", "") or "").lower()
            if "image" not in ctype:
                raise RuntimeError(f"Non-image response: Content-Type={ctype}")

            expected = int(r.headers.get("Content-Length", "0") or 0)

            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    n += len(chunk)

            if expected and n < expected:
                raise RuntimeError(f"Truncated download: got {n} bytes < expected {expected}")

        tmp.replace(out_path)
        return n, ctype

    finally:
        # if we failed before replace(), try to clean up the .part file
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    last_err = None

    with requests.Session() as s:
        for attempt in range(1, RETRIES + 1):
            try:
                n, ctype = _download_once(s, URL, OUT)
                print(f"OK {OUT} {n} {ctype}")
                return 0
            except Exception as e:
                last_err = e
                if attempt >= RETRIES:
                    break
                print(f"WARN snapshot download failed (attempt {attempt}/{RETRIES}): {e}")
                time.sleep(1.0 * attempt)

    raise SystemExit(f"ERROR snapshot download failed after {RETRIES} attempts: {last_err}")


if __name__ == "__main__":
    raise SystemExit(main())