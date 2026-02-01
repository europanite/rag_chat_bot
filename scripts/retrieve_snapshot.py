import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

# Output image used as INPUT_IMAGE for img2img.
OUT = Path(os.environ.get("SNAP_OUT", "snapshot.jpg"))
RETRIES = int(os.environ.get("SNAP_RETRIES", "3"))
TIMEOUT = float(os.environ.get("SNAP_TIMEOUT", "30"))

# If REF_IMAGE_URL is set, use it as the reference image URL.
# Otherwise, try to find an image URL inside LATEST_PATH (latest.json).
REF_IMAGE_URL = (os.environ.get("REF_IMAGE_URL", "") or "").strip()
LATEST_PATH = (os.environ.get("LATEST_PATH", "") or "").strip()

# Fallback: fixed camera snapshot.
SNAP_URL = (os.environ.get("SNAP_URL", "") or "").strip()

MAX_REF_CANDIDATES = int(os.environ.get("REF_MAX_CANDIDATES", "8"))

HEADERS = {
    "User-Agent": "snapshot-safe-http/1.2 (+github-actions)",
    "Accept": "image/*",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

_IMAGE_URL_RE = re.compile(r"^https?://\S+\.(?:jpe?g|png|webp|gif)(?:\?\S*)?$", re.IGNORECASE)
_URL_IN_TEXT_RE = re.compile(r"https?://\S+")


def _cache_bust(u: str) -> str:
    sep = "&" if "?" in u else "?"
    # ms to reduce accidental caching collisions
    return f"{u}{sep}t={int(time.time() * 1000)}"


def _looks_like_image_url(u: str) -> bool:
    return bool(_IMAGE_URL_RE.match(u.strip()))


def _candidate_urls_from_latest(latest_path: Path) -> list[str]:
    """Extract candidate image URLs from latest.json.

    We keep this conservative: only direct image URLs are returned.
    """
    try:
        obj: Any = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(obj, dict):
        return []

    cands: list[str] = []

    # 1) explicit fields (if you later decide to add one)
    for k in ("ref_image_url", "image_ref_url", "image_url", "thumbnail", "thumb", "og_image"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip().startswith(("http://", "https://")):
            cands.append(v.strip())

    # 2) links (string list or dict list)
    links_obj = obj.get("links", obj.get("link"))
    if isinstance(links_obj, list):
        items = links_obj[:MAX_REF_CANDIDATES]
    else:
        items = [links_obj] if links_obj is not None else []

    for item in items:
        if isinstance(item, str):
            cands.append(item.strip())
        elif isinstance(item, dict):
            for kk in ("image_url", "image", "url", "href", "link"):
                vv = item.get(kk)
                if isinstance(vv, str) and vv.strip().startswith(("http://", "https://")):
                    cands.append(vv.strip())

    # 3) scan text fields for direct image URLs
    for field in ("text", "body", "content", "caption"):
        v = obj.get(field)
        if isinstance(v, str):
            for u in _URL_IN_TEXT_RE.findall(v):
                cands.append(u.strip().rstrip(").,]\"'"))

    # filter to unique image urls (preserve order)
    out: list[str] = []
    seen: set[str] = set()
    for u in cands:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if uu in seen:
            continue
        seen.add(uu)
        if _looks_like_image_url(uu):
            out.append(uu)
    return out


def _pick_source_url() -> tuple[str, str, bool]:
    """Return (url, source, cache_bust)."""
    if REF_IMAGE_URL and _looks_like_image_url(REF_IMAGE_URL):
        return REF_IMAGE_URL, "REF_IMAGE_URL", False

    if LATEST_PATH:
        p = Path(LATEST_PATH)
        if p.exists():
            cands = _candidate_urls_from_latest(p)
            if cands:
                return cands[0], f"LATEST_PATH:{p}", False

    if SNAP_URL:
        return SNAP_URL, "SNAP_URL", True

    raise SystemExit(
        "ERROR: No usable image URL. Set REF_IMAGE_URL (direct image URL) "
        "or ensure LATEST_PATH contains an image URL, or set SNAP_URL as fallback."
    )


def _download_once(
    session: requests.Session, url: str, out_path: Path, *, cache_bust: bool
) -> tuple[int, str]:
    tmp = out_path.parent / (out_path.name + ".part")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass

    u = _cache_bust(url) if cache_bust else url
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

    url, source, cache_bust = _pick_source_url()
    print(f"Snapshot source: {source}")
    print(f"Snapshot url: {url}")

    with requests.Session() as s:
        for attempt in range(1, RETRIES + 1):
            try:
                n, ctype = _download_once(s, url, OUT, cache_bust=cache_bust)
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
