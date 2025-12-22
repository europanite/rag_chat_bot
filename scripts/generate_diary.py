#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a tweet-style diary entry via backend RAG API and write JSON outputs.

This is a Python port of `scripts/generate_diary.sh`.

Behavior summary (keeps the bash contract):
- Reads env vars (LAT/LON required; many optional with defaults)
- Fetches a live weather snapshot via `scripts/fetch_weather.py`
- Waits for backend `/rag/status`, triggers `/rag/reindex` if empty
- Calls `/rag/query` to generate today's tweet text (with retries)
- Writes:
  - latest.json
  - time-stamped feed JSON (append/replace today's entry)
  - weather snapshot file next to latest: snapshot/snapshot_{now_local}.json

The output path contract matches the bash script:
- FEED_PATH + LATEST_PATH (single) OR FEED_PATHS + LATEST_PATHS (colon-separated)
- If FEED_PATHS/LATEST_PATHS are unset/blank, falls back to:
  FEED_PATH={FEED_PATH}/feed/feed_{now_local}.json
  LATEST_PATH={LATEST_PATH:-frontend/app/public/latest.json}

Stdlib-only.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# -----------------------------
# Utilities
# -----------------------------
def env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v


def is_blank(s: str) -> bool:
    return not s or not s.strip()


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def local_stamp(tz_name: str) -> str:
    if ZoneInfo is None:
        # Fallback: best-effort (still returns something stable)
        return datetime.now().strftime("%Y%m%d_%H%M%S_LOCAL")
    try:
        dt = datetime.now(ZoneInfo(tz_name))
        return dt.strftime("%Y%m%d_%H%M%S_%Z")
    except Exception:
        return datetime.now().strftime("%Y%m%d_%H%M%S_LOCAL")


def split_paths(colon_separated: str) -> List[str]:
    return [p.strip() for p in colon_separated.split(":") if p.strip()]


def normalize_answer(text: str) -> str:
    # Mirror bash: collapse all whitespace/newlines to single spaces
    return re.sub(r"\\s+", " ", (text or "").strip()).strip()


def parse_possibly_concatenated_json(s: str) -> List[Any]:
    s = (s or "").strip()
    if not s:
        raise ValueError("empty body")
    # Trim to first JSON start if there are logs before it
    start_candidates = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if start_candidates:
        s = s[min(start_candidates) :]
    dec = json.JSONDecoder()
    objs = []
    while s:
        obj, end = dec.raw_decode(s)
        objs.append(obj)
        s = s[end:].lstrip()
    return objs


# -----------------------------
# HTTP helpers (stdlib only)
# -----------------------------
@dataclass
class HttpConfig:
    max_time_s: int = 512
    retries: int = 2
    debug: bool = False
    bearer_token: str = ""


def http_json(method: str, url: str, payload: Optional[Dict[str, Any]], cfg: HttpConfig) -> Dict[str, Any]:
    body = ""
    last_exc: Optional[BaseException] = None

    headers = {"Accept": "application/json"}
    data: Optional[bytes] = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if cfg.bearer_token:
        headers["Authorization"] = f"Bearer {cfg.bearer_token}"

    attempts = cfg.retries + 1
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, method=method.upper(), headers=headers, data=data)
            with urllib.request.urlopen(req, timeout=cfg.max_time_s) as resp:
                raw = resp.read()
            body = raw.decode("utf-8", errors="replace")
            if cfg.debug:
                head = body[:200]
                print(f"DEBUG http_json: attempt={attempt} bytes={len(body)} url={url}", file=sys.stderr)
                if head:
                    print(f"DEBUG http_json body head: {head}", file=sys.stderr)
            if body:
                objs = parse_possibly_concatenated_json(body)
                last = objs[-1]
                if isinstance(last, dict):
                    return last
                # If last is not dict, still return wrapped
                return {"_value": last}
        except BaseException as e:
            last_exc = e
            if cfg.debug:
                print(f"DEBUG http_json: attempt={attempt} error={e}", file=sys.stderr)
            time.sleep(2)

    # Match bash behavior: fail if body empty or not JSON
    if not body and last_exc is not None:
        raise RuntimeError(f"backend response body is empty ({url})") from last_exc
    raise RuntimeError(f"backend response is not usable JSON: {body[:200]}") from last_exc


def wait_for_backend(api_base: str, cfg: HttpConfig, tries: int = 60, sleep_s: int = 2) -> None:
    url = f"{api_base}/rag/status"
    for i in range(1, tries + 1):
        try:
            _ = http_json("GET", url, None, cfg)
            print(f"OK: {url}")
            return
        except Exception:
            print(f"Waiting for backend /rag/status ... ({i}/{tries})")
            time.sleep(sleep_s)
    raise RuntimeError(f"backend not ready: {api_base}")


# -----------------------------
# Domain logic
# -----------------------------
def fetch_weather_snapshot(lat: str, lon: str, tz_name: str, place: str) -> Tuple[str, Dict[str, Any]]:
    # Calls the existing script (keeps the same behavior/format)
    cmd = [
        sys.executable,
        "scripts/fetch_weather.py",
        "--format",
        "json",
        "--lat",
        lat,
        "--lon",
        lon,
        "--tz",
        tz_name,
        "--place",
        place,
    ]
    # Use subprocess without importing subprocess? It's stdlib; fine.
    import subprocess

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_weather.py failed: {proc.stderr.strip()}")
    raw = proc.stdout.strip()
    try:
        snap = json.loads(raw) if raw else {}
    except Exception:
        snap = {"raw": raw}
    return raw, snap


def build_question(max_chars: str) -> str:
    # Mirror the bash multi-line QUESTION (including guidance)
    return (
        "Write short tweet-style post starting with greeting on time.\n"
        "Use the live weather JSON for the weather facts.\n"
        "Mention upcoming events suitable for the datetime, weather and season from the RAG context.\n"
        "Show URLs if a topic includes them."
        f"Keep this article within about {max_chars} characters.\n"
        "Use Emoji.\n"
    )


def build_payload(question: str, top_k: int, snap_json_raw: str) -> Dict[str, Any]:
    # Keep current bash behavior: send snapshot as extra_context string; use_live_weather=False
    return {
        "question": question,
        "top_k": int(top_k),
        "extra_context": snap_json_raw,
        "use_live_weather": False,
    }


def extract_tweet(resp_obj: Dict[str, Any]) -> str:
    ans = (resp_obj.get("answer") or "").strip()
    return normalize_answer(ans)


def extract_detail(resp_obj: Dict[str, Any]) -> str:
    d = resp_obj.get("detail")
    if isinstance(d, (list, dict)):
        return json.dumps(d, ensure_ascii=False)
    if d is None:
        return ""
    return str(d)


def build_entry(today: str, now_iso: str, tweet: str, place: str, snap_obj: Dict[str, Any]) -> Dict[str, Any]:
    if not today or not now_iso or not tweet:
        missing = [k for k, v in [("today", today), ("now_iso", now_iso), ("tweet", tweet)] if not v]
        raise RuntimeError(f"missing values for entry: {', '.join(missing)}")
    return {
        "date": today,
        "generated_at": now_iso,
        "text": tweet,
        "place": place or "",
        "weather": snap_obj,
    }


def to_item(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    date = str(entry.get("date") or "")
    text = str(entry.get("text") or "")
    if not date or not text:
        return None
    _id = entry.get("id") or entry.get("generated_at") or date
    place = entry.get("place") or ""
    return {
        "id": str(_id),
        "date": date,
        "text": text,
        "place": str(place),
        "generated_at": entry.get("generated_at"),
        "weather": entry.get("weather"),
    }


def update_feed(feed_path: Path, entry: Dict[str, Any]) -> None:
    entry_item = to_item(entry)
    if not entry_item:
        raise RuntimeError("entry JSON missing required fields (date/text)")

    feed_obj: Dict[str, Any] = {"items": []}
    if feed_path.exists() and feed_path.stat().st_size > 0:
        try:
            loaded = json.loads(feed_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("items"), list):
                feed_obj = loaded
            elif isinstance(loaded, list):
                # legacy format: list of entries
                items = [to_item(x) for x in loaded]
                items2 = [i for i in items if i]
                feed_obj = {
                    "items": items2,
                    "updated_at": (loaded[-1].get("generated_at") if loaded else None),
                    "place": (loaded[-1].get("place") if loaded else ""),
                }
            else:
                feed_obj = {"items": []}
        except Exception:
            feed_obj = {"items": []}

    items = feed_obj.get("items") if isinstance(feed_obj.get("items"), list) else []
    # replace today
    items = [i for i in items if isinstance(i, dict) and i.get("date") != entry_item.get("date")]
    items.append(entry_item)
    items.sort(key=lambda x: x.get("date", ""))
    feed_obj["items"] = items
    feed_obj["updated_at"] = entry.get("generated_at")
    if not feed_obj.get("place"):
        feed_obj["place"] = entry.get("place", "")

    feed_path.write_text(json.dumps(feed_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {feed_path} ({len(items)} entries)")


def write_outputs(feed_path: str, latest_path: str, entry: Dict[str, Any], snap_json_raw: str, now_local: str) -> None:
    fp = Path(feed_path)
    lp = Path(latest_path)

    fp.parent.mkdir(parents=True, exist_ok=True)
    lp.parent.mkdir(parents=True, exist_ok=True)

    entry_txt = json.dumps(entry, ensure_ascii=False, indent=2) + "\n"
    lp.write_text(entry_txt, encoding="utf-8")

    update_feed(fp, entry)

    # Also write weather snapshot next to latest (for debugging / transparency)
    snap_path = lp.parent / "snapshot" / f"snapshot_{now_local}.json"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(snap_json_raw.strip() + "\n", encoding="utf-8")

    print(f"Wrote: {lp}")
    print(f"Wrote: {snap_path}")


def pair_paths(feeds: List[str], latests: List[str]) -> List[Tuple[str, str]]:
    if not feeds or not latests:
        raise RuntimeError("FEED_PATHS / LATEST_PATHS resolved to empty.")
    if len(feeds) == len(latests):
        return list(zip(feeds, latests))
    # bash fallback: pair each feed with first latest
    first = latests[0]
    return [(f, first) for f in feeds]


def main() -> int:
    debug = env("DEBUG", "0") == "1"
    cfg = HttpConfig(
        max_time_s=int(env("CURL_MAX_TIME", "512") or "512"),
        retries=int(env("CURL_RETRIES", "2") or "2"),
        debug=debug,
        bearer_token=env("RAG_TOKEN", ""),
    )

    api_base = env("BACKEND_URL", "http://localhost:8000")
    tz_name = env("TZ_NAME", "Asia/Tokyo")
    place = env("PLACE", "")

    lat = env("LAT", "")
    lon = env("LON", "")
    if is_blank(lat) or is_blank(lon):
        print("ERROR: LAT and LON must be set.", file=sys.stderr)
        print("Example: export LAT=35.2810 LON=139.6720", file=sys.stderr)
        return 1

    # Tweet config
    top_k = int(env("RAG_TOP_K", "3") or "3")
    max_chars = env("MAX_CHARS", "512") or "512"
    hashtags = env("HASHTAGS", "")

    now_local = local_stamp(tz_name)
    # Output paths (match bash behavior)
    feed_path_dir = env("FEED_PATH", "")
    latest_path_default = env("LATEST_PATH", "frontend/app/public/latest.json") or "frontend/app/public/latest.json"
    computed_feed_path = str(Path(feed_path_dir) / "feed" / f"feed_{now_local}.json") if feed_path_dir else ""

    feed_paths_raw = env("FEED_PATHS", "")
    if is_blank(feed_paths_raw):
        feed_paths_raw = computed_feed_path
    latest_paths_raw = env("LATEST_PATHS", "")
    if is_blank(latest_paths_raw):
        latest_paths_raw = latest_path_default

    feeds = split_paths(feed_paths_raw)
    latests = split_paths(latest_paths_raw)

    print(tz_name)
    print(f"API_BASE: {api_base}")
    print(f"WEATHER: lat={lat} lon={lon} tz={tz_name} place='{place}'")
    print(f"TOP_K={top_k} MAX_CHARS={max_chars}")
    print(f"FEED_PATHS={feed_paths_raw}")
    print(f"LATEST_PATHS={latest_paths_raw}")

    # 1) Weather snapshot
    print("Fetching weather snapshot (Open-Meteo)...")
    snap_json_raw, snap_obj = fetch_weather_snapshot(lat=lat, lon=lon, tz_name=tz_name, place=place)
    if debug:
        print(f"DEBUG: SNAP_JSON bytes={len(snap_json_raw.encode('utf-8'))}", file=sys.stderr)

    # 2) Ensure backend ready + index
    wait_for_backend(api_base, cfg)

    status = http_json("GET", f"{api_base}/rag/status", None, cfg)
    chunks_in_store = int(status.get("chunks_in_store", 0) or 0)
    if chunks_in_store == 0:
        print("No chunks in store -> POST /rag/reindex")
        _ = http_json("POST", f"{api_base}/rag/reindex", {}, cfg)
        if debug:
            status2 = {}
            try:
                status2 = http_json("GET", f"{api_base}/rag/status", None, cfg)
            except Exception:
                status2 = {"_error": "failed to re-check status"}
            print(f"DEBUG: status(after reindex)={json.dumps(status2, ensure_ascii=False)}", file=sys.stderr)

    # Warm up once (ignore failures)
    try:
        _ = http_json(
            "POST",
            f"{api_base}/rag/query",
            {"question": "ping", "top_k": 1, "extra_context": "{}", "use_live_weather": False},
            cfg,
        )
    except Exception:
        pass

    # 3) Query backend for today's tweet
    question = build_question(max_chars=max_chars)
    payload = build_payload(question=question, top_k=top_k, snap_json_raw=snap_json_raw)

    if debug:
        print(f"DEBUG: JSON_PAYLOAD={json.dumps(payload, ensure_ascii=False)}", file=sys.stderr)

    tweet = ""
    resp_obj: Dict[str, Any] = {}
    detail = ""
    # bash: retries are CURL_RETRIES+2 here
    for attempt in range(1, cfg.retries + 2 + 1):
        try:
            resp_obj = http_json("POST", f"{api_base}/rag/query", payload, cfg)
        except Exception as e:
            print(f"WARN: /rag/query call failed (attempt {attempt}/{cfg.retries+2}). Retrying...", file=sys.stderr)
            time.sleep(2)
            continue

        tweet = extract_tweet(resp_obj)
        if tweet:
            break

        detail = extract_detail(resp_obj)
        print(
            f"WARN: backend did not return an answer (attempt {attempt}/{cfg.retries+2}). detail={detail}",
            file=sys.stderr,
        )
        time.sleep(2)

    if not tweet:
        print("ERROR: tweet is empty after parsing (after retries).", file=sys.stderr)
        if detail:
            print(f"Last backend detail: {detail}", file=sys.stderr)
        print("---- last response ----", file=sys.stderr)
        print(json.dumps(resp_obj, ensure_ascii=False), file=sys.stderr)
        return 1

    if hashtags and "#" not in tweet:
        tweet = f"{tweet} {hashtags}"

    # 4) Write outputs
    today = utc_date()
    now_iso = utc_now_iso_z()
    entry = build_entry(today=today, now_iso=now_iso, tweet=tweet, place=place, snap_obj=snap_obj)

    for feed_p, latest_p in pair_paths(feeds, latests):
        write_outputs(feed_p, latest_p, entry=entry, snap_json_raw=snap_json_raw, now_local=now_local)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
