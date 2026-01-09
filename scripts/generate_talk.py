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
import hashlib
import random
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


# -----------------------------
# Utilities
# -----------------------------
def env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v




def env_bool(name: str, default: bool = False) -> bool:
    v = env(name, None)
    if v is None:
        return bool(default)
    v2 = str(v).strip().lower()
    if v2 in ("1", "true", "yes", "y", "on"):
        return True
    if v2 in ("0", "false", "no", "n", "off"):
        return False
    # fallback: any non-empty value means True
    return True


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
        except urllib.error.HTTPError as e:
            # IMPORTANT: read FastAPI error JSON (e.g. {"detail":"..."})
            try:
                raw = e.read()
                body = raw.decode("utf-8", errors="replace")
                if body:
                    try:
                        objs = parse_possibly_concatenated_json(body)
                        last = objs[-1]
                        if isinstance(last, dict):
                            return last
                        return {"_value": last}
                    except Exception:
                        pass
            finally:
                last_exc = e
            time.sleep(2)
        except BaseException as e:
            last_exc = e
            if cfg.debug:
                print(f"DEBUG http_json: attempt={attempt} error={e}", file=sys.stderr)
            time.sleep(2)

    # Match bash behavior: fail if body empty or not JSON
    if not body and last_exc is not None:
        raise RuntimeError(f"backend response body is empty ({url}); last_exc={last_exc!r}") from last_exc
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


def _as_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _time_of_day_bucket(hour: int) -> str:
    # Local time bucket for topic bias (NOT greeting; greeting remains model-side from weather JSON)
    if 5 <= hour <= 10:
        return "morning"
    if 11 <= hour <= 15:
        return "afternoon"
    if 16 <= hour <= 20:
        return "evening"
    return "night"


def _season_bucket(month: int) -> str:
    # Simple JP season mapping
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _seasonal_event_keywords(now_local: datetime) -> str:
    """Keywords for seasonal events.

    Avoid hard-biasing 'new year' unless we are near that period, to reduce retrieval of past New Year events.
    """
    base = "seasonal holiday event"
    mth = now_local.month
    day = now_local.day
    extra: list[str] = []
    # Christmas window (approx): Dec 15 - Dec 26
    if mth == 12 and 15 <= day <= 26:
        extra.append("christmas")
    # New Year window (approx): Dec 26 - Jan 5
    if (mth == 12 and day >= 26) or (mth == 1 and day <= 5):
        extra.append("new year")
    if extra:
        return base + " " + " ".join(extra)
    return base


def _weather_hint(cur: Dict[str, Any]) -> str:
    temp = _as_float(cur.get("temp_c"))
    precip = _as_float(cur.get("precip_mm"))
    cloud = _as_float(cur.get("cloud_cover_pct"))
    wind = _as_float(cur.get("wind_kmh"))
    code = cur.get("weather_code")

    snow_codes = {71, 73, 75, 77, 85, 86}
    thunder_codes = {95, 96, 99}

    tags: list[str] = []
    if isinstance(code, int) and code in thunder_codes:
        tags.append("thunder")
    if isinstance(code, int) and code in snow_codes:
        tags.append("snowy")
    if precip is not None and precip >= 0.2:
        tags.append("rainy")
    if wind is not None and wind >= 20:
        tags.append("windy")
    if cloud is not None and cloud >= 70:
        tags.append("cloudy")

    if temp is not None:
        if temp <= 8:
            tags.append("cold")
        elif temp >= 26:
            tags.append("hot")
        else:
            tags.append("mild")

    return ", ".join(tags) if tags else "unknown"

def _weather_brief_for_llm(snap_obj: Dict[str, Any]) -> str:
    """Return a tiny JSON string for LLM: only condition words + temperature.

    The tweet prompt requires weather words to be among:
    sunny / cloudy / windy / chilly / rainy.
    """
    cur = (snap_obj or {}).get("current") or {}
    temp = _as_float(cur.get("temp_c"))
    precip = _as_float(cur.get("precip_mm"))
    cloud = _as_float(cur.get("cloud_cover_pct"))
    wind = _as_float(cur.get("wind_kmh"))
    code = cur.get("weather_code")

    # WMO weather codes (Open-Meteo compatible)
    precip_codes = set(range(51, 68)) | {80, 81, 82, 95, 96, 99}
    clear_codes = {0, 1}
    cloudy_codes = {2, 3}

    tags: set[str] = set()

    # Rain first (covers showers / thunderstorms as well)
    if precip is not None and precip >= 0.2:
        tags.add("rainy")
    elif isinstance(code, int) and code in precip_codes:
        tags.add("rainy")

    # Sky condition if not rainy
    if "rainy" not in tags:
        if cloud is not None:
            if cloud >= 70:
                tags.add("cloudy")
            elif cloud <= 30:
                tags.add("sunny")
        elif isinstance(code, int):
            if code in clear_codes:
                tags.add("sunny")
            elif code in cloudy_codes:
                tags.add("cloudy")

    # Wind and temperature modifiers
    if wind is not None and wind >= 20:
        tags.add("windy")
    if temp is not None and temp <= 8:
        tags.add("chilly")

    # Guarantee at least one of {sunny, cloudy, rainy}
    if not ({"sunny", "cloudy", "rainy"} & tags):
        tags.add("cloudy")

    temp_i = int(round(temp)) if temp is not None else None
    order = ["rainy", "sunny", "cloudy", "windy", "chilly"]
    weather = [t for t in order if t in tags]
    return json.dumps({"weather": weather, "temp_c": temp_i}, ensure_ascii=False)


def pick_topic(now_local: datetime, snap_obj: Dict[str, Any]) -> Tuple[str, str]:
    """
    Pick topic as (family, mode) where family ∈ {event, place, chat}.
    mode adds variety while keeping the 3-family constraint.
    """
    cur = (snap_obj or {}).get("current") or {}
    temp = _as_float(cur.get("temp_c"))
    precip = _as_float(cur.get("precip_mm"))
    code = cur.get("weather_code")

    tod = _time_of_day_bucket(now_local.hour)
    season = _season_bucket(now_local.month)
    hint = _weather_hint(cur)

    # 1) Choose family: event/place/chat (as requested)
    family_w = {"event": 34, "place": 33, "chat": 33}

    # Weather bias
    if precip is not None and precip >= 0.2:
        family_w["place"] -= 12
        family_w["chat"] += 8
        family_w["event"] += 4
    if temp is not None and temp <= 8:
        family_w["place"] -= 6
        family_w["chat"] += 6
        # winter events (illumination, seasonal) remain viable
        family_w["event"] += 0
    if temp is not None and temp >= 24 and (precip is None or precip < 0.2):
        family_w["place"] += 6
        family_w["event"] += 3
        family_w["chat"] -= 4

    # Time-of-day bias
    if tod in ("evening", "night"):
        family_w["event"] += 10
        family_w["chat"] += 4
        family_w["place"] -= 6
    elif tod == "morning":
        family_w["place"] += 6
        family_w["chat"] += 2
        family_w["event"] -= 2
    elif tod == "afternoon":
        family_w["place"] += 4
        family_w["event"] += 2

    # Seasonal bias
    if season == "winter":
        family_w["event"] += 6   # illumination / holiday / new year
        family_w["chat"] += 2    # warm food/drinks
    elif season == "summer":
        family_w["place"] += 6   # coast / outdoor
        family_w["event"] += 4   # matsuri
        family_w["chat"] -= 4

    for k in list(family_w.keys()):
        family_w[k] = max(1, int(family_w[k]))

    # Deterministic per-hour, but still "random" and condition-aware
    seed_str = f"{now_local.strftime('%Y-%m-%d-%H')}|{season}|{tod}|{hint}|{code}"
    seed = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)

    families = list(family_w.keys())
    family = rng.choices(families, weights=[family_w[f] for f in families], k=1)[0]

    # 2) Choose mode inside family
    if family == "place":
        modes = ["coast", "park", "viewpoint", "shrine", "museum", "walk"]
        w =     [  20,     18,       18,       16,       14,     14]
        if precip is not None and precip >= 0.2:
            # rain -> indoor-ish / short walk
            w = [8, 10, 10, 14, 38, 20]
        if season == "winter" and tod in ("evening", "night"):
            # winter evening -> viewpoints for lights / night scenery
            w = [14, 10, 26, 12, 20, 18]
    elif family == "event":
        modes = ["illumination", "festival", "market", "exhibition", "seasonal"]
        w =     [      26,        20,       18,          18,        18]
        if season == "winter":
            w = [40, 12, 12, 16, 20]
        if tod in ("evening", "night"):
            w = [38, 16, 14, 14, 18]
        if precip is not None and precip >= 0.2:
            # rain -> exhibitions/indoor events
            w = [14, 10, 10, 42, 24]
    else:  # chat
        modes = ["food", "trivia", "history", "activity"]
        w =     [  30,      25,       20,        25]
        if precip is not None and precip >= 0.2:
            w = [40, 25, 25, 10]
        if temp is not None and temp <= 8:
            w = [44, 22, 24, 10]
        if temp is not None and temp >= 24 and (precip is None or precip < 0.2):
            w = [24, 18, 18, 40]
        if tod in ("evening", "night"):
            w = [42, 24, 20, 14]

    mode = rng.choices(modes, weights=w, k=1)[0]
    return family, mode


def build_question(
    max_chars: int,
    topic_family: str,
    topic_mode: str,
    now_local: datetime,
    snap_obj: dict,
    links: list[str] | None = None,
    datetime: str | None = None,
) -> str:

    cur = (snap_obj or {}).get("current") or {}
    tod = _time_of_day_bucket(now_local.hour)
    season = _season_bucket(now_local.month)
    hint = _weather_hint(cur)

    # Keywords help retrieval; family/mode enforce "event/place/chat"
    keyword_map: Dict[Tuple[str, str], str] = {
        # place
        ("place", "coast"): "coast beach sea bay port waterfront",
        ("place", "park"): "park garden green nature trail",
        ("place", "viewpoint"): "viewpoint hill lookout skyline sunset night-view",
        ("place", "shrine"): "shrine temple heritage tradition",
        ("place", "museum"): "museum exhibition indoor history culture",
        ("place", "walk"): "walk stroll promenade shopping street",
        # event
        ("event", "illumination"): "illumination lights winter evening night-view",
        ("event", "festival"): "festival matsuri parade performance",
        ("event", "market"): "market fair flea local vendors",
        ("event", "exhibition"): "exhibition art museum gallery indoor",
        ("event", "seasonal"): "seasonal holiday event",
        # chat
        ("chat", "food"): "curry ramen cafe bakery warm drink",
        ("chat", "trivia"): "fun fact local tip small story",
        ("chat", "history"): "history navy port heritage museum",
        ("chat", "activity"): "running fishing hike workout",
    }
    topic_keywords = keyword_map.get((topic_family, topic_mode), "local tip short story")
    if topic_family == "event" and topic_mode == "seasonal":
        topic_keywords = _seasonal_event_keywords(now_local)

    # Strong guardrails: prevent "upcoming" from referencing past events.
    event_guard = ""
    if topic_family == "event":
        event_guard = (
            "IMPORTANT(event): 'upcoming' means today or later (based on datetime above). "
            "Do NOT mention past events or any date earlier than today. "
            "If you cannot find a future event in the RAG Context, write about a local spot instead (still from RAG Context).\n"
        )


    return (
        "Write a tweet in English.\n"
        "Format.\n"
        "- GREETING FIRST\n"
        "- WEATHER_TOPIC(Simply descrive weather(use only sunny, cloudy, windy, chilly, rainy with temperarure, No humid)\n"
        "- Local Spots or upcoming events from ONLY RAG CONTEXT\n"
        f"datetime: {datetime}.\n"
        "\n"
        "If you use words like 'tonight', 'this evening', 'later tonight', 'later today', they must match NOW.\n"
        "If the event date is not today, say “tomorrow” or include an explicit date (e.g., Dec 31).\n"
        f"{event_guard}"
        f"TOPIC FAMILY: {topic_family} (event/place/chat).\n"
        f"SUBTOPIC: {topic_mode} (keywords: {topic_keywords}).\n"
        f"HINTS: time_of_day={tod}, season={season}, weather_hint={hint}.\n"
        "Pick up ONE topic and mention only that one from RAG Context that fits the HINTS.\n"
        "You may include at most one official URL only if it exists in the chosen text.\n"
        f"Write up to {max_chars} characters.\n"
    )


def build_payload(
    question: str,
    top_k: int,
    snap_obj: Dict[str, Any] | None,
    max_chars: int,
    include_debug: bool,
    datetime: str | None = None,
    links: list[str] | None = None,
    *,
    audit: bool | None = None,
    audit_rewrite: bool | None = None,
) -> dict:
    # Send only a compact weather summary (condition words + temp) to the LLM.
    payload: dict = {
        "question": question,
        "top_k": top_k,
        "max_chars": max_chars,
        "include_debug": include_debug,
        "output_style": "tweet_bot",
        "extra_context": _weather_brief_for_llm(snap_obj),
        "datetime": datetime,
        "links": links,
    }
    if audit is not None:
        payload["audit"] = audit
    if audit_rewrite is not None:
        payload["audit_rewrite"] = audit_rewrite
    return payload
def extract_tweet(resp_obj: Dict[str, Any]) -> str:
    ans = (resp_obj.get("answer") or "").strip()
    return normalize_answer(ans)

def extract_links(resp_obj: Any) -> List[str]:
    if not isinstance(resp_obj, dict):
        return []
    links_obj = resp_obj.get("links")
    links: List[str] = []
    if isinstance(links_obj, list):
        links = [x.strip() for x in links_obj if isinstance(x, str) and x.strip()]
    elif isinstance(links_obj, str):
        v = links_obj.strip()
        if v:
            links = [v]
    return links[:5]

def extract_detail(resp_obj: Dict[str, Any]) -> str:
    d = resp_obj.get("detail")
    if isinstance(d, (list, dict)):
        return json.dumps(d, ensure_ascii=False)
    if d is None:
        return ""
    return str(d)


def build_entry(today: str, now_iso: str, tweet: str, place: str, snap_obj: Dict[str, Any], links: Optional[List[str]] = None) -> Dict[str, Any]:
    if not today or not now_iso or not tweet:
        missing = [k for k, v in [("today", today), ("now_iso", now_iso), ("tweet", tweet)] if not v]
        raise RuntimeError(f"missing values for entry: {', '.join(missing)}")
    if not links:
        links = []
    return {
        "date": today,
        "generated_at": now_iso,
        "text": tweet,
        "place": place or "",
        "weather": snap_obj,
        "links": links
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
        "links": entry.get("links") or [],
    }


def write_outputs(feed_path: str, latest_path: str, entry: Dict[str, Any], snap_json_raw: str, now_local: str) -> None:
    fp = Path(feed_path)
    lp = Path(latest_path)

    fp.parent.mkdir(parents=True, exist_ok=True)
    lp.parent.mkdir(parents=True, exist_ok=True)

    # Keep a single-item object shape (date/text at top-level) for latest.json compatibility
    item = to_item(entry) or dict(entry)

    # 1) feed snapshot (per-run)
    fp.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {fp}")

    # 2) latest.json
    lp.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {lp}")

    # 3) raw weather snapshot next to latest (debug/transparency)
    snap_path = lp.parent / "snapshot" / f"snapshot_{now_local}.json"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        snap_obj = json.loads(snap_json_raw)
        snap_path.write_text(json.dumps(snap_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        snap_path.write_text(str(snap_json_raw).rstrip() + "\n", encoding="utf-8")

    print(f"Wrote: {fp}")
    print(f"Wrote: {lp}")
    print(f"Wrote: {snap_path}")
    return fp, lp, snap_path

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
    top_k = int(env("RAG_TOP_K", "16") or "16")
    if top_k > 128:
        print(f"ERROR: RAG_TOP_K={top_k} is invalid. Backend requires top_k <= 128.", file=sys.stderr)
        return 1
    max_chars = int(env("MAX_CHARS"))
    hashtags = env("HASHTAGS", "")

    now_local = local_stamp(tz_name)
    # Output paths (match bash behavior)
    feed_path_dir = env("FEED_PATH", "")
    latest_path_default = env("LATEST_PATH", "frontend/app/public/latest.json")
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
    print(f"TOP_K={top_k} max_chars={max_chars}")
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

    # Build query URL with location hints (place is NOT a QueryRequest field)
    q = {"place": place, "lat": str(lat), "lon": str(lon), "tz": tz_name}
    query_url = f"{api_base}/rag/query?{urllib.parse.urlencode(q)}"

    # Warm up once (ignore failures)
    try:
        _ = http_json(
            "POST",
            query_url,
            {"question": "ping", "top_k": 1, "extra_context": "{}"},
            cfg,
        )
    except Exception:
        pass

    # 3) Query backend for today's tweet
    now_dt_local = datetime.now(ZoneInfo(tz_name))
    topic_family, topic_mode = pick_topic(now_local=now_dt_local, snap_obj=snap_obj)
    req_links: list[str] = []
    req_datetime = now_dt_local.isoformat()
    question = build_question(
        max_chars=max_chars,
        topic_family=topic_family,
        topic_mode=topic_mode,
        now_local=now_dt_local,
        snap_obj=snap_obj,
        links=req_links,
        datetime=req_datetime,
    )
    include_debug=1
    audit = env_bool("RAG_AUDIT", default=True)
    audit_rewrite = env_bool("RAG_AUDIT_REWRITE", default=True)
    payload = build_payload(
        question=question,
        top_k=top_k,
        snap_obj=snap_obj,
        max_chars=max_chars,
        include_debug=include_debug,
        datetime=req_datetime,
        links=req_links,
        audit=audit,
        audit_rewrite=audit_rewrite,
    )
    print(payload)

    if debug:
        print(f"DEBUG: JSON_PAYLOAD={json.dumps(payload, ensure_ascii=False)}", file=sys.stderr)

    tweet = ""
    links: List[str] = []
    resp_obj: Dict[str, Any] = {}
    detail = ""
    # bash: retries are CURL_RETRIES+2 here
    for attempt in range(1, cfg.retries + 2 + 1):
        try:
            resp_obj = http_json("POST", query_url, payload, cfg)
        except Exception as e:
            print(f"WARN: /rag/query call failed (attempt {attempt}/{cfg.retries+2}). Retrying... err={e!r}", file=sys.stderr)
            time.sleep(2)
            continue
        
        links = extract_links(resp_obj)

        detail = extract_detail(resp_obj)
        tweet = extract_tweet(resp_obj)
        if tweet:
            break

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
    entry = build_entry(today=today, now_iso=now_iso, tweet=tweet, place=place, snap_obj=snap_obj,links=links)

    for feed_p, latest_p in pair_paths(feeds, latests):
        write_outputs(feed_p, latest_p, entry=entry, snap_json_raw=snap_json_raw, now_local=now_local)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
