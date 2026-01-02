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


def build_question(max_words: str, topic_family: str, topic_mode: str, now_local: datetime, snap_obj: Dict[str, Any]) -> str:
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
        ("event", "seasonal"): "seasonal holiday christmas new year event",
        # chat
        ("chat", "food"): "curry ramen cafe bakery warm drink",
        ("chat", "trivia"): "fun fact local tip small story",
        ("chat", "history"): "history navy port heritage museum",
        ("chat", "activity"): "running fishing hike workout",
    }
    topic_keywords = keyword_map.get((topic_family, topic_mode), "local tip short story")

    return (
        "Write a tweet in English.\n"
        f"NOW (local, reference): {now_local.strftime('%Y-%m-%d %H:%M')} {now_local.tzname() or ''} ({now_local:%a}).\n"
        "TIME & GREETING (IMPORTANT):\n"
        "- Determine the local datetime from LIVE WEATHER JSON.\n"
        "- Prefer LIVE WEATHER.current.time and LIVE WEATHER.timezone.\n"
        "- HOLIDAY OVERRIDE (date-based, day-limited):\n"
        "  * 12-24 => 'Merry Christmas Eve'\n"
        "  * 12-25 => 'Merry Christmas'\n"
        "  * 12-31 => \"Happy New Year's Eve\"\n"
        "  * from 01-01  to 01-04 => 'Happy New Year'\n"
        "  If today's local date matches one of these, start with that greeting and do NOT use the hour-based greetings.\n"
        "- Otherwise, start with exactly one greeting based on local hour:\n"
        "  * 05:00-11:59 => 'Good morning'\n"
        "  * 12:00-16:59 => 'Good afternoon'\n"
        "  * 17:00-23:59 => 'Good evening'\n"
        "  * 00:00-04:59 => 'Good night'\n"
        "\n"
        "Decide the greeting using the local time in LIVE WEATHER JSON (current.time + timezone; assume Asia/Tokyo if missing).\n"
        "Summarize the weather using ONLY LIVE WEATHER facts.\n"
        "If you use words like 'tonight', 'this evening', 'later tonight', 'later today', they must match NOW.\n"
        "If the event date is not today, say “tomorrow” or include an explicit date (e.g., Dec 31).\n"
        f"TOPIC FAMILY: {topic_family} (event/place/chat).\n"
        f"SUBTOPIC: {topic_mode} (keywords: {topic_keywords}).\n"
        f"HINTS: time_of_day={tod}, season={season}, weather_hint={hint}.\n"
        "Pick up ONE topic and mention only that one from RAG Context that fits the HINTS.\n"
        "You may include at most one official URL only if it exists in the chosen text.\n"
        f"Keep within {max_words} characters.\n"
    )


def build_payload(question: str, top_k: int, snap_json_raw: str) -> Dict[str, Any]:
    # Keep current bash behavior: send snapshot as extra_context string;
    return {
        "question": question,
        "top_k": int(top_k),
        "extra_context": snap_json_raw,
    }


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
        links = ""
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
        "permalink": entry.get("permalink") or f"./?post={urllib.parse.quote(str(_id), safe='')}",
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

    # Make permalink stable by using feed filename stem as the canonical post id.
    post_id = fp.stem
    entry_out = dict(entry)
    entry_out["id"] = post_id
    entry_out["permalink"] = f"./?post={urllib.parse.quote(str(post_id), safe='')}"

    entry_txt = json.dumps(entry_out, ensure_ascii=False, indent=2) + "\n"
    lp.write_text(entry_txt, encoding="utf-8")
    update_feed(fp, entry_out)

    # Also write weather snapshot next to latest (for debugging / transparency)
    snap_path = lp.parent / "snapshot" / f"snapshot_{now_local}.json"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps(entry_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    max_words = env("MAX_WORDS", "128") or "128"
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
    print(f"TOP_K={top_k} MAX_WORDS={max_words}")
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
            {"question": "ping", "top_k": 1, "extra_context": "{}"},
            cfg,
        )
    except Exception:
        pass

    # 3) Query backend for today's tweet
    now_dt_local = datetime.now(ZoneInfo(tz_name))
    topic_family, topic_mode = pick_topic(now_local=now_dt_local, snap_obj=snap_obj)
    question = build_question(max_words=max_words, topic_family=topic_family, topic_mode=topic_mode, now_local=now_dt_local, snap_obj=snap_obj)
    payload = build_payload(question=question, top_k=top_k, snap_json_raw=snap_json_raw)

    if debug:
        print(f"DEBUG: JSON_PAYLOAD={json.dumps(payload, ensure_ascii=False)}", file=sys.stderr)

    tweet = ""
    links: List[str] = []
    resp_obj: Dict[str, Any] = {}
    detail = ""
    # bash: retries are CURL_RETRIES+2 here
    for attempt in range(1, cfg.retries + 2 + 1):
        try:
            resp_obj = http_json("POST", f"{api_base}/rag/query", payload, cfg)
        except Exception as e:
            print(f"WARN: /rag/query call failed (attempt {attempt}/{cfg.retries+2}). Retrying... err={e!r}", file=sys.stderr)
            time.sleep(2)
            continue

        tweet = extract_tweet(resp_obj)
        if tweet:
            break
        
        links = extract_links(resp_obj)

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
    entry = build_entry(today=today, now_iso=now_iso, tweet=tweet, place=place, snap_obj=snap_obj,links=links)

    for feed_p, latest_p in pair_paths(feeds, latests):
        write_outputs(feed_p, latest_p, entry=entry, snap_json_raw=snap_json_raw, now_local=now_local)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
