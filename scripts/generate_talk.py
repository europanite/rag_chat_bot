from __future__ import annotations

import json
import os
import re
import sys
import time
import random
import zlib
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
GARBAGE_REMINDER_URL = "https://www.city.yokosuka.kanagawa.jp/4105/kurashi/documents/english2023.pdf"

def _pick_hashtags(s: str, k: int, seed: str) -> str:
    tags = [t.strip() for t in (s or "").split() if t.strip().startswith("#")]
    uniq = []
    seen = set()
    for t in tags:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    if not uniq:
        return ""

    rng = random.Random(seed) 
    if len(uniq) <= k:
        return " ".join(uniq)
    return " ".join(rng.sample(uniq, k))

def _append_if_fit(tweet: str, extra: str, max_chars: int) -> str:
    if not extra:
        return tweet
    candidate = f"{tweet} {extra}".strip()
    return candidate if len(candidate) <= max_chars else tweet

def build_garbage_post(*, place: str, date: str) -> str:
    # URL is provided via links[] (NOT in text).
    return f"🗑️ Garbage reminder: please check {date}'s collection and sorting rules."


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



def env_int(name: str, default: int) -> int:
    v = env(name, None)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(str(v).strip())
    except Exception:
        print(f"WARN: invalid int env {name}={v!r}; using default {default}", file=sys.stderr)
        return int(default)

def env_float(name: str, default: float) -> float:
    v = env(name, None)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(str(v).strip())
    except Exception:
        print(f"WARN: invalid float env {name}={v!r}; using default {default}", file=sys.stderr)
        return float(default)

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
    # Keep line structure (greeting/weather/body). Normalize spaces per-line.
    raw = (text or "").strip()
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


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
def fetch_weather_snapshot(lat: str | float, lon: str | float, tz_name: str, place: str) -> Tuple[str, Dict[str, Any]]:
    # Calls the existing script (keeps the same behavior/format)
    cmd = [
        sys.executable,
        "scripts/fetch_weather.py",
        "--format",
        "json",
        "--lat",
        str(lat),
        "--lon",
        str(lon),
        "--tz",
        str(tz_name),
        "--place",
        str(place),
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
    # JSON from Open-Meteo should already be numeric, but be tolerant of strings.
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _as_int(x: Any) -> Optional[int]:
    # Be tolerant: numbers may appear as float or numeric strings.
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        try:
            return int(x)
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:
            return None
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


def _weather_condition_and_temp(obj: Dict[str, Any]) -> tuple[str, int]:
    """Derive a single weather condition + temperature.

    Accepts either:
      - the full snapshot dict (with key "current")
      - the "current" dict itself (already flattened)

    Output condition is one of: sunny, cloudy, rainy, windy, chilly
    """

    if not isinstance(obj, dict):
        cur = {}
    else:
        cur = obj.get("current") if isinstance(obj.get("current"), dict) else obj

    temp_c = _as_float(cur.get("temp_c"))
    precip_mm = _as_float(cur.get("precip_mm"))
    cloud_pct = _as_float(cur.get("cloud_cover_pct"))
    wind_kmh = _as_float(cur.get("wind_kmh"))
    weather_code = _as_int(cur.get("weather_code"))

    # Priority: rainy > windy > chilly > sunny > cloudy
    is_rainy = False
    if precip_mm is not None and precip_mm > 0.1:
        is_rainy = True
    if weather_code is not None:
        # WMO weather codes: treat any precipitation / thunder / snow as "rainy"
        if 51 <= weather_code <= 67:
            is_rainy = True
        if 71 <= weather_code <= 86:
            is_rainy = True
        if 95 <= weather_code <= 99:
            is_rainy = True

    if is_rainy:
        condition = "rainy"
    elif wind_kmh is not None and wind_kmh >= 25.0:
        condition = "windy"
    elif temp_c is not None and temp_c <= 5.0:
        condition = "chilly"
    else:
        is_sunny = False
        if weather_code is not None and weather_code == 0:
            is_sunny = True
        if cloud_pct is not None and cloud_pct <= 20.0:
            is_sunny = True
        condition = "sunny" if is_sunny else "cloudy"

    temp_i = int(round(temp_c)) if temp_c is not None else 0
    return condition, temp_i


def _weather_brief_for_llm(snap_obj: Dict[str, Any]) -> str:
    """Return a *minimal* weather summary for the LLM.

    We intentionally pass only:
      - condition: one of {sunny, cloudy, rainy, windy, chilly}
      - temp_c: integer Celsius

    Everything else is used only to derive the condition, but is NOT forwarded.
    """

    condition, temp_i = _weather_condition_and_temp(snap_obj)
    return f"weather={condition}\ntemp_c={temp_i}\n"


def _topic_prompt_for_kind(*, kind: str, weather: str, temp_c: int) -> Tuple[str, str]:
    """Return (main_sentence, preference_line) for the prompt.

    The goal is to make the retrieval query *specific* (event vs restaurant vs spot vs activity),
    so the vector search does not keep hitting the same generic chunk.
    """
    k = (kind or "").strip().lower()
    w = (weather or "").strip().lower()

    # Simple comfort heuristics to nudge the model without forcing unsupported claims.
    chilly = (w == "chilly") or (isinstance(temp_c, int) and temp_c <= 7)
    rainy = (w == "rainy")
    windy = (w == "windy")
    sunny = (w == "sunny")

    if k == "restaurant":
        pref = "a cozy place / warm food" if (rainy or chilly) else ("a place with a nice view" if sunny else "a popular local place")
        main = "Introduce ONE local restaurant (or cafe)."
        return main, f"   - Prefer {pref}.\n"

    if k == "event":
        pref = "an indoor event" if rainy else "a fun upcoming event"
        main = "Introduce ONE upcoming event."
        return main, f"   - Prefer {pref} if possible.\n"

    if k == "activity":
        pref = "an indoor activity idea" if rainy else "an outdoor activity idea"
        main = "Introduce ONE local activity idea (something to do today)."
        return main, f"   - Prefer {pref}.\n"

    # Default: spot / attraction
    pref = "an indoor-friendly spot" if rainy else ("a scenic spot" if not windy else "a spot sheltered from wind")
    main = "Introduce ONE local spot/attraction."
    return main, f"   - Prefer {pref}.\n"


def _compute_variety_seed(*, now_local: datetime, scheduled_kind: str, snap_obj: Optional[Dict[str, Any]]) -> int:
    """Derive a stable-ish seed that changes across time-of-day, weather, and scheduled kind.

    This helps the backend's 'anchor reordering' pick different chunks even when the same
    generic context would otherwise be returned.
    """
    base = env_int("TOPIC_VARIANT", 0)
    cur = (snap_obj or {}).get("current") or {}
    tod = _time_of_day_bucket(now_local.hour)
    season = _season_bucket(now_local.month)
    condition, temp_i = _weather_condition_and_temp(cur)

    key = f"{now_local:%Y%m%d%H}|{scheduled_kind}|{tod}|{season}|{condition}|{temp_i}"
    h = zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF
    return int((base + h) & 0x7FFFFFFF)


def build_question(*, now_local: datetime, snap_obj: Optional[Dict[str, Any]], scheduled_kind: str) -> str:
    cur = (snap_obj or {}).get("current") or {}
    tod = _time_of_day_bucket(now_local.hour)
    season = _season_bucket(now_local.month)
    condition, temp_i = _weather_condition_and_temp(cur)

    main_sentence, pref_line = _topic_prompt_for_kind(kind=scheduled_kind, weather=condition, temp_c=temp_i)

    # Fallback instruction depends on the scheduled kind.
    if (scheduled_kind or "").strip().lower() == "event":
        fallback = (
            "If you cannot find a future event in the RAG Context, write about a local spot or restaurant instead (still from RAG Context).\n"
        )
    else:
        fallback = (
            f"If you cannot find a good {scheduled_kind} in the RAG Context, write about a local spot instead (still from RAG Context).\n"
        )

    return (
        "Write a tweet in English.\n"
        "Follow this format EXACTLY (EXACTLY 3 sentences, single paragraph, no line breaks, no URLs in the text):\n"
        "1) Greeting sentence: Good morning/Good afternoon/Good evening/Good night (match HINTS.time_of_day).\n"
        "2) Weather sentence: '<sunny|cloudy|windy|chilly|rainy> with your feelings'.\n"
        f"3) Main sentence: {main_sentence} from ONLY the RAG context.\n"
        "   - Mention the name explicitly.\n"
        f"{pref_line}"
        "   - Make it fit the situation (HINTS.weather/temp/time_of_day/season).\n"
        "Rules: Use emojis. Keep it punchy.\n"
        "Do NOT mention past events.\n"
        f"{fallback}"
        f"TOPIC_FAMILY: {scheduled_kind}\n"
        f"datetime: {now_local}.\n"
        f"HINTS: topic_kind={scheduled_kind}, time_of_day={tod}, season={season}, weather={condition}, temp_c={temp_i}.\n"
    )

def build_payload(
    question: str,
    top_k: int,
    snap_obj: Dict[str, Any] | None,
    max_chars: int,
    include_debug: bool,
    blocked_urls: List[str] | None = None,
    blocked_terms: List[str] | None = None,
    datetime: str | None = None,
    seed: int | None = None,
    anchor_top_n: int | None = None,
) -> dict:
    # Send only a compact weather summary (condition words + temp) to the LLM.

    # anchor selection for variety should look beyond top_k. Default is bigger than top_k, but still capped by backend.
    default_anchor_top_n = max(8, min(20, int(top_k or 5) * 3))
    anchor_top_n_final = int(anchor_top_n) if anchor_top_n is not None else env_int("RAG_ANCHOR_TOP_N", default_anchor_top_n)
    seed_final = int(seed) if seed is not None else env_int("TOPIC_VARIANT", 0)

    payload: dict = {
        "question": question,
        "top_k": top_k,
        "max_chars": max_chars,
        "include_debug": include_debug,
        "extra_context": _weather_brief_for_llm(snap_obj),
        "datetime": datetime,
        "variety": env_float("RAG_VARIETY", 0.0),
        "seed": seed_final,
        "anchor_top_n": anchor_top_n_final,
    }
    if blocked_urls:
        payload["blocked_urls"] = [u for u in blocked_urls if isinstance(u, str) and u.strip()]
    if blocked_terms:
        payload["blocked_terms"] = [t for t in blocked_terms if isinstance(t, str) and t.strip()]
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
    return links[:1]


def _read_json_file(path: Path) -> Any:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            return None
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            # tolerate accidental multiple JSON objects in one file
            objs = parse_json_objects(txt)
            return objs[-1] if objs else None
    except Exception:
        return None

def _urls_from_item(obj: Any) -> List[str]:
    if not isinstance(obj, dict):
        return []
    urls: List[str] = []
    urls.extend(extract_links(obj))
    # Legacy/alternate shapes
    link = obj.get("link")
    if isinstance(link, str) and link.strip():
        urls.append(link.strip())
    url = obj.get("url")
    if isinstance(url, str) and url.strip():
        urls.append(url.strip())
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u2 = u.strip()
        if not u2 or u2 in seen:
            continue
        seen.add(u2)
        out.append(u2)
    return out

def collect_recent_urls(latest_paths: List[str], feed_paths: List[str], max_files: int = 20, max_urls: int = 64) -> List[str]:
    """Collect recently used URLs from latest.json and recent feed_*.json files (to avoid repeats)."""
    seen: set[str] = set()
    out: List[str] = []

    def add(u: str) -> None:
        if not u or u in seen:
            return
        seen.add(u)
        out.append(u)

    # 1) latest.json first (usually the most recent)
    for lp in latest_paths:
        p = Path(lp)
        if not p.exists():
            continue
        obj = _read_json_file(p)
        for u in _urls_from_item(obj):
            add(u)
        if len(out) >= max_urls:
            return out[:max_urls]

    # 2) then recent feed snapshots
    dirs: List[Path] = []
    dir_seen: set[Path] = set()
    for fp in feed_paths:
        d = Path(fp).parent
        if d not in dir_seen:
            dir_seen.add(d)
            dirs.append(d)
    for lp in latest_paths:
        d = Path(lp).parent / "feed"
        if d not in dir_seen:
            dir_seen.add(d)
            dirs.append(d)

    for d in dirs:
        if not d.exists():
            continue
        files = sorted(d.glob("feed_*.json"))
        if not files:
            continue
        for f in reversed(files[-max_files:]):
            obj = _read_json_file(f)
            for u in _urls_from_item(obj):
                add(u)
            if len(out) >= max_urls:
                return out[:max_urls]

    return out[:max_urls]

def extract_required_mention(resp_obj: Any) -> str:
    if not isinstance(resp_obj, dict):
        return ""
    # Prefer top-level fields (Phase5 backend), fallback to debug
    rm = resp_obj.get("required_mention")
    if isinstance(rm, str) and rm.strip():
        return rm.strip()
    dbg = resp_obj.get("debug")
    if isinstance(dbg, dict):
        rm2 = dbg.get("required_mention")
        if isinstance(rm2, str) and rm2.strip():
            return rm2.strip()
    return ""

_BLOCK_TERM_STOP = {
    "yokosuka",
    "yokosuka city",
    "miura peninsula",
    "kanagawa",
    "japan",
    "tokyo",
    "vrchat",
    "metaverse yokosuka",
}

def _normalize_term(s: str) -> str:
    s2 = (s or "").strip()
    s2 = re.sub(r"\s+", " ", s2)
    return s2

def _term_ok(term: str) -> bool:
    t = _normalize_term(term)
    if not t:
        return False
    low = t.lower()
    if low in _BLOCK_TERM_STOP:
        return False
    # Avoid blocking too-short generic ASCII tokens (but allow short JP names)
    if len(low) < 3 and all("a" <= ch <= "z" for ch in low):
        return False
    return True

def _terms_from_item(obj: Any) -> List[str]:
    if not isinstance(obj, dict):
        return []
    out: List[str] = []
    rm = obj.get("required_mention")
    if isinstance(rm, str) and rm.strip():
        out.append(rm.strip())
    meta = obj.get("meta")
    if isinstance(meta, dict):
        rm2 = meta.get("required_mention")
        if isinstance(rm2, str) and rm2.strip():
            out.append(rm2.strip())
    tp = obj.get("topic")
    if isinstance(tp, str) and tp.strip():
        out.append(tp.strip())

    seen: set[str] = set()
    uniq: List[str] = []
    for t in out:
        nt = _normalize_term(t)
        if not _term_ok(nt):
            continue
        if nt in seen:
            continue
        seen.add(nt)
        uniq.append(nt)
    return uniq

def collect_recent_terms(latest_paths: List[str], feed_paths: List[str], max_files: int = 20, max_terms: int = 32) -> List[str]:
    """Collect recently used topic terms (required_mention) from latest.json and feed snapshots."""
    seen: set[str] = set()
    out: List[str] = []

    def add(t: str) -> None:
        if not t:
            return
        nt = _normalize_term(t)
        if not _term_ok(nt):
            return
        if nt in seen:
            return
        seen.add(nt)
        out.append(nt)

    for lp in latest_paths:
        p = Path(lp)
        if not p.exists():
            continue
        obj = _read_json_file(p)
        for t in _terms_from_item(obj):
            add(t)
        if len(out) >= max_terms:
            return out[:max_terms]

    dirs: List[Path] = []
    dir_seen: set[Path] = set()
    for fp in feed_paths:
        d = Path(fp).parent
        if d not in dir_seen:
            dir_seen.add(d)
            dirs.append(d)
    for lp in latest_paths:
        d = Path(lp).parent / "feed"
        if d not in dir_seen:
            dir_seen.add(d)
            dirs.append(d)

    for d in dirs:
        if not d.exists():
            continue
        files = sorted(d.glob("feed_*.json"))
        if not files:
            continue
        for f in reversed(files[-max_files:]):
            obj = _read_json_file(f)
            for t in _terms_from_item(obj):
                add(t)
            if len(out) >= max_terms:
                return out[:max_terms]

    return out[:max_terms]

def extract_detail(resp_obj: Dict[str, Any]) -> str:
    d = resp_obj.get("detail")
    if isinstance(d, (list, dict)):
        return json.dumps(d, ensure_ascii=False)
    if d is None:
        return ""
    return str(d)


def build_entry(
        today: str, 
        now_iso: str, 
        tweet: str, 
        place: str, 
        snap_obj: Dict[str, Any], 
        links: Optional[List[str]] = None, 
        required_mention: str = "", 
        kind: str = "", 
        image_fixed: str = "",
        avatar_image: str = "",
        ) -> Dict[str, Any]:
    if not today or not now_iso or not tweet:
        missing = [k for k, v in [("today", today), ("now_iso", now_iso), ("tweet", tweet)] if not v]
        raise RuntimeError(f"missing values for entry: {', '.join(missing)}")
    if not links:
        links = []

    obj = {
         "date": today,
         "generated_at": now_iso,
         "text": tweet,
         "place": place or "",
         "weather": snap_obj,
         "links": links,
         "required_mention": (required_mention or "").strip(),
     }
    if (avatar_image or "").strip():
        obj["avatar_image"] = str(avatar_image).strip()
    if (kind or "").strip():
        obj["kind"] = str(kind).strip()
    if (image_fixed or "").strip():
        obj["image_fixed"] = str(image_fixed).strip()
    return obj

def build_fixed_post_entry(
        *, 
        today: str, 
        now_iso: str, 
        tweet: str, 
        place: str, 
        snap_obj: Dict[str, Any], 
        links: Optional[List[str]] = None, 
        kind: str, 
        fixed_image: str,
        avatar_image: str = "",
        ) -> Dict[str, Any]:
    """Build an entry that uses a fixed image (copied by scripts/illustrate.py) instead of text2img."""
    return build_entry(
        today=today,
        now_iso=now_iso,
        tweet=tweet,
        place=place,
        snap_obj=snap_obj,
        links=links,
        kind=kind,
        image_fixed=fixed_image,
        avatar_image=avatar_image,
    )

def to_item(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    date = str(entry.get("date") or "")
    text = str(entry.get("text") or "")
    if not date or not text:
        return None
    _id = entry.get("id") or entry.get("generated_at") or date
    place = entry.get("place") or ""
    item = {
         "id": str(_id),
         "date": date,
         "text": text,
         "place": str(place),
         "generated_at": entry.get("generated_at"),
         "weather": entry.get("weather"),
         "links": entry.get("links") or [],
         "required_mention": (entry.get("required_mention") or "").strip(),
     }
    if (entry.get("kind") or "").strip():
        item["kind"] = str(entry.get("kind")).strip()
    if (entry.get("image_fixed") or "").strip():
        item["image_fixed"] = str(entry.get("image_fixed")).strip()
    if (entry.get("avatar_image") or "").strip():
        item["avatar_image"] = str(entry.get("avatar_image")).strip()
    return item


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

def collect_recent_primary_urls(feed_dir: Path, *, max_files: int = 40, max_urls: int = 25) -> list[str]:
    urls: list[str] = []
    if not feed_dir.exists():
        return urls
    files = sorted(feed_dir.glob("feed_*.json"), reverse=True)[:max_files]
    for p in files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            links = obj.get("links") or []
            for u in links:
                if isinstance(u, str) and u.startswith("http"):
                    if u not in urls:
                        urls.append(u)
                if len(urls) >= max_urls:
                    return urls
        except Exception:
            continue
    return urls

def main() -> int:
    debug = env("DEBUG", "0") == "1"
    cfg = HttpConfig(
        max_time_s=int(env("CURL_MAX_TIME")),
        retries=int(env("CURL_RETRIES")),
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
    top_k = int(env("RAG_TOP_K"))
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


    # Schedule kind (python-side)
    now_dt_local = datetime.now(ZoneInfo(tz_name))
    scheduled_kind = env("SCHEDULED_KIND").strip()

    if debug:
        print(
            f"DEBUG: now_local={now_dt_local.isoformat()} scheduled_kind={scheduled_kind}",
            file=sys.stderr,
        )

    # garbage posts: DO NOT call RAG. Just reminder + 1 URL in links[].
    if scheduled_kind == "garbage_tomorrow":
        tweet = build_garbage_post(place=place, date="tomorrow")
        links = [GARBAGE_REMINDER_URL]

        if hashtags and "#" not in tweet:
            picked = _pick_hashtags(hashtags, k=3, seed=now_local)
            tweet = _append_if_fit(tweet, picked, max_chars)

        today = utc_date()
        now_iso = utc_now_iso_z()
        entry = build_fixed_post_entry(
            today=today, 
            now_iso=now_iso, 
            tweet=tweet, 
            place=place, 
            snap_obj=snap_obj, 
            links=links, 
            kind=scheduled_kind, 
            fixed_image="img/cheetsheet.png",
            avatar_image=env("AVATAR_IMAGE", "image/avatar/normal.png"),
            )
        for feed_path, latest_path in pair_paths(feeds, latests):
            write_outputs(feed_path=feed_path, latest_path=latest_path, entry=entry, snap_json_raw=snap_json_raw, now_local=now_local)
        return 0
    
    if scheduled_kind == "garbage_today":
        tweet = build_garbage_post(place=place, date="today")
        links = [GARBAGE_REMINDER_URL]
            
        if hashtags and "#" not in tweet:
            picked = _pick_hashtags(hashtags, k=3, seed=now_local)
            tweet = _append_if_fit(tweet, picked, max_chars)

        today = utc_date()
        now_iso = utc_now_iso_z()
        entry = build_fixed_post_entry(
            today=today, 
            now_iso=now_iso, 
            tweet=tweet, 
            place=place, 
            snap_obj=snap_obj, 
            links=links, 
            kind=scheduled_kind, 
            fixed_image="image/avatar/garbage.png",
            avatar_image=env("AVATAR_IMAGE", "image/avatar/normal.png"),
            )
        for feed_path, latest_path in pair_paths(feeds, latests):
            write_outputs(feed_path=feed_path, latest_path=latest_path, entry=entry, snap_json_raw=snap_json_raw, now_local=now_local)
        return 0

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

    # Warm up once: DO NOT bypass build_payload; keep request schema consistent.
    # Avoid repeating the same URL/content across runs by blocking recently used links
    blocked_urls = collect_recent_urls(
        latest_paths=latests,
        feed_paths=feeds,
        max_files=env_int("RAG_BLOCKED_FILES_MAX", 20),
        max_urls=env_int("RAG_BLOCKED_URLS_MAX", 64),
    )
    blocked_terms = collect_recent_terms(
        latest_paths=latests,
        feed_paths=feeds,
        max_files=env_int("RAG_BLOCKED_FILES_MAX", 20),
        max_terms=env_int("RAG_BLOCKED_TERMS_MAX", 32),
    )
    warm_payload = build_payload(
        question="ping",
        top_k=1,
        snap_obj=snap_obj,
        max_chars=max_chars,
        include_debug=True,
        blocked_urls=blocked_urls,
        blocked_terms=blocked_terms,
        datetime=now_dt_local.isoformat(),
    )
    # Hard fail if payload shape regresses again
    if warm_payload.get("max_chars") is None:
        raise RuntimeError("BUG: warmup payload missing max_chars")
    _ = http_json("POST", query_url, warm_payload, cfg)

    question = build_question(
        now_local=now_dt_local,
        snap_obj=snap_obj,
        scheduled_kind=scheduled_kind,
    )

    include_debug = 1 if debug else 0

    variety_seed = _compute_variety_seed(now_local=now_dt_local, scheduled_kind=scheduled_kind, snap_obj=snap_obj)

    payload = build_payload(
        question=question,
        top_k=top_k,
        snap_obj=snap_obj,
        max_chars=max_chars,
        include_debug=include_debug,
        seed=variety_seed,
        blocked_urls=blocked_urls,
        blocked_terms=blocked_terms,
        datetime=str(now_dt_local), 
    )

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
        picked = _pick_hashtags(hashtags, k=3, seed=now_local)
        tweet = _append_if_fit(tweet, picked, max_chars)

    # 4) Write outputs
    today = utc_date()
    now_iso = utc_now_iso_z()
    required_mention = extract_required_mention(resp_obj)
    entry = build_entry(
        today=today, 
        now_iso=now_iso, 
        tweet=tweet, 
        place=place, 
        snap_obj=snap_obj,
        links=links,
        required_mention=required_mention,
        kind=scheduled_kind,
        avatar_image=env("AVATAR_IMAGE", f"image/avatar/normal.png"),
    )

    for feed_p, latest_p in pair_paths(feeds, latests):
        write_outputs(feed_p, latest_p, entry=entry, snap_json_raw=snap_json_raw, now_local=now_local)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
