#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a daily diary/feed item (text) from local JSON sources + weather, then write:
  - a timestamped feed file: FEED_PATH/feed/feed_{now_local}.json
  - a "latest" pointer:      LATEST_PATH/latest.json

Design notes:
  - We store `generated_at` in UTC ISO8601 (Z).
  - We store `date` as *local date* (TZ_NAME) so that runs around midnight JST do not lag by 1 day.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # type: ignore


# ----------------------------
# Utilities
# ----------------------------

def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def utc_now_iso_z() -> str:
    # e.g. "2025-12-27T15:17:02Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_date() -> str:
    # NOTE: kept for compatibility (rarely useful for "daily" content in JST)
    return datetime.now(timezone.utc).date().isoformat()


def local_stamp(tz_name: str) -> str:
    """Return a local timestamp string (YYYYMMDD_HHMMSS_TZ)."""
    if ZoneInfo is None:
        dt = datetime.now()
        return dt.strftime("%Y%m%d_%H%M%S_LOCAL")

    try:
        dt = datetime.now(ZoneInfo(tz_name))
        # %Z => "JST" for Asia/Tokyo
        return dt.strftime("%Y%m%d_%H%M%S_%Z")
    except Exception:
        dt = datetime.now()
        return dt.strftime("%Y%m%d_%H%M%S_LOCAL")


def local_date(tz_name: str) -> str:
    """Return today's date in the given timezone (e.g., Asia/Tokyo).

    NOTE:
      - We intentionally use *local* date for the diary/feed 'date' field.
      - Using UTC here causes the date to lag by one day for runs before 09:00 JST.
    """
    if ZoneInfo is None:
        return datetime.now().strftime("%Y-%m-%d")
    try:
        return datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def split_paths(raw: str) -> List[Path]:
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    return [Path(p) for p in parts]


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(p: Path, data: Any) -> None:
    ensure_parent(p)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^0-9A-Za-z._-]+", "", s)
    return s or "x"


def stable_hash8(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:8]


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    # keep newlines, normalize CRLF
    return str(s).replace("\r\n", "\n").replace("\r", "\n").strip()


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Place:
    title: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None


# ----------------------------
# Weather + Source loading
# ----------------------------

def read_weather_snapshot(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return load_json(path) or {}
    except Exception as ex:
        eprint(f"[warn] failed to read weather snapshot: {path} ({ex})")
        return {}


def read_source_docs(docs_dir: Path) -> List[Dict[str, Any]]:
    if not docs_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(docs_dir.glob("*.json")):
        try:
            d = load_json(p)
            if isinstance(d, dict):
                d["_file"] = p.name
                out.append(d)
        except Exception as ex:
            eprint(f"[warn] failed to read doc: {p} ({ex})")
    return out


# ----------------------------
# Feed update
# ----------------------------

def to_item(entry: Dict[str, Any], place: Optional[Place] = None) -> Dict[str, Any]:
    # Feed item schema used by frontend
    it: Dict[str, Any] = {}
    it["id"] = entry.get("id") or entry.get("generated_at") or entry.get("date") or stable_hash8(entry.get("text", ""))
    it["date"] = entry.get("date", "")
    it["generated_at"] = entry.get("generated_at", "")
    it["text"] = entry.get("text", "")

    if place:
        it["place"] = {
            "title": place.title or "",
            "lat": place.lat,
            "lon": place.lon,
        }
    else:
        it["place"] = entry.get("place") or {}

    # Optional image fields (might be patched later by workflows)
    if "image_url" in entry:
        it["image_url"] = entry["image_url"]
    if "image_prompt" in entry:
        it["image_prompt"] = entry["image_prompt"]
    if "image_generated_at" in entry:
        it["image_generated_at"] = entry["image_generated_at"]

    return it


def update_feed(feed_path: Path, entry: Dict[str, Any], place: Optional[Place] = None) -> None:
    """
    - If feed file exists: prepend new item; keep max length (FEED_MAX_ITEMS).
    - If not: create.
    """
    max_items = int(os.environ.get("FEED_MAX_ITEMS", "50"))
    data: Dict[str, Any] = {"items": []}

    if feed_path.exists():
        try:
            d = load_json(feed_path)
            if isinstance(d, dict):
                data = d
        except Exception as ex:
            eprint(f"[warn] failed to read feed, will recreate: {feed_path} ({ex})")

    items = data.get("items", [])
    if not isinstance(items, list):
        items = []

    new_item = to_item(entry, place=place)

    # Deduplicate by (date,text) within this file (keep newest first)
    def key_of(it: Dict[str, Any]) -> Tuple[str, str]:
        return (str(it.get("date", "")), normalize_text(str(it.get("text", ""))))

    seen: set = set()
    merged: List[Dict[str, Any]] = []

    merged.append(new_item)
    seen.add(key_of(new_item))

    for it in items:
        if not isinstance(it, dict):
            continue
        k = key_of(it)
        if k in seen:
            continue
        merged.append(it)
        seen.add(k)
        if len(merged) >= max_items:
            break

    data["items"] = merged
    dump_json(feed_path, data)


# ----------------------------
# Diary text generation
# ----------------------------

def build_prompt(today: str, weather: Dict[str, Any], docs: List[Dict[str, Any]]) -> str:
    # Keep prompt stable; user requested "no text in images" elsewhere but this is for text generation.
    parts: List[str] = []
    parts.append(f"Today is {today}.")
    if weather:
        # keep short
        cur = weather.get("current") or {}
        parts.append(f"Weather snapshot time: {cur.get('time','')}.")
        parts.append(f"Temperature: {cur.get('temperature_2m','')}°C. Precip: {cur.get('precipitation','')}.")
    if docs:
        parts.append("Reference docs:")
        for d in docs:
            title = d.get("title") or d.get("name") or d.get("_file") or "doc"
            detail = d.get("detail") or d.get("text") or ""
            detail = normalize_text(detail)
            if len(detail) > 800:
                detail = detail[:800] + "..."
            parts.append(f"- {title}: {detail}")
    parts.append("Write a short feed item for social posting. Output plain text only.")
    return "\n".join(parts).strip()


def generate_text(prompt: str) -> str:
    """
    Placeholder for LLM call.
    In this repo, generation may be done elsewhere; keep deterministic fallback.
    """
    # If no LLM, fallback:
    # Use first 240 chars of prompt summary.
    # (In practice, your workflow likely replaces this with an API call.)
    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    # Try to extract 1-2 doc bullets if present
    bullets = [ln[2:] for ln in lines if ln.startswith("- ")]
    out = []
    if bullets:
        out.append(bullets[0])
    else:
        out.append(lines[0] if lines else "Daily update.")
    return normalize_text("\n".join(out))[:500]


def build_entry(today: str, weather: Dict[str, Any], docs: List[Dict[str, Any]], tz_name: str) -> Dict[str, Any]:
    generated_at = utc_now_iso_z()
    prompt = build_prompt(today, weather, docs)
    text = generate_text(prompt)

    entry: Dict[str, Any] = {
        "id": generated_at,                 # stable unique id for this run
        "date": today,                      # IMPORTANT: local date (TZ_NAME)
        "generated_at": generated_at,       # UTC instant
        "text": text,
        "weather": weather or {},
        "prompt": prompt,
        "tz_name": tz_name,
    }
    return entry


# ----------------------------
# Output paths
# ----------------------------

def resolve_output_paths(now_local: str) -> Tuple[Path, Path]:
    """
    FEED_PATH:
      - if directory: write FEED_PATH/feed/feed_{now_local}.json
      - if file: write exactly FEED_PATH
    LATEST_PATH:
      - if directory: write LATEST_PATH/latest.json
      - if file: write exactly LATEST_PATH
    """
    feed_path_env = os.environ.get("FEED_PATH", "frontend/app/public")
    latest_path_env = os.environ.get("LATEST_PATH", "frontend/app/public/latest.json")

    feed_base = Path(feed_path_env)
    if feed_base.suffix.lower() == ".json":
        feed_path = feed_base
    else:
        feed_path = feed_base / "feed" / f"feed_{now_local}.json"

    latest_base = Path(latest_path_env)
    if latest_base.suffix.lower() == ".json":
        latest_path = latest_base
    else:
        latest_path = latest_base / "latest.json"

    return feed_path, latest_path


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    tz_name = os.environ.get("TZ_NAME", "Asia/Tokyo")

    docs_dir = Path(os.environ.get("DOCS_DIR", "frontend/app/public/data/json"))
    weather_snapshot_path = Path(os.environ.get("WEATHER_SNAPSHOT_PATH", "frontend/app/public/weather_snapshot.json"))

    now_local = local_stamp(tz_name)
    feed_path, latest_path = resolve_output_paths(now_local)

    weather = read_weather_snapshot(weather_snapshot_path)
    docs = read_source_docs(docs_dir)

    # IMPORTANT: use local date for "today"
    today = local_date(tz_name)

    entry = build_entry(today=today, weather=weather, docs=docs, tz_name=tz_name)

    place = None
    # If you store place info somewhere, load here.
    # place = Place(title="...", lat=..., lon=...)

    update_feed(feed_path, entry, place=place)
    dump_json(latest_path, entry)

    print(f"[ok] wrote feed:   {feed_path}")
    print(f"[ok] wrote latest: {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
