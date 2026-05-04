#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

try:
    from artifact_paths import artifact_feed_dir, artifact_latest_path, artifact_public_dir
except ImportError:
    from scripts.artifact_paths import artifact_feed_dir, artifact_latest_path, artifact_public_dir


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"weather json is not an object: {path}")
    return obj


def utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def local_now(tz_name: str) -> datetime:
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        return datetime.now()


def local_stamp(now_dt: datetime) -> str:
    return now_dt.strftime("%Y%m%d_%H%M%S")


def first_text(*values: Any) -> str:
    for value in values:
        s = str(value or "").strip()
        if s:
            return s
    return ""


def greeting_for_hour(hour: int) -> str:
    if 5 <= hour <= 10:
        return "Good morning."
    if 11 <= hour <= 16:
        return "Good afternoon."
    if 17 <= hour <= 21:
        return "Good evening."
    return "Good night."


def condition_from_code(code: Any) -> str:
    try:
        c = int(code)
    except Exception:
        return "weather update"

    if c == 0:
        return "sunny"
    if c in (1, 2, 3):
        return "cloudy"
    if 45 <= c <= 48:
        return "foggy"
    if 51 <= c <= 67 or 80 <= c <= 82:
        return "rainy"
    if 71 <= c <= 77 or 85 <= c <= 86:
        return "snowy"
    if 95 <= c <= 99:
        return "stormy"
    return "weather update"


def normalize_condition(day: Dict[str, Any], current: Dict[str, Any]) -> str:
    raw = first_text(
        day.get("weather_text_en"),
        day.get("weather_text"),
        day.get("weather_text_ja"),
        current.get("weather_text_en"),
        current.get("weather_text"),
        current.get("weather_text_ja"),
    ).lower()

    if raw:
        if "雷" in raw or "thunder" in raw or "storm" in raw:
            return "stormy"
        if "雪" in raw or "snow" in raw:
            return "snowy"
        if "雨" in raw or "rain" in raw or "shower" in raw:
            return "rainy"
        if "霧" in raw or "fog" in raw or "mist" in raw:
            return "foggy"
        if "晴" in raw or "clear" in raw or "sun" in raw:
            return "sunny"
        if "曇" in raw or "くもり" in raw or "cloud" in raw:
            return "cloudy"

    return condition_from_code(first_text(day.get("weather_code"), current.get("weather_code")))


def overview_sentence(condition: str, place: str) -> str:
    city = place.split(",")[0].strip() or "Yokosuka"

    if condition == "sunny":
        return f"{city} looks bright today, so it feels like a good day to be outside."
    if condition == "cloudy":
        return f"{city} looks mostly cloudy today, with a calm and subdued mood."
    if condition == "rainy":
        return f"{city} looks rainy today, so outdoor plans may need a little flexibility."
    if condition == "windy":
        return f"{city} looks breezy today, especially around the waterfront."
    if condition == "foggy":
        return f"{city} looks hazy today, so the scenery may feel softer than usual."
    if condition == "snowy":
        return f"{city} looks wintry today, so it is better to move carefully."
    if condition == "stormy":
        return f"{city} looks unsettled today, so indoor plans may be the safer choice."

    return f"{city} has a changing sky today, so it is worth keeping plans flexible."


def closing_sentence(condition: str) -> str:
    if condition == "sunny":
        return "A short walk, a local lunch, or a harbor stop should fit the day well."
    if condition == "cloudy":
        return "A relaxed walk, a cafe break, or a quiet local stop should fit the mood."
    if condition == "rainy":
        return "A restaurant guide, shopping street, or indoor stop may work better today."
    if condition == "windy":
        return "It may be better to choose a route with easy breaks along the way."
    if condition in {"foggy", "snowy", "stormy"}:
        return "It may be better to keep the plan simple and leave room to adjust."
    return "A flexible local plan should work best today."


def build_text(snapshot: Dict[str, Any], place: str, now_dt: datetime) -> str:
    current = snapshot.get("current") if isinstance(snapshot.get("current"), dict) else {}
    today = snapshot.get("today") if isinstance(snapshot.get("today"), dict) else {}

    condition = normalize_condition(today, current)
    return " ".join(
        [
            greeting_for_hour(now_dt.hour),
            overview_sentence(condition, place),
            closing_sentence(condition),
        ]
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Convert weather snapshot JSON into latest/feed JSON.")
    p.add_argument("--weather-json", required=True)
    p.add_argument("--place", default="Yokosuka, Kanagawa, JP")
    p.add_argument("--tz", default=os.environ.get("TZ_NAME", "Asia/Tokyo"))
    p.add_argument("--avatar-image", default="image/avatar/weather.png")
    p.add_argument("--public-dir", default="")
    p.add_argument("--latest-path", default="")
    args = p.parse_args()

    public_dir = Path(args.public_dir) if args.public_dir.strip() else artifact_public_dir()
    latest_path = Path(args.latest_path) if args.latest_path.strip() else artifact_latest_path(public_dir)
    feed_dir = artifact_feed_dir(public_dir)

    snapshot = load_json(Path(args.weather_json))
    generated_at = utc_now_iso_z()
    now_dt = local_now(args.tz)
    feed_stem = f"feed_{local_stamp(now_dt)}"

    place = first_text(args.place, snapshot.get("place"), "Yokosuka, Kanagawa, JP")
    text = build_text(snapshot, place, now_dt)

    today = snapshot.get("today") if isinstance(snapshot.get("today"), dict) else {}
    current = snapshot.get("current") if isinstance(snapshot.get("current"), dict) else {}

    entry: Dict[str, Any] = {
        "id": feed_stem,
        "date": generated_at,
        "generated_at": generated_at,
        "place": place,
        "kind": "weather",
        "category": "weather",
        "selection_mode": "weather_overview",
        "text": text,
        "title": "Yokosuka weather overview",
        "summary": text,
        "avatar_image": args.avatar_image,
        "weather": {
            "source": snapshot.get("source"),
            "current": current,
            "today": today,
            "tomorrow": snapshot.get("tomorrow"),
        },
    }

    feed_path = feed_dir / f"{feed_stem}.json"
    write_json(feed_path, entry)
    write_json(latest_path, entry)

    print(f"[weather] wrote {latest_path}")
    print(f"[weather] wrote {feed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
