#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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


def local_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def fmt_num(value: Any, unit: str = "", digits: int = 0) -> str:
    try:
        n = float(value)
    except Exception:
        return ""
    if digits <= 0:
        s = str(int(round(n)))
    else:
        s = f"{n:.{digits}f}"
    return f"{s}{unit}"


def first_text(*values: Any) -> str:
    for value in values:
        s = str(value or "").strip()
        if s:
            return s
    return ""


def condition_label(day: Dict[str, Any]) -> str:
    ja = first_text(day.get("weather_text_ja"))
    if ja:
        return ja
    code = day.get("weather_code")
    try:
        c = int(code)
    except Exception:
        return "weather update"
    if c == 0:
        return "clear"
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


def build_text(snapshot: Dict[str, Any], place: str) -> str:
    current = snapshot.get("current") if isinstance(snapshot.get("current"), dict) else {}
    today = snapshot.get("today") if isinstance(snapshot.get("today"), dict) else {}
    tomorrow = snapshot.get("tomorrow") if isinstance(snapshot.get("tomorrow"), dict) else {}

    temp = fmt_num(current.get("temp_c"), "°C")
    humidity = fmt_num(current.get("humidity_pct"), "%")
    precip_now = fmt_num(current.get("precip_mm"), "mm", 1)
    wind = fmt_num(current.get("wind_kmh"), "km/h")
    today_max = fmt_num(today.get("temp_max_c"), "°C")
    today_min = fmt_num(today.get("temp_min_c"), "°C")
    pop = fmt_num(today.get("precip_prob_pct"), "%")
    tomorrow_label = condition_label(tomorrow)

    parts = []
    headline = f"Weather note for {place}: {condition_label(today)}"
    if temp:
        headline += f", currently around {temp}"
    headline += "."
    parts.append(headline)

    details = []
    if today_min or today_max:
        details.append(f"today's range {today_min or '?'}–{today_max or '?'}")
    if pop:
        details.append(f"rain chance {pop}")
    if humidity:
        details.append(f"humidity {humidity}")
    if precip_now:
        details.append(f"recent rain {precip_now}")
    if wind:
        details.append(f"wind {wind}")

    if details:
        parts.append("Key points: " + ", ".join(details) + ".")

    if tomorrow_label and tomorrow_label != "weather update":
        parts.append(f"Tomorrow looks like {tomorrow_label}, so it is worth checking again before going out.")

    return " ".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description="Convert weather snapshot JSON into latest/feed JSON.")
    p.add_argument("--weather-json", required=True)
    p.add_argument("--place", default="Yokosuka, Kanagawa, JP")
    p.add_argument("--avatar-image", default="image/avatar/weather.png")
    p.add_argument("--public-dir", default="")
    p.add_argument("--latest-path", default="")
    args = p.parse_args()

    public_dir = Path(args.public_dir) if args.public_dir.strip() else artifact_public_dir()
    latest_path = Path(args.latest_path) if args.latest_path.strip() else artifact_latest_path(public_dir)
    feed_dir = artifact_feed_dir(public_dir)

    snapshot = load_json(Path(args.weather_json))
    generated_at = utc_now_iso_z()
    now_local = local_stamp()
    feed_stem = f"feed_{now_local}"

    place = first_text(args.place, snapshot.get("place"), "Yokosuka, Kanagawa, JP")
    text = build_text(snapshot, place)

    today = snapshot.get("today") if isinstance(snapshot.get("today"), dict) else {}
    current = snapshot.get("current") if isinstance(snapshot.get("current"), dict) else {}

    entry: Dict[str, Any] = {
        "id": feed_stem,
        "date": generated_at,
        "generated_at": generated_at,
        "place": place,
        "kind": "weather",
        "category": "weather",
        "selection_mode": "weather_snapshot",
        "text": text,
        "title": "Yokosuka weather note",
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
