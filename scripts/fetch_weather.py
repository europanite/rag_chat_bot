#!/usr/bin/env python3
"""Fetch a small weather snapshot from Open-Meteo (stdlib-only).

This writes a compact JSON snapshot that can be passed to the backend as
`extra_context` (client-provided live weather).

Example:
  python scripts/fetch_weather.py \
    --lat 35.2810 --lon 139.6720 --tz Asia/Tokyo --place Yokosuka \
    --format json --out frontend/app/public/weather_snapshot.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"


def http_get_json(url: str, timeout_s: int = 30) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "rag-bot-container-weather/1.0 (+github-actions)",
            "Accept": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from Open-Meteo: {e}") from e


def build_url(lat: float, lon: float, tz: str) -> str:
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "timezone": tz,
        "forecast_days": "2",
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),
        "daily": ",".join(
            [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "wind_speed_10m_max",
            ]
        ),
    }
    return f"{OPEN_METEO_BASE}?{urllib.parse.urlencode(params)}"


def make_snapshot(raw: Dict[str, Any], *, lat: float, lon: float, tz: str, place: str) -> Dict[str, Any]:
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    current = raw.get("current") or {}
    daily = raw.get("daily") or {}

    times = daily.get("time") or []

    def day(i: int) -> Optional[Dict[str, Any]]:
        if not isinstance(times, list) or len(times) <= i:
            return None

        def pick(key: str) -> Any:
            arr = daily.get(key)
            if isinstance(arr, list) and len(arr) > i:
                return arr[i]
            return None

        return {
            "date": times[i],
            "weather_code": pick("weather_code"),
            "temp_max_c": pick("temperature_2m_max"),
            "temp_min_c": pick("temperature_2m_min"),
            "precip_sum_mm": pick("precipitation_sum"),
            "wind_max_kmh": pick("wind_speed_10m_max"),
        }

    return {
        "source": "open-meteo",
        "generated_at": now_iso,
        "place": place,
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "current": {
            "time": current.get("time"),
            "temp_c": current.get("temperature_2m"),
            "apparent_temp_c": current.get("apparent_temperature"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "precip_mm": current.get("precipitation"),
            "weather_code": current.get("weather_code"),
            "cloud_cover_pct": current.get("cloud_cover"),
            "wind_kmh": current.get("wind_speed_10m"),
            "wind_dir_deg": current.get("wind_direction_10m"),
        },
        "today": day(0),
        "tomorrow": day(1),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch weather snapshot from Open-Meteo.")
    p.add_argument("--lat", type=float, required=True, help="Latitude, e.g. 35.2810")
    p.add_argument("--lon", type=float, required=True, help="Longitude, e.g. 139.6720")
    p.add_argument("--tz", type=str, default="Asia/Tokyo", help="Timezone name, e.g. Asia/Tokyo")
    p.add_argument("--place", type=str, default="", help="Optional place name to embed in output JSON")
    p.add_argument("--format", type=str, default="json", choices=["json"], help="Output format (json only)")
    p.add_argument("--out", type=str, default="", help="Write JSON to this path (default: stdout)")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    url = build_url(args.lat, args.lon, args.tz)

    try:
        raw = http_get_json(url)
        snap = make_snapshot(raw, lat=args.lat, lon=args.lon, tz=args.tz, place=args.place)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out = json.dumps(snap, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
