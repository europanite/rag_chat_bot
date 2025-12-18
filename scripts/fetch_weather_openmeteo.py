from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    # Local-only helper (no dependencies): detect approximate location from public IP
    from detect_location_ip import detect_location
except Exception:  # pragma: no cover
    detect_location = None  # type: ignore[assignment]



# Open-Meteo weather code reference (condensed)
# https://open-meteo.com/en/docs (we keep a small mapping for human-readable text)
_WEATHER_CODE_TEXT: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@dataclass(frozen=True)
class WeatherSnapshot:
    fetched_at: str
    timezone: str
    lat: float
    lon: float
    place: str | None = None
    geo_source: str | None = None
    today: dict[str, Any]
    tomorrow: dict[str, Any] | None
    current: dict[str, Any] | None


def _code_text(code: int | None) -> str:
    if code is None:
        return "Unknown"
    return _WEATHER_CODE_TEXT.get(int(code), f"Code {code}")


def _build_url(*, lat: str, lon: str, tz: str) -> str:
    # Use the "forecast" endpoint with daily summary + current_weather (widely supported).
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "current_weather": "true",
        "daily": ",".join(
            [
                "weathercode",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "windspeed_10m_max",
            ]
        ),
    }
    return "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)


def fetch_weather(*, lat: str, lon: str, tz: str, place: str | None = None, geo_source: str | None = None) -> WeatherSnapshot:
    url = _build_url(lat=lat, lon=lon, tz=tz)

    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    now_local = datetime.now(ZoneInfo(tz))
    today_key = now_local.date().isoformat()
    daily = raw.get("daily") or {}

    times: list[str] = daily.get("time") or []
    try:
        idx_today = times.index(today_key)
    except ValueError:
        idx_today = 0

    def day_at(i: int) -> dict[str, Any] | None:
        if not times or i < 0 or i >= len(times):
            return None
        return {
            "date": times[i],
            "weather_code": int((daily.get("weathercode") or [None])[i]),
            "weather": _code_text((daily.get("weathercode") or [None])[i]),
            "temp_max_c": (daily.get("temperature_2m_max") or [None])[i],
            "temp_min_c": (daily.get("temperature_2m_min") or [None])[i],
            "precip_mm": (daily.get("precipitation_sum") or [None])[i],
            "wind_max_kmh": (daily.get("windspeed_10m_max") or [None])[i],
        }

    today = day_at(idx_today) or {"date": today_key, "weather": "Unknown"}
    tomorrow = day_at(idx_today + 1)

    current_weather = raw.get("current_weather")
    current: dict[str, Any] | None = None
    if isinstance(current_weather, dict):
        current = {
            "time": current_weather.get("time"),
            "temp_c": current_weather.get("temperature"),
            "wind_kmh": current_weather.get("windspeed"),
            "wind_dir_deg": current_weather.get("winddirection"),
            "weather_code": current_weather.get("weathercode"),
            "weather": _code_text(current_weather.get("weathercode")),
        }

return WeatherSnapshot(
    fetched_at=now_local.isoformat(timespec="seconds"),
    timezone=tz,
    lat=float(lat),
    lon=float(lon),
    place=place,
    geo_source=geo_source,
    today=today,
    tomorrow=tomorrow,
    current=current,
)


def render_text(snap: WeatherSnapshot) -> str:
    t = snap.today
    cur = snap.current

    lines: list[str] = []
    head = f"[Weather] {t.get('date')} ({snap.timezone}) @ {snap.lat:.4f},{snap.lon:.4f}"
    if snap.place:
        head += f" [{snap.place}]"
    if snap.geo_source:
        head += f" (geo: {snap.geo_source})"
    lines.append(head)
    if cur:
        lines.append(
            f"- Current: {cur.get('temp_c')}°C, {cur.get('weather')} (wind {cur.get('wind_kmh')} km/h)"
        )
    lines.append(
        f"- Today: {t.get('weather')}, high {t.get('temp_max_c')}°C / low {t.get('temp_min_c')}°C, "
        f"precip {t.get('precip_mm')} mm"
    )
    if snap.tomorrow:
        tm = snap.tomorrow
        lines.append(
            f"- Tomorrow: {tm.get('weather')}, high {tm.get('temp_max_c')}°C / low {tm.get('temp_min_c')}°C, "
            f"precip {tm.get('precip_mm')} mm"
        )
    lines.append(f"- Fetched at: {snap.fetched_at}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch short-lived weather context (Open-Meteo).")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the raw snapshot JSON (useful for debugging).",
    )
parser.add_argument("--lat", default=None, help="Override latitude (optional).")
parser.add_argument("--lon", default=None, help="Override longitude (optional).")
parser.add_argument("--tz", default=None, help="Override timezone, e.g., Asia/Tokyo (optional).")
parser.add_argument(
    "--auto-locate",
    action="store_true",
    help="Detect approximate location from public IP (no API key).",
)
    args = parser.parse_args()

# Priority: CLI override > env var > auto-locate > fallback default (Yokosuka)
lat = args.lat or os.environ.get("WEATHER_LAT")
lon = args.lon or os.environ.get("WEATHER_LON")
tz = args.tz or os.environ.get("WEATHER_TZ")

want_auto = args.auto_locate or os.environ.get("WEATHER_AUTO", "").lower() in {"1", "true", "yes", "on"}
place: str | None = os.environ.get("WEATHER_PLACE") or None
geo_source: str | None = os.environ.get("WEATHER_GEO_SOURCE") or None

if want_auto and (not lat or not lon):
    if detect_location is not None:
        try:
            loc = detect_location(timeout=10)
            lat = lat or str(loc.lat)
            lon = lon or str(loc.lon)
            tz = tz or (loc.timezone or "Asia/Tokyo")
            # best-effort human hint
            place_bits = [b for b in [loc.city, loc.region, loc.country] if b]
            place = place or (", ".join(place_bits) if place_bits else None)
            geo_source = geo_source or loc.source
        except Exception:
            # If auto-locate fails, fall back to default coordinates.
            pass

# Final fallback: Yokosuka area by default (stable for local/dev)
lat = lat or "35.2810"
lon = lon or "139.6722"
tz = tz or "Asia/Tokyo"

snap = fetch_weather(lat=lat, lon=lon, tz=tz, place=place, geo_source=geo_source)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(snap), ensure_ascii=False, indent=2), encoding="utf-8")

    if args.format == "json":
        print(json.dumps(asdict(snap), ensure_ascii=False))
    else:
        print(render_text(snap))


if __name__ == "__main__":
    main()