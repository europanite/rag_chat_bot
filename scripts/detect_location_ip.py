from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class DetectedLocation:
    lat: float
    lon: float
    timezone: str | None
    city: str | None
    region: str | None
    country: str | None
    source: str
    fetched_at: str


def _fetch_json(url: str, *, timeout: int = 10) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "weather-rag/1.0 (+https://github.com/)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _try_ipapi_co(*, timeout: int) -> DetectedLocation | None:
    # https://ipapi.co/json/ -> { latitude, longitude, timezone, city, region, country_name, ... }
    try:
        data = _fetch_json("https://ipapi.co/json/", timeout=timeout)
        lat = data.get("latitude")
        lon = data.get("longitude")
        if lat is None or lon is None:
            return None
        return DetectedLocation(
            lat=float(lat),
            lon=float(lon),
            timezone=data.get("timezone"),
            city=data.get("city"),
            region=data.get("region"),
            country=data.get("country_name"),
            source="ipapi.co",
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    except Exception:
        return None


def _try_ipinfo(*, timeout: int) -> DetectedLocation | None:
    # https://ipinfo.io/json -> { loc:"lat,lon", timezone:"Asia/Tokyo", city, region, country, ... }
    try:
        data = _fetch_json("https://ipinfo.io/json", timeout=timeout)
        loc = data.get("loc") or ""
        if not isinstance(loc, str) or "," not in loc:
            return None
        lat_s, lon_s = loc.split(",", 1)
        return DetectedLocation(
            lat=float(lat_s.strip()),
            lon=float(lon_s.strip()),
            timezone=data.get("timezone"),
            city=data.get("city"),
            region=data.get("region"),
            country=data.get("country"),
            source="ipinfo.io",
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    except Exception:
        return None


def _try_ip_api_com(*, timeout: int) -> DetectedLocation | None:
    # http://ip-api.com/json/?fields=... (HTTP only, but reliable fallback)
    try:
        data = _fetch_json(
            "http://ip-api.com/json/?fields=status,message,lat,lon,timezone,city,regionName,country",
            timeout=timeout,
        )
        if data.get("status") != "success":
            return None
        lat = data.get("lat")
        lon = data.get("lon")
        if lat is None or lon is None:
            return None
        return DetectedLocation(
            lat=float(lat),
            lon=float(lon),
            timezone=data.get("timezone"),
            city=data.get("city"),
            region=data.get("regionName"),
            country=data.get("country"),
            source="ip-api.com",
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    except Exception:
        return None


def detect_location(*, timeout: int = 10) -> DetectedLocation:
    """
    Detect an *approximate* location from your public IP.
    - Pros: no API key, no browser permissions.
    - Cons: may be off by a city or more (especially on mobile/VPN).

    Returns DetectedLocation or raises RuntimeError on failure.
    """
    for fn in (_try_ipapi_co, _try_ipinfo, _try_ip_api_com):
        loc = fn(timeout=timeout)
        if loc is not None:
            return loc
    raise RuntimeError("Failed to detect location via public IP (all providers failed).")


def _to_env(loc: DetectedLocation) -> str:
    lines = [
        f"WEATHER_LAT={loc.lat}",
        f"WEATHER_LON={loc.lon}",
    ]
    if loc.timezone:
        lines.append(f"TZ_NAME={loc.timezone}")
    # Optional human hint
    place_bits = [b for b in [loc.city, loc.region, loc.country] if b]
    if place_bits:
        lines.append(f"WEATHER_PLACE={', '.join(place_bits)}")
    lines.append(f"WEATHER_GEO_SOURCE={loc.source}")
    lines.append(f"WEATHER_GEO_FETCHED_AT={loc.fetched_at}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect approximate location from public IP.")
    parser.add_argument("--format", choices=["env", "json"], default="env")
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    try:
        loc = detect_location(timeout=args.timeout)
    except Exception as e:
        print(f"[detect_location_ip] {e}", file=sys.stderr)
        raise SystemExit(1)

    if args.format == "json":
        print(json.dumps(asdict(loc), ensure_ascii=False))
    else:
        print(_to_env(loc))


if __name__ == "__main__":
    main()
