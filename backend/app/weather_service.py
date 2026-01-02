from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from typing import Any

import requests
from fastapi import Request


class WeatherError(RuntimeError):
    """Raised when live weather context cannot be produced."""


@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float
    timezone: str | None = None
    place: str | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class WeatherSnapshot:
    observed_at: str  # ISO string from API
    temperature_c: float | None
    apparent_temperature_c: float | None
    humidity_pct: float | None
    precipitation_mm: float | None
    wind_speed_kmh: float | None
    wind_direction_deg: float | None
    weather_code: int | None


_WEATHER_CODE = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _is_public_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_reserved)


def _extract_client_ip(req: Request) -> str | None:
    # Respect a common reverse-proxy header. If multiple, take the left-most.
    xff = req.headers.get("x-forwarded-for") or req.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    if req.client:
        return req.client.host
    return None


def _location_from_request_overrides(req: Request) -> Location | None:
    # 1) query params (useful for local dev): /rag/query?lat=...&lon=...
    lat = _parse_float(req.query_params.get("lat"))
    lon = _parse_float(req.query_params.get("lon"))

    # 2) headers (optional): X-Weather-Lat / X-Weather-Lon
    if lat is None:
        lat = _parse_float(req.headers.get("X-Weather-Lat") or req.headers.get("x-weather-lat"))
    if lon is None:
        lon = _parse_float(req.headers.get("X-Weather-Lon") or req.headers.get("x-weather-lon"))

    if lat is not None and lon is not None:
        tz = req.query_params.get("tz") or req.headers.get("X-Weather-Tz")
        place = req.query_params.get("place") or req.headers.get("X-Weather-Place")
        return Location(latitude=lat, longitude=lon, timezone=tz, place=place, source="request")

    return None


def _location_from_env() -> Location | None:
    lat = _parse_float(os.getenv("LAT"))
    lon = _parse_float(os.getenv("LON"))
    if lat is None or lon is None:
        return None
    return Location(
        latitude=lat,
        longitude=lon,
        timezone=os.getenv("TZ_NAME"),
        place=os.getenv("PLACE"),
        source="env",
    )


def _lookup_location_by_ip(ip: str, session: requests.Session, timeout_s: float) -> Location | None:
    # Try a few public, no-key endpoints. Each may rate-limit.
    candidates = [
        ("ipapi", f"https://ipapi.co/{ip}/json/"),
        ("ipinfo", f"https://ipinfo.io/{ip}/json"),
        ("ip-api", f"http://ip-api.com/json/{ip}?fields=status,message,lat,lon,city,regionName,country,timezone"),
    ]

    for name, url in candidates:
        try:
            r = session.get(url, timeout=timeout_s)
            r.raise_for_status()
            data: dict[str, Any] = r.json()
        except Exception:
            continue

        # Normalize the 3 providers.
        lat = data.get("latitude") or data.get("lat")
        lon = data.get("longitude") or data.get("lon")
        tz = data.get("timezone")
        city = data.get("city")
        region = data.get("region") or data.get("regionName")
        country = data.get("country_name") or data.get("country")

        if isinstance(lat, str):
            lat = _parse_float(lat)
        if isinstance(lon, str):
            lon = _parse_float(lon)

        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            place_parts = [p for p in [city, region, country] if isinstance(p, str) and p.strip()]
            place = ", ".join(place_parts) if place_parts else None
            return Location(latitude=float(lat), longitude=float(lon), timezone=tz if isinstance(tz, str) else None, place=place, source=f"ip:{name}")

    return None


def get_location(req: Request, session: requests.Session, timeout_s: float = 5.0) -> Location:
    # Request overrides first (nice for local dev).
    override = _location_from_request_overrides(req)
    if override:
        return override

    # Then env vars (nice for local dev + GitHub Actions).
    env_loc = _location_from_env()
    if env_loc:
        return env_loc

    # Finally IP lookup (works in real deployments; often not in local Docker).
    ip = _extract_client_ip(req)
    if not ip:
        raise WeatherError("Could not determine client IP for geolocation. Provide lat/lon via env or request params.")
    if not _is_public_ip(ip):
        raise WeatherError(
            f"Client IP '{ip}' is not a public IP (likely local/private). "
            "Provide lat/lon via env (LAT/LON) or request params (?lat=&lon=)."
        )

    loc = _lookup_location_by_ip(ip, session=session, timeout_s=timeout_s)
    if not loc:
        raise WeatherError("IP geolocation failed. Provide lat/lon via env or request params.")
    return loc


def fetch_current_weather(location: Location, session: requests.Session, timeout_s: float = 8.0) -> WeatherSnapshot:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "timezone": "auto" if not location.timezone else location.timezone,
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),
    }

    try:
        r = session.get(url, params=params, timeout=timeout_s)
        r.raise_for_status()
        data: dict[str, Any] = r.json()
    except Exception as exc:
        raise WeatherError(f"Weather API request failed: {exc}") from exc

    current = data.get("current") or {}
    return WeatherSnapshot(
        observed_at=str(current.get("time") or ""),
        temperature_c=_as_float(current.get("temperature_2m")),
        apparent_temperature_c=_as_float(current.get("apparent_temperature")),
        humidity_pct=_as_float(current.get("relative_humidity_2m")),
        precipitation_mm=_as_float(current.get("precipitation")),
        wind_speed_kmh=_as_float(current.get("wind_speed_10m")),
        wind_direction_deg=_as_float(current.get("wind_direction_10m")),
        weather_code=_as_int(current.get("weather_code")),
    )


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return _parse_float(v)
    return None


def _as_int(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        try:
            return int(float(v))
        except ValueError:
            return None
    return None


def format_live_weather_context(location: Location, weather: WeatherSnapshot) -> str:
    desc = _WEATHER_CODE.get(weather.weather_code, f"weather_code={weather.weather_code}" if weather.weather_code is not None else "unknown")
    where = location.place or f"{location.latitude:.4f},{location.longitude:.4f}"
    parts: list[str] = [
        f"Location: {where} (source={location.source})",
    ]
    if weather.observed_at:
        parts.append(f"Observed at: {weather.observed_at}")
    parts.append(f"Condition: {desc}")

    if weather.temperature_c is not None:
        t = f"{weather.temperature_c:.1f}°C"
        if weather.apparent_temperature_c is not None:
            t += f" (feels {weather.apparent_temperature_c:.1f}°C)"
        parts.append(f"Temperature: {t}")

    if weather.humidity_pct is not None:
        parts.append(f"Humidity: {weather.humidity_pct:.0f}%")

    if weather.precipitation_mm is not None:
        parts.append(f"Precipitation: {weather.precipitation_mm:.1f} mm")

    if weather.wind_speed_kmh is not None:
        wind = f"{weather.wind_speed_kmh:.1f} km/h"
        if weather.wind_direction_deg is not None:
            wind += f" @ {weather.wind_direction_deg:.0f}°"
        parts.append(f"Wind: {wind}")

    # A reminder for the LLM: this is ephemeral and should not be stored.
    parts.append("Note: This is live, short-lived context; do NOT store it in the vector DB.")

    return "\n".join(parts)


def get_live_weather_context(*, http_request: Request, session: requests.Session) -> str:
    """Return a short, human-readable block that can be appended to the RAG prompt."""
    loc = get_location(http_request, session=session)
    snap = fetch_current_weather(loc, session=session)
    return format_live_weather_context(loc, snap)
