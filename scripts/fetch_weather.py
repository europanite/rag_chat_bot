#!/usr/bin/env python3
"""
Fetch a small "weather snapshot" JSON used by this repo (diary/feed generation).

Providers:
  - jma       : Japan Meteorological Agency (AMeDAS + forecast)  ✅ Japan-first
  - open-meteo: Open-Meteo (kept as fallback / compatibility)

This script keeps the snapshot shape compatible with the existing Open-Meteo snapshot:
- source, generated_at, place, latitude, longitude, timezone
- current: time, temp_c, apparent_temp_c, humidity_pct, precip_mm, weather_code, cloud_cover_pct, wind_kmh, wind_dir_deg
- today/tomorrow: date, weather_code, precip_sum_mm, wind_max_kmh, temp_max_c, temp_min_c

Notes:
- JMA does not provide "cloud cover %" or "apparent temp" directly, so those are null.
- JMA forecast "weatherCodes" are not WMO; we also store a best-effort WMO-ish weather_code by heuristics on the JP weather text.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

# ---- Open-Meteo (existing) ----
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

# ---- JMA (Japan Meteorological Agency) ----
JMA_FORECAST_BASE = "https://www.jma.go.jp/bosai/forecast/data/forecast"
JMA_AMEDAS_LATEST_TIME = "https://www.jma.go.jp/bosai/amedas/data/latest_time.txt"
JMA_AMEDAS_TABLE = "https://www.jma.go.jp/bosai/amedas/const/amedastable.json"
JMA_AMEDAS_POINT_BASE = "https://www.jma.go.jp/bosai/amedas/data/point"


# -------------------- HTTP helpers --------------------
def http_get_text(url: str, timeout_s: float = 10.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "rag-chat-bot/1.0 (+weather)"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8")


def http_get_json(url: str, timeout_s: float = 10.0) -> Any:
    return json.loads(http_get_text(url, timeout_s=timeout_s))


# -------------------- Open-Meteo --------------------
def build_open_meteo_url(lat: float, lon: float, tz: str) -> str:
    params = {
        "latitude": f"{lat:.5f}",
        "longitude": f"{lon:.5f}",
        "timezone": tz,
        "forecast_days": "2",
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
    }
    return OPEN_METEO_BASE + "?" + urllib.parse.urlencode(params)


def fetch_open_meteo_snapshot(place: str, lat: float, lon: float, tz: str) -> Dict[str, Any]:
    data = http_get_json(build_open_meteo_url(lat, lon, tz))
    cur = data.get("current", {})

    def _safe_float(v: Any) -> Optional[float]:
        try:
            return None if v is None else float(v)
        except Exception:
            return None

    def daily(i: int) -> Dict[str, Any]:
        d = data.get("daily", {})
        dates = d.get("time") or []
        if i >= len(dates):
            return {
                "date": None,
                "weather_code": None,
                "precip_sum_mm": None,
                "wind_max_kmh": None,
                "temp_max_c": None,
                "temp_min_c": None,
            }
        return {
            "date": dates[i],
            "weather_code": int(d.get("weather_code", [None])[i]) if d.get("weather_code") else None,
            "precip_sum_mm": _safe_float((d.get("precipitation_sum") or [None])[i]),
            "wind_max_kmh": _safe_float((d.get("wind_speed_10m_max") or [None])[i]),
            "temp_max_c": _safe_float((d.get("temperature_2m_max") or [None])[i]),
            "temp_min_c": _safe_float((d.get("temperature_2m_min") or [None])[i]),
        }

    snapshot = {
        "source": "open-meteo",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "place": place,
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "current": {
            "time": cur.get("time"),
            "temp_c": _safe_float(cur.get("temperature_2m")),
            "apparent_temp_c": _safe_float(cur.get("apparent_temperature")),
            "humidity_pct": _safe_float(cur.get("relative_humidity_2m")),
            "precip_mm": _safe_float(cur.get("precipitation")),
            "weather_code": int(cur["weather_code"]) if "weather_code" in cur and cur["weather_code"] is not None else None,
            "cloud_cover_pct": _safe_float(cur.get("cloud_cover")),
            "wind_kmh": _safe_float(cur.get("wind_speed_10m")),
            "wind_dir_deg": _safe_float(cur.get("wind_direction_10m")),
        },
        "today": daily(0),
        "tomorrow": daily(1),
    }
    return snapshot


# -------------------- JMA helpers --------------------
def _degmin_to_deg(v: Any) -> Optional[float]:
    # JMA amedastable uses [deg, minutes] e.g. [35, 26.4]
    try:
        if isinstance(v, list) and len(v) == 2:
            return float(v[0]) + float(v[1]) / 60.0
    except Exception:
        return None
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


_AMEDAS_TABLE_CACHE: Optional[Dict[str, Any]] = None


def load_amedas_table() -> Dict[str, Any]:
    global _AMEDAS_TABLE_CACHE
    if _AMEDAS_TABLE_CACHE is None:
        _AMEDAS_TABLE_CACHE = http_get_json(JMA_AMEDAS_TABLE)
    return _AMEDAS_TABLE_CACHE


def pick_nearest_station(lat: float, lon: float) -> str:
    table = load_amedas_table()
    best_id: Optional[str] = None
    best_dist = 10**9
    for sid, info in table.items():
        slat = _degmin_to_deg(info.get("lat"))
        slon = _degmin_to_deg(info.get("lon"))
        if slat is None or slon is None:
            continue
        d = _haversine_km(lat, lon, slat, slon)
        if d < best_dist:
            best_dist = d
            best_id = str(sid)
    if not best_id:
        raise RuntimeError("Failed to choose an AMeDAS station")
    return best_id


def jma_latest_time() -> dt.datetime:
    # Example: "2025-12-26T04:30:00Z" or with timezone; parse via fromisoformat after normalization.
    s = http_get_text(JMA_AMEDAS_LATEST_TIME).strip()
    # Python's fromisoformat doesn't accept trailing Z before 3.11 reliably; normalize.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return dt.datetime.fromisoformat(s)


def _station_point_block_url(station: str, when_utc: dt.datetime) -> str:
    # JMA point data is grouped in 3-hour blocks: /point/{station}/{yyyymmdd}_{h3}.json
    # h3 is 00,03,06,...,21 (no 24).
    when_jst = when_utc.astimezone(dt.timezone(dt.timedelta(hours=9)))
    yyyymmdd = when_jst.strftime("%Y%m%d")
    h3 = (when_jst.hour // 3) * 3
    return f"{JMA_AMEDAS_POINT_BASE}/{station}/{yyyymmdd}_{h3:02d}.json"


def _latest_obs_from_point_block(block: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    # block keys are timestamps like "20220705030000"
    keys = [k for k in block.keys() if k.isdigit()]
    if not keys:
        raise RuntimeError("Unexpected AMeDAS point payload (no timestamp keys)")
    k = max(keys)  # lexicographically == chronologically
    return k, block[k]


def _val(entry: Dict[str, Any], key: str) -> Optional[float]:
    v = entry.get(key)
    if isinstance(v, list) and v:
        try:
            return float(v[0]) if v[0] is not None else None
        except Exception:
            return None
    return None


def _wind_dir_code_to_deg(code: Optional[float]) -> Optional[float]:
    if code is None:
        return None
    try:
        c = int(code)
    except Exception:
        return None
    if c <= 0:
        return None  # calm / unknown
    return (c % 16) * 22.5


def _jp_weather_to_wmo_code(text_ja: str) -> int:
    """
    Best-effort mapping from JP forecast text to WMO-ish codes (Open-Meteo style).
    Not perfect, but good enough for tags like rain/snow/thunder.
    """
    t = text_ja or ""
    if "雷" in t:
        return 95
    if "みぞれ" in t:
        return 68
    if "雪" in t:
        return 71
    if "雨" in t:
        return 61
    if "晴" in t and "くもり" in t:
        return 2
    if "晴" in t:
        return 0
    if "くもり" in t:
        return 3
    return 3


def fetch_jma_current(lat: float, lon: float, station: Optional[str] = None) -> Dict[str, Any]:
    latest = jma_latest_time()
    station_id = station or pick_nearest_station(lat, lon)
    url = _station_point_block_url(station_id, latest)
    block = http_get_json(url)
    ts_key, obs = _latest_obs_from_point_block(block)

    temp_c = _val(obs, "temp")
    humidity = _val(obs, "humidity")
    precip_1h = _val(obs, "precipitation1h")
    wind_ms = _val(obs, "wind")
    wind_kmh = None if wind_ms is None else wind_ms * 3.6
    wind_dir_deg = _wind_dir_code_to_deg(_val(obs, "windDirection"))

    # JMA point data may include a "weather" field (JMA internal code). We'll keep it as-is for debugging.
    jma_weather_code = _val(obs, "weather")

    # Observations are in JST when interpreted as yyyymmddHHMMSS; format to ISO with JST offset.
    when_jst = dt.datetime.strptime(ts_key, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone(dt.timedelta(hours=9)))

    return {
        "station": station_id,
        "time": when_jst.isoformat(),
        "temp_c": temp_c,
        "humidity_pct": humidity,
        "precip_mm": precip_1h,
        "wind_kmh": wind_kmh,
        "wind_dir_deg": wind_dir_deg,
        "jma_weather_obs_code": jma_weather_code,
    }


def fetch_jma_forecast(
    office_code: str,
    area_code: Optional[str],
    temp_city_code: Optional[str],
) -> Dict[str, Any]:
    data = http_get_json(f"{JMA_FORECAST_BASE}/{office_code}.json")
    if not isinstance(data, list) or not data:
        raise RuntimeError("Unexpected JMA forecast payload")

    short = data[0]  # 3-day-ish
    weekly = data[1] if len(data) > 1 else None

    # Select area for weather/pops (e.g., Kanagawa East/West)
    ts_weather = short["timeSeries"][0]
    areas = ts_weather["areas"]
    area = areas[0]
    if area_code:
        for a in areas:
            if a.get("area", {}).get("code") == str(area_code):
                area = a
                break

    time_defines = ts_weather["timeDefines"]
    weathers = area.get("weathers") or []
    weather_codes_jma = area.get("weatherCodes") or []

    # Build date-indexed dict for today/tomorrow selection
    by_date: Dict[str, Dict[str, Any]] = {}
    for i, t in enumerate(time_defines):
        try:
            d = dt.datetime.fromisoformat(t).date().isoformat()
        except Exception:
            continue
        wtxt = weathers[i] if i < len(weathers) else ""
        wmo = _jp_weather_to_wmo_code(wtxt)
        by_date[d] = {
            "date": d,
            "weather_text_ja": wtxt,
            "jma_weather_code": weather_codes_jma[i] if i < len(weather_codes_jma) else None,
            "weather_code": wmo,
        }

    # Pops (precip probability) by date (max for the day)
    pops_ts = short["timeSeries"][1]
    pops_area = pops_ts["areas"][0]
    if area_code:
        for a in pops_ts["areas"]:
            if a.get("area", {}).get("code") == str(area_code):
                pops_area = a
                break
    pop_time = pops_ts["timeDefines"]
    pops = pops_area.get("pops") or []
    pops_by_date: Dict[str, int] = {}
    for i, t in enumerate(pop_time):
        try:
            d = dt.datetime.fromisoformat(t).date().isoformat()
        except Exception:
            continue
        try:
            p = int(pops[i])
        except Exception:
            continue
        pops_by_date[d] = max(pops_by_date.get(d, 0), p)

    # Temps from weekly part (usually includes tempsMin/Max by city code like Yokohama 46106)
    temps_by_date: Dict[str, Dict[str, Optional[float]]] = {}
    if weekly:
        # find a timeSeries with tempsMin/Max
        ts_candidates = [ts for ts in weekly.get("timeSeries", []) if ts.get("areas")]
        for ts in ts_candidates:
            # tempsMin/Max exist here
            if "tempsMax" in (ts["areas"][0] or {}) or "tempsMin" in (ts["areas"][0] or {}):
                # pick area
                t_area = ts["areas"][0]
                if temp_city_code:
                    for a in ts["areas"]:
                        if a.get("area", {}).get("code") == str(temp_city_code):
                            t_area = a
                            break
                tdefs = ts.get("timeDefines") or []
                tmin = t_area.get("tempsMin") or []
                tmax = t_area.get("tempsMax") or []
                for i, t in enumerate(tdefs):
                    try:
                        d = dt.datetime.fromisoformat(t).date().isoformat()
                    except Exception:
                        continue
                    def _to_float(x: Any) -> Optional[float]:
                        try:
                            return float(x) if x not in ("", None) else None
                        except Exception:
                            return None
                    temps_by_date[d] = {
                        "temp_min_c": _to_float(tmin[i] if i < len(tmin) else None),
                        "temp_max_c": _to_float(tmax[i] if i < len(tmax) else None),
                    }
                break

    return {
        "publishing_office": short.get("publishingOffice"),
        "report_datetime": short.get("reportDatetime"),
        "by_date": by_date,
        "pops_by_date": pops_by_date,
        "temps_by_date": temps_by_date,
    }


def make_jma_snapshot(
    place: str,
    lat: float,
    lon: float,
    tz: str,
    office_code: str,
    area_code: Optional[str],
    temp_city_code: Optional[str],
    station: Optional[str],
) -> Dict[str, Any]:
    current = fetch_jma_current(lat, lon, station=station)
    forecast = fetch_jma_forecast(office_code=office_code, area_code=area_code, temp_city_code=temp_city_code)

    # Determine today & tomorrow dates in the requested tz
    tz_offset = dt.timezone(dt.timedelta(hours=9)) if tz == "Asia/Tokyo" else dt.timezone.utc
    today = dt.datetime.now(tz_offset).date().isoformat()
    tomorrow = (dt.datetime.now(tz_offset).date() + dt.timedelta(days=1)).isoformat()

    def daily(d: str) -> Dict[str, Any]:
        base = forecast["by_date"].get(d, {"date": d, "weather_code": None, "weather_text_ja": None, "jma_weather_code": None})
        temps = forecast["temps_by_date"].get(d, {})
        pop = forecast["pops_by_date"].get(d)
        # Keep compatible keys; add extra JP fields without breaking consumers.
        return {
            "date": d,
            "weather_code": base.get("weather_code"),
            "precip_sum_mm": None,      # JMA doesn't provide daily precipitation amount here
            "wind_max_kmh": None,       # same
            "temp_max_c": temps.get("temp_max_c"),
            "temp_min_c": temps.get("temp_min_c"),
            "weather_text_ja": base.get("weather_text_ja"),
            "jma_weather_code": base.get("jma_weather_code"),
            "precip_prob_pct": pop,
        }

    # Use today's forecast weather (if available) to assign a best-effort condition code
    today_forecast = forecast["by_date"].get(today) or {}
    wmo_code = today_forecast.get("weather_code")

    snapshot = {
        "source": "jma",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "place": place,
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "current": {
            "time": current.get("time"),
            "temp_c": current.get("temp_c"),
            "apparent_temp_c": None,
            "humidity_pct": current.get("humidity_pct"),
            "precip_mm": current.get("precip_mm"),
            "weather_code": wmo_code,
            "cloud_cover_pct": None,
            "wind_kmh": current.get("wind_kmh"),
            "wind_dir_deg": current.get("wind_dir_deg"),
            # extra debug/meta
            "amedas_station": current.get("station"),
            "jma_weather_obs_code": current.get("jma_weather_obs_code"),
        },
        "today": daily(today),
        "tomorrow": daily(tomorrow),
        "jma_meta": {
            "publishing_office": forecast.get("publishing_office"),
            "report_datetime": forecast.get("report_datetime"),
            "office_code": office_code,
            "area_code": area_code,
            "temp_city_code": temp_city_code,
        },
    }
    return snapshot


def fetch_weather_snapshot(
    provider: str,
    place: str,
    lat: float,
    lon: float,
    tz: str,
    office_code: str,
    area_code: Optional[str],
    temp_city_code: Optional[str],
    station: Optional[str],
) -> Dict[str, Any]:
    provider = (provider or "").strip().lower()
    if provider in ("jma", "japan", "japan-meteorological-agency"):
        return make_jma_snapshot(place, lat, lon, tz, office_code, area_code, temp_city_code, station)
    if provider in ("open-meteo", "openmeteo", "meteo"):
        return fetch_open_meteo_snapshot(place, lat, lon, tz)
    raise SystemExit(f"Unknown provider: {provider!r}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--place", default="Yokosuka, JP")
    p.add_argument("--lat", type=float, default=35.2813)
    p.add_argument("--lon", type=float, default=139.6720)
    p.add_argument("--tz", default="Asia/Tokyo")

    p.add_argument("--provider", default=os.getenv("WEATHER_PROVIDER", "jma"), choices=["jma", "open-meteo"])

    # JMA knobs (optional but recommended for better forecast labeling)
    p.add_argument("--jma-office-code", default=os.getenv("JMA_OFFICE_CODE", "140000"))
    p.add_argument("--jma-area-code", default=os.getenv("JMA_AREA_CODE"))          # e.g., Kanagawa East: 140010
    p.add_argument("--jma-temp-city-code", default=os.getenv("JMA_TEMP_CITY_CODE"))  # e.g., Yokohama: 46106
    p.add_argument("--jma-amedas-station", default=os.getenv("JMA_AMEDAS_STATION"))  # 5-digit, optional

    args = p.parse_args(argv)

    try:
        snap = fetch_weather_snapshot(
            provider=args.provider,
            place=args.place,
            lat=args.lat,
            lon=args.lon,
            tz=args.tz,
            office_code=str(args.jma_office_code),
            area_code=str(args.jma_area_code) if args.jma_area_code else None,
            temp_city_code=str(args.jma_temp_city_code) if args.jma_temp_city_code else None,
            station=str(args.jma_amedas_station) if args.jma_amedas_station else None,
        )
    except Exception as e:
        # If Japan-first fails, fallback to Open-Meteo unless provider was explicitly open-meteo.
        if args.provider == "jma":
            snap = fetch_open_meteo_snapshot(args.place, args.lat, args.lon, args.tz)
            snap["jma_fallback_error"] = repr(e)
        else:
            raise

    print(json.dumps(snap, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
