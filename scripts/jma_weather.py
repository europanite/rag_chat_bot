"""
backend/app/jma_weather.py

A small helper to fetch "current weather" from JMA (AMeDAS point + forecast)
and return a dict compatible with backend/app/weather_service.py's expectations.

This is intended to be imported by weather_service.py when WEATHER_PROVIDER=jma.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, Optional, Tuple

import requests

JMA_FORECAST_BASE = "https://www.jma.go.jp/bosai/forecast/data/forecast"
JMA_AMEDAS_LATEST_TIME = "https://www.jma.go.jp/bosai/amedas/data/latest_time.txt"
JMA_AMEDAS_TABLE = "https://www.jma.go.jp/bosai/amedas/const/amedastable.json"
JMA_AMEDAS_POINT_BASE = "https://www.jma.go.jp/bosai/amedas/data/point"

_AMEDAS_TABLE_CACHE: Optional[Dict[str, Any]] = None


def _degmin_to_deg(v: Any) -> Optional[float]:
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
        return None
    return (c % 16) * 22.5


def _jp_weather_to_wmo_code(text_ja: str) -> int:
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


def load_amedas_table(session: requests.Session, timeout_s: float = 8.0) -> Dict[str, Any]:
    global _AMEDAS_TABLE_CACHE
    if _AMEDAS_TABLE_CACHE is None:
        r = session.get(JMA_AMEDAS_TABLE, timeout=timeout_s)
        r.raise_for_status()
        _AMEDAS_TABLE_CACHE = r.json()
    return _AMEDAS_TABLE_CACHE


def pick_nearest_station(session: requests.Session, lat: float, lon: float, timeout_s: float = 8.0) -> str:
    table = load_amedas_table(session, timeout_s=timeout_s)
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


def jma_latest_time_utc(session: requests.Session, timeout_s: float = 8.0) -> dt.datetime:
    s = session.get(JMA_AMEDAS_LATEST_TIME, timeout=timeout_s).text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return dt.datetime.fromisoformat(s)


def _station_point_block_url(station: str, when_utc: dt.datetime) -> str:
    when_jst = when_utc.astimezone(dt.timezone(dt.timedelta(hours=9)))
    yyyymmdd = when_jst.strftime("%Y%m%d")
    h3 = (when_jst.hour // 3) * 3
    return f"{JMA_AMEDAS_POINT_BASE}/{station}/{yyyymmdd}_{h3:02d}.json"


def _latest_obs_from_point_block(block: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    keys = [k for k in block.keys() if k.isdigit()]
    if not keys:
        raise RuntimeError("Unexpected AMeDAS point payload (no timestamp keys)")
    k = max(keys)
    return k, block[k]


def fetch_current_from_jma(
    session: requests.Session,
    lat: float,
    lon: float,
    office_code: str,
    area_code: Optional[str],
    amedas_station: Optional[str],
    timeout_s: float = 8.0,
) -> Dict[str, Any]:
    # 1) AMeDAS observation
    latest = jma_latest_time_utc(session, timeout_s=timeout_s)
    station = amedas_station or pick_nearest_station(session, lat, lon, timeout_s=timeout_s)

    url = _station_point_block_url(station, latest)
    r = session.get(url, timeout=timeout_s)
    r.raise_for_status()
    block = r.json()
    ts_key, obs = _latest_obs_from_point_block(block)

    temp_c = _val(obs, "temp")
    humidity = _val(obs, "humidity")
    precip_1h = _val(obs, "precipitation1h")
    wind_ms = _val(obs, "wind")
    wind_kmh = None if wind_ms is None else wind_ms * 3.6
    wind_dir_deg = _wind_dir_code_to_deg(_val(obs, "windDirection"))

    when_jst = dt.datetime.strptime(ts_key, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone(dt.timedelta(hours=9)))

    # 2) Forecast text to infer condition code (optional but improves UX)
    wmo_code: Optional[int] = None
    weather_text_ja: Optional[str] = None
    try:
        fr = session.get(f"{JMA_FORECAST_BASE}/{office_code}.json", timeout=timeout_s)
        fr.raise_for_status()
        data = fr.json()
        short = data[0]
        ts_weather = short["timeSeries"][0]
        areas = ts_weather["areas"]
        area = areas[0]
        if area_code:
            for a in areas:
                if a.get("area", {}).get("code") == str(area_code):
                    area = a
                    break
        weathers = area.get("weathers") or []
        if weathers:
            weather_text_ja = weathers[0]
            wmo_code = _jp_weather_to_wmo_code(weather_text_ja)
    except Exception:
        pass

    return {
        "time": when_jst.isoformat(),
        "temp_c": temp_c,
        "humidity_pct": humidity,
        "precip_mm": precip_1h,
        "wind_kmh": wind_kmh,
        "wind_dir_deg": wind_dir_deg,
        "weather_code": wmo_code,
        "weather_text_ja": weather_text_ja,
        "amedas_station": station,
    }
