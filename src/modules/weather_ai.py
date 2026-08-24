from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from modules.models import WeatherSnapshot, Waypoint


def _fetch_live_weather(lat: float, lon: float, when_utc: datetime, timeout_sec: float = 6.0) -> WeatherSnapshot:
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation_probability",
        "timezone": "UTC",
        "forecast_days": "2",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url, headers={"User-Agent": "pilotless-aircraft-sim/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    hourly = payload["hourly"]
    timestamps = hourly["time"]

    target = when_utc.replace(minute=0, second=0, microsecond=0)
    target_iso = target.isoformat().replace("+00:00", "")
    if target_iso not in timestamps:
        # Fallback to first available forecast hour.
        idx = 0
    else:
        idx = timestamps.index(target_iso)

    obs_time = datetime.fromisoformat(timestamps[idx]).replace(tzinfo=timezone.utc)
    return WeatherSnapshot(
        timestamp_utc=obs_time,
        latitude=lat,
        longitude=lon,
        temperature_c=float(hourly["temperature_2m"][idx]),
        wind_speed_kph=float(hourly["wind_speed_10m"][idx]),
        wind_direction_deg=float(hourly["wind_direction_10m"][idx]),
        precip_probability=float(hourly["precipitation_probability"][idx]),
        source="open-meteo",
    )


def _load_fallback_rows(csv_path: str | Path) -> List[dict[str, float]]:
    rows: List[dict[str, float]] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "temperature_c": float(row["temperature_c"]),
                    "wind_speed_kph": float(row["wind_speed_kph"]),
                    "wind_direction_deg": float(row["wind_direction_deg"]),
                    "precip_probability": float(row["precip_probability"]),
                }
            )
    return rows


def _fallback_weather(lat: float, lon: float, when_utc: datetime, csv_path: str | Path) -> WeatherSnapshot:
    rows = _load_fallback_rows(csv_path)

    nearest = min(
        rows,
        key=lambda row: (row["latitude"] - lat) ** 2 + (row["longitude"] - lon) ** 2,
    )

    return WeatherSnapshot(
        timestamp_utc=when_utc,
        latitude=lat,
        longitude=lon,
        temperature_c=nearest["temperature_c"],
        wind_speed_kph=nearest["wind_speed_kph"],
        wind_direction_deg=nearest["wind_direction_deg"],
        precip_probability=nearest["precip_probability"],
        source="fallback-csv",
    )


def collect_route_weather(
    waypoints: Iterable[Waypoint],
    when_utc: datetime,
    fallback_csv: str | Path,
    prefer_live: bool = True,
) -> tuple[List[WeatherSnapshot], bool]:
    snapshots: List[WeatherSnapshot] = []
    all_live = True

    for wp in waypoints:
        if not prefer_live:
            all_live = False
            snapshots.append(_fallback_weather(wp.latitude, wp.longitude, when_utc, fallback_csv))
            continue

        try:
            snapshots.append(_fetch_live_weather(wp.latitude, wp.longitude, when_utc))
        except Exception:
            all_live = False
            snapshots.append(_fallback_weather(wp.latitude, wp.longitude, when_utc, fallback_csv))

    return snapshots, all_live
