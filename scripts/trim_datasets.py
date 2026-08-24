#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "src" / "datasets"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

REQUIRED_IATA = ["SFO", "LAX", "SEA", "LAS", "JFK", "LHR", "DXB", "SIN"]
DEFAULT_ROUTES = [
    ("SFO", "LAX"),
    ("SEA", "LAS"),
    ("JFK", "LAX"),
    ("LHR", "DXB"),
    ("SIN", "DXB"),
]


@dataclass(frozen=True)
class BBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.max_lat < other.min_lat
            or self.min_lat > other.max_lat
            or self.max_lon < other.min_lon
            or self.min_lon > other.max_lon
        )


def _download_json(url: str, timeout: float = 20.0) -> dict:
    req = Request(url, headers={"User-Agent": "pilotless-aircraft-sim/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_open_meteo(lat: float, lon: float, when_utc: datetime) -> dict | None:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation_probability",
        "timezone": "UTC",
        "forecast_days": 2,
    }
    url = f"{OPEN_METEO_URL}?{urlencode(params)}"
    payload = _download_json(url)
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return None

    target = when_utc.replace(minute=0, second=0, microsecond=0)
    target_iso = target.isoformat().replace("+00:00", "")
    if target_iso in times:
        idx = times.index(target_iso)
    else:
        idx = 0

    return {
        "temperature_c": float(hourly["temperature_2m"][idx]),
        "wind_speed_kph": float(hourly["wind_speed_10m"][idx]),
        "wind_direction_deg": float(hourly["wind_direction_10m"][idx]),
        "precip_probability": float(hourly["precipitation_probability"][idx]),
    }


def trim_airports() -> int:
    path = DATASETS_DIR / "airports.csv"
    df = pd.read_csv(path)
    df = df[df["iata"].isin(REQUIRED_IATA)].copy()
    df.to_csv(path, index=False)
    return len(df)


def trim_cmapss(max_units: int) -> int:
    path = DATASETS_DIR / "cmapss_engine_health_sample.csv"
    df = pd.read_csv(path)
    units = sorted(df["unit_nr"].unique())
    keep_units = set(units[:max_units])
    df = df[df["unit_nr"].isin(keep_units)].copy()
    df.to_csv(path, index=False)
    return len(df)


def trim_opensky(limit: int) -> int:
    path = DATASETS_DIR / "opensky_traffic_sample.csv"
    df = pd.read_csv(path).head(limit)
    df.to_csv(path, index=False)
    return len(df)


def _bbox_for_feature(feature: dict) -> BBox | None:
    geom = feature.get("geometry") or {}
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return None

    polys = []
    if gtype == "Polygon":
        polys = [coords]
    elif gtype == "MultiPolygon":
        polys = coords
    else:
        return None

    min_lat = math.inf
    max_lat = -math.inf
    min_lon = math.inf
    max_lon = -math.inf

    for poly in polys:
        for ring in poly:
            for lon, lat in ring:
                min_lat = min(min_lat, lat)
                max_lat = max(max_lat, lat)
                min_lon = min(min_lon, lon)
                max_lon = max(max_lon, lon)

    if not math.isfinite(min_lat):
        return None
    return BBox(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)


def trim_no_fly_zones(buffer_deg: float) -> int:
    airports_path = DATASETS_DIR / "airports.csv"
    nf_path = DATASETS_DIR / "no_fly_zones.geojson"
    if not nf_path.exists():
        return 0

    airports = pd.read_csv(airports_path).set_index("iata")

    route_boxes: list[BBox] = []
    for dep, arr in DEFAULT_ROUTES:
        if dep not in airports.index or arr not in airports.index:
            continue
        dep_row = airports.loc[dep]
        arr_row = airports.loc[arr]
        min_lat = min(dep_row["latitude"], arr_row["latitude"]) - buffer_deg
        max_lat = max(dep_row["latitude"], arr_row["latitude"]) + buffer_deg
        min_lon = min(dep_row["longitude"], arr_row["longitude"]) - buffer_deg
        max_lon = max(dep_row["longitude"], arr_row["longitude"]) + buffer_deg
        route_boxes.append(BBox(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon))

    payload = json.loads(nf_path.read_text(encoding="utf-8"))
    features = payload.get("features", [])

    kept = []
    for feature in features:
        bbox = _bbox_for_feature(feature)
        if bbox is None:
            continue
        if any(bbox.intersects(rbox) for rbox in route_boxes):
            kept.append(feature)

    if not kept:
        kept = features[:10]

    payload["features"] = kept
    nf_path.write_text(json.dumps(payload), encoding="utf-8")
    return len(kept)


def trim_weather() -> int:
    airports_path = DATASETS_DIR / "airports.csv"
    weather_path = DATASETS_DIR / "weather_fallback.csv"

    airports = pd.read_csv(airports_path)
    existing = None
    if weather_path.exists():
        try:
            existing = pd.read_csv(weather_path)
        except Exception:
            existing = None

    when_utc = datetime.now(timezone.utc)
    rows = []
    for _, row in airports.iterrows():
        data = None
        try:
            data = _fetch_open_meteo(float(row["latitude"]), float(row["longitude"]), when_utc)
        except Exception:
            data = None

        if data is None and existing is not None and not existing.empty:
            ex = existing.copy()
            ex["dist"] = (ex["latitude"] - float(row["latitude"])) ** 2 + (
                ex["longitude"] - float(row["longitude"])
            ) ** 2
            nearest = ex.sort_values("dist").iloc[0]
            data = {
                "temperature_c": float(nearest["temperature_c"]),
                "wind_speed_kph": float(nearest["wind_speed_kph"]),
                "wind_direction_deg": float(nearest["wind_direction_deg"]),
                "precip_probability": float(nearest["precip_probability"]),
            }

        if data is None:
            continue

        rows.append(
            {
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                **data,
            }
        )
        time.sleep(0.1)

    if not rows:
        raise RuntimeError("No weather rows generated")

    pd.DataFrame(rows).to_csv(weather_path, index=False)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim datasets to only what the code uses")
    parser.add_argument("--opensky-limit", type=int, default=50)
    parser.add_argument("--cmapss-units", type=int, default=20)
    parser.add_argument("--no-fly-buffer-deg", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"airports.csv: {trim_airports()} rows")
    print(f"cmapss_engine_health_sample.csv: {trim_cmapss(args.cmapss_units)} rows")
    print(f"opensky_traffic_sample.csv: {trim_opensky(args.opensky_limit)} rows")
    print(f"no_fly_zones.geojson: {trim_no_fly_zones(args.no_fly_buffer_deg)} features")
    print(f"weather_fallback.csv: {trim_weather()} rows")


if __name__ == "__main__":
    main()
