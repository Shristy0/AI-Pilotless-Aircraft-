#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "src" / "datasets"

OURAIRPORTS_URL = "https://ourairports.com/data/airports.csv"
OURAIRPORTS_RUNWAYS_URL = "https://ourairports.com/data/runways.csv"
CMAPSS_ZIP_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
OPENSKY_URL = "https://opensky-network.org/api/states/all"
SUA_ARCGIS_URL = (
    "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/"
    "Special_Use_Airspace/FeatureServer/0/query"
)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _download_bytes(url: str, timeout: float = 45.0) -> bytes:
    req = Request(url, headers={"User-Agent": "pilotless-aircraft-sim/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise RuntimeError(f"No rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_airports(out_path: Path) -> int:
    data = _download_bytes(OURAIRPORTS_URL)
    df = pd.read_csv(io.BytesIO(data))
    runway_col = None
    for candidate in ["longest_runway_length_ft", "longest_runway_length"]:
        if candidate in df.columns:
            runway_col = candidate
            break

    if runway_col is None:
        runways = pd.read_csv(io.BytesIO(_download_bytes(OURAIRPORTS_RUNWAYS_URL)))
        runway_lengths = runways.groupby("airport_ident")["length_ft"].max().reset_index()
        df = df.merge(runway_lengths, left_on="ident", right_on="airport_ident", how="left")
        runway_col = "length_ft"

    df = df[
        df["iata_code"].notna()
        & df["latitude_deg"].notna()
        & df["longitude_deg"].notna()
        & df["elevation_ft"].notna()
        & df[runway_col].notna()
    ].copy()

    df = df[
        [
            "iata_code",
            "name",
            "latitude_deg",
            "longitude_deg",
            "elevation_ft",
            runway_col,
        ]
    ]
    df.rename(
        columns={
            "iata_code": "iata",
            "latitude_deg": "latitude",
            "longitude_deg": "longitude",
            "elevation_ft": "elevation_ft",
            runway_col: "runway_ft",
        },
        inplace=True,
    )
    df["elevation_m"] = df["elevation_ft"] * 0.3048
    df["runway_m"] = df["runway_ft"] * 0.3048
    df = df[["iata", "name", "latitude", "longitude", "elevation_m", "runway_m"]]

    df.to_csv(out_path, index=False)
    return len(df)


def build_cmapss(out_path: Path) -> int:
    data = _download_bytes(CMAPSS_ZIP_URL)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        target = None
        for name in zf.namelist():
            if name.lower().endswith("train_fd001.txt"):
                target = name
                break

        if target is None:
            nested_name = None
            for name in zf.namelist():
                if name.lower().endswith("cmapssdata.zip"):
                    nested_name = name
                    break
            if nested_name is None:
                raise RuntimeError("Could not find train_FD001.txt or CMAPSSData.zip in C-MAPSS zip")

            with zf.open(nested_name) as nested_handle:
                nested_bytes = nested_handle.read()
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested_zip:
                for name in nested_zip.namelist():
                    if name.lower().endswith("train_fd001.txt"):
                        target = name
                        break
                if target is None:
                    raise RuntimeError("Could not find train_FD001.txt in nested CMAPSSData.zip")
                with nested_zip.open(target) as handle:
                    raw = pd.read_csv(handle, sep=r"\s+", header=None)
        else:
            with zf.open(target) as handle:
                raw = pd.read_csv(handle, sep=r"\s+", header=None)

    columns = [
        "unit_nr",
        "time_cycles",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
    ] + [f"sensor_{i}" for i in range(1, 22)]
    raw.columns = columns

    max_cycles = raw.groupby("unit_nr")["time_cycles"].transform("max")
    raw["failure_within_30"] = ((max_cycles - raw["time_cycles"]) <= 30).astype(int)

    keep = [
        "unit_nr",
        "time_cycles",
        "op_setting_1",
        "op_setting_2",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_11",
        "sensor_12",
        "sensor_15",
        "sensor_21",
        "failure_within_30",
    ]
    out = raw[keep]
    out.to_csv(out_path, index=False)
    return len(out)


def build_opensky(out_path: Path, limit: int = 200) -> int:
    params = {
        "lamin": 32.0,
        "lomin": -125.0,
        "lamax": 38.5,
        "lomax": -115.0,
    }
    url = f"{OPENSKY_URL}?{urlencode(params)}"
    payload = json.loads(_download_bytes(url))
    states = payload.get("states") or []

    rows = []
    for state in states:
        if len(state) < 14:
            continue
        icao24 = state[0]
        callsign = (state[1] or "").strip() or icao24
        lon = state[5]
        lat = state[6]
        baro_alt = state[7]
        velocity = state[9]
        heading = state[10]
        geo_alt = state[13]

        if lat is None or lon is None or velocity is None or heading is None:
            continue
        altitude = geo_alt if geo_alt is not None else baro_alt
        if altitude is None:
            continue

        rows.append(
            {
                "callsign": callsign,
                "latitude": float(lat),
                "longitude": float(lon),
                "altitude_m": float(altitude),
                "velocity_mps": float(velocity),
                "heading_deg": float(heading),
            }
        )
        if len(rows) >= limit:
            break

    _write_csv(out_path, rows)
    return len(rows)


def _guess_name(properties: dict) -> str:
    for key in ["NAME", "NAME_", "SUA_NAME", "AIRSPACE", "AREA_NAME", "TYPE", "DESIG", "ID"]:
        if key in properties and properties[key]:
            return str(properties[key])
    return "Restricted Area"


def build_no_fly_zones(out_path: Path, limit: int = 120) -> int:
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson",
        "outSR": 4326,
        "resultRecordCount": limit,
    }
    url = f"{SUA_ARCGIS_URL}?{urlencode(params)}"
    payload = json.loads(_download_bytes(url))
    features = payload.get("features", [])

    cleaned = []
    for feature in features:
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not coords:
            continue

        if gtype == "MultiPolygon":
            coords = coords[0]
            gtype = "Polygon"
        if gtype != "Polygon":
            continue

        props = feature.get("properties") or {}
        cleaned.append(
            {
                "type": "Feature",
                "properties": {
                    "name": _guess_name(props),
                    "type": props.get("TYPE") or props.get("CLASS") or "SUA",
                },
                "geometry": {"type": "Polygon", "coordinates": coords},
            }
        )

    out_geojson = {"type": "FeatureCollection", "features": cleaned}
    out_path.write_text(json.dumps(out_geojson), encoding="utf-8")
    return len(cleaned)


def _fetch_open_meteo(lat: float, lon: float, when_utc: datetime) -> dict | None:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation_probability",
        "timezone": "UTC",
        "forecast_days": 2,
    }
    url = f"{OPEN_METEO_URL}?{urlencode(params)}"
    payload = json.loads(_download_bytes(url, timeout=20.0))
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


def build_weather(out_path: Path, airports_path: Path, limit: int = 30) -> int:
    df = pd.read_csv(airports_path)
    df = df.sort_values("runway_m", ascending=False).head(limit)
    when_utc = datetime.now(timezone.utc)

    rows = []
    for _, row in df.iterrows():
        try:
            data = _fetch_open_meteo(float(row["latitude"]), float(row["longitude"]), when_utc)
        except Exception:
            data = None

        if not data:
            continue

        rows.append(
            {
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                **data,
            }
        )
        time.sleep(0.15)

    _write_csv(out_path, rows)
    return len(rows)


@dataclass
class EngineGrid:
    weight_kg: list[float]
    speed_kts: list[float]
    altitude_ft: list[float]
    delta_t_c: list[float]


def _isa_temp_c(alt_ft: float) -> float:
    # ISA lapse rate up to 36k ft: 2C per 1000 ft.
    return 15.0 - 2.0 * (alt_ft / 1000.0)


def build_engine_performance(out_path: Path, aircraft: str = "E190") -> int:
    try:
        from openap import FuelFlow
    except Exception as exc:  # pragma: no cover - requires optional dependency
        raise RuntimeError(
            "openap is required to build engine_performance.csv. "
            "Install it with: pip install openap"
        ) from exc

    grid = EngineGrid(
        weight_kg=[20000.0, 24000.0, 28000.0, 32000.0],
        speed_kts=[210.0, 230.0, 250.0, 270.0],
        altitude_ft=[10000.0, 20000.0, 24000.0, 30000.0],
        delta_t_c=[-10.0, 0.0, 10.0],
    )

    ff = FuelFlow(ac=aircraft)
    rows = []
    for weight in grid.weight_kg:
        mass_kg = max(10000.0, weight)
        for speed in grid.speed_kts:
            tas_mps = speed * 0.514444
            for alt in grid.altitude_ft:
                for dT in grid.delta_t_c:
                    isa_temp = _isa_temp_c(alt)
                    oat_c = isa_temp + dT
                    # openap FuelFlow returns kg/s
                    kgps = float(ff.enroute(mass=mass_kg, tas=tas_mps, alt=alt * 0.3048, dT=dT))
                    rows.append(
                        {
                            "weight_kg": weight,
                            "cruise_speed_kts": speed,
                            "altitude_ft": alt,
                            "outside_temp_c": round(oat_c, 2),
                            "fuel_flow_kgph": round(kgps * 3600.0, 2),
                        }
                    )

    _write_csv(out_path, rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and build datasets for the pilotless aircraft simulation")
    parser.add_argument("--skip-airports", action="store_true")
    parser.add_argument("--skip-cmapss", action="store_true")
    parser.add_argument("--skip-opensky", action="store_true")
    parser.add_argument("--skip-no-fly", action="store_true")
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--skip-engine", action="store_true")
    parser.add_argument("--opensky-limit", type=int, default=200)
    parser.add_argument("--weather-limit", type=int, default=30)
    parser.add_argument("--engine-aircraft", default="E190")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_airports:
        count = build_airports(DATASETS_DIR / "airports.csv")
        print(f"airports.csv: {count} rows")

    if not args.skip_cmapss:
        count = build_cmapss(DATASETS_DIR / "cmapss_engine_health_sample.csv")
        print(f"cmapss_engine_health_sample.csv: {count} rows")

    if not args.skip_engine:
        count = build_engine_performance(DATASETS_DIR / "engine_performance.csv", aircraft=args.engine_aircraft)
        print(f"engine_performance.csv: {count} rows")

    if not args.skip_opensky:
        count = build_opensky(DATASETS_DIR / "opensky_traffic_sample.csv", limit=args.opensky_limit)
        print(f"opensky_traffic_sample.csv: {count} rows")

    if not args.skip_no_fly:
        count = build_no_fly_zones(DATASETS_DIR / "no_fly_zones.geojson")
        print(f"no_fly_zones.geojson: {count} features")

    if not args.skip_weather:
        airports_path = DATASETS_DIR / "airports.csv"
        if not airports_path.exists():
            raise RuntimeError("airports.csv is required to build weather_fallback.csv")
        count = build_weather(DATASETS_DIR / "weather_fallback.csv", airports_path, limit=args.weather_limit)
        print(f"weather_fallback.csv: {count} rows")


if __name__ == "__main__":
    main()
