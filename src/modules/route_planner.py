from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

from modules.models import Airport, RoutePlan, Waypoint


EARTH_RADIUS_NM = 3440.065


def load_airports(csv_path: str | Path) -> Dict[str, Airport]:
    airports: Dict[str, Airport] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            airport = Airport(
                iata=row["iata"].strip().upper(),
                name=row["name"].strip(),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                elevation_m=float(row["elevation_m"]),
                runway_m=float(row["runway_m"]),
            )
            airports[airport.iata] = airport
    return airports


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_NM * c


def _interpolate_waypoints(departure: Airport, arrival: Airport, waypoint_count: int) -> List[Waypoint]:
    if waypoint_count < 2:
        raise ValueError("waypoint_count must be >= 2")

    total_distance = haversine_nm(departure.latitude, departure.longitude, arrival.latitude, arrival.longitude)
    waypoints: List[Waypoint] = []

    for idx in range(waypoint_count):
        frac = idx / (waypoint_count - 1)
        lat = departure.latitude + (arrival.latitude - departure.latitude) * frac
        lon = departure.longitude + (arrival.longitude - departure.longitude) * frac
        dist = total_distance * frac
        waypoints.append(Waypoint(idx=idx, latitude=lat, longitude=lon, distance_from_start_nm=dist))

    return waypoints


def plan_route(airports: Dict[str, Airport], departure_iata: str, arrival_iata: str, waypoint_count: int = 16) -> RoutePlan:
    departure = airports[departure_iata.upper()]
    arrival = airports[arrival_iata.upper()]
    waypoints = _interpolate_waypoints(departure, arrival, waypoint_count=waypoint_count)

    return RoutePlan(
        departure=departure,
        arrival=arrival,
        waypoints=waypoints,
        total_distance_nm=waypoints[-1].distance_from_start_nm,
    )
