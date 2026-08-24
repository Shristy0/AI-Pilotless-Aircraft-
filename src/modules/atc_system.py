from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from modules.models import RoutePlan


def _point_in_polygon(lon: float, lat: float, polygon: list[list[float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > lat) != (yj > lat)
        if intersects:
            x_cross = (xj - xi) * (lat - yi) / ((yj - yi) or 1e-9) + xi
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def check_no_fly_conflicts(route: RoutePlan, no_fly_geojson: str | Path) -> List[str]:
    with Path(no_fly_geojson).open(encoding="utf-8") as handle:
        features = json.load(handle)["features"]

    conflicts: List[str] = []

    for wp in route.waypoints:
        for feature in features:
            geom = feature.get("geometry", {})
            if geom.get("type") != "Polygon":
                continue

            polygon = geom["coordinates"][0]
            if _point_in_polygon(wp.longitude, wp.latitude, polygon):
                conflicts.append(
                    f"Waypoint {wp.idx} intersects no-fly zone: {feature['properties'].get('name', 'unknown')}"
                )

    return conflicts


def build_atc_messages(route: RoutePlan, conflicts: Iterable[str]) -> List[str]:
    messages = [
        f"ATC PRE-CLEARANCE: {route.departure.iata} -> {route.arrival.iata}",
        f"Filed route length: {route.total_distance_nm:.1f} NM",
    ]

    conflict_list = list(conflicts)
    if conflict_list:
        messages.append("ATC HOLD: route intersects restricted airspace")
        messages.extend(conflict_list)
    else:
        messages.append("ATC CLEAR: no restricted-airspace conflicts detected")

    return messages
