from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from modules.models import RoutePlan
from modules.route_planner import haversine_nm


@dataclass(frozen=True)
class TrafficContact:
    callsign: str
    latitude: float
    longitude: float
    altitude_m: float
    velocity_mps: float
    heading_deg: float


@dataclass(frozen=True)
class CollisionAlert:
    waypoint_idx: int
    callsign: str
    horizontal_nm: float
    vertical_ft: float
    severity: str
    avoidance_action: str


def load_traffic(csv_path: str | Path) -> List[TrafficContact]:
    contacts: List[TrafficContact] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            contacts.append(
                TrafficContact(
                    callsign=row["callsign"].strip(),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    altitude_m=float(row["altitude_m"]),
                    velocity_mps=float(row["velocity_mps"]),
                    heading_deg=float(row["heading_deg"]),
                )
            )
    return contacts


def detect_collision_risks(
    route: RoutePlan,
    nav_altitude_ft: List[float],
    contacts: List[TrafficContact],
) -> List[CollisionAlert]:
    alerts: List[CollisionAlert] = []

    for idx, wp in enumerate(route.waypoints[:-1]):
        ownship_alt_ft = nav_altitude_ft[min(idx, len(nav_altitude_ft) - 1)]

        for contact in contacts:
            horizontal_nm = haversine_nm(wp.latitude, wp.longitude, contact.latitude, contact.longitude)
            contact_alt_ft = contact.altitude_m * 3.28084
            vertical_ft = abs(ownship_alt_ft - contact_alt_ft)

            if horizontal_nm < 5.0 and vertical_ft < 1000.0:
                severity = "critical"
                action = "Immediate: turn +20 deg and climb +1500 ft"
            elif horizontal_nm < 8.0 and vertical_ft < 1500.0:
                severity = "warning"
                action = "Preventive: turn +10 deg and climb +700 ft"
            else:
                continue

            alerts.append(
                CollisionAlert(
                    waypoint_idx=idx,
                    callsign=contact.callsign,
                    horizontal_nm=round(horizontal_nm, 2),
                    vertical_ft=round(vertical_ft, 1),
                    severity=severity,
                    avoidance_action=action,
                )
            )

    return alerts
