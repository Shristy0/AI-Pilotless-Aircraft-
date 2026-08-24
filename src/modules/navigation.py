from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from modules.models import RoutePlan, WeatherSnapshot


@dataclass(frozen=True)
class NavigationStep:
    waypoint_idx: int
    target_altitude_ft: float
    true_airspeed_kts: float
    heading_deg: float
    estimated_groundspeed_kts: float


@dataclass(frozen=True)
class NavigationProfile:
    steps: List[NavigationStep]
    avg_groundspeed_kts: float


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)

    y = math.sin(dlon_r) * math.cos(lat2_r)
    x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def build_navigation_profile(
    route: RoutePlan,
    weather: List[WeatherSnapshot],
    cruise_altitude_ft: float = 24000.0,
    cruise_tas_kts: float = 250.0,
) -> NavigationProfile:
    steps: List[NavigationStep] = []

    for idx in range(len(route.waypoints) - 1):
        cur = route.waypoints[idx]
        nxt = route.waypoints[idx + 1]
        wx = weather[idx]

        heading = _bearing_deg(cur.latitude, cur.longitude, nxt.latitude, nxt.longitude)

        # Approximate tailwind/headwind projection onto route heading.
        wind_to_deg = (wx.wind_direction_deg + 180.0) % 360.0
        diff = math.radians(wind_to_deg - heading)
        tailwind_kts = (wx.wind_speed_kph / 1.852) * math.cos(diff)
        groundspeed = max(90.0, cruise_tas_kts + tailwind_kts)

        # Simple climb-cruise-descent profile.
        phase = idx / max(1, len(route.waypoints) - 2)
        if phase < 0.25:
            altitude = 5000.0 + phase / 0.25 * (cruise_altitude_ft - 5000.0)
        elif phase > 0.80:
            descent_frac = (phase - 0.80) / 0.20
            altitude = cruise_altitude_ft - descent_frac * (cruise_altitude_ft - 4000.0)
        else:
            altitude = cruise_altitude_ft

        steps.append(
            NavigationStep(
                waypoint_idx=idx,
                target_altitude_ft=round(altitude, 1),
                true_airspeed_kts=cruise_tas_kts,
                heading_deg=round(heading, 1),
                estimated_groundspeed_kts=round(groundspeed, 1),
            )
        )

    avg_gs = sum(s.estimated_groundspeed_kts for s in steps) / max(1, len(steps))
    return NavigationProfile(steps=steps, avg_groundspeed_kts=round(avg_gs, 1))
