from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass(frozen=True)
class Airport:
    iata: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float
    runway_m: float


@dataclass(frozen=True)
class Waypoint:
    idx: int
    latitude: float
    longitude: float
    distance_from_start_nm: float


@dataclass(frozen=True)
class WeatherSnapshot:
    timestamp_utc: datetime
    latitude: float
    longitude: float
    temperature_c: float
    wind_speed_kph: float
    wind_direction_deg: float
    precip_probability: float
    source: str


@dataclass(frozen=True)
class RoutePlan:
    departure: Airport
    arrival: Airport
    waypoints: List[Waypoint]
    total_distance_nm: float


@dataclass(frozen=True)
class TakeoffDecision:
    approved: bool
    required_runway_m: float
    reason: str


@dataclass(frozen=True)
class LandingDecision:
    approved: bool
    required_runway_m: float
    reason: str


@dataclass(frozen=True)
class FuelEstimate:
    cruise_fuel_kg: float
    reserve_fuel_kg: float
    total_required_kg: float
    predicted_burn_kgph: float


@dataclass(frozen=True)
class SecurityAlert:
    level: str
    message: str
    at_index: int
