from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from modules.models import FuelEstimate


@dataclass(frozen=True)
class EnginePoint:
    weight_kg: float
    cruise_speed_kts: float
    altitude_ft: float
    outside_temp_c: float
    fuel_flow_kgph: float


def load_engine_dataset(csv_path: str | Path) -> List[EnginePoint]:
    points: List[EnginePoint] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append(
                EnginePoint(
                    weight_kg=float(row["weight_kg"]),
                    cruise_speed_kts=float(row["cruise_speed_kts"]),
                    altitude_ft=float(row["altitude_ft"]),
                    outside_temp_c=float(row["outside_temp_c"]),
                    fuel_flow_kgph=float(row["fuel_flow_kgph"]),
                )
            )
    return points


def _distance_sq(point: EnginePoint, weight_kg: float, speed_kts: float, altitude_ft: float, oat_c: float) -> float:
    return (
        ((point.weight_kg - weight_kg) / 6000.0) ** 2
        + ((point.cruise_speed_kts - speed_kts) / 60.0) ** 2
        + ((point.altitude_ft - altitude_ft) / 15000.0) ** 2
        + ((point.outside_temp_c - oat_c) / 30.0) ** 2
    )


def predict_fuel_flow_kgph(
    dataset: List[EnginePoint],
    weight_kg: float,
    speed_kts: float,
    altitude_ft: float,
    oat_c: float,
    k: int = 4,
) -> float:
    ranked = sorted(
        dataset,
        key=lambda p: _distance_sq(p, weight_kg=weight_kg, speed_kts=speed_kts, altitude_ft=altitude_ft, oat_c=oat_c),
    )[: max(1, k)]
    return sum(p.fuel_flow_kgph for p in ranked) / len(ranked)


def estimate_mission_fuel(
    route_distance_nm: float,
    avg_groundspeed_kts: float,
    predicted_burn_kgph: float,
    reserve_minutes: float = 45.0,
) -> FuelEstimate:
    flight_hours = route_distance_nm / max(80.0, avg_groundspeed_kts)
    cruise_fuel = flight_hours * predicted_burn_kgph
    reserve_fuel = (reserve_minutes / 60.0) * predicted_burn_kgph
    total = cruise_fuel + reserve_fuel

    return FuelEstimate(
        cruise_fuel_kg=round(cruise_fuel, 1),
        reserve_fuel_kg=round(reserve_fuel, 1),
        total_required_kg=round(total, 1),
        predicted_burn_kgph=round(predicted_burn_kgph, 1),
    )
