from __future__ import annotations

from typing import Iterable, List

from modules.models import FuelEstimate, LandingDecision, TakeoffDecision, WeatherSnapshot


def evaluate_emergency_risks(
    weather: Iterable[WeatherSnapshot],
    takeoff: TakeoffDecision,
    landing: LandingDecision,
    fuel: FuelEstimate,
    initial_fuel_kg: float,
    collision_alerts_count: int = 0,
    maintenance_failure_probability: float = 0.0,
) -> List[str]:
    alerts: List[str] = []

    if not takeoff.approved:
        alerts.append("Critical: takeoff not approved by performance checks")
    if not landing.approved:
        alerts.append("Critical: landing not approved for destination conditions")

    max_wind = max((w.wind_speed_kph for w in weather), default=0.0)
    max_precip = max((w.precip_probability for w in weather), default=0.0)

    if max_wind > 65.0:
        alerts.append(f"High wind risk on route ({max_wind:.1f} kph)")
    if max_precip > 70.0:
        alerts.append(f"Severe precipitation probability detected ({max_precip:.0f}%)")

    if collision_alerts_count > 0:
        alerts.append(f"Collision avoidance triggered for {collision_alerts_count} traffic conflicts")

    margin = initial_fuel_kg - fuel.total_required_kg
    if margin < 0:
        alerts.append(f"Fuel deficit of {abs(margin):.1f} kg")
    elif margin < 300:
        alerts.append("Low fuel margin (<300 kg)")

    if maintenance_failure_probability >= 0.8:
        alerts.append("Engine failure risk critical: maintenance required before mission")
    elif maintenance_failure_probability >= 0.5:
        alerts.append("Engine failure risk elevated: shorten mission window and inspect")

    if not alerts:
        alerts.append("No emergency triggers detected")

    return alerts
