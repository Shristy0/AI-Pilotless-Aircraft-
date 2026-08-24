from __future__ import annotations

from modules.models import Airport, TakeoffDecision, WeatherSnapshot


def evaluate_takeoff(
    airport: Airport,
    weather: WeatherSnapshot,
    gross_weight_kg: float,
) -> TakeoffDecision:
    # Baseline requirement for a medium UAV/aircraft class.
    base_required = 1050.0
    weight_factor = (gross_weight_kg - 18000.0) * 0.03
    temp_factor = max(0.0, weather.temperature_c - 15.0) * 6.0

    # Assume headwind helps and tailwind hurts.
    headwind_component = weather.wind_speed_kph * 0.15
    required = base_required + weight_factor + temp_factor - headwind_component
    required = max(700.0, required)

    approved = airport.runway_m >= required
    reason = "Takeoff approved" if approved else "Runway too short for current weight/weather"

    return TakeoffDecision(approved=approved, required_runway_m=round(required, 1), reason=reason)
