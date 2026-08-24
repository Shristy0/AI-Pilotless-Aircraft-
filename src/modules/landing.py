from __future__ import annotations

from modules.models import Airport, LandingDecision, WeatherSnapshot


def evaluate_landing(
    airport: Airport,
    weather: WeatherSnapshot,
    landing_weight_kg: float,
) -> LandingDecision:
    base_required = 920.0
    weight_factor = (landing_weight_kg - 16000.0) * 0.025
    precip_factor = weather.precip_probability * 1.8
    wind_penalty = max(0.0, weather.wind_speed_kph - 35.0) * 2.5

    required = base_required + weight_factor + precip_factor + wind_penalty
    required = max(700.0, required)

    approved = airport.runway_m >= required
    reason = "Landing approved" if approved else "Landing risk high for runway/weather"

    return LandingDecision(approved=approved, required_runway_m=round(required, 1), reason=reason)
