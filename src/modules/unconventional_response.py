from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from modules.models import Airport, RoutePlan
from modules.route_planner import haversine_nm


@dataclass(frozen=True)
class UnconventionalEvent:
    name: str
    severity: str
    detection_signal: str
    autonomous_reaction: str


@dataclass(frozen=True)
class ScenarioResponse:
    mission_mode: str
    revised_distance_nm: float
    revised_fuel_required_kg: float
    revised_fuel_margin_kg: float
    diversion_airport_iata: str
    activated_events: List[UnconventionalEvent]
    autonomous_actions: List[str]


def _nearest_alternate(
    airports: Dict[str, Airport],
    lat: float,
    lon: float,
    exclude_iata: set[str],
    min_runway_m: float = 2200.0,
) -> Optional[Airport]:
    candidates = [a for a in airports.values() if a.iata not in exclude_iata and a.runway_m >= min_runway_m]
    if not candidates:
        return None
    return min(candidates, key=lambda a: haversine_nm(lat, lon, a.latitude, a.longitude))


def run_unconventional_scenario(
    route: RoutePlan,
    airports: Dict[str, Airport],
    nominal_fuel_required_kg: float,
    initial_fuel_kg: float,
    event_names: List[str],
) -> ScenarioResponse:
    events: List[UnconventionalEvent] = []
    actions: List[str] = []

    distance_factor = 1.0
    fuel_factor = 1.0
    mission_mode = "nominal"
    diversion = ""

    for raw in event_names:
        name = raw.strip().lower()
        if not name:
            continue

        if name == "gnss_spoofing":
            events.append(
                UnconventionalEvent(
                    name="GNSS spoofing attempt",
                    severity="high",
                    detection_signal="GNSS/INS position residual exceeded 0.35 NM",
                    autonomous_reaction="Switch to INS + vision-aided navigation and tighten heading envelope",
                )
            )
            actions.append("Cyber mode hardening enabled; external nav packets isolated")
            fuel_factor *= 1.03

        elif name == "engine_thrust_loss":
            events.append(
                UnconventionalEvent(
                    name="Engine thrust degradation",
                    severity="critical",
                    detection_signal="N1/EGT trend mismatch with commanded thrust",
                    autonomous_reaction="Reduce speed to 220 kts and descend to efficient contingency altitude",
                )
            )
            actions.append("Predictive maintenance escalated and emergency power profile enabled")
            fuel_factor *= 1.18
            mission_mode = "degraded"

        elif name == "convective_weather_burst":
            events.append(
                UnconventionalEvent(
                    name="Convective weather burst",
                    severity="high",
                    detection_signal="Wind shear and precipitation spike along route corridor",
                    autonomous_reaction="Reroute around weather cell with avoidance corridor",
                )
            )
            actions.append("Dynamic reroute generated with storm-cell standoff")
            distance_factor *= 1.12
            fuel_factor *= 1.06

        elif name == "atc_link_loss":
            events.append(
                UnconventionalEvent(
                    name="ATC data-link loss",
                    severity="medium",
                    detection_signal="CPDLC timeout > 120 seconds",
                    autonomous_reaction="Hold last valid clearance, fallback to contingency comm protocol",
                )
            )
            actions.append("Autonomy set to procedural hold until comms restored")
            mission_mode = "degraded" if mission_mode == "nominal" else mission_mode

        elif name == "destination_runway_blocked":
            events.append(
                UnconventionalEvent(
                    name="Destination runway blocked",
                    severity="critical",
                    detection_signal="Runway occupancy alert at destination",
                    autonomous_reaction="Compute nearest suitable alternate and execute diversion plan",
                )
            )
            alt = _nearest_alternate(
                airports,
                lat=route.arrival.latitude,
                lon=route.arrival.longitude,
                exclude_iata={route.departure.iata, route.arrival.iata},
            )
            diversion = alt.iata if alt else ""
            actions.append(
                f"Diversion activated to {diversion}" if diversion else "Diversion requested but no suitable alternate in dataset"
            )
            distance_factor *= 1.15
            fuel_factor *= 1.10
            mission_mode = "divert"

        elif name == "cyber_intrusion_attempt":
            events.append(
                UnconventionalEvent(
                    name="Avionics network intrusion attempt",
                    severity="high",
                    detection_signal="Abnormal command pattern and signature mismatch",
                    autonomous_reaction="Segment network, reject unsigned commands, switch to trusted control plane",
                )
            )
            actions.append("Security lockdown applied; non-essential interfaces disabled")
            mission_mode = "degraded" if mission_mode == "nominal" else mission_mode
            fuel_factor *= 1.02

    revised_distance = route.total_distance_nm * distance_factor
    revised_fuel = nominal_fuel_required_kg * distance_factor * fuel_factor
    revised_margin = initial_fuel_kg - revised_fuel

    if revised_margin < 0:
        actions.append("Fuel-critical procedure triggered: priority landing and emergency declaration")
        if mission_mode == "nominal":
            mission_mode = "degraded"

    if not actions:
        actions.append("No unconventional conditions triggered")

    return ScenarioResponse(
        mission_mode=mission_mode,
        revised_distance_nm=round(revised_distance, 1),
        revised_fuel_required_kg=round(revised_fuel, 1),
        revised_fuel_margin_kg=round(revised_margin, 1),
        diversion_airport_iata=diversion,
        activated_events=events,
        autonomous_actions=actions,
    )


def response_to_dict(response: ScenarioResponse) -> dict:
    payload = asdict(response)
    payload["activated_events"] = [asdict(e) for e in response.activated_events]
    return payload
