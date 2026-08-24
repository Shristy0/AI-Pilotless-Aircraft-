from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from modules.atc_system import build_atc_messages, check_no_fly_conflicts
from modules.collision_avoidance import detect_collision_risks, load_traffic
from modules.cybersecurity import detect_telemetry_anomalies
from modules.emergency_system import evaluate_emergency_risks
from modules.engine_ai import estimate_mission_fuel, load_engine_dataset, predict_fuel_flow_kgph
from modules.hmi import build_hmi_summary
from modules.landing import evaluate_landing
from modules.navigation import build_navigation_profile
from modules.predictive_maintenance import load_engine_health_dataset, predict_failure_probabilities
from modules.route_planner import load_airports, plan_route
from modules.takeoff import evaluate_takeoff
from modules.unconventional_response import response_to_dict, run_unconventional_scenario
from modules.weather_ai import collect_route_weather
from simulators import check_backend, launch_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pilotless aircraft proposal implementation with real datasets")
    parser.add_argument("--departure", default="SFO", help="Departure airport IATA")
    parser.add_argument("--arrival", default="LAX", help="Arrival airport IATA")
    parser.add_argument("--weight-kg", type=float, default=24000.0, help="Initial gross weight")
    parser.add_argument("--initial-fuel-kg", type=float, default=5400.0, help="Fuel at departure")
    parser.add_argument(
        "--mission-time-utc",
        default="",
        help="Mission datetime in UTC, ISO format (example: 2026-03-09T12:00:00)",
    )
    parser.add_argument(
        "--offline-weather",
        action="store_true",
        help="Use local weather fallback CSV only (faster, deterministic)",
    )
    parser.add_argument(
        "--scenario",
        choices=["normal", "unconventional"],
        default="normal",
        help="Run nominal mission or inject unconventional event responses",
    )
    parser.add_argument(
        "--events",
        default="",
        help=(
            "Comma-separated unconventional events. "
            "Options: gnss_spoofing,engine_thrust_loss,convective_weather_burst,"
            "atc_link_loss,destination_runway_blocked,cyber_intrusion_attempt"
        ),
    )
    parser.add_argument(
        "--sim-backend",
        choices=["local", "jsbsim", "flightgear"],
        default="local",
        help="Simulation backend (local python simulation, JSBSim, or FlightGear)",
    )
    parser.add_argument(
        "--sim-config",
        default="",
        help="Path to simulator config JSON (required for JSBSim/FlightGear launch)",
    )
    parser.add_argument(
        "--launch-sim",
        action="store_true",
        help="Launch external simulator if sim-config is provided",
    )
    return parser.parse_args()


def _resolve_mission_time(raw: str) -> datetime:
    if not raw.strip():
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _build_navigation_fuel_profile(route, navigation, initial_fuel_kg: float, predicted_burn_kgph: float) -> list[dict]:
    rows: list[dict] = []
    cumulative_fuel_kg = 0.0
    elapsed_hr = 0.0

    for idx, step in enumerate(navigation.steps):
        seg_nm = max(
            0.0,
            route.waypoints[idx + 1].distance_from_start_nm - route.waypoints[idx].distance_from_start_nm,
        )
        seg_hr = seg_nm / max(90.0, step.estimated_groundspeed_kts)
        seg_fuel_kg = predicted_burn_kgph * seg_hr

        elapsed_hr += seg_hr
        cumulative_fuel_kg += seg_fuel_kg

        rows.append(
            {
                "waypoint_idx": idx + 1,
                "distance_nm": round(route.waypoints[idx + 1].distance_from_start_nm, 2),
                "elapsed_min": round(elapsed_hr * 60.0, 2),
                "target_altitude_ft": step.target_altitude_ft,
                "groundspeed_kts": step.estimated_groundspeed_kts,
                "segment_fuel_kg": round(seg_fuel_kg, 2),
                "cumulative_fuel_kg": round(cumulative_fuel_kg, 2),
                "remaining_fuel_kg": round(initial_fuel_kg - cumulative_fuel_kg, 2),
            }
        )

    return rows


def _save_profile_csv(rows: list[dict], out_csv_path: Path) -> None:
    if not rows:
        return
    with out_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_mission_profile_plot(rows: list[dict], out_png_path: Path) -> bool:
    if not rows:
        return False

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    elapsed = [r["elapsed_min"] for r in rows]
    altitude = [r["target_altitude_ft"] for r in rows]
    remaining_fuel = [r["remaining_fuel_kg"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(elapsed, altitude, color="#0c7da8", linewidth=2)
    axes[0].set_ylabel("Altitude (ft)")
    axes[0].set_title("Navigation Profile")
    axes[0].grid(alpha=0.25)

    axes[1].plot(elapsed, remaining_fuel, color="#2a9d8f", linewidth=2)
    axes[1].set_xlabel("Elapsed Time (min)")
    axes[1].set_ylabel("Remaining Fuel (kg)")
    axes[1].set_title("Fuel Profile")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=140)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    sim_status = check_backend(args.sim_backend)
    if not sim_status.available:
        raise SystemExit(f"Simulation backend '{args.sim_backend}' unavailable: {sim_status.message}")
    sim_launch = launch_backend(
        args.sim_backend,
        config_path=args.sim_config if args.sim_config.strip() else None,
        launch=args.launch_sim,
    )

    airports = load_airports(root / "datasets" / "airports.csv")
    route = plan_route(airports, args.departure, args.arrival, waypoint_count=18)

    mission_time = _resolve_mission_time(args.mission_time_utc)
    weather, used_live_data = collect_route_weather(
        route.waypoints,
        when_utc=mission_time,
        fallback_csv=root / "datasets" / "weather_fallback.csv",
        prefer_live=not args.offline_weather,
    )

    navigation = build_navigation_profile(route, weather, cruise_altitude_ft=24000.0, cruise_tas_kts=250.0)

    avg_temp = sum(w.temperature_c for w in weather) / max(1, len(weather))
    engine_dataset = load_engine_dataset(root / "datasets" / "engine_performance.csv")
    predicted_burn = predict_fuel_flow_kgph(
        engine_dataset,
        weight_kg=args.weight_kg,
        speed_kts=250.0,
        altitude_ft=24000.0,
        oat_c=avg_temp,
    )

    fuel = estimate_mission_fuel(route.total_distance_nm, navigation.avg_groundspeed_kts, predicted_burn)
    scenario_mode = args.scenario
    nav_fuel_profile = _build_navigation_fuel_profile(
        route=route,
        navigation=navigation,
        initial_fuel_kg=args.initial_fuel_kg,
        predicted_burn_kgph=fuel.predicted_burn_kgph,
    )

    # Advanced feature: unconventional-event response simulation.
    if scenario_mode == "unconventional":
        default_events = ["gnss_spoofing", "engine_thrust_loss", "convective_weather_burst", "atc_link_loss"]
        parsed = [e.strip() for e in args.events.split(",") if e.strip()]
        selected_events = parsed if parsed else default_events
        unconventional = run_unconventional_scenario(
            route=route,
            airports=airports,
            nominal_fuel_required_kg=fuel.total_required_kg,
            initial_fuel_kg=args.initial_fuel_kg,
            event_names=selected_events,
        )
    else:
        selected_events = []
        unconventional = run_unconventional_scenario(
            route=route,
            airports=airports,
            nominal_fuel_required_kg=fuel.total_required_kg,
            initial_fuel_kg=args.initial_fuel_kg,
            event_names=[],
        )

    # Feature 1 and 3: automatic takeoff and landing.
    takeoff = evaluate_takeoff(route.departure, weather[0], gross_weight_kg=args.weight_kg)
    est_landing_weight = max(12000.0, args.weight_kg - fuel.cruise_fuel_kg)
    landing = evaluate_landing(route.arrival, weather[-1], landing_weight_kg=est_landing_weight)

    # Feature 6: ATC integration and restricted-airspace checks.
    conflicts = check_no_fly_conflicts(route, root / "datasets" / "no_fly_zones.geojson")
    atc_messages = build_atc_messages(route, conflicts)

    # Feature 5: collision avoidance using OpenSky-style traffic snapshots.
    contacts = load_traffic(root / "datasets" / "opensky_traffic_sample.csv")
    altitude_series = [step.target_altitude_ft for step in navigation.steps]
    groundspeed_series = [step.estimated_groundspeed_kts for step in navigation.steps]
    collision_alerts = detect_collision_risks(route, nav_altitude_ft=altitude_series, contacts=contacts)

    # Feature 10 (cybersecurity): anomaly detection on telemetry streams.
    security_alerts = detect_telemetry_anomalies(altitude_series, groundspeed_series)

    # Feature 4: predictive maintenance using C-MAPSS-like engine data.
    health_dataset = load_engine_health_dataset(root / "datasets" / "cmapss_engine_health_sample.csv")
    maintenance_predictions = predict_failure_probabilities(health_dataset)
    top_maintenance_risk = maintenance_predictions[0].failure_probability_30_cycles if maintenance_predictions else 0.0

    # Feature 9: emergency handling rules combining all safety modules.
    emergency_alerts = evaluate_emergency_risks(
        weather=weather,
        takeoff=takeoff,
        landing=landing,
        fuel=fuel,
        initial_fuel_kg=args.initial_fuel_kg,
        collision_alerts_count=len(collision_alerts),
        maintenance_failure_probability=top_maintenance_risk,
    )
    if unconventional.mission_mode != "nominal":
        emergency_alerts.append(f"Unconventional scenario mode: {unconventional.mission_mode}")
    if unconventional.revised_fuel_margin_kg < 0:
        emergency_alerts.append(
            f"Unconventional fuel margin deficit: {abs(unconventional.revised_fuel_margin_kg):.1f} kg"
        )

    # Feature 8: human-machine interface status output.
    hmi_messages = build_hmi_summary(
        mission_label=f"{route.departure.iata}-{route.arrival.iata}",
        takeoff_ok=takeoff.approved,
        landing_ok=landing.approved,
        atc_conflicts=len(conflicts),
        collision_alerts=len(collision_alerts),
        cyber_alerts=len(security_alerts),
        emergency_alerts=emergency_alerts,
    )

    report = {
        "mission": {
            "departure": route.departure.iata,
            "arrival": route.arrival.iata,
            "planned_utc": mission_time.isoformat(),
            "distance_nm": round(route.total_distance_nm, 1),
            "scenario": scenario_mode,
        },
        "simulation_backend": {
            "name": sim_status.backend,
            "status": sim_status.message,
            "launched": sim_launch.launched,
            "launch_status": sim_launch.message,
            "launch_command": sim_launch.command,
        },
        "data_source": {
            "weather": "live-open-meteo" if used_live_data else "fallback-csv",
            "airports": "datasets/airports.csv",
            "engine_performance": "datasets/engine_performance.csv",
            "engine_health": "datasets/cmapss_engine_health_sample.csv",
            "traffic": "datasets/opensky_traffic_sample.csv",
        },
        "feature_coverage": {
            "automatic_takeoff": True,
            "autonomous_flight_navigation": True,
            "automatic_landing": True,
            "engine_failure_prediction": True,
            "collision_avoidance": True,
            "flight_path_management": True,
            "atc_integration": True,
            "weather_forecasting_adaptation": True,
            "emergency_handling": True,
            "human_machine_interface_and_cybersecurity": True,
        },
        "takeoff": asdict(takeoff),
        "landing": asdict(landing),
        "fuel": asdict(fuel),
        "atc_messages": atc_messages,
        "collision_alerts": [asdict(c) for c in collision_alerts],
        "maintenance_predictions": [asdict(m) for m in maintenance_predictions],
        "emergency_alerts": emergency_alerts,
        "security_alerts": [asdict(a) for a in security_alerts],
        "hmi_messages": hmi_messages,
        "unconventional_response": response_to_dict(unconventional),
        "fuel_profile": nav_fuel_profile,
        "navigation_preview": [asdict(s) for s in navigation.steps[:5]],
    }

    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_path = out_dir / f"mission_report_{timestamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "latest_mission_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fuel_profile_csv = out_dir / "latest_fuel_profile.csv"
    _save_profile_csv(nav_fuel_profile, fuel_profile_csv)
    profile_plot_path = out_dir / "latest_mission_profile.png"
    plot_ok = _save_mission_profile_plot(nav_fuel_profile, profile_plot_path)

    print("Mission simulation completed")
    print(f"Route: {route.departure.iata} -> {route.arrival.iata} ({route.total_distance_nm:.1f} NM)")
    print(f"Weather source: {'live-open-meteo' if used_live_data else 'fallback-csv'}")
    print(f"Fuel required: {fuel.total_required_kg:.1f} kg")
    print(f"Takeoff approved: {takeoff.approved} | Landing approved: {landing.approved}")
    print(f"ATC conflicts: {len(conflicts)} | Collision alerts: {len(collision_alerts)}")
    print(f"Maintenance units analyzed: {len(maintenance_predictions)}")
    print(f"Security alerts: {len(security_alerts)}")
    if selected_events:
        print(f"Unconventional events: {', '.join(selected_events)}")
    print(f"Mission mode: {unconventional.mission_mode}")
    print(f"Fuel profile CSV: {fuel_profile_csv}")
    if plot_ok:
        print(f"Mission profile plot: {profile_plot_path}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
