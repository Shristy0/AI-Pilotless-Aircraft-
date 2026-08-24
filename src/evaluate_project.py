from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai_training.engine_training import train_predictive_maintenance_classifier
from modules.atc_system import check_no_fly_conflicts
from modules.collision_avoidance import detect_collision_risks, load_traffic
from modules.cybersecurity import detect_telemetry_anomalies
from modules.emergency_system import evaluate_emergency_risks
from modules.engine_ai import estimate_mission_fuel, load_engine_dataset, predict_fuel_flow_kgph
from modules.landing import evaluate_landing
from modules.navigation import build_navigation_profile
from modules.predictive_maintenance import load_engine_health_dataset, predict_failure_probabilities
from modules.route_planner import load_airports, plan_route
from modules.takeoff import evaluate_takeoff
from modules.weather_ai import collect_route_weather


def _run_scenario(root: Path, dep: str, arr: str) -> dict:
    timings_ms = {}
    t0 = time.perf_counter() 

    airports = load_airports(root / "datasets" / "airports.csv")
    route = plan_route(airports, dep, arr, waypoint_count=16)
    timings_ms["route"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    weather, _ = collect_route_weather(
        route.waypoints,
        when_utc=pd.Timestamp.utcnow().to_pydatetime(),
        fallback_csv=root / "datasets" / "weather_fallback.csv",
        prefer_live=False,
    )
    timings_ms["weather"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    nav = build_navigation_profile(route, weather, cruise_altitude_ft=24000.0, cruise_tas_kts=250.0)
    timings_ms["navigation"] = (time.perf_counter() - t0) * 1000.0

    avg_temp = sum(w.temperature_c for w in weather) / max(1, len(weather))

    t0 = time.perf_counter()
    engine = load_engine_dataset(root / "datasets" / "engine_performance.csv")
    burn = predict_fuel_flow_kgph(engine, weight_kg=24000.0, speed_kts=250.0, altitude_ft=24000.0, oat_c=avg_temp)
    fuel = estimate_mission_fuel(route.total_distance_nm, nav.avg_groundspeed_kts, burn)
    timings_ms["engine"] = (time.perf_counter() - t0) * 1000.0
    initial_fuel_kg = max(5400.0, fuel.total_required_kg + 800.0)

    t0 = time.perf_counter()
    takeoff = evaluate_takeoff(route.departure, weather[0], gross_weight_kg=24000.0)
    landing = evaluate_landing(route.arrival, weather[-1], landing_weight_kg=max(12000.0, 24000.0 - fuel.cruise_fuel_kg))
    timings_ms["takeoff_landing"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    atc_conflicts = check_no_fly_conflicts(route, root / "datasets" / "no_fly_zones.geojson")
    timings_ms["atc"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    contacts = load_traffic(root / "datasets" / "opensky_traffic_sample.csv")
    altitude_series = [step.target_altitude_ft for step in nav.steps]
    collision_alerts = detect_collision_risks(route, nav_altitude_ft=altitude_series, contacts=contacts)
    timings_ms["collision"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    groundspeed_series = [step.estimated_groundspeed_kts for step in nav.steps]
    security_alerts = detect_telemetry_anomalies(altitude_series, groundspeed_series)
    timings_ms["cyber"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    maint = predict_failure_probabilities(load_engine_health_dataset(root / "datasets" / "cmapss_engine_health_sample.csv"))
    maint_risk = maint[0].failure_probability_30_cycles if maint else 0.0
    emergency = evaluate_emergency_risks(
        weather,
        takeoff,
        landing,
        fuel,
        initial_fuel_kg=initial_fuel_kg,
        collision_alerts_count=len(collision_alerts),
        maintenance_failure_probability=maint_risk,
    )
    timings_ms["maintenance_emergency"] = (time.perf_counter() - t0) * 1000.0

    critical_cyber = sum(1 for a in security_alerts if a.level == "critical")
    critical_collision = sum(1 for a in collision_alerts if a.severity == "critical")
    fuel_margin = initial_fuel_kg - fuel.total_required_kg

    success = (
        takeoff.approved
        and landing.approved
        and len(atc_conflicts) == 0
        and critical_collision == 0
        and critical_cyber == 0
        and fuel_margin >= 0
    )

    return {
        "route": f"{dep}-{arr}",
        "distance_nm": round(route.total_distance_nm, 2),
        "fuel_required_kg": fuel.total_required_kg,
        "fuel_margin_kg": round(fuel_margin, 2),
        "atc_conflicts": len(atc_conflicts),
        "collision_alerts": len(collision_alerts),
        "critical_cyber_alerts": critical_cyber,
        "maintenance_failure_prob": maint_risk,
        "mission_success": int(success),
        "response_ms_total": round(sum(timings_ms.values()), 3),
        **{f"response_ms_{k}": round(v, 3) for k, v in timings_ms.items()},
        "emergency_top": emergency[0] if emergency else "None",
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)

    scenarios = [("SFO", "LAX"), ("SEA", "LAS"), ("JFK", "LAX"), ("LHR", "DXB"), ("SIN", "DXB")]
    rows = [_run_scenario(root, dep, arr) for dep, arr in scenarios]
    df = pd.DataFrame(rows)

    csv_path = out_dir / "evaluation_scenarios.csv"
    df.to_csv(csv_path, index=False)

    success_rate = float(df["mission_success"].mean())
    avg_response_ms = float(df["response_ms_total"].mean())
    p95_response_ms = float(df["response_ms_total"].quantile(0.95))

    maintenance_metrics = train_predictive_maintenance_classifier(root / "datasets" / "cmapss_engine_health_sample.csv")

    deep_learning_metrics = {}
    dl_path = out_dir / "deep_learning_metrics.json"
    if dl_path.exists():
        deep_learning_metrics = json.loads(dl_path.read_text(encoding="utf-8"))

    summary = {
        "quantitative_analysis": {
            "scenario_count": len(df),
            "mission_success_rate": round(success_rate, 4),
            "avg_response_efficiency_ms": round(avg_response_ms, 3),
            "p95_response_efficiency_ms": round(p95_response_ms, 3),
            "avg_fuel_margin_kg": round(float(df["fuel_margin_kg"].mean()), 2),
        },
        "prediction_reliability": maintenance_metrics,
        "deep_learning_metrics": deep_learning_metrics,
    }

    json_path = out_dir / "evaluation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(df["route"], df["response_ms_total"], color="#1f77b4")
    axes[0].set_title("Response Efficiency by Route")
    axes[0].set_ylabel("Milliseconds")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(df["route"], df["fuel_margin_kg"], color="#2ca02c")
    axes[1].set_title("Fuel Margin by Route")
    axes[1].set_ylabel("kg")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig_path = out_dir / "evaluation_plots.png"
    fig.savefig(fig_path, dpi=140)

    print(f"Saved scenario table: {csv_path}")
    print(f"Saved summary: {json_path}")
    print(f"Saved plots: {fig_path}")


if __name__ == "__main__":
    main()
