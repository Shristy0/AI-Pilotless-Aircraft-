from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Keep matplotlib headless and writable for reproducible CLI runs.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from modules.unconventional_response import ScenarioResponse, run_unconventional_scenario
from modules.weather_ai import collect_route_weather


EVENT_PROBABILITIES: dict[str, float] = {
    "gnss_spoofing": 0.08,
    "engine_thrust_loss": 0.06,
    "convective_weather_burst": 0.10,
    "atc_link_loss": 0.07,
    "destination_runway_blocked": 0.03,
    "cyber_intrusion_attempt": 0.06,
}

RESEARCH_PROFILES = {
    "full",
    "no_contingency",
    "no_collision",
    "no_cyber",
    "no_maintenance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-grade Monte Carlo study for pilotless aircraft simulation")
    parser.add_argument("--trials", type=int, default=120, help="Trials per route")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--routes",
        default="SFO-LAX,SEA-LAS,JFK-LAX,LHR-DXB,SIN-DXB",
        help="Comma-separated departure-arrival pairs",
    )
    parser.add_argument(
        "--offline-weather",
        action="store_true",
        help="Use fallback weather only (recommended for reproducible research runs)",
    )
    parser.add_argument(
        "--research-profile",
        choices=sorted(RESEARCH_PROFILES),
        default="full",
        help="Ablation profile for component-wise research experiments",
    )
    parser.add_argument("--event-prob-scale", type=float, default=1.0, help="Scale factor for unconventional event probabilities")
    parser.add_argument("--temp-noise-std", type=float, default=1.8, help="Std-dev (deg C) for temperature perturbation")
    parser.add_argument("--wind-noise-frac", type=float, default=0.12, help="Fractional wind-speed perturbation")
    parser.add_argument("--wind-dir-noise-std", type=float, default=8.0, help="Std-dev (deg) for wind-direction perturbation")
    parser.add_argument("--precip-noise-std", type=float, default=7.5, help="Std-dev (%) for precipitation perturbation")
    parser.add_argument("--output-prefix", default="research", help="Output filename prefix")
    return parser.parse_args()


def _bootstrap_ci(data: np.ndarray, rng: np.random.Generator, n_boot: int = 1200, alpha: float = 0.05) -> tuple[float, float]:
    if data.size == 0:
        return (0.0, 0.0)

    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(data, size=data.size, replace=True)
        boots[i] = float(sample.mean())

    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def _scaled_event_probabilities(scale: float) -> dict[str, float]:
    return {k: float(np.clip(v * scale, 0.0, 0.95)) for k, v in EVENT_PROBABILITIES.items()}


def _sample_events(rng: np.random.Generator, event_probabilities: dict[str, float]) -> list[str]:
    return [event for event, p in event_probabilities.items() if rng.random() < p]


def _perturb_weather_snapshots(
    weather: Iterable,
    rng: np.random.Generator,
    temp_noise_std: float,
    wind_noise_frac: float,
    wind_dir_noise_std: float,
    precip_noise_std: float,
) -> list:
    perturbed = []
    for w in weather:
        t = w.temperature_c + rng.normal(0.0, temp_noise_std)
        ws = max(0.0, w.wind_speed_kph * (1.0 + rng.normal(0.0, wind_noise_frac)))
        wd = (w.wind_direction_deg + rng.normal(0.0, wind_dir_noise_std)) % 360.0
        pp = float(np.clip(w.precip_probability + rng.normal(0.0, precip_noise_std), 0.0, 100.0))
        perturbed.append(
            type(w)(
                timestamp_utc=w.timestamp_utc,
                latitude=w.latitude,
                longitude=w.longitude,
                temperature_c=float(t),
                wind_speed_kph=float(ws),
                wind_direction_deg=float(wd),
                precip_probability=pp,
                source=f"{w.source}+stochastic",
            )
        )
    return perturbed


def _unmanaged_response(route, initial_fuel_kg: float, nominal_fuel_required_kg: float, event_names: list[str]) -> ScenarioResponse:
    distance_factor = 1.0
    fuel_factor = 1.0
    mission_mode = "unmanaged"

    for event in event_names:
        if event == "engine_thrust_loss":
            fuel_factor *= 1.22
        elif event == "convective_weather_burst":
            distance_factor *= 1.08
            fuel_factor *= 1.07
        elif event == "gnss_spoofing":
            fuel_factor *= 1.03
        elif event == "atc_link_loss":
            fuel_factor *= 1.02
        elif event == "destination_runway_blocked":
            distance_factor *= 1.10
            fuel_factor *= 1.05
        elif event == "cyber_intrusion_attempt":
            fuel_factor *= 1.03

    revised_distance = route.total_distance_nm * distance_factor
    revised_fuel = nominal_fuel_required_kg * distance_factor * fuel_factor
    revised_margin = initial_fuel_kg - revised_fuel

    return ScenarioResponse(
        mission_mode=mission_mode,
        revised_distance_nm=round(revised_distance, 1),
        revised_fuel_required_kg=round(revised_fuel, 1),
        revised_fuel_margin_kg=round(revised_margin, 1),
        diversion_airport_iata="",
        activated_events=[],
        autonomous_actions=["Contingency/autonomous response module disabled"],
    )


def _simulate_trial(
    root: Path,
    dep: str,
    arr: str,
    rng: np.random.Generator,
    prefer_live_weather: bool,
    research_profile: str,
    event_probabilities: dict[str, float],
    temp_noise_std: float,
    wind_noise_frac: float,
    wind_dir_noise_std: float,
    precip_noise_std: float,
) -> dict:
    airports = load_airports(root / "datasets" / "airports.csv")
    route = plan_route(airports, dep, arr, waypoint_count=18)

    weather, _ = collect_route_weather(
        route.waypoints,
        when_utc=datetime.now(timezone.utc),
        fallback_csv=root / "datasets" / "weather_fallback.csv",
        prefer_live=prefer_live_weather,
    )
    weather = _perturb_weather_snapshots(
        weather,
        rng,
        temp_noise_std=temp_noise_std,
        wind_noise_frac=wind_noise_frac,
        wind_dir_noise_std=wind_dir_noise_std,
        precip_noise_std=precip_noise_std,
    )

    nav = build_navigation_profile(route, weather, cruise_altitude_ft=24000.0, cruise_tas_kts=250.0)
    avg_temp = float(np.mean([w.temperature_c for w in weather]))

    burn = predict_fuel_flow_kgph(
        load_engine_dataset(root / "datasets" / "engine_performance.csv"),
        weight_kg=24000.0,
        speed_kts=250.0,
        altitude_ft=24000.0,
        oat_c=avg_temp,
    )
    burn *= float(np.clip(rng.normal(1.0, 0.06), 0.88, 1.18))

    fuel = estimate_mission_fuel(route.total_distance_nm, nav.avg_groundspeed_kts, burn)
    initial_fuel = max(5600.0, fuel.total_required_kg + rng.uniform(350.0, 1100.0))

    takeoff = evaluate_takeoff(route.departure, weather[0], gross_weight_kg=24000.0)
    landing = evaluate_landing(route.arrival, weather[-1], landing_weight_kg=max(12000.0, 24000.0 - fuel.cruise_fuel_kg))

    atc_conflicts = check_no_fly_conflicts(route, root / "datasets" / "no_fly_zones.geojson")

    altitude_series = [step.target_altitude_ft for step in nav.steps]
    groundspeed_series = [step.estimated_groundspeed_kts for step in nav.steps]
    events = _sample_events(rng, event_probabilities)

    if "gnss_spoofing" in events and len(groundspeed_series) > 5:
        j = int(rng.integers(2, len(groundspeed_series) - 2))
        groundspeed_series[j] += 135.0

    latent_collision_alerts = detect_collision_risks(
        route,
        nav_altitude_ft=altitude_series,
        contacts=load_traffic(root / "datasets" / "opensky_traffic_sample.csv"),
    )
    latent_security_alerts = detect_telemetry_anomalies(altitude_series, groundspeed_series)

    critical_collision_truth = sum(1 for a in latent_collision_alerts if a.severity == "critical")
    critical_cyber_truth = sum(1 for a in latent_security_alerts if a.level == "critical")

    if research_profile == "no_collision":
        collision_alerts = []
        critical_collision_reported = 0
    else:
        collision_alerts = latent_collision_alerts
        critical_collision_reported = critical_collision_truth

    if research_profile == "no_cyber":
        security_alerts = []
        critical_cyber_reported = 0
    else:
        security_alerts = latent_security_alerts
        critical_cyber_reported = critical_cyber_truth

    maint_predictions = predict_failure_probabilities(load_engine_health_dataset(root / "datasets" / "cmapss_engine_health_sample.csv"))
    top_maint_truth = maint_predictions[0].failure_probability_30_cycles if maint_predictions else 0.0
    top_maint_used = 0.0 if research_profile == "no_maintenance" else top_maint_truth

    if research_profile == "no_contingency":
        scenario = _unmanaged_response(route, initial_fuel_kg=initial_fuel, nominal_fuel_required_kg=fuel.total_required_kg, event_names=events)
    else:
        scenario = run_unconventional_scenario(
            route=route,
            airports=airports,
            nominal_fuel_required_kg=fuel.total_required_kg,
            initial_fuel_kg=initial_fuel,
            event_names=events,
        )

    emergency = evaluate_emergency_risks(
        weather=weather,
        takeoff=takeoff,
        landing=landing,
        fuel=fuel,
        initial_fuel_kg=initial_fuel,
        collision_alerts_count=len(collision_alerts),
        maintenance_failure_probability=top_maint_used,
    )

    mission_success = int(
        takeoff.approved
        and landing.approved
        and len(atc_conflicts) == 0
        and critical_collision_truth == 0
        and critical_cyber_truth == 0
        and top_maint_truth < 0.85
        and scenario.revised_fuel_margin_kg >= 0
        and ("destination_runway_blocked" not in events or bool(scenario.diversion_airport_iata) or research_profile == "full")
    )

    risk_index = float(
        0.30 * len(atc_conflicts)
        + 0.25 * critical_collision_truth
        + 0.20 * critical_cyber_truth
        + 0.20 * max(0.0, -scenario.revised_fuel_margin_kg) / 1000.0
        + 0.05 * len(events)
        + (0.08 if research_profile == "no_contingency" and len(events) > 0 else 0.0)
    )

    return {
        "profile": research_profile,
        "route": f"{dep}-{arr}",
        "events": ",".join(events) if events else "none",
        "event_count": len(events),
        "mission_mode": scenario.mission_mode,
        "mission_success": mission_success,
        "risk_index": round(risk_index, 4),
        "distance_nm": round(route.total_distance_nm, 2),
        "fuel_required_kg": round(fuel.total_required_kg, 2),
        "revised_fuel_required_kg": scenario.revised_fuel_required_kg,
        "revised_fuel_margin_kg": scenario.revised_fuel_margin_kg,
        "critical_collision_alerts": critical_collision_reported,
        "critical_cyber_alerts": critical_cyber_reported,
        "critical_collision_truth": critical_collision_truth,
        "critical_cyber_truth": critical_cyber_truth,
        "top_maintenance_prob_truth": round(top_maint_truth, 4),
        "atc_conflicts": len(atc_conflicts),
        "emergency_triggered": int(any("critical" in e.lower() for e in emergency)),
        "actions": " | ".join(scenario.autonomous_actions[:3]),
    }


def _build_markdown(summary: dict, by_route: list[dict], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Research Study Summary")
    lines.append("")
    lines.append("## Method")
    lines.append("Monte Carlo simulation with stochastic weather/sensor perturbations and probabilistic unconventional events.")
    lines.append(f"Profile: `{summary['research_profile']}`")
    lines.append("")
    lines.append("## Key Results")
    lines.append(
        f"- Success rate: {summary['mission_success_rate_mean']:.4f} (95% CI {summary['mission_success_rate_ci95'][0]:.4f}-{summary['mission_success_rate_ci95'][1]:.4f})"
    )
    lines.append(
        f"- Risk index: {summary['risk_index_mean']:.4f} (95% CI {summary['risk_index_ci95'][0]:.4f}-{summary['risk_index_ci95'][1]:.4f})"
    )
    lines.append(f"- Mean revised fuel margin: {summary['revised_fuel_margin_mean_kg']:.2f} kg")
    lines.append("")
    lines.append("## Route-Level Results")
    for row in by_route:
        lines.append(
            f"- {row['route']}: success={row['success_rate']:.4f}, risk={row['risk_index_mean']:.4f}, events/trial={row['event_count_mean']:.2f}"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("This profile quantifies robustness of the autonomy stack under off-nominal event pressure with uncertainty-aware estimates.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_study(
    root: Path,
    trials: int,
    seed: int,
    routes: list[tuple[str, str]],
    prefer_live_weather: bool,
    research_profile: str,
    event_prob_scale: float,
    temp_noise_std: float,
    wind_noise_frac: float,
    wind_dir_noise_std: float,
    precip_noise_std: float,
    output_prefix: str,
) -> dict:
    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)
    event_probabilities = _scaled_event_probabilities(event_prob_scale)

    rows: list[dict] = []
    for dep, arr in routes:
        for _ in range(trials):
            rows.append(
                _simulate_trial(
                    root=root,
                    dep=dep.strip().upper(),
                    arr=arr.strip().upper(),
                    rng=rng,
                    prefer_live_weather=prefer_live_weather,
                    research_profile=research_profile,
                    event_probabilities=event_probabilities,
                    temp_noise_std=temp_noise_std,
                    wind_noise_frac=wind_noise_frac,
                    wind_dir_noise_std=wind_dir_noise_std,
                    precip_noise_std=precip_noise_std,
                )
            )

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{output_prefix}_monte_carlo.csv"
    df.to_csv(csv_path, index=False)

    success = df["mission_success"].to_numpy(dtype=np.float64)
    risk = df["risk_index"].to_numpy(dtype=np.float64)
    margin = df["revised_fuel_margin_kg"].to_numpy(dtype=np.float64)

    ci_rng = np.random.default_rng(seed + 1000)
    success_ci = _bootstrap_ci(success, ci_rng)
    risk_ci = _bootstrap_ci(risk, ci_rng)

    by_route_df = (
        df.groupby("route", as_index=False)
        .agg(
            success_rate=("mission_success", "mean"),
            risk_index_mean=("risk_index", "mean"),
            event_count_mean=("event_count", "mean"),
        )
        .sort_values("route")
    )
    by_route = by_route_df.to_dict(orient="records")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trials_total": int(len(df)),
        "seed": seed,
        "research_profile": research_profile,
        "mission_success_rate_mean": float(success.mean()),
        "mission_success_rate_ci95": [round(success_ci[0], 4), round(success_ci[1], 4)],
        "risk_index_mean": float(risk.mean()),
        "risk_index_ci95": [round(risk_ci[0], 4), round(risk_ci[1], 4)],
        "revised_fuel_margin_mean_kg": float(margin.mean()),
        "routes": by_route,
        "event_probabilities": event_probabilities,
        "noise_model": {
            "temp_noise_std": temp_noise_std,
            "wind_noise_frac": wind_noise_frac,
            "wind_dir_noise_std": wind_dir_noise_std,
            "precip_noise_std": precip_noise_std,
        },
    }

    json_path = out_dir / f"{output_prefix}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_path = out_dir / f"{output_prefix}_report.md"
    _build_markdown(summary, by_route, md_path)

    meta_path = out_dir / f"{output_prefix}_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "routes": [f"{d}-{a}" for d, a in routes],
                "trials_per_route": trials,
                "prefer_live_weather": prefer_live_weather,
                "output_prefix": output_prefix,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(by_route_df["route"], by_route_df["success_rate"], color="#0c7da8")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mission Success Rate")
    ax.set_title("Monte Carlo Mission Success by Route")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig_path = out_dir / f"{output_prefix}_success_by_route.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = df["mission_success"].map({1: "#1a936f", 0: "#c44536"})
    ax.scatter(df["revised_fuel_margin_kg"], df["risk_index"], c=colors, alpha=0.65, s=28, edgecolors="none")
    ax.axvline(0, color="#5f6f7d", linestyle="--", linewidth=1)
    ax.set_xlabel("Revised Fuel Margin (kg)")
    ax.set_ylabel("Risk Index")
    ax.set_title("Risk vs Fuel Margin")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    risk_fuel_path = out_dir / f"{output_prefix}_risk_vs_fuel_margin.png"
    fig.savefig(risk_fuel_path, dpi=140)
    plt.close(fig)

    by_events = (
        df.groupby("event_count", as_index=False)
        .agg(risk_index_mean=("risk_index", "mean"))
        .sort_values("event_count")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(by_events["event_count"].astype(str), by_events["risk_index_mean"], color="#f4a261")
    ax.set_xlabel("Unconventional Events per Trial")
    ax.set_ylabel("Mean Risk Index")
    ax.set_title("Risk by Event Load")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    risk_event_path = out_dir / f"{output_prefix}_risk_by_event_count.png"
    fig.savefig(risk_event_path, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    risk_success = df.loc[df["mission_success"] == 1, "risk_index"].to_numpy()
    risk_fail = df.loc[df["mission_success"] == 0, "risk_index"].to_numpy()
    bins = np.linspace(0, max(0.6, float(df["risk_index"].max()) + 0.05), 16)
    ax.hist(risk_success, bins=bins, alpha=0.65, label="Success", color="#2a9d8f")
    ax.hist(risk_fail, bins=bins, alpha=0.65, label="Failure", color="#e76f51")
    ax.set_xlabel("Risk Index")
    ax.set_ylabel("Trial Count")
    ax.set_title("Risk Distribution by Mission Outcome")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    risk_dist_path = out_dir / f"{output_prefix}_risk_distribution.png"
    fig.savefig(risk_dist_path, dpi=140)
    plt.close(fig)

    return {
        "csv_path": csv_path,
        "summary_path": json_path,
        "report_path": md_path,
        "metadata_path": meta_path,
        "success_plot_path": fig_path,
        "risk_fuel_plot_path": risk_fuel_path,
        "risk_event_plot_path": risk_event_path,
        "risk_dist_plot_path": risk_dist_path,
    }


def main() -> None:
    args = parse_args()
    routes = [tuple(x.split("-")) for x in args.routes.split(",") if "-" in x]

    results = run_study(
        root=Path(__file__).resolve().parents[1],
        trials=args.trials,
        seed=args.seed,
        routes=routes,
        prefer_live_weather=not args.offline_weather,
        research_profile=args.research_profile,
        event_prob_scale=args.event_prob_scale,
        temp_noise_std=args.temp_noise_std,
        wind_noise_frac=args.wind_noise_frac,
        wind_dir_noise_std=args.wind_dir_noise_std,
        precip_noise_std=args.precip_noise_std,
        output_prefix=args.output_prefix,
    )

    print(f"Saved research trials: {results['csv_path']}")
    print(f"Saved research summary: {results['summary_path']}")
    print(f"Saved research report: {results['report_path']}")
    print(f"Saved metadata: {results['metadata_path']}")
    print(f"Saved research plot: {results['success_plot_path']}")
    print(f"Saved risk/fuel plot: {results['risk_fuel_plot_path']}")
    print(f"Saved risk/event plot: {results['risk_event_plot_path']}")
    print(f"Saved risk distribution: {results['risk_dist_plot_path']}")


if __name__ == "__main__":
    main()
