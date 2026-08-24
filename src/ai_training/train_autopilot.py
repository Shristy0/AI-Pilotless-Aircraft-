from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.route_planner import load_airports, plan_route


def build_autopilot_profile() -> dict:
    airports = load_airports(ROOT / "datasets" / "airports.csv")
    route = plan_route(airports, "SFO", "LAX", waypoint_count=12)

    # Baseline control gains for heading and altitude hold.
    gains = {
        "heading_kp": 0.85,
        "heading_ki": 0.08,
        "heading_kd": 0.02,
        "altitude_kp": 0.92,
        "altitude_ki": 0.11,
        "altitude_kd": 0.04,
    }

    return {
        "route_nm": round(route.total_distance_nm, 1),
        "training_scenario": "SFO-LAX",
        "controller_gains": gains,
    }


def main() -> None:
    profile = build_autopilot_profile()
    out_path = ROOT / "outputs" / "autopilot_controller_profile.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Saved autopilot profile to {out_path}")


if __name__ == "__main__":
    main()
