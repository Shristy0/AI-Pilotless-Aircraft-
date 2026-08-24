from __future__ import annotations

from typing import Iterable, List


def build_hmi_summary(
    mission_label: str,
    takeoff_ok: bool,
    landing_ok: bool,
    atc_conflicts: int,
    collision_alerts: int,
    cyber_alerts: int,
    emergency_alerts: Iterable[str],
) -> List[str]:
    emergency_list = list(emergency_alerts)
    has_critical_text = any("critical" in e.lower() for e in emergency_list)

    status = (
        "GREEN"
        if all([takeoff_ok, landing_ok, atc_conflicts == 0, collision_alerts == 0, cyber_alerts == 0]) and not has_critical_text
        else "AMBER"
    )
    if (not takeoff_ok) or (not landing_ok) or has_critical_text:
        status = "RED"

    lines = [
        f"HMI MISSION STATUS [{mission_label}] => {status}",
        f"Takeoff: {'OK' if takeoff_ok else 'NOT OK'} | Landing: {'OK' if landing_ok else 'NOT OK'}",
        f"ATC conflicts: {atc_conflicts} | Collision risks: {collision_alerts} | Cyber alerts: {cyber_alerts}",
    ]

    lines.append("Emergency monitor: " + ("; ".join(emergency_list[:2]) if emergency_list else "No alerts"))
    lines.append("Operator action: review alerts and approve autonomous continuation if all critical checks are clear.")
    return lines
