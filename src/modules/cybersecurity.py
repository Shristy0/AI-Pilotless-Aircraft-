from __future__ import annotations

import statistics
from typing import Iterable, List

from modules.models import SecurityAlert


def detect_telemetry_anomalies(
    altitude_series_ft: Iterable[float],
    groundspeed_series_kts: Iterable[float],
    z_threshold: float = 2.8,
) -> List[SecurityAlert]:
    alt = list(altitude_series_ft)
    gs = list(groundspeed_series_kts)
    alerts: List[SecurityAlert] = []

    if len(alt) < 3 or len(gs) < 3:
        return alerts

    alt_mean = statistics.fmean(alt)
    alt_std = statistics.pstdev(alt) or 1.0
    gs_mean = statistics.fmean(gs)
    gs_std = statistics.pstdev(gs) or 1.0

    for i, (a, g) in enumerate(zip(alt, gs)):
        alt_z = abs((a - alt_mean) / alt_std)
        gs_z = abs((g - gs_mean) / gs_std)

        if alt_z > z_threshold:
            alerts.append(SecurityAlert(level="warning", message="Altitude anomaly", at_index=i))
        if gs_z > z_threshold:
            alerts.append(SecurityAlert(level="warning", message="Groundspeed anomaly", at_index=i))

        if i > 0:
            if abs(alt[i] - alt[i - 1]) > 9000:
                alerts.append(SecurityAlert(level="critical", message="Abrupt altitude jump", at_index=i))
            if abs(gs[i] - gs[i - 1]) > 120:
                alerts.append(SecurityAlert(level="critical", message="Abrupt groundspeed jump", at_index=i))

    return alerts
