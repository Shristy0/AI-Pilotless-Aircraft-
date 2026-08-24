from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class EngineCycle:
    unit_nr: int
    time_cycles: int
    op_setting_1: float
    op_setting_2: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_11: float
    sensor_12: float
    sensor_15: float
    sensor_21: float
    failure_within_30: int


@dataclass(frozen=True)
class MaintenancePrediction:
    unit_nr: int
    failure_probability_30_cycles: float
    predicted_failure_within_30: bool
    recommended_action: str


def load_engine_health_dataset(csv_path: str | Path) -> List[EngineCycle]:
    rows: List[EngineCycle] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for r in reader:
            rows.append(
                EngineCycle(
                    unit_nr=int(r["unit_nr"]),
                    time_cycles=int(r["time_cycles"]),
                    op_setting_1=float(r["op_setting_1"]),
                    op_setting_2=float(r["op_setting_2"]),
                    sensor_2=float(r["sensor_2"]),
                    sensor_3=float(r["sensor_3"]),
                    sensor_4=float(r["sensor_4"]),
                    sensor_7=float(r["sensor_7"]),
                    sensor_11=float(r["sensor_11"]),
                    sensor_12=float(r["sensor_12"]),
                    sensor_15=float(r["sensor_15"]),
                    sensor_21=float(r["sensor_21"]),
                    failure_within_30=int(r["failure_within_30"]),
                )
            )
    return rows


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _group_by_engine(dataset: List[EngineCycle]) -> Dict[int, List[EngineCycle]]:
    grouped: Dict[int, List[EngineCycle]] = {}
    for row in dataset:
        grouped.setdefault(row.unit_nr, []).append(row)
    for unit in grouped:
        grouped[unit] = sorted(grouped[unit], key=lambda r: r.time_cycles)
    return grouped


def _trend(last: EngineCycle, prev: EngineCycle, attr: str) -> float:
    return getattr(last, attr) - getattr(prev, attr)


def predict_failure_probabilities(dataset: List[EngineCycle]) -> List[MaintenancePrediction]:
    grouped = _group_by_engine(dataset)
    preds: List[MaintenancePrediction] = []

    for unit_nr, series in grouped.items():
        if len(series) < 2:
            continue

        prev, last = series[-2], series[-1]
        sensor_rise = (
            _trend(last, prev, "sensor_2") * 0.8
            + _trend(last, prev, "sensor_3") * 0.1
            + _trend(last, prev, "sensor_4") * 0.12
            + _trend(last, prev, "sensor_11") * 8.0
            + _trend(last, prev, "sensor_21") * 10.0
        )

        score = (
            -19.0
            + 0.17 * last.time_cycles
            + 0.8 * (last.op_setting_1 - 0.58) * 10
            + 0.8 * (last.op_setting_2 - 0.04) * 10
            + 0.06 * (last.sensor_2 - 640.0)
            + 0.015 * (last.sensor_3 - 1585.0)
            + 0.005 * (last.sensor_4 - 1390.0)
            + 0.3 * (last.sensor_11 - 46.5)
            + 0.06 * (last.sensor_21 - 38.0)
            + 0.3 * sensor_rise
        )

        prob = _sigmoid(score)
        high_risk = prob >= 0.5
        if prob >= 0.8:
            action = "Immediate maintenance required before next sortie"
        elif high_risk:
            action = "Schedule maintenance within 10 flight cycles"
        else:
            action = "Continue operation with standard monitoring"

        preds.append(
            MaintenancePrediction(
                unit_nr=unit_nr,
                failure_probability_30_cycles=round(prob, 4),
                predicted_failure_within_30=high_risk,
                recommended_action=action,
            )
        )

    return sorted(preds, key=lambda p: p.failure_probability_30_cycles, reverse=True)
