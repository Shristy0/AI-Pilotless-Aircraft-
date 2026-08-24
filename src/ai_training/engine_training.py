from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def train_predictive_maintenance_classifier(dataset_csv: Path) -> dict[str, float]:
    df = pd.read_csv(dataset_csv)
    feature_cols = [
        "time_cycles",
        "op_setting_1",
        "op_setting_2",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_11",
        "sensor_12",
        "sensor_15",
        "sensor_21",
    ]

    X = df[feature_cols]
    y = df["failure_within_30"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14, stratify=y)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "rows": int(len(df)),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "auc": round(float(roc_auc_score(y_test, probs)), 4),
        "brier": round(float(brier_score_loss(y_test, probs)), 4),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics = train_predictive_maintenance_classifier(root / "datasets" / "cmapss_engine_health_sample.csv")

    out_path = root / "outputs" / "engine_maintenance_metrics.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved maintenance metrics to {out_path}")


if __name__ == "__main__":
    main()
