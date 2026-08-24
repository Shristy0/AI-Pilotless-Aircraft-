from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


def configure_accelerator() -> dict[str, float | str]:
    gpus = tf.config.list_physical_devices("GPU")
    accelerator = "CPU"

    if gpus:
        accelerator = "GPU"
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                # Safe fallback if memory growth is unsupported on this backend.
                pass

    return {
        "accelerator": accelerator,
        "gpu_count": float(len(gpus)),
    }


def _prepare_collision_cnn_data(traffic_df: pd.DataFrame, n_aug: int = 22) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    base = traffic_df.copy()

    records = []
    labels = []

    for _, row in base.iterrows():
        for _ in range(n_aug):
            lat = float(row["latitude"]) + rng.normal(0, 0.08)
            lon = float(row["longitude"]) + rng.normal(0, 0.08)
            alt = float(row["altitude_m"]) + rng.normal(0, 120)
            vel = float(row["velocity_mps"]) + rng.normal(0, 8)
            hdg = float(row["heading_deg"]) + rng.normal(0, 5)

            feature_grid = np.array(
                [
                    [lat / 90.0, lon / 180.0, alt / 12000.0],
                    [vel / 300.0, np.sin(np.radians(hdg)), np.cos(np.radians(hdg))],
                    [lat * lon / 10000.0, vel * alt / 1_000_000.0, 1.0],
                ],
                dtype=np.float32,
            )

            # Label: proximity to dense terminal region and low separation altitude.
            risk = int((34.0 <= lat <= 36.2) and (-120.0 <= lon <= -118.1) and (alt < 8500))
            records.append(feature_grid)
            labels.append(risk)

    X = np.stack(records)
    y = np.array(labels, dtype=np.float32)
    return X[..., np.newaxis], y


def train_cnn_collision_model(traffic_csv: Path, out_dir: Path) -> dict[str, float]:
    df = pd.read_csv(traffic_csv)
    X, y = _prepare_collision_cnn_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(3, 3, 1)),
            tf.keras.layers.Conv2D(12, (2, 2), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    tb_dir = out_dir / "tensorboard" / "cnn_collision"
    tb_dir.mkdir(parents=True, exist_ok=True)
    callback = tf.keras.callbacks.TensorBoard(log_dir=str(tb_dir), histogram_freq=0)

    start = time.perf_counter()
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, callbacks=[callback])
    train_seconds = time.perf_counter() - start

    probs = model.predict(X_test, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)

    model.save(out_dir / "cnn_collision_model.keras", overwrite=True)

    return {
        "cnn_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "cnn_f1": round(float(f1_score(y_test, preds)), 4),
        "cnn_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "cnn_train_seconds": round(train_seconds, 3),
    }


def _sequence_windows(cmapss_df: pd.DataFrame, seq_len: int = 3) -> tuple[np.ndarray, np.ndarray]:
    features = [
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

    windows = []
    labels = []

    for _, grp in cmapss_df.groupby("unit_nr"):
        grp = grp.sort_values("time_cycles")
        values = grp[features].to_numpy(dtype=np.float32)
        y = grp["failure_within_30"].to_numpy(dtype=np.float32)

        for i in range(len(grp) - seq_len + 1):
            windows.append(values[i : i + seq_len])
            labels.append(y[i + seq_len - 1])

    X = np.stack(windows)
    y = np.array(labels, dtype=np.float32)

    # Normalize features globally for stable training.
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True)
    X = (X - mean) / (std + 1e-6)
    return X, y


def train_rnn_maintenance_model(cmapss_csv: Path, out_dir: Path) -> dict[str, float]:
    df = pd.read_csv(cmapss_csv)
    X, y = _sequence_windows(df, seq_len=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11, stratify=y)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
            tf.keras.layers.LSTM(24, return_sequences=False),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    tb_dir = out_dir / "tensorboard" / "rnn_maintenance"
    tb_dir.mkdir(parents=True, exist_ok=True)
    callback = tf.keras.callbacks.TensorBoard(log_dir=str(tb_dir), histogram_freq=0)

    start = time.perf_counter()
    model.fit(X_train, y_train, epochs=16, batch_size=8, verbose=0, callbacks=[callback])
    train_seconds = time.perf_counter() - start

    probs = model.predict(X_test, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)

    model.save(out_dir / "rnn_maintenance_model.keras", overwrite=True)

    return {
        "rnn_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "rnn_f1": round(float(f1_score(y_test, preds)), 4),
        "rnn_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "rnn_train_seconds": round(train_seconds, 3),
    }


def train_rl_navigation_qlearning(episodes: int = 400) -> dict[str, float]:
    rng = np.random.default_rng(17)
    n_states = 5  # route deviation buckets
    n_actions = 3  # left, straight, right corrections
    q = np.zeros((n_states, n_actions), dtype=np.float32)

    alpha = 0.2
    gamma = 0.95
    epsilon = 0.20

    rewards = []
    for _ in range(episodes):
        state = int(rng.integers(0, n_states))
        episode_reward = 0.0

        for _step in range(20):
            if rng.random() < epsilon:
                action = int(rng.integers(0, n_actions))
            else:
                action = int(np.argmax(q[state]))

            desired = 1  # keep centered track (straight)
            weather_penalty = abs(state - 2) * 0.2
            reward = 1.0 - 0.6 * abs(action - desired) - weather_penalty
            next_state = int(np.clip(state + (action - 1), 0, n_states - 1))

            q[state, action] += alpha * (reward + gamma * np.max(q[next_state]) - q[state, action])
            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
        epsilon = max(0.03, epsilon * 0.995)

    return {
        "rl_episodes": episodes,
        "rl_avg_reward_last50": round(float(np.mean(rewards[-50:])), 4),
        "rl_policy_stability": round(float(np.std(np.argmax(q, axis=1))), 4),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)

    metrics: dict[str, float | str] = {}
    metrics.update(configure_accelerator())
    metrics.update(train_cnn_collision_model(root / "datasets" / "opensky_traffic_sample.csv", out_dir))
    metrics.update(train_rnn_maintenance_model(root / "datasets" / "cmapss_engine_health_sample.csv", out_dir))
    metrics.update(train_rl_navigation_qlearning(episodes=400))

    result_path = out_dir / "deep_learning_metrics.json"
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Accelerator: {metrics['accelerator']} (GPU count: {int(metrics['gpu_count'])})")
    print(f"Saved deep learning metrics to {result_path}")


if __name__ == "__main__":
    main()
