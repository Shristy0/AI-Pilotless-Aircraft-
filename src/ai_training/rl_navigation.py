from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def run_rl_navigation_qlearning(episodes: int = 500) -> dict:
    rng = np.random.default_rng(123)
    n_states = 7
    n_actions = 3
    q_table = np.zeros((n_states, n_actions), dtype=np.float32)

    alpha = 0.18
    gamma = 0.93
    epsilon = 0.25

    rewards = []
    for _ in range(episodes):
        state = int(rng.integers(0, n_states))
        total_reward = 0.0

        for _ in range(25):
            action = int(rng.integers(0, n_actions)) if rng.random() < epsilon else int(np.argmax(q_table[state]))

            # Reward centered path-following with smooth control actions.
            center_penalty = abs(state - (n_states // 2)) * 0.15
            action_penalty = abs(action - 1) * 0.08
            reward = 1.0 - center_penalty - action_penalty

            next_state = int(np.clip(state + (action - 1), 0, n_states - 1))
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        epsilon = max(0.03, epsilon * 0.996)

    return {
        "episodes": episodes,
        "avg_reward": round(float(np.mean(rewards[-100:])), 4),
        "best_policy_actions": np.argmax(q_table, axis=1).tolist(),
        "q_table_mean": round(float(np.mean(q_table)), 4),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results = run_rl_navigation_qlearning()

    out_path = root / "outputs" / "rl_navigation_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved RL navigation summary to {out_path}")


if __name__ == "__main__":
    main()
