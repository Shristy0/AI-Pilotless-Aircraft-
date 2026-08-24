from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_research_study import run_study


PROFILES = ["full", "no_contingency", "no_collision", "no_cyber", "no_maintenance"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation study with significance tests")
    parser.add_argument("--trials", type=int, default=120, help="Trials per route")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--routes",
        default="SFO-LAX,SEA-LAS,JFK-LAX,LHR-DXB,SIN-DXB",
        help="Comma-separated departure-arrival pairs",
    )
    parser.add_argument("--offline-weather", action="store_true", help="Use fallback weather only")
    parser.add_argument("--output-prefix", default="ablation", help="Prefix for ablation output artifacts")
    return parser.parse_args()


def _permutation_pvalue(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_perm: int = 2500) -> float:
    observed = abs(float(a.mean() - b.mean()))
    combined = np.concatenate([a, b])
    count = 0

    for _ in range(n_perm):
        rng.shuffle(combined)
        a_p = combined[: len(a)]
        b_p = combined[len(a) :]
        stat = abs(float(a_p.mean() - b_p.mean()))
        if stat >= observed:
            count += 1

    return (count + 1) / (n_perm + 1)


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    gt = 0
    lt = 0
    for av in a:
        gt += int(np.sum(av > b))
        lt += int(np.sum(av < b))
    total = len(a) * len(b)
    if total == 0:
        return 0.0
    return (gt - lt) / total


def _effect_label(delta: float) -> str:
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)

    routes = [tuple(x.split("-")) for x in args.routes.split(",") if "-" in x]
    trial_frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict] = {}

    for idx, profile in enumerate(PROFILES):
        prefix = f"{args.output_prefix}_{profile}"
        run_results = run_study(
            root=root,
            trials=args.trials,
            seed=args.seed + idx * 137,
            routes=routes,
            prefer_live_weather=not args.offline_weather,
            research_profile=profile,
            event_prob_scale=1.0,
            temp_noise_std=1.8,
            wind_noise_frac=0.12,
            wind_dir_noise_std=8.0,
            precip_noise_std=7.5,
            output_prefix=prefix,
        )

        trial_frames[profile] = pd.read_csv(run_results["csv_path"])
        summaries[profile] = json.loads(Path(run_results["summary_path"]).read_text(encoding="utf-8"))

    baseline = trial_frames["full"]
    baseline_success = baseline["mission_success"].to_numpy(dtype=np.float64)
    baseline_risk = baseline["risk_index"].to_numpy(dtype=np.float64)

    rows = []
    rng = np.random.default_rng(args.seed + 9000)
    for profile in PROFILES:
        df = trial_frames[profile]
        success = df["mission_success"].to_numpy(dtype=np.float64)
        risk = df["risk_index"].to_numpy(dtype=np.float64)

        if profile == "full":
            p_success = 1.0
            p_risk = 1.0
            delta_success = 0.0
            delta_risk = 0.0
        else:
            p_success = _permutation_pvalue(baseline_success, success, rng)
            p_risk = _permutation_pvalue(baseline_risk, risk, rng)
            delta_success = _cliffs_delta(success, baseline_success)
            delta_risk = _cliffs_delta(risk, baseline_risk)

        rows.append(
            {
                "profile": profile,
                "success_rate": float(success.mean()),
                "risk_index_mean": float(risk.mean()),
                "delta_success_vs_full": float(success.mean() - baseline_success.mean()),
                "delta_risk_vs_full": float(risk.mean() - baseline_risk.mean()),
                "pvalue_success_vs_full": p_success,
                "pvalue_risk_vs_full": p_risk,
                "cliffs_delta_success": delta_success,
                "cliffs_delta_risk": delta_risk,
                "effect_success": _effect_label(delta_success),
                "effect_risk": _effect_label(delta_risk),
            }
        )

    cmp_df = pd.DataFrame(rows).sort_values("profile")

    cmp_csv = out_dir / f"{args.output_prefix}_comparison.csv"
    cmp_df.to_csv(cmp_csv, index=False)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profiles": PROFILES,
        "baseline_profile": "full",
        "trials_per_route": args.trials,
        "routes": [f"{d}-{a}" for d, a in routes],
        "results": rows,
    }

    summary_json = out_dir / f"{args.output_prefix}_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_md = out_dir / f"{args.output_prefix}_report.md"
    lines = [
        "# Ablation Study Report",
        "",
        f"Trials per route: {args.trials}",
        f"Baseline profile: full",
        "",
        "## Findings",
    ]
    for r in rows:
        lines.append(
            f"- {r['profile']}: success={r['success_rate']:.4f}, risk={r['risk_index_mean']:.4f}, "
            f"p_success={r['pvalue_success_vs_full']:.4f}, p_risk={r['pvalue_risk_vs_full']:.4f}, "
            f"effect(success)={r['effect_success']}, effect(risk)={r['effect_risk']}"
        )

    report_md.write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order_df = cmp_df.copy()

    axes[0].bar(order_df["profile"], order_df["success_rate"], color="#2a9d8f")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Ablation: Success Rate")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(order_df["profile"], order_df["risk_index_mean"], color="#e76f51")
    axes[1].set_title("Ablation: Mean Risk Index")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig_path = out_dir / f"{args.output_prefix}_comparison.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    print(f"Saved ablation comparison CSV: {cmp_csv}")
    print(f"Saved ablation summary JSON: {summary_json}")
    print(f"Saved ablation report: {report_md}")
    print(f"Saved ablation plot: {fig_path}")


if __name__ == "__main__":
    main()
