from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-style report from research outputs")
    parser.add_argument("--research-prefix", default="research", help="Prefix for main research outputs")
    parser.add_argument("--ablation-prefix", default="ablation", help="Prefix for ablation outputs")
    return parser.parse_args()


def _safe_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs"
    docs = root / "docs"

    research_summary = _safe_json(out / f"{args.research_prefix}_summary.json")
    ablation_summary = _safe_json(out / f"{args.ablation_prefix}_summary.json")
    literature_path = docs / "literature_review.md"
    ethics_path = docs / "ethics_regulatory.md"

    now = datetime.now(timezone.utc).isoformat()

    lines = [
        "# Pilotless Aircraft Research Report",
        "",
        f"Generated: {now}",
        "",
        "## 1. Research Question",
        "Can an integrated AI autonomy stack maintain safe mission performance under off-nominal aviation events?",
        "",
        "## 2. Experimental Method",
        "- Monte Carlo simulation with stochastic weather and sensor perturbations",
        "- Probabilistic unconventional event injection",
        "- Metrics: mission success rate, risk index, fuel margin, event burden",
        "",
    ]

    if research_summary:
        lines.extend(
            [
                "## 3. Main Study Results",
                f"- Trials: {research_summary.get('trials_total', 'n/a')}",
                (
                    f"- Success rate: {research_summary.get('mission_success_rate_mean', 0):.4f} "
                    f"(95% CI {research_summary.get('mission_success_rate_ci95', ['n/a','n/a'])[0]}-"
                    f"{research_summary.get('mission_success_rate_ci95', ['n/a','n/a'])[1]})"
                ),
                (
                    f"- Risk index mean: {research_summary.get('risk_index_mean', 0):.4f} "
                    f"(95% CI {research_summary.get('risk_index_ci95', ['n/a','n/a'])[0]}-"
                    f"{research_summary.get('risk_index_ci95', ['n/a','n/a'])[1]})"
                ),
                f"- Mean revised fuel margin: {research_summary.get('revised_fuel_margin_mean_kg', 0):.2f} kg",
                "",
                "### Route-Level Summary",
            ]
        )

        for row in research_summary.get("routes", []):
            lines.append(
                f"- {row['route']}: success={row['success_rate']:.4f}, risk={row['risk_index_mean']:.4f}, events/trial={row['event_count_mean']:.2f}"
            )
        lines.append("")

    if ablation_summary:
        lines.extend(["## 4. Ablation and Significance", ""])
        for row in ablation_summary.get("results", []):
            lines.append(
                f"- {row['profile']}: success={row['success_rate']:.4f}, risk={row['risk_index_mean']:.4f}, "
                f"p_success={row['pvalue_success_vs_full']:.4f}, p_risk={row['pvalue_risk_vs_full']:.4f}, "
                f"effect(success)={row['effect_success']}, effect(risk)={row['effect_risk']}"
            )
        lines.append("")

    lines.extend(
        [
            "## 5. Discussion",
            "The full autonomy profile should be interpreted against ablations to identify which modules materially improve robustness.",
            "",
            "## 6. Literature Review (Summary)",
        ]
    )

    if literature_path.exists():
        lines.extend(literature_path.read_text(encoding="utf-8").splitlines())
    else:
        lines.extend(
            [
                "Literature review not found. Add `docs/literature_review.md` to include cited sources.",
            ]
        )

    lines.extend(
        [
            "",
            "## 7. Ethics, Cybersecurity, and Regulatory Implications",
        ]
    )

    if ethics_path.exists():
        lines.extend(ethics_path.read_text(encoding="utf-8").splitlines())
    else:
        lines.extend(
            [
                "Ethics/regulatory summary not found. Add `docs/ethics_regulatory.md` to include cited sources.",
            ]
        )

    lines.extend(
        [
            "",
            "## 8. Threats to Validity",
            "- Open-source/sample datasets may not capture all real-world edge cases",
            "- Simulation simplifications may under-represent aerodynamics and certification constraints",
            "",
            "## 9. Reproducibility",
            "Use fixed seeds, archived outputs, and profile-specific metadata JSON files under `src/outputs`.",
        ]
    )

    out_path = out / "paper_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved paper-style report: {out_path}")


if __name__ == "__main__":
    main()
