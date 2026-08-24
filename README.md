# Pilotless Aircraft AI Project - Practical Package

## Contents
- `src/`: runnable code (`main.py`, `evaluate_project.py`, `modules/`, `ai_training/`)
- `src/datasets/`: datasets used by simulation and training
- `src/outputs/`: generated mission/evaluation results
- `requirements.txt`: Python dependencies
- `requirements-mac-gpu.txt`: optional Apple GPU acceleration dependency
 - `requirements-optional.txt`: optional PyTorch/OpenCV/OpenAP dependencies
- `docs/`: literature review and ethics/regulatory drafts

## Datasets (download/build)
This project expects the following files under `src/datasets/`:
- `airports.csv` (OurAirports)
- `weather_fallback.csv` (Open-Meteo snapshot)
- `engine_performance.csv` (OpenAP performance grid)
- `no_fly_zones.geojson` (US Special Use Airspace, ArcGIS open data)
- `opensky_traffic_sample.csv` (OpenSky Network snapshot)
- `cmapss_engine_health_sample.csv` (NASA turbofan engine degradation dataset)

To build them automatically:
```bash
cd /Users/shristy/Desktop/AI_Pilotless_Aircraft
python3 scripts/build_datasets.py
```

Optional dependencies for dataset generation and extra demos:
```bash
python3 -m pip install -r requirements-optional.txt
```

## Optional GPU setup (Mac)
```bash
cd /Users/shristy/Desktop/AI_Pilotless_Aircraft
python3 -m pip install -r requirements-mac-gpu.txt
python3 src/ai_training/deep_learning_suite.py
```

`deep_learning_suite.py` now prints detected accelerator (CPU/GPU) and GPU count.

## Run (VS Code Terminal)
```bash
cd /Users/shristy/Desktop/AI_Pilotless_Aircraft
python3 src/main.py --offline-weather
python3 src/ai_training/deep_learning_suite.py
python3 src/evaluate_project.py
```

Optional PyTorch/OpenCV vision demo:
```bash
python3 src/ai_training/vision_baseline.py
```

Optional external simulators (JSBSim/FlightGear):
```bash
python3 src/main.py --sim-backend jsbsim
python3 src/main.py --sim-backend flightgear
```

Example simulator config (for external launch):
```bash
python3 src/main.py --sim-backend flightgear --sim-config docs/sim_config_example.json --launch-sim
```

## Unconventional scenario demo
```bash
python3 src/main.py --offline-weather --scenario unconventional
python3 src/main.py --offline-weather --scenario unconventional --events gnss_spoofing,engine_thrust_loss,destination_runway_blocked
```

## Research mode (more realistic)
```bash
python3 src/research/run_research_study.py --offline-weather --trials 120 --seed 42
```

This runs Monte Carlo trials with stochastic weather/sensor uncertainty and probabilistic unconventional events, then writes:
- `src/outputs/research_monte_carlo.csv`
- `src/outputs/research_summary.json` (includes 95% confidence intervals)
- `src/outputs/research_metadata.json`
- `src/outputs/research_report.md`
- `src/outputs/research_success_by_route.png`
- `src/outputs/research_risk_vs_fuel_margin.png`
- `src/outputs/research_risk_by_event_count.png`
- `src/outputs/research_risk_distribution.png`

Advanced options:
```bash
python3 src/research/run_research_study.py \
  --offline-weather \
  --trials 200 \
  --seed 42 \
  --research-profile full \
  --event-prob-scale 1.2 \
  --temp-noise-std 2.2 \
  --output-prefix research
```

## Ablation + significance tests
```bash
python3 src/research/run_ablation_study.py --offline-weather --trials 120 --seed 42
```

Outputs:
- `src/outputs/ablation_comparison.csv`
- `src/outputs/ablation_summary.json`
- `src/outputs/ablation_report.md`
- `src/outputs/ablation_comparison.png`

## Publication-style report
```bash
python3 src/research/generate_paper_report.py --research-prefix research --ablation-prefix ablation
```

Output:
- `src/outputs/paper_report.md`

## Main practical outputs
- `src/outputs/latest_mission_report.json`
- `src/outputs/latest_fuel_profile.csv`
- `src/outputs/latest_mission_profile.png`
- `src/outputs/evaluation_summary.json`
- `src/outputs/evaluation_scenarios.csv`
- `src/outputs/deep_learning_metrics.json`
- `src/outputs/evaluation_plots.png`
