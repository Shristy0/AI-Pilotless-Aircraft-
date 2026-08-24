# AI Pilotless Aircraft — Autonomous Flight Research Platform

> A research-oriented Python framework for evaluating AI-assisted autonomous flight, navigation, aircraft health, collision avoidance, contingency handling, and unconventional-event response.

## Research scope

This repository packages a reproducible simulation and evaluation workflow for pilotless-aircraft research. The project combines modular flight systems with AI training components, scenario simulation, Monte Carlo studies, ablation experiments, and publication-style reporting.

### Core research areas

- Autonomous navigation and route planning
- Air-traffic awareness and collision avoidance
- Weather-aware decision making
- Engine health and predictive maintenance
- Emergency and unconventional-event response
- Cybersecurity-aware flight scenarios
- Deep-learning and reinforcement-learning experiments
- Monte Carlo evaluation and ablation studies

## Repository structure

```text
.
├── src/
│   ├── main.py                    # Main simulation entry point
│   ├── evaluate_project.py        # Project-level evaluation
│   ├── modules/                   # Autonomous flight subsystems
│   ├── ai_training/               # DL/RL training and baselines
│   ├── datasets/                  # Research/sample datasets
│   ├── simulators/                # Simulation backends
│   ├── research/                  # Research and ablation workflows
│   └── outputs/                   # Generated reports, metrics, and figures
├── scripts/                       # Dataset/build utilities
├── figures/                       # Repository figures
├── requirements.txt               # Core dependencies
├── requirements-optional.txt      # Optional ML/vision/simulator dependencies
├── requirements-mac-gpu.txt       # Optional Apple GPU dependencies
└── README.md
```

## Reproducible setup

### 1. Clone the repository

```bash
git clone https://github.com/Shristy0/AI-Pilotless-Aircraft-.git
cd AI-Pilotless-Aircraft-
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Optional dependencies:

```bash
python3 -m pip install -r requirements-optional.txt
```

For supported Mac GPU experiments:

```bash
python3 -m pip install -r requirements-mac-gpu.txt
```

## Quick start

Run the offline-weather baseline:

```bash
python3 src/main.py --offline-weather
```

Run project evaluation:

```bash
python3 src/evaluate_project.py
```

Run the deep-learning suite:

```bash
python3 src/ai_training/deep_learning_suite.py
```

## Research experiments

### Unconventional-event scenarios

```bash
python3 src/main.py --offline-weather --scenario unconventional
```

Example multi-event scenario:

```bash
python3 src/main.py --offline-weather \
  --scenario unconventional \
  --events gnss_spoofing,engine_thrust_loss,destination_runway_blocked
```

### Monte Carlo research study

```bash
python3 src/research/run_research_study.py \
  --offline-weather \
  --trials 120 \
  --seed 42
```

The research workflow produces summary statistics, confidence intervals, Monte Carlo data, and figures under `src/outputs/`.

### Ablation study

```bash
python3 src/research/run_ablation_study.py \
  --offline-weather \
  --trials 120 \
  --seed 42
```

### Publication-style report

```bash
python3 src/research/generate_paper_report.py \
  --research-prefix research \
  --ablation-prefix ablation
```

## Datasets

The project includes or expects research/sample data for:

- Airport locations
- Weather fallback data
- Aircraft/engine performance
- Restricted/no-fly zones
- Air-traffic samples
- Turbofan engine health/degradation

Dataset-generation utilities are provided in `scripts/`. When redistributing or publishing results, verify the license and attribution requirements of each upstream dataset.

## Reproducibility notes

For comparable experiments, record:

- Python version and dependency versions
- Experiment profile/configuration
- Random seed
- Number of Monte Carlo trials
- Scenario/event probabilities
- Sensor/weather uncertainty parameters
- Output/report prefix

Example:

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

## Research outputs

Key generated artifacts include:

- `research_summary.json`
- `research_metadata.json`
- `research_monte_carlo.csv`
- `research_report.md`
- risk-distribution and route-success figures
- ablation comparison and summary files
- evaluation metrics and scenario tables

Generated TensorBoard logs and other machine-specific artifacts should remain untracked.

## Limitations and responsible use

This repository is a research and simulation platform. Its outputs should **not** be interpreted as evidence that an autonomous aircraft is safe for real-world operation or as a substitute for certified avionics, flight testing, airworthiness analysis, operational approval, or regulatory compliance.

Simulation assumptions, datasets, model limitations, uncertainty, and failure cases should be documented before using results in academic or engineering claims.

## Suggested research workflow

1. Define the research question and evaluation metric.
2. Establish a reproducible baseline.
3. Run controlled scenarios with fixed seeds.
4. Compare model/system variants through ablation.
5. Quantify uncertainty and report confidence intervals where appropriate.
6. Inspect failure cases, not only aggregate success rates.
7. Preserve configurations, metadata, and generated reports.
8. Clearly distinguish simulated evidence from real-world validation.

## Author

**Srishti Neupane**  
AI Pilotless Aircraft Research Project

## License

No open-source license has been declared yet. Until a license is added, normal copyright restrictions apply to reuse of the repository contents.
