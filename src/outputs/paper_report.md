# Pilotless Aircraft Research Report

Generated: 2026-03-09T21:17:09.501970+00:00

## 1. Research Question
Can an integrated AI autonomy stack maintain safe mission performance under off-nominal aviation events?

## 2. Experimental Method
- Monte Carlo simulation with stochastic weather and sensor perturbations
- Probabilistic unconventional event injection
- Metrics: mission success rate, risk index, fuel margin, event burden

## 3. Main Study Results
- Trials: 75
- Success rate: 0.8267 (95% CI 0.7467-0.9067)
- Risk index mean: 0.0940 (95% CI 0.0491-0.1492)
- Mean revised fuel margin: 665.14 kg

### Route-Level Summary
- JFK-LAX: success=0.8000, risk=0.1054, events/trial=0.40
- LHR-DXB: success=0.8667, risk=0.1030, events/trial=0.40
- SEA-LAS: success=0.6667, risk=0.0841, events/trial=0.47
- SFO-LAX: success=1.0000, risk=0.0167, events/trial=0.33
- SIN-DXB: success=0.8000, risk=0.1607, events/trial=0.27

## 4. Ablation and Significance

- full: success=0.8000, risk=0.0917, p_success=1.0000, p_risk=1.0000, effect(success)=negligible, effect(risk)=negligible
- no_contingency: success=0.8200, risk=0.1451, p_success=1.0000, p_risk=0.3175, effect(success)=negligible, effect(risk)=negligible
- no_collision: success=0.7200, risk=0.1713, p_success=0.4806, p_risk=0.1251, effect(success)=negligible, effect(risk)=negligible
- no_cyber: success=0.8000, risk=0.1311, p_success=1.0000, p_risk=0.4330, effect(success)=negligible, effect(risk)=negligible
- no_maintenance: success=0.7600, risk=0.1454, p_success=0.8141, p_risk=0.2687, effect(success)=negligible, effect(risk)=negligible

## 5. Discussion
The full autonomy profile should be interpreted against ablations to identify which modules materially improve robustness.

## 6. Threats to Validity
- Open-source/sample datasets may not capture all real-world edge cases
- Simulation simplifications may under-represent aerodynamics and certification constraints

## 7. Reproducibility
Use fixed seeds, archived outputs, and profile-specific metadata JSON files under `src/outputs`.