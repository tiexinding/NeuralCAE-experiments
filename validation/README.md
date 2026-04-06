# NPM Validation Experiments

Empirical validation of the Neural Percolation Model (NPM) framework predictions on MMLU benchmarks.

[中文版](README_CN.md)

---

## Published Results

**NPM Framework Study #1: Predicting Emergence Order**
- Zenodo: [DOI: 10.5281/zenodo.19440042](https://doi.org/10.5281/zenodo.19440042)
- arXiv: submit/7447724 (cs.LG)

### Key Findings

1. **Task emergence order is metric-independent.** Spearman = 0.984 (p ~ 10⁻³⁹) between accuracy and Brier Score ranking across 53 MMLU sub-categories.
2. **Data quality dominates model scale.** Phi-2 (2.78B) scores 54.5% vs Pythia-2.8B (2.80B) at 25.2% — 29-point gap at identical parameter count.
3. **170x parameter scaling yields no emergence.** Pythia 70M–12B all remain near random baseline (22.9%–25.6%).

---

## Directory Structure

```
validation/
├── scripts/
│   ├── npm_validation_fit.py        # NPM parameter fitting + prediction
│   └── beta_dimensionless_test.py   # Dimensionless β verification
├── results/
│   ├── results_pythia_70m/          # Pythia-70M MMLU results (lm_eval JSON)
│   ├── results_pythia_160m/         # Pythia-160M
│   ├── results_pythia_410m/         # Pythia-410M
│   ├── results_pythia_1b/           # Pythia-1B
│   ├── results_pythia_1.4b/         # Pythia-1.4B
│   ├── results_pythia_2.8b/         # Pythia-2.8B
│   ├── results_pythia_6.9b/         # Pythia-6.9B
│   ├── results_pythia_12b/          # Pythia-12B
│   ├── results_phi2_fp16/           # Phi-2 FP16
│   ├── pilot/                       # Pilot experiments
│   └── pilot_phi2/                  # Phi-2 pilot
├── figures/
│   ├── fig1_acc_vs_brier_EN.png     # Main scatter plot (English)
│   ├── fig1_acc_vs_brier_CN.png     # Main scatter plot (Chinese)
│   └── acc_vs_brier_scatter.png     # Draft version
├── configs/                         # Experiment configurations
└── requirements.txt                 # Python dependencies
```

## Models Evaluated

| Model | Params (B) | Training Data | MMLU |
|-------|-----------|---------------|------|
| Pythia-70m | 0.07 | The Pile 300B | 22.9% |
| Pythia-160m | 0.16 | The Pile 300B | 23.0% |
| Pythia-410m | 0.41 | The Pile 300B | 23.1% |
| Pythia-1b | 1.00 | The Pile 300B | 23.1% |
| Pythia-1.4b | 1.40 | The Pile 300B | 24.3% |
| Pythia-2.8b | 2.80 | The Pile 300B | 25.2% |
| Pythia-6.9b | 6.90 | The Pile 300B | 25.6% |
| Pythia-12b | 12.00 | The Pile 300B | 24.2% |
| Phi-2 | 2.78 | Curated ~1.4T | 54.5% |

## Computing

- AutoDL RTX 4090 (24GB)
- lm-evaluation-harness v0.4.11
- 0-shot evaluation on all 57 MMLU sub-categories

---

**License:** CC BY-NC-ND 4.0 | **Author:** Tiexin Ding | **NeuralCAE**
