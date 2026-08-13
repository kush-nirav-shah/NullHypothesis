# Spatial Misalignment Framework: From Epidemiology to Precision Agriculture

**A Comparative Study of Classical Correction Methods and Modern Machine Learning Approaches**

**Authors:** Kush Shah (4006420), Akash Patil (4006939), Md Tusar Ahmed (4001610)
**Course:** Probabilistic Modelling — Prof. Burkhardt Funk
**Institution:** M.Sc. Data Science, Leuphana University of Lüneburg, Germany

---

## Overview

Spatial misalignment — the discrepancy between where explanatory variables are measured and where outcomes
are observed — is a pervasive problem in environmental epidemiology and precision agriculture. This project
independently replicates the measurement-error framework of Gryparis et al. (2009), extends it through a
monitor density sensitivity analysis, introduces three modern machine learning methods (Gaussian Process
Regression, Random Forest, Gradient Boosting), and evaluates whether findings transfer to a second domain
(precision agriculture) with genuinely different spatial parameters.

Across **145 simulation runs** spanning four studies, the Plug-in estimator achieves the lowest MSE in three
of four studies, Random Forest emerges as the strongest modern method, and Gaussian Process Regression shows
the largest cross-domain sensitivity (improving from rank 5 in epidemiology to rank 2 in agriculture).

---

## Folder Structure

```
final_research_project/
├── code/
│   ├── framework.py            # Core simulation: data generation, kriging, all 7 methods
│   └── generate_figure.py      # Generates all 10 publication figures from results
├── results/
│   ├── figures/                # 10 output figures (fig1.png ... fig10.png)
│   └── tables/
│       └── all_results.csv     # Raw results from all 4 studies (145 simulation runs)
├── validation/
│   ├── parameter_comparison.json   # Epidemiology vs. agriculture scenario parameters
│   ├── results_summary.json        # Cross-domain summary statistics
│   └── validation_report.txt       # Replication validation notes
└── README.md
```

---

## Requirements

- Python 3.10+
- Dependencies:
  ```bash
  pip install numpy pandas matplotlib scipy scikit-learn
  ```

---

## How to Reproduce

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/final_research_project.git
   cd final_research_project
   ```

2. **Run the simulation framework**
   ```bash
   python code/framework.py
   ```
   This runs all four studies (Replication, Monitor Density, Modern Methods, Agriculture Transfer) and
   writes:
   - `results/tables/all_results.csv`
   - `validation/parameter_comparison.json`

   > **Note:** `framework.py` currently points to a local Windows path
   > (`C:\Users\kushn\OneDrive\Desktop\final\...`) for output. Update the `ROOT` variable near the top of the
   > script to a relative path (e.g. `ROOT = "."`) before running on another machine.

3. **Generate figures**
   ```bash
   python code/generate_figure.py
   ```
   This reads `results/tables/all_results.csv` and produces all 10 figures in `results/figures/`.

Total runtime: approximately 8 minutes on a single CPU core.

---

## Studies

| Study | Description | Configuration |
|---|---|---|
| **1 — Replication** | Independent replication of Gryparis et al. (2009) | 7 methods × scenarios A–D, 82 monitors, 200 subjects |
| **2 — Monitor Density** | Sensitivity to monitor count | Densities 10–80, Scenario C, 10 reps each |
| **3 — Modern Methods** | Classical vs. ML method comparison | 40 monitors, Scenario C, 15 reps |
| **4 — Agriculture Transfer** | Cross-domain validation | Densities 10–80, agriculture Scenario C, independent random seed |

---

## Methods Evaluated

**Classical (Gryparis et al. 2009):** Plug-in, RC-OOS (Regression Calibration, Out-of-Sample), Exposure
Simulation, Two-Stage Bayesian.

**Modern (this study):** Gaussian Process Regression (GPR), Random Forest (RF), Gradient Boosting (GBR).

---

## Key Results

- **Replication confirmed:** Exposure Simulation catastrophically fails (0% coverage) due to Berkson-to-classical
  error reconversion — a structural limitation, not an implementation bug.
- **Practical monitor threshold:** ~40 monitors is where alternative methods become competitive with Plug-in.
- **Cross-domain transfer:** The framework generalises to precision agriculture; method rankings are broadly
  stable (Plug-in #1, Exp.Simulation/Bayes #6–7 in both domains) but absolute MSE values are not portable
  across domains.

Full methodology, results, and discussion are available in the accompanying paper (`NullHypothesis.tex`),
submitted separately via the course documentation portal.

---

## Data

All data used in this study are **simulated** (drawn from a Matérn Gaussian Process), not real-world
measurements — consistent with the original Gryparis et al. (2009) design. This allows the true effect and
exposure surface to be known exactly, which is required to compute bias, MSE, and coverage against ground
truth. See `validation/validation_report.txt` for replication validation notes and
`validation/parameter_comparison.json` for the full epidemiology vs. agriculture parameter comparison.

---

## Citation

If referencing this work, please cite the original framework:

> Gryparis, A., Paciorek, C. J., Zeka, A., Schwartz, J., & Coull, B. A. (2009). Measurement error caused by
> spatial misalignment in environmental epidemiology. *Biostatistics*, 10(2), 258–274.

---

## Contact

For questions about this implementation, contact any of the authors listed above via their university email
addresses.
