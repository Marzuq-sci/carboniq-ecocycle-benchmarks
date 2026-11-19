# CarbonIQ + EcoCycle: Hybrid Combinatorial Optimization Benchmark Suite

Reproducible synthetic datasets, exact ground-state reference files (`GS_energies_*.txt`), an interactive Jupyter/Colab explorer notebook, and multi-solver benchmarks for:

> **CarbonIQ + EcoCycle: A Three-Tier Urban Carbon Economy and Hybrid Combinatorial Optimization Framework for Emerging Cities**

Repository organization and exact ground-state reference tracking are inspired by [VectorInstitute/VariationalNeuralAnnealing](https://github.com/VectorInstitute/VariationalNeuralAnnealing).

## Repository Layout

```text
carboniq-ecocycle-benchmarks/
├── notebooks/
│   └── CarbonIQ_Explorer.ipynb       # Interactive Jupyter / Google Colab explorer (P1-P5, MRV, sliders, plots)
├── tools/
│   ├── generate_synthetic_data.py    # Seeded Accra CSV generator (small / medium / large)
│   └── generate_gs_references.py     # Exact 25-seed GS_energies_*.txt generator
├── data/
│   ├── starter_small/                # 7 CSV schemas (exact-truth verifiable scale)
│   ├── starter_medium/               # 7 CSV schemas (50-site Q-Energic parity scale)
│   ├── starter_large/                # 7 CSV schemas (city-scale stress test)
│   ├── P1_routing/GS_energies_*.txt  # 25-seed exact ground states (1v N=8, 2v N=16)
│   ├── P2_vpp/GS_energies_p2_5x6.txt # 25-seed exact ground states (5x6 N=30, 6^5 manifold)
│   ├── P3_ev_corridor/GS_energies_*.txt
│   └── P4_hub_siting/GS_energies_p4_14sites.txt
├── src/
│   ├── qubo_builders.py              # Eq. (1) MRV ledger, Eqs. (2)-(3) W_ij, Eqs. (5)-(8) QUBOs
│   ├── solvers.py                    # Tier 3A (Dijkstra, Hungarian, Exact) & Tier 3B (SA, Subspace SA, SB+SRWS)
│   └── binary_vca.py                 # Tier 3C Binary Variational Classical Annealing (Dilated RNN + Adam, CPU/CUDA)
├── benchmarks/
│   └── run_all_benchmarks.py         # Unified CSV-driven multi-solver benchmark runner
├── tests/
│   └── test_suite.py                 # Algebraic, phase-transition, and feasibility unit tests
└── results/                          # Generated benchmark CSV summaries & JSON reports
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Verify QUBO algebra, anti-fraud gate, P3 feasibility, and analytical phase transition
python tests/test_suite.py

# 2. Launch the interactive explorer in JupyterLab or Jupyter Notebook
jupyter lab notebooks/CarbonIQ_Explorer.ipynb

# 3. Run unified multi-solver benchmarks across P1-P5 (CLI)
python benchmarks/run_all_benchmarks.py --data-dir data/starter_small --scale small
python benchmarks/run_all_benchmarks.py --data-dir data/starter_medium --scale medium --skip-vca
python benchmarks/run_all_benchmarks.py --data-dir data/starter_large --scale large --skip-vca
```

### Running in Google Colab

1. Upload `carboniq-ecocycle-benchmarks-v2.zip` to `/content/` in Colab and open `notebooks/CarbonIQ_Explorer.ipynb` (or upload the `.ipynb` directly into Colab alongside the zip).
2. Run **Cell 1**: it automatically extracts the archive, configures `sys.path`, detects CUDA GPU availability (`VCA_DEVICE = "auto"`), and enables `ipywidgets` sliders.

## Ground-State Reference Format (`GS_energies_*.txt`)

Each file in `data/P*/GS_energies_*.txt` contains 25 deterministic seeds (`seed = 1..25`) with space-separated columns:

```text
# Columns: seed  gs_energy  gs_energy_per_var  N  feasible  gs_bitstring
```

Load directly with NumPy (`np.loadtxt(path, comments='#', usecols=(0, 1))`) to evaluate residual energy per variable `(E - E_GS) / N`.
