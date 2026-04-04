# IST-04: Financial Metabolomics Series

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19339552.svg)](https://doi.org/10.5281/zenodo.19339552)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

**Repository for the Financial Metabolomics Series (IST-04)**  
*Faculty of Economic and Management Sciences, North-West University, South Africa*

**Author:** Prof. Ntebogang Dinah Moroke  
**ORCID:** [0000-0001-8545-1860](https://orcid.org/0000-0001-8545-1860)  
**Email:** Ntebo.Moroke@nwu.ac.za

---

## Series Overview

The Financial Metabolomics Series applies non-equilibrium thermodynamics,
information geometry, and hierarchical transformer architectures to systemic
stress detection in the Johannesburg Stock Exchange (JSE). The series treats
financial markets as metabolic systems far from equilibrium, where stress
propagates as directed information flux between sector-level nodes.

| Paper | Title | Journal | Status |
|-------|-------|---------|--------|
| **Paper 1** | Gaussian-Weighted Swin Networks with Contractive Attention and Metabolic Saliency: An Intrinsically Interpretable Architecture for Systemic Stress Detection in JSE Equity Markets | *Mathematics* (MDPI) | Submitted |
| **Paper 2** | Metabolic Saliency as a KL-Divergence Estimator: Information-Geometric Attribution of Systemic Stress in JSE Equity Networks | *Entropy* (MDPI) | In preparation |
| **Paper 3** | Scale-Invariant Metabolic Stress Propagation in JSE Equity Networks: A Fractal Conservation Law for the GWS-PMNet System | *Chaos, Solitons and Fractals* (Elsevier) | Planned |

---

## Dataset

**JSE Canonical Panel (IST-04)**

| Property | Value |
|----------|-------|
| Securities | 87 continuously listed JSE equities |
| Period | 5 January 2015 – 31 December 2025 |
| Trading days | 2,731 |
| Features | Log-return, demeaned log-volume, 21-day realised variance |
| Exogenous | Eskom load-shedding stages 0–6 (EskomSePush API) |
| Splits | Train: Jan 2015–Dec 2018 (992 days) · Walk-forward: Jan 2019–Dec 2023 (1,242 days) · Hold-out: Jan 2024–Dec 2025 (497 days) |

Derived data (aggregated, anonymised) available on Zenodo:  
**DOI:** [10.5281/zenodo.19339552](https://doi.org/10.5281/zenodo.19339552)

Raw individual security data are proprietary to the JSE and are not redistributed.

---

## Repository Structure

```
IST-04-Financial-Metabolomics/
├── README.md
├── LICENSE
├── CITATION.cff
│
├── paper1/                        # GWS-STNet (Mathematics MDPI)
│   ├── main.tex                   # LaTeX source
│   ├── figures/                   # All 5 publication figures (PDF)
│   │   ├── fig0_architecture.pdf
│   │   ├── fig1_saliency_heatmap.pdf
│   │   ├── fig2_phase_portrait.pdf
│   │   ├── fig3_pareto_front.pdf
│   │   └── fig4_glassbox_network.pdf
│   └── supplementary/
│
├── paper2/                        # Entropy-Saliency (Entropy MDPI)
│   ├── main.tex
│   └── figures/
│       ├── fig_stif_network.pdf
│       └── fig2_glassbox_kl_tracking.pdf
│
├── paper3/                        # Fractal Conservation Law (planned)
│   └── README.md
│
├── data/
│   ├── README.md                  # Data description and access instructions
│   └── sample/                    # Small anonymised sample for code testing
│
├── figures/
│   └── generate_all_figures.py    # Reproduces all publication figures
│
└── code/
    ├── requirements.txt
    ├── model/
    │   ├── gws_stnet.py           # GWS-STNet architecture (PyTorch)
    │   ├── pmnet.py               # Power Mapping Network
    │   └── metabolic_loss.py      # Entropy-weighted metabolic loss
    ├── attribution/
    │   ├── metabolic_saliency.py  # Metabolic Saliency computation
    │   └── transfer_entropy.py    # KSG transfer entropy estimator
    ├── evaluation/
    │   ├── metrics.py             # GCS, MER, HED, RMSE, MAE
    │   └── diebold_mariano.py     # DM test with Harvey correction
    └── experiments/
        ├── train.py               # Full training pipeline
        ├── evaluate.py            # Hold-out evaluation
        └── bandwidth_ablation.py  # Conjecture 1 verification
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ntebo40/IST-04-Financial-Metabolomics.git
cd IST-04-Financial-Metabolomics

# Install dependencies
pip install -r code/requirements.txt

# Reproduce all publication figures
python figures/generate_all_figures.py

# Run bandwidth sensitivity ablation (Table 7, Paper 1)
python code/experiments/bandwidth_ablation.py --sigma 0.10 0.20 0.30 0.39 0.45
```

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{Moroke2026gwsstnet,
  author  = {Moroke, Ntebogang Dinah},
  title   = {Gaussian-Weighted Swin Networks with Contractive Attention
             and Metabolic Saliency: An Intrinsically Interpretable
             Architecture for Systemic Stress Detection in
             {JSE} Equity Markets},
  journal = {Mathematics},
  year    = {2026},
  note    = {Submitted. Financial Metabolomics Series, Paper 1.
             DOI: 10.5281/zenodo.19339552}
}
```

---

## Related Repositories

| Series | Repository | DOI |
|--------|------------|-----|
| IST-02 (WOW-E-W Quadrilogy) | [ntebo40/IST03\_Quadrilogy](https://github.com/ntebo40/IST03_Quadrilogy) | [10.5281/zenodo.19339552](https://doi.org/10.5281/zenodo.19339552) |
| SHREDI (IRFA) | Manuscript: FINANA-D-26-01285 | — |
| GEODEX (J. Stat. Phys.) | Co-author: L.D. Metsileng | — |

---

## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

Code is additionally released under the [MIT License](LICENSE).

---

## Acknowledgements

The author acknowledges institutional support from the Faculty of Economic
and Management Sciences, North-West University (Mafikeng Campus), South Africa.
Eskom load-shedding data sourced via the EskomSePush API.
