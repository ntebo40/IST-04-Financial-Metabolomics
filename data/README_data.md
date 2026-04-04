# Data Description — IST-04 Financial Metabolomics

## JSE Canonical Panel

| Attribute | Value |
|-----------|-------|
| Exchange | Johannesburg Stock Exchange (JSE) |
| Securities | N = 87 continuously listed equities |
| Period | 5 January 2015 – 31 December 2025 |
| Trading days | T = 2,731 |
| Splits | Training 992d / Walk-forward 1,242d / Hold-out 497d |

## Features
- **r_{i,t}** — log-return = log(P_{i,t} / P_{i,t-1})
- **v_{i,t}** — demeaned log-volume = log(V_{i,t} / V̄_i)
- **σ²_{i,t}** — 21-day realised variance

## Eskom Load-Shedding
`eskom_stages.csv` — daily stages ε_t ∈ {0,...,6}, sourced from
Eskom schedule archive + EskomSePush API cross-validation.

## Transfer Entropy Matrix
`transfer_entropy/te_matrix_training.npy` — 87×87 KSG-estimated
pairwise TE values, lag ℓ*=3, training baseline only (no look-ahead).

## Raw Data Access
Raw JSE prices are proprietary. Contact: datasales@jse.co.za
Derived quantities for reproducibility: https://doi.org/10.5281/zenodo.19072906
