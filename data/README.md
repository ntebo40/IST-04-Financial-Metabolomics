# Data — IST-04 Financial Metabolomics

## JSE Canonical Panel

**Full dataset:** Zenodo DOI [10.5281/zenodo.19072906](https://doi.org/10.5281/zenodo.19072906)

| File | Description |
|------|-------------|
| `jse_returns.parquet` | Daily log-returns, 87 × 2,731 |
| `jse_volume.parquet` | Demeaned log-volume, 87 × 2,731 |
| `jse_realised_var.parquet` | 21-day realised variance, 87 × 2,731 |
| `eskom_stages.csv` | Eskom load-shedding stages 0–6, 2,731 days |
| `sector_weights.csv` | Energy-intensity weights ωᵢ (GICS) |
| `transfer_entropy.parquet` | Pairwise TE matrix, 87 × 87, training baseline |

**Period:** 5 January 2015 – 31 December 2025

| Split | Period | Days |
|-------|--------|------|
| Training baseline | Jan 2015 – Dec 2018 | 992 |
| Walk-forward | Jan 2019 – Dec 2023 | 1,242 |
| Hold-out | Jan 2024 – Dec 2025 | 497 |
