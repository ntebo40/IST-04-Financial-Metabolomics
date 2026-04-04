"""
Evaluation Metrics — IST-04 Financial Metabolomics Series

GCS: Gaussian Calibration Score (2-Wasserstein based)
MER: Metabolic Efficiency Ratio
HED: Hurst Exponent Deviation
DM:  Diebold-Mariano test (see diebold_mariano.py)
"""
import numpy as np
from scipy import stats

def gcs(residuals_test, sigma0_sq):
    """
    Gaussian Calibration Score.
    GCS = exp(-W2^2(q_hat, N(0, sigma0^2)) / sigma_test^2)
    W2^2(p,q) = (mu_p - mu_q)^2 + (sigma_p - sigma_q)^2  [univariate]
    """
    mu_t    = residuals_test.mean()
    sig_t   = residuals_test.std()
    sig0    = sigma0_sq ** 0.5
    w2_sq   = mu_t**2 + (sig_t - sig0)**2
    return float(np.exp(-w2_sq / residuals_test.var()))

def mer(rmse, mean_saliency, mean_entropy):
    """Metabolic Efficiency Ratio: (1-RMSE) / (1 + S_bar * H_bar)"""
    return (1 - rmse) / (1 + mean_saliency * mean_entropy)

def hurst(ts, min_n=10):
    """Hurst exponent via R/S analysis."""
    n = len(ts)
    ns = [n // k for k in range(2, 20) if n // k >= min_n]
    rs_vals = []
    for sub_n in ns:
        chunks = [ts[i:i+sub_n] for i in range(0, n - sub_n + 1, sub_n)]
        rs_chunk = []
        for c in chunks:
            mean_c = np.mean(c)
            dev    = np.cumsum(c - mean_c)
            rs_chunk.append((dev.max() - dev.min()) / (np.std(c) + 1e-12))
        rs_vals.append(np.mean(rs_chunk))
    if len(rs_vals) < 2:
        return 0.5
    log_ns = np.log(ns)
    log_rs = np.log(rs_vals)
    slope, *_ = np.polyfit(log_ns, log_rs, 1)
    return float(slope)

def hed(predicted, actual):
    """Hurst Exponent Deviation."""
    return abs(hurst(predicted) - hurst(actual))
