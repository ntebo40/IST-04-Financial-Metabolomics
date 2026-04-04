"""
metrics.py — Evaluation metrics for GWS-STNet (IST-04 Paper 1)

Implements: GCS (Gaussian Calibration Score), MER (Metabolic
Efficiency Ratio), HED (Hurst Exponent Deviation), DM test.

Author: Prof. Ntebogang Dinah Moroke
"""

import numpy as np
from scipy import stats


def gaussian_calibration_score(residuals_test, residuals_train):
    """
    Gaussian Calibration Score (GCS) based on squared 2-Wasserstein distance.

    GCS = exp(-W2²(q_test, N(0, σ²_train)) / σ²_test)
    GCS ∈ [0, 1]; higher is better.
    """
    mu_test  = np.mean(residuals_test)
    s2_test  = np.var(residuals_test)
    s2_train = np.var(residuals_train)
    # Closed-form W2² for univariate: (μ_p - μ_q)² + (σ_p - σ_q)²
    w2_sq = mu_test**2 + (np.sqrt(s2_test) - np.sqrt(s2_train))**2
    gcs = np.exp(-w2_sq / max(s2_test, 1e-10))
    return float(np.clip(gcs, 0, 1))


def metabolic_efficiency_ratio(rmse, saliency_mean, entropy_mean):
    """
    Metabolic Efficiency Ratio (MER).
    MER = (1 - RMSE) / (1 + E[S_ms] * H_test)
    Higher is better.
    """
    return (1 - rmse) / (1 + saliency_mean * entropy_mean)


def hurst_exponent(series, min_lag=2, max_lag=100):
    """
    Hurst exponent via rescaled range (R/S) analysis.
    H > 0.5: long memory. H = 0.5: random walk. H < 0.5: mean-reverting.
    """
    lags = np.arange(min_lag, min(max_lag, len(series)//4))
    rs_vals = []
    for lag in lags:
        chunks = [series[i:i+lag] for i in range(0, len(series)-lag, lag)]
        rs_per_chunk = []
        for chunk in chunks:
            mean = np.mean(chunk)
            cum_dev = np.cumsum(chunk - mean)
            R = np.max(cum_dev) - np.min(cum_dev)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_per_chunk.append(R / S)
        if rs_per_chunk:
            rs_vals.append(np.mean(rs_per_chunk))
    if len(rs_vals) < 2:
        return 0.5
    log_lags = np.log(lags[:len(rs_vals)])
    log_rs   = np.log(rs_vals)
    H, _, _, _, _ = stats.linregress(log_lags, log_rs)
    return float(H)


def hurst_exponent_deviation(predicted, actual):
    """HED = |H_predicted - H_actual|. Lower is better."""
    return abs(hurst_exponent(predicted) - hurst_exponent(actual))


def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction.
    H0: equal predictive accuracy.
    Returns: DM statistic, p-value (two-sided).
    """
    d  = e1**2 - e2**2        # loss differential (MSE-based)
    T  = len(d)
    d_bar = np.mean(d)
    # Newey-West variance with h-1 lags
    gamma0 = np.var(d, ddof=1)
    gammas = sum(2*(1 - j/(h)) * np.cov(d[:-j], d[j:])[0,1]
                 for j in range(1, h)) if h > 1 else 0
    var_d = (gamma0 + gammas) / T
    # HLN correction factor
    hln = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    dm_stat = d_bar / (np.sqrt(var_d) * hln + 1e-12)
    p_val = 2 * stats.t.sf(abs(dm_stat), df=T-1)
    return float(dm_stat), float(p_val)
