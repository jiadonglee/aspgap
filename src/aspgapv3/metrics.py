"""Evaluation metrics, computed in RAW label units: per-label masked RMSE / bias / robust scatter / R^2 and
prediction-error calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def masked_stats(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                 mask: np.ndarray, label: str) -> dict:
    m = mask.astype(bool)
    n = int(m.sum())
    if n == 0:
        return {"label": label, "n": 0}
    r = mu[m] - y[m]
    rms = float(np.sqrt(np.mean(r ** 2)))
    mad = float(1.4826 * np.median(np.abs(r - np.median(r))))
    var_y = float(np.var(y[m]))
    return {
        "label": label, "n": n,
        "rmse": rms, "bias": float(np.mean(r)),
        "scatter_mad": mad,
        "r2": float(1.0 - np.mean(r ** 2) / var_y) if var_y > 0 else np.nan,
        "sigma_med": float(np.median(sigma[m])),
        # calibration: predicted sigma should match the actual residual scale
        "calibration_ratio": float(np.sqrt(np.mean(r ** 2)) / np.sqrt(
            np.mean(sigma[m] ** 2))) if np.all(np.isfinite(sigma[m])) else np.nan,
    }


def metrics_table(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                  mask: np.ndarray, labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame([masked_stats(mu[:, k], sigma[:, k], y[:, k],
                                      mask[:, k], lab)
                         for k, lab in enumerate(labels)])
