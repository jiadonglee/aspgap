"""Evaluation metrics. Everything is computed in RAW label units.

- per-label masked RMSE / bias / robust scatter / R^2 / prediction-error
  calibration;
- derived [Mg/Fe] = [Mg/H] - [Fe/H] with propagated (independent) errors;
- the conditional chemical-information experiment: regress [alpha/M] (or
  [Mg/Fe]) on (Teff, logg, [M/H]) with a quadratic relation fit on TRAIN,
  define the residual delta, and score how much of delta the model recovers
  beyond the atmospheric correlations.
"""

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


def derived_mgfe(mg_h_mu: np.ndarray, mg_h_sigma: np.ndarray,
                 fe_h_mu: np.ndarray, fe_h_sigma: np.ndarray):
    """[Mg/Fe] and its error assuming independent [Mg/H], [Fe/H] errors."""
    return mg_h_mu - fe_h_mu, np.sqrt(mg_h_sigma ** 2 + fe_h_sigma ** 2)


# ------------------- conditional chemical-information test ------------------

def _design(teff: np.ndarray, logg: np.ndarray, mh: np.ndarray) -> np.ndarray:
    """Standardized quadratic basis of (Teff, logg, [M/H]) incl. cross terms."""
    cols = [teff, logg, mh]
    m = [np.isfinite(c) for c in cols]
    ok = np.logical_and.reduce(m)
    X = np.stack([c[ok] for c in cols], 1)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    a, b, c = Xs[:, 0], Xs[:, 1], Xs[:, 2]
    return np.stack([np.ones_like(a), a, b, c, a * b, a * c, b * c,
                     a ** 2, b ** 2, c ** 2], 1), ok


def fit_conditional_mean(teff: np.ndarray, logg: np.ndarray, mh: np.ndarray,
                         chem: np.ndarray):
    """Fit chem ~ f(Teff, logg, [M/H]) on finite rows; returns (predict_fn,
    residual array with NaN where not applicable)."""
    D, ok = _design(teff, logg, mh)
    t = chem[ok]
    finite = np.isfinite(t)
    beta, *_ = np.linalg.lstsq(D[finite], t[finite], rcond=None)

    pred = np.full_like(chem, np.nan, dtype=float)
    p = D @ beta
    pred[ok] = p
    finite_full = ok.copy()          # `finite` lives on the ok-subset; map back
    finite_full[ok] = finite
    delta = np.where(finite_full, chem - pred, np.nan)

    def predict(teff_: np.ndarray, logg_: np.ndarray, mh_: np.ndarray) -> np.ndarray:
        teff_ = np.asarray(teff_, float)
        D_, ok_ = _design(teff_, np.asarray(logg_, float),
                          np.asarray(mh_, float))
        out = np.full(len(teff_), np.nan)
        out[ok_] = D_ @ beta
        return out

    r2 = float(1.0 - np.var(t[finite] - p[finite]) / np.var(t[finite])) \
        if np.var(t[finite]) > 0 else np.nan
    return predict, delta, {"r2_conditional_mean": r2, "n_fit": int(finite.sum())}


def residual_recovery_score(delta_true: np.ndarray, delta_pred: np.ndarray):
    """How much of the conditional residual does the model see in the XP?
    Reported three ways on the common finite set: raw R^2 (delta_pred used
    as-is), the Pearson r, and the recalibrated R^2 after fitting the optimal
    affine map delta_pred -> delta_true (isolates 'is the signal there at all'
    from 'is the model's sigma scale right')."""
    ok = np.isfinite(delta_true) & np.isfinite(delta_pred)
    n = int(ok.sum())
    if n < 10:
        return {"r2_delta": np.nan, "n": n}
    t, p = delta_true[ok], delta_pred[ok]
    A = np.stack([p, np.ones_like(p)], 1)
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    p_cal = A @ coef
    var_t = np.var(t)
    return {
        "r2_delta": float(1.0 - np.mean((t - p) ** 2) / var_t),
        "r2_delta_recal": float(1.0 - np.mean((t - p_cal) ** 2) / var_t),
        "pearson": float(np.corrcoef(t, p)[0, 1]),
        "rmse_delta": float(np.sqrt(np.mean((t - p) ** 2))),
        "n": n,
    }
