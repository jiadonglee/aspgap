"""Red-clump (RC) A0 anchors for the distance head, defined in the model's own G-band convention.

The existing A0 anchors (nearby A0 = 0, high-latitude A0 = SFD) only cover low-extinction lines of sight.
RC stars are a standard candle that is visible through the disk, so every RC star with a usable parallax
gives a (noisy) per-star A0 measurement at ANY extinction:

    A0 = (G - M_G^RC([M/H], dTeff) - mu_plx) / K_G,   mu_plx = 5 log10(1000 / (plx - zp)) - 5,   K_G = 0.789.

Selection (APOGEE labels only, TRAIN split, source == "apogee"):
  Bovy+2014-like Teff-logg-[M/H] window, dT = Teff - (4607 - 382.5 [M/H]):
      2.2 <= logg <= min(0.0018 dT + 2.5, 2.9),  -150 < dT < 250 K,  -0.8 < [M/H] < 0.5.
  (The lower logg edge 2.2 and the dT window are empirical: outside them the low-extinction M_G distribution
  is no longer a single peak, the upper line removes the RGB / subgiants; [C/N] is not in the processed
  catalogue for the external rows and showed no gain once this window is applied.)

Calibration (self-consistent with the parallax loss, which uses the same K_G, zero-point and G):
  low-extinction RC (A_SFD = 3.1 * 0.86 * E(B-V)_SFD < 0.1, parallax S/N > 10, RUWE < 1.4) give
      M_G = G + 5 log10(plx_corr) - 10 - K_G * A_SFD
  fitted with a robust (3 sigma-MAD clipped) least squares as  c0 + c1 m + c2 m^2 + c3 dT/100  (m = [M/H]);
  sigma_RC is the maximum-likelihood intrinsic scatter after subtracting the parallax-error term.

Anchor: stars of the training split that are RC, have parallax S/N > 5 (any extinction), are not already a
nearby / high-latitude anchor.  sigma_A0 = sqrt(sigma_RC^2 + max(sigma_mu, floor)^2) / K_G with
sigma_mu = 5/ln10 * plx_err / plx_corr.  Targets are NOT clipped at 0 (a negative target is only disfavoured
through its sigma).
"""

from __future__ import annotations

import numpy as np

K_G = 0.789
RC_SET = 3
A_SFD_FAC = 3.1 * 0.86          # A0(SFD) = 3.1 * 0.86 * E(B-V)_SFD, as in the high-latitude anchors
_5_LN10 = 5.0 / np.log(10.0)
SELECTION = {"logg_lo": 2.2, "logg_hi": 2.9, "dT_lo": -150.0, "dT_hi": 250.0, "mh_lo": -0.8, "mh_hi": 0.5,
             "slope_logg": 0.0018, "logg_off": 2.5, "teff_ref0": 4607.0, "teff_ref_slope": -382.5}
CALIB_CUTS = {"a_sfd_max": 0.1, "snr_min": 10.0, "ruwe_max": 1.4, "clip": 3.0}


def rc_dT(teff, m_h, sel: dict = SELECTION):
    return np.asarray(teff, float) - (sel["teff_ref0"] + sel["teff_ref_slope"] * np.asarray(m_h, float))


def rc_select(teff, logg, m_h, sel: dict = SELECTION) -> np.ndarray:
    """Boolean RC flag from APOGEE labels (NaN label -> False)."""
    teff, logg, m_h = (np.asarray(v, float) for v in (teff, logg, m_h))
    dT = rc_dT(teff, m_h, sel)
    with np.errstate(invalid="ignore"):
        return ((logg >= sel["logg_lo"]) & (logg <= np.minimum(sel["slope_logg"] * dT + sel["logg_off"], sel["logg_hi"]))
                & (dT > sel["dT_lo"]) & (dT < sel["dT_hi"]) & (m_h > sel["mh_lo"]) & (m_h < sel["mh_hi"]))


def _design(m_h, teff):
    m = np.asarray(m_h, float)
    return np.stack([np.ones_like(m), m, m ** 2, rc_dT(teff, m) / 100.0], 1)


def rc_mg(coef, m_h, teff):
    """M_G^RC([M/H], Teff) from the calibration coefficients."""
    return _design(m_h, teff) @ np.asarray(coef, float)


def mu_plx(plx_corr, plx_err):
    """Distance modulus 5 log10(1000/plx_corr) - 5 and its error (mag); NaN where plx_corr <= 0."""
    plx_corr, plx_err = np.asarray(plx_corr, float), np.asarray(plx_err, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(plx_corr > 0, 5.0 * np.log10(1000.0 / plx_corr) - 5.0, np.nan), \
            np.where(plx_corr > 0, _5_LN10 * plx_err / plx_corr, np.nan)


def calibrate_rc(g, plx_corr, plx_err, m_h, teff, logg, ebv_sfd, ruwe, cuts: dict = CALIB_CUTS,
                 sel: dict = SELECTION) -> dict:
    """Self-calibrate M_G^RC on low-extinction RC stars (arrays of TRAIN APOGEE stars).  Returns
    {"coef", "sigma_rc", "n", "n_clipped", "scatter_mad", "sel", "cuts"}."""
    from scipy.optimize import minimize_scalar
    g, plx_corr, plx_err, m_h, teff, logg, ebv, ruwe = (np.asarray(v, float) for v in
                                                         (g, plx_corr, plx_err, m_h, teff, logg, ebv_sfd, ruwe))
    a_sfd = A_SFD_FAC * ebv
    with np.errstate(invalid="ignore", divide="ignore"):
        use = (rc_select(teff, logg, m_h, sel) & (plx_corr > 0) & (plx_corr / plx_err > cuts["snr_min"])
               & (ruwe < cuts["ruwe_max"]) & (a_sfd < cuts["a_sfd_max"]) & np.isfinite(g))
        mg = g + 5.0 * np.log10(plx_corr) - 10.0 - K_G * np.nan_to_num(a_sfd)
    use &= np.isfinite(mg)
    X, y = _design(m_h[use], teff[use]), mg[use]
    _, smu = mu_plx(plx_corr[use], plx_err[use])
    keep = np.ones(len(y), bool)
    for _ in range(10):
        coef = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        r = y - X @ coef
        s = 1.4826 * np.median(np.abs(r[keep]))
        keep = np.abs(r) < cuts["clip"] * s
    nll = lambda sr: np.sum(0.5 * (r[keep] ** 2 / (sr ** 2 + smu[keep] ** 2) + np.log(sr ** 2 + smu[keep] ** 2)))
    sig = float(minimize_scalar(nll, bounds=(0.01, 0.6), method="bounded").x)
    return {"coef": [float(c) for c in coef], "sigma_rc": sig, "n": int(keep.sum()), "n_selected": int(use.sum()),
            "n_clipped": int((~keep).sum()), "scatter_mad": float(s), "sel": dict(sel), "cuts": dict(cuts)}


def rc_anchor_targets(g, plx_corr, plx_err, m_h, teff, logg, calib: dict, snr_min: float = 5.0,
                      sig_mu_floor: float = 0.1, sel: dict | None = None):
    """Per-star RC A0 anchors.  Returns (target, sigma, ok) (numpy); ok = RC & parallax S/N > snr_min.
    Targets are not clipped at zero."""
    g, plx_corr, plx_err = (np.asarray(v, float) for v in (g, plx_corr, plx_err))
    sel = calib.get("sel", SELECTION) if sel is None else sel
    mu, smu = mu_plx(plx_corr, plx_err)
    mg_rc = rc_mg(calib["coef"], m_h, teff)
    with np.errstate(invalid="ignore", divide="ignore"):
        tgt = (g - mg_rc - mu) / K_G
        sig = np.sqrt(calib["sigma_rc"] ** 2 + np.maximum(smu, sig_mu_floor) ** 2) / K_G
        ok = (rc_select(teff, logg, m_h, sel) & (plx_corr > 0) & (plx_corr / plx_err > snr_min)
              & np.isfinite(tgt) & np.isfinite(sig))
    return np.where(ok, tgt, 0.0), np.where(ok, sig, 1.0), ok
