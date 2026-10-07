"""Evaluation of the M_G / A0 outputs (pure numpy; used by scripts/evaluate.py).

All parallax quantities are in mas.  `ar` is APOGEEDataset.astro_arr (original-star G, parallax,
errors, zero-point, ok masks, magnitudes), `tab` APOGEEDataset.astro_table (l, b, ra, dec ...).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .models.distance import BANDS, EXT_K, LN10_5

G_BINS = (0.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 99.0)
SFD_DIR = "data/dustmaps/sfd"          # SFD_dust_4096_{ngp,sgp}.fits (see README / build notes)


def rstd(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def med(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def parallax_arrays(mg, logvar_mg, a0, ar: dict) -> dict:
    """Predicted parallax, total sigma and pull, plus the M_G + A_G / parallax-based comparison."""
    s_mg = np.sqrt(np.exp(np.clip(logvar_mg, -8, 8)))
    pp = 10.0 ** ((10.0 - ar["g"] + mg + EXT_K["G"] * a0) / 5.0)
    sig = np.sqrt(ar["plx_err"] ** 2 + (pp * LN10_5 * s_mg) ** 2)
    corr = ar["plx"] - ar["zp"]
    ok = ar["plx_ok"]
    pull = np.where(ok, (corr - pp) / sig, np.nan)
    snr = np.where(ok, corr / ar["plx_err"], np.nan)
    hi = ok & (snr > 10)
    mg_plx = np.where(hi, ar["g"] + 5 * np.log10(np.where(hi, corr, 1.0)) - 10.0, np.nan)   # = M_G + A_G
    resid = np.where(hi, (mg + EXT_K["G"] * a0) - mg_plx, np.nan)
    return dict(pp=pp, sig=sig, pull=pull, hi=hi, resid=resid, s_mg=s_mg, corr=corr, ok=ok,
                lnratio=np.where(hi, np.log(np.where(hi, corr, 1.0) / pp), np.nan))


def distance_tables(mg, logvar_mg, a0, ar: dict) -> tuple[pd.DataFrame, dict]:
    """Per-G-bin parallax pull / M_G residual / distance-precision table + overall dict."""
    d = parallax_arrays(mg, logvar_mg, a0, ar)
    rows = []
    for lo, hi_ in [(0.0, 99.0)] + list(zip(G_BINS[:-1], G_BINS[1:])):
        sel = (ar["g"] >= lo) & (ar["g"] < hi_)
        sp = sel & d["ok"]
        sh = sel & d["hi"]
        rows.append({"G_lo": lo, "G_hi": hi_, "n": int(sp.sum()),
                     "pull_median": med(d["pull"][sp]), "pull_rstd": rstd(d["pull"][sp]),
                     "n_hi_snr": int(sh.sum()),
                     "MG_resid_rmse": float(np.sqrt(np.nanmean(d["resid"][sh] ** 2))) if sh.any() else np.nan,
                     "MG_resid_mad": rstd(d["resid"][sh]), "MG_resid_median": med(d["resid"][sh]),
                     "frac_dist_pred_median": float(np.median(LN10_5 * d["s_mg"][sel])) if sel.any() else np.nan,
                     "frac_dist_obs_rstd_hi": rstd(d["lnratio"][sh])})
    t = pd.DataFrame(rows)
    return t, d


def red_clump(y_raw, mask, labels, ar: dict, mg, a0, k_ok) -> dict:
    """APOGEE red-clump locus (2.3<logg<2.6, 4500<Teff<5100, [M/H]>-0.6): predicted
    M_K = K - mu - A_K with mu = G - M_G - A_G (should be ~ -1.6, small scatter)."""
    ix = {n: i for i, n in enumerate(labels)}
    teff, logg, mh = (y_raw[:, ix[n]] for n in ("teff", "logg", "m_h"))
    ok = mask[:, ix["teff"]] & mask[:, ix["logg"]] & mask[:, ix["m_h"]]
    rc = ok & (logg > 2.3) & (logg < 2.6) & (teff > 4500) & (teff < 5100) & (mh > -0.6) & k_ok
    kmag = ar["mag"][:, BANDS.index("K")]
    mu = ar["g"] - mg - EXT_K["G"] * a0
    mk = kmag - mu - EXT_K["K"] * a0
    v = mk[rc]
    return {"n": int(rc.sum()), "MK_median": med(v), "MK_rstd": rstd(v),
            "MK_std": float(np.nanstd(v)) if v.size else np.nan}


def _sfd_ebv(ra, dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from dustmaps.sfd import SFDQuery
    q = SFDQuery(map_dir=SFD_DIR)
    return np.asarray(q(SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")), float)


def _spearman(a, b) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def extinction_report(a0, tab: pd.DataFrame, teff_label, teff_pred, nearby=None) -> dict:
    """A0 vs SFD (x0.86, R_V=3.1) at |b|>30, plus the A0-Teff degeneracy checks."""
    out = {"a0_min": float(np.min(a0)), "a0_median": float(np.median(a0)), "frac_a0_lt0": float(np.mean(a0 < 0))}
    if nearby is not None and nearby.any():
        out.update({"n_nearby": int(nearby.sum()), "a0_nearby_median": med(a0[nearby]),
                    "a0_nearby_p90": float(np.percentile(a0[nearby], 90))})
    try:
        e = _sfd_ebv(tab["ra"].to_numpy(float), tab["dec"].to_numpy(float))
    except Exception as exc:                     # dustmaps / map files unavailable
        out["sfd_error"] = repr(exc)
        return out
    a_sfd = 3.1 * 0.86 * e
    b = tab["b"].to_numpy(float)
    hb = np.abs(b) > 30
    out.update({"spearman_a0_teff_hb": _spearman(a0[hb], teff_label[hb]), "n_hb": int(hb.sum()), "a0_hb_median": med(a0[hb]), "a_sfd_hb_median": med(a_sfd[hb]),
                "pearson_a0_sfd_hb": float(np.corrcoef(a0[hb], a_sfd[hb])[0, 1]),
                "spearman_a0_sfd_hb": _spearman(a0[hb], a_sfd[hb]),
                "a0_minus_sfd_median": med(a0[hb] - a_sfd[hb]), "a0_minus_sfd_rstd": rstd(a0[hb] - a_sfd[hb])})
    lo = (np.abs(b) > 40) & (e < 0.04)
    out.update({"n_lowext": int(lo.sum()),
                "spearman_a0_teff_lowext": _spearman(a0[lo], teff_label[lo]),
                "spearman_a0_teffresid_lowext": _spearman(a0[lo], (teff_pred - teff_label)[lo])})
    # fixed sky position: subtract the median A0 of each 2x2 deg (l, sin b) cell, then correlate with Teff
    l = tab["l"].to_numpy(float)
    cell = (np.floor(l / 2.0).astype(int) * 1000 + np.floor((np.sin(np.radians(b)) + 1) * 90).astype(int))
    df = pd.DataFrame({"c": cell, "a": a0, "t": teff_label, "r": teff_pred - teff_label})
    g = df.groupby("c")["a"]
    df = df[g.transform("size") >= 8]
    da = df["a"] - df.groupby("c")["a"].transform("median")
    out["spearman_a0cellresid_teff"] = _spearman(da.to_numpy(), df["t"].to_numpy())
    out["spearman_a0cellresid_teffresid"] = _spearman(da.to_numpy(), df["r"].to_numpy())
    out["n_cellstars"] = int(len(df))
    edges = np.arange(3500, 8001, 500)
    tb = pd.DataFrame({"teff_bin": pd.cut(teff_label[lo], edges), "a0": a0[lo]}).groupby(
        "teff_bin", observed=True)["a0"].agg(["size", "median"])
    out["a0_by_teff_lowext"] = {str(k): (int(r["size"]), round(float(r["median"]), 3)) for k, r in tb.iterrows()}
    return out
