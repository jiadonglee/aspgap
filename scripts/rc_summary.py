#!/usr/bin/env python
"""runs/<new>/rc_summary.md: base (ext_r2) | new (ext2_rc) on identical rows.

    python scripts/rc_summary.py --base aspgap_dist_ext_r2 --new aspgap_dist_ext2_rc

Inputs (nothing is overwritten except <new>/rc_summary.md): evaluate.py outputs of both runs (APOGEE tests,
predictions_dist_*), `eval_ood.py --holdout` outputs (<run>/ood_holdout/pred_*.parquet, metrics_*.csv), the OOD
catalogues, and the RC calibration of the ext2 data root (src/aspgapv3/rc_anchor.py)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from aspgapv3.data.dataset import APOGEEDataset  # noqa: E402
from aspgapv3.models.distance import anchor_targets  # noqa: E402
from aspgapv3.rc_anchor import A_SFD_FAC, K_G, rc_anchor_targets  # noqa: E402

RUNS = Path("/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3/runs")
ROOT = REPO / "data/processed/aspcap_v0.6.0_ext2"
OOD = REPO / "data/processed/ood"
LAB = ["teff", "logg", "m_h", "alpha_m"]
ABINS = [(-9, 0.5, "<=0.5"), (0.5, 1, "0.5-1"), (1, 2, "1-2"), (2, 4, "2-4"), (4, 1e9, ">4")]
GBINS = [(0, 13, "G<13"), (13, 14, "13-14"), (14, 15, "14-15"), (15, 16, "15-16"), (16, 17, "16-17"), (17, 18, "G>17")]
SPLITS = ("iid_test", "sky_test")


def md(df, fmt=".3f"):
    if df is None or len(df) == 0:
        return "(empty)\n"
    c = list(df.columns)
    o = ["| " + " | ".join(map(str, c)) + " |", "|" + "|".join("---" for _ in c) + "|"]
    for _, r in df.iterrows():
        o.append("| " + " | ".join(f"{v:{fmt}}" if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                   else ("" if isinstance(v, (float, np.floating)) else str(v)) for v in r) + " |")
    return "\n".join(o) + "\n"


def mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def side(rows_by_run: dict, keys: list[str], vals: list[str]) -> pd.DataFrame:
    """rows_by_run: tag -> DataFrame with keys + vals; -> keys, then val_<tag> for each tag."""
    out = None
    tags = list(rows_by_run)
    for t, d in rows_by_run.items():
        d = d[keys + vals].rename(columns={v: f"{v}|{t}" for v in vals})
        out = d if out is None else out.merge(d, on=keys, how="outer")
    cols = list(keys)
    for v in vals:
        cols += [f"{v}|{t}" for t in tags]
    return out[cols]


# ---------------------------------------------------------------- APOGEE tests
def apogee_tables(tags):
    S = []
    for sp in SPLITS:
        d = {t: pd.read_csv(RUNS / r / f"metrics_{sp}.csv") for t, r in tags.items()}
        S += [f"### APOGEE {sp}: label rmse / bias (must not degrade)\n", md(side(d, ["label"], ["n", "rmse", "bias", "scatter_mad"]), ".4f")]
    for sp in SPLITS:
        d = {t: pd.read_csv(RUNS / r / f"metrics_distance_{sp}.csv") for t, r in tags.items()}
        S += [f"### APOGEE {sp}: distance by G bin\n",
              md(side(d, ["G_lo", "G_hi"], ["n", "pull_median", "pull_rstd", "MG_resid_mad", "MG_resid_rmse", "frac_dist_pred_median"]), ".3f")]
    return S


# ---------------------------------------------------------------- APOGEE test: RC / anchors-style checks
def test_stars(tags):
    """Test-split APOGEE stars (iid + sky) with predictions of every run and the RC calibration."""
    parts, calib = [], None
    for sp in SPLITS:
        ds = APOGEEDataset(ROOT, sp, input="coef110n", labels=LAB, astro=True, require_xp_valid=True,
                           rc_anchor={"enabled": True, "snr_min": 5.0, "sig_mu_floor": 0.1})
        calib = ds.rc_calib
        t, ar = ds.astro_table.copy(), ds.astro_arr
        t["split"] = sp
        t["source_id"] = ds.catalog["source_id"].to_numpy()
        t["source"] = ds.catalog["source"].to_numpy() if "source" in ds.catalog else "apogee"
        for i, l in enumerate(LAB):
            t[l] = np.where(ds.mask[:, i], ds.y_raw[:, i], np.nan)
        t["g"], t["plx_corr"], t["plx_err"] = ar["g"], ar["plx"] - ar["zp"], ar["plx_err"]
        t["plx_ok"] = ar["plx_ok"]
        for tag, r in tags.items():
            z = np.load(RUNS / r / f"predictions_dist_{sp}.npz")
            assert np.array_equal(z["source_id"], t["source_id"].to_numpy()), (sp, r)
            t["a0|" + tag], t["mg|" + tag] = z["a0"], z["mg"]
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    t = t[t.source == "apogee"].reset_index(drop=True)
    t["a_sfd"] = A_SFD_FAC * t["ebv_sfd"]
    tg, sg, ok = rc_anchor_targets(t.g, t.plx_corr, t.plx_err.where(t.plx_ok), t.m_h, t.teff, t.logg, calib)
    t["rc_t"], t["rc_s"], t["rc_ok"] = tg, sg, ok
    return t, calib


def binned(x, var, bins=ABINS):
    return [(n, (var > lo) & (var <= hi)) for lo, hi, n in bins]


def rc_test_table(t, tags):
    rows = []
    for n, s in binned(None, t.a_sfd.to_numpy()):
        s = s & t.rc_ok.to_numpy()
        r = {"A_SFD bin": n, "n_test_RC": int(s.sum()), "median_target": med(t.rc_t[s]), "median_sigma": med(t.rc_s[s])}
        for tag in tags:
            d = t["a0|" + tag].to_numpy()[s] - t.rc_t.to_numpy()[s]
            r[f"A0-target med|{tag}"] = med(d)
            r[f"MAD|{tag}"] = mad(d)
            r[f"pull med|{tag}"] = med(d / t.rc_s.to_numpy()[s])
            r[f"pull MAD|{tag}"] = mad(d / t.rc_s.to_numpy()[s])
        rows.append(r)
    return pd.DataFrame(rows)


def anchors_like_table(t, tags):
    """High-latitude A0 vs SFD, nearby-star A0 (test split)."""
    tgt, sig, k = anchor_targets(t.plx_corr, t.plx_err, t.b, t.ebv_sfd)
    k = np.where(t.plx_ok, k, 0)
    rows = []
    for nm, s, ref in (("nearby (d<100pc, S/N>10): A0", k == 1, np.zeros(len(t))),
                       ("high-lat |b|>30, z>300pc: A0 - SFD", k == 2, tgt)):
        r = {"set": nm, "n": int(s.sum())}
        for tag in tags:
            d = t["a0|" + tag].to_numpy()[s] - ref[s]
            r[f"median|{tag}"], r[f"MAD|{tag}"] = med(d), mad(d)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- OOD
def ood_df(comp, tags):
    t = pq.read_table(OOD / f"ood_{comp}.parquet").drop(["x"]).to_pandas()
    sp = pd.read_parquet(OOD / "ood_split.parquet")
    keep = ~t.source_id.isin(sp.source_id[sp.role == "train"])
    if comp in ("galah", "cluster_gc", "lamost_vmp"):
        keep &= t.source_id.isin(sp.source_id[(sp.component == comp) & (sp.role == "heldout")])
    t = t[keep].reset_index(drop=True)
    for tag, r in tags.items():
        p = pd.read_parquet(RUNS / r / "ood_holdout" / f"pred_{comp}.parquet")
        assert np.array_equal(p.source_id.to_numpy(), t.source_id.to_numpy()), (comp, r)
        t["a0|" + tag], t["mg|" + tag], t["smg|" + tag] = p["a0"].to_numpy(), p["mg"].to_numpy(), p["s_mg"].to_numpy()
        t["dmu|" + tag] = t.G - t["mg|" + tag] - K_G * t["a0|" + tag]
    plx = t.parallax - t.zpt
    hi = (plx / t.parallax_error > 10) & (t.ruwe < 1.4)
    t["mu_plx"] = np.where(hi, 5 * np.log10(1000.0 / plx.where(plx > 0)) - 5, np.nan)
    t["a_sfd"] = A_SFD_FAC * t.ebv_sfd
    return t


def comp_table(t, tags, bvar, bins, cols):
    """cols: list of (name, fn(tag) -> array of residuals).  Median and MAD per bin per tag."""
    rows = []
    for n, s in binned(None, t[bvar].to_numpy(), bins):
        r = {"bin": n, "n": int(s.sum())}
        for cn, fn in cols:
            for tag in tags:
                d = fn(t, tag)[s]
                r[f"{cn} med|{tag}"] = med(d)
                r[f"{cn} MAD|{tag}"] = mad(d)
        rows.append(r)
    return pd.DataFrame(rows)


def ood_tables(tags):
    S = []
    sh = ood_df("starhorse", tags)
    cols_a = [("A0-A_V(SH)", lambda t, g: t["a0|" + g] - t.a0_ref),
              ("dmu vs SH", lambda t, g: t["dmu|" + g] - t.mu_true),
              ("dmu vs plx(S/N>10)", lambda t, g: t["dmu|" + g] - t.mu_plx)]
    S += ["### StarHorse (held-out rows), by A_SFD bin, all G\n", md(comp_table(sh, tags, "a_sfd", ABINS, cols_a))]
    S += ["### StarHorse, by G bin, all A_SFD\n", md(comp_table(sh, tags, "G", GBINS, cols_a))]
    faint = sh[sh.G >= 16].reset_index(drop=True)
    S += ["### StarHorse faint (G >= 16), by A_SFD bin\n", md(comp_table(faint, tags, "a_sfd", ABINS, cols_a))]
    oc = ood_df("cluster_oc", tags)
    cols_o = [("A0-A_V(HR23)", lambda t, g: t["a0|" + g] - t.a0_ref), ("dmu vs mu_true", lambda t, g: t["dmu|" + g] - t.mu_true)]
    S += ["### Open clusters (held-out rows = all OC), by A_SFD bin\n", md(comp_table(oc, tags, "a_sfd", ABINS, cols_o))]
    allb = [(-9, 1e9, "all")]
    S += ["### OC all\n", md(comp_table(oc, tags, "a_sfd", allb, cols_o))]
    gc = ood_df("cluster_gc", tags)
    S += ["### Globular clusters (held-out), dmu vs mu_true by G\n",
          md(comp_table(gc, tags, "G", [(0, 16, "G<16"), (16, 17, "16-17"), (17, 18, "G>17"), (0, 99, "all")],
                        [("dmu", lambda t, g: t["dmu|" + g] - t.mu_true)]))]
    gl = ood_df("galah", tags)
    S += ["### GALAH (held-out), dmu vs parallax (S/N>10)\n",
          md(comp_table(gl, tags, "G", [(0, 13, "G<13"), (13, 14, "13-14"), (14, 99, "G>14"), (0, 99, "all")],
                        [("dmu vs plx", lambda t, g: t["dmu|" + g] - t.mu_plx)]))]
    return S


def label_ood_tables(tags):
    S = []
    c = {t: pd.read_csv(RUNS / r / "ood_holdout" / "metrics_components.csv") for t, r in tags.items()}
    S += ["### OOD components: label bias / rmse (model - external label)\n",
          md(side(c, ["component", "label"], ["n", "bias", "rmse"]), ".4f")]
    b = {t: pd.read_csv(RUNS / r / "ood_holdout" / "metrics_binned.csv") for t, r in tags.items()}
    for comp, lab in (("cluster_oc", "m_h"), ("cluster_gc", "m_h"), ("galah", "m_h"), ("lamost_vmp", "m_h")):
        d = {t: x[(x.component == comp) & (x.label == lab) & (x.bin_var == "G")] for t, x in b.items()}
        if all(len(v) for v in d.values()):
            S += [f"### {comp} [M/H] by G: median error and MAD\n", md(side(d, ["bin"], ["n", "median", "mad"]), ".4f")]
    return S


# ---------------------------------------------------------------- fixed-coefficient diagnostic
def coef_diag(t, tags):
    """RC stars (S/N>10): effective G and (BP-RP) reddening per unit RJCE A0, vs A0; and A0_pred - target vs A0_pred / Teff."""
    S = []
    s = t.rc_ok & (t.plx_corr / t.plx_err > 10) & (t.ruwe < 1.4)
    tt = t[s].copy()
    # intrinsic (BP-RP)_0 of the RC from low-extinction stars: linear in [M/H], dT
    from aspgapv3.rc_anchor import rc_dT
    tt["bprp"] = tt.phot_bp_mean_mag - tt.phot_rp_mean_mag
    X = np.stack([np.ones(len(tt)), tt.m_h, rc_dT(tt.teff, tt.m_h) / 100.0], 1)
    lo = (tt.a_sfd < 0.1).to_numpy()
    y = (tt.bprp - 0.413 * tt.a_sfd).to_numpy()
    cf = np.linalg.lstsq(X[lo], y[lo], rcond=None)[0]
    tt["E_bprp"] = tt.bprp - X @ cf
    ok = (tt.tmass_ok_h & tt.wise_ok_w2).to_numpy(bool)
    hw = (tt.h_m - tt.w2mpro).to_numpy()
    tt["a0_rjce"] = 0.918 * (hw - 0.08) / 0.078
    tt["AG_obs"] = tt.g - rc_mg_t(tt) - 5 * np.log10(1000.0 / tt.plx_corr) + 5     # A_G implied by the RC calibration
    rows = []
    edges = [(0.0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 99)]
    for lo_, hi_ in edges:
        m = ok & (tt.a0_rjce > lo_).to_numpy() & (tt.a0_rjce <= hi_).to_numpy()
        r = {"A0_RJCE bin": f"{lo_}-{hi_}", "n": int(m.sum()), "median A0_RJCE": med(tt.a0_rjce[m]),
             "k_G,eff = A_G(RC)/A0_RJCE": med(tt.AG_obs[m] / tt.a0_rjce[m]),
             "k_(BP-RP),eff = E(BP-RP)/A0_RJCE": med(tt.E_bprp[m] / tt.a0_rjce[m]),
             "E(BP-RP)/A_G (model 0.523)": med(tt.E_bprp[m]) / med(tt.AG_obs[m]) if m.any() else np.nan}
        rows.append(r)
    S += ["RC stars, S/N>10, test+iid/sky split APOGEE; A_G(RC) = G - M_G^RC - mu_plx (so k_G,eff is relative to the RC calibration); "
          "model fixed coefficients: k_G = 0.789, k_BP-RP = 0.413 (ratio 0.523, independent of the RJCE scale; the RJCE-based columns depend on the (H-W2)_0 = 0.08 and A_K/A0 = 0.078 conventions, read them only as trends).\n", md(pd.DataFrame(rows))]
    rows = []
    for tag in tags:
        for what, var, edges2 in (("A0_pred", tt["a0|" + tag], [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 4), (4, 99)]),
                                   ("Teff", tt.teff, [(4300, 4650), (4650, 4750), (4750, 4850), (4850, 5100)])):
            for lo_, hi_ in edges2:
                m = ((var > lo_) & (var <= hi_)).to_numpy()
                d = (tt["a0|" + tag] - tt.rc_t).to_numpy()[m]
                rows.append({"run": tag, "binned by": what, "bin": f"{lo_}-{hi_}", "n": int(m.sum()),
                             "median(A0_pred - target)": med(d), "MAD": mad(d)})
    S += ["Predicted A0 minus RC target for test RC (S/N>10), by predicted A0 and by Teff.\n", md(pd.DataFrame(rows))]
    return S


def rc_mg_t(tt):
    from aspgapv3.rc_anchor import rc_mg
    return rc_mg(CAL["coef"], tt.m_h.to_numpy(), tt.teff.to_numpy())


CAL = None


def main():
    global CAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="aspgap_dist_ext_r2")
    ap.add_argument("--new", default="aspgap_dist_ext2_rc")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-ood", action="store_true")
    a = ap.parse_args()
    tags = {"ext_r2": a.base, "ext2_rc": a.new} if a.base != a.new else {"A": a.base, "B": a.new + ""}
    if a.base == a.new:
        tags = {"ext_r2": a.base}
    out = Path(a.out) if a.out else RUNS / a.new / "rc_summary.md"
    S = [f"# RC anchors: {' | '.join(tags.values())}\n", "Columns `x|tag`: tag = " + ", ".join(tags) + ".  Errors = model - reference, identical rows.\n"]
    chk = RUNS / a.new / "rc_anchor_check.md"
    if chk.exists():
        S += ["## RC selection, calibration and anchor sanity (rc_anchor_check.py, before training)\n", chk.read_text()]
    S += ["## APOGEE tests\n"] + apogee_tables(tags)
    t, CAL = test_stars(tags)
    S += ["## Independent A0 check: APOGEE test-split RC stars (never anchors), predicted A0 - RC-target A0\n",
          md(rc_test_table(t, tags), ".3f"),
          "## High-latitude / nearby anchors (test split)\n", md(anchors_like_table(t, tags), ".3f")]
    if not a.no_ood:
        S += ["## OOD distance / A0 (held-out rows)\n"] + ood_tables(tags) + ["## OOD labels (nothing should change)\n"] + label_ood_tables(tags)
    S += ["## Fixed-extinction-coefficient diagnostic\n"] + coef_diag(t, tags)
    out.write_text("\n".join(S))
    print("wrote", out)


if __name__ == "__main__":
    main()
