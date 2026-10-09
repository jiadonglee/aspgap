#!/usr/bin/env python
"""runs/<ext>/ext_summary.md: r3 (APOGEE-only) vs ext_r1 (APOGEE + external labels).

    python scripts/ext_summary.py --base aspgap_dist_r3 --ext aspgap_dist_ext_r1

Needs: evaluate.py outputs of both runs (APOGEE iid/sky tests, distance), `eval_ood.py --holdout` outputs of both
(<run>/ood_holdout), the extended dataset (training-set composition, [M/H] histogram).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS = Path("/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3/runs")
ROOT = REPO / "data/processed/aspcap_v0.6.0"
EXT = REPO / "data/processed/aspcap_v0.6.0_ext"
OOD = REPO / "data/processed/ood"
LABELS = ["teff", "logg", "m_h", "alpha_m"]


def md(df: pd.DataFrame, fmt: str = ".4f") -> str:
    if df is None or len(df) == 0:
        return "(empty)\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join((f"{v:{fmt}}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                                      ("" if isinstance(v, (float, np.floating)) else str(v))) for v in r) + " |")
    return "\n".join(out) + "\n"


def two(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], vals: list[str], na="r3", nb="ext_r1") -> pd.DataFrame:
    m = a.merge(b, on=keys, suffixes=(f"_{na}", f"_{nb}"))
    cols = list(keys)
    for v in vals:
        m[f"d_{v}"] = m[f"{v}_{nb}"] - m[f"{v}_{na}"]
        cols += [f"{v}_{na}", f"{v}_{nb}", f"d_{v}"]
    return m[cols]


def read(run: str, f: str) -> pd.DataFrame:
    p = RUNS / run / f
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def composition() -> str:
    cat = pd.read_parquet(EXT / "catalog.parquet", columns=["row", "split", "source", "m_h"])
    z = np.load(EXT / "labels.npz")
    valid = np.load(EXT / "xpcov_valid.npy")
    names = [str(s) for s in z["labels"]]
    tr = (cat.split == "train").to_numpy() & valid                # the rows the training set actually contains (require_xp_valid)
    rows = []
    for src in ("apogee", "galah", "gc", "lamost_vmp"):
        sel = tr & (cat.source == src).to_numpy()
        r = dict(source=src, n_train_rows=int(sel.sum()))
        for l in LABELS:
            r[l] = int(z["mask"][sel, names.index(l)].sum())
        r["all4"] = int(z["mask"][sel][:, [names.index(l) for l in LABELS]].all(1).sum())
        r["m_h_only"] = int((z["mask"][sel, names.index("m_h")] & ~z["mask"][sel][:, [names.index(l) for l in ("teff", "logg", "alpha_m")]].any(1)).sum())
        rows.append(r)
    t = pd.DataFrame(rows)
    tot = t.sum(numeric_only=True)
    tot["source"] = "total"
    t = pd.concat([t, tot.to_frame().T], ignore_index=True)
    mh = z["y"][:, names.index("m_h")]
    ok = z["mask"][:, names.index("m_h")]
    h = []
    for nm, sel in (("APOGEE only (r3 training set)", tr & (cat.source == "apogee").to_numpy()), ("APOGEE + external (ext_r1)", tr)):
        r = {"training set": nm, "n with [M/H]": int((sel & ok).sum())}
        for c in (-1.0, -1.5, -2.0, -2.5):
            r[f"[M/H] < {c}"] = int((sel & ok & (mh < c)).sum())
        h.append(r)
    h2 = []
    for src in ("apogee", "galah", "gc", "lamost_vmp"):
        sel = tr & (cat.source == src).to_numpy() & ok
        r = {"source": src}
        for c in (-1.0, -1.5, -2.0, -2.5):
            r[f"< {c}"] = int((sel & (mh < c)).sum())
        h2.append(r)
    return ("### Training rows by source and label availability (after the xp-valid cut; counts are rows with the label unmasked)\n\n"
            + md(t, ".0f") + "\n### Training-set [M/H] histogram before / after (cumulative counts)\n\n" + md(pd.DataFrame(h), ".0f")
            + "\nper source:\n\n" + md(pd.DataFrame(h2), ".0f"))


def apogee_tables(base: str, ext: str) -> str:
    out = []
    for s in ("iid_test", "sky_test"):
        a, b = read(base, f"metrics_{s}.csv"), read(ext, f"metrics_{s}.csv")
        out += [f"### APOGEE {s}: atmospheric labels (raw units)\n",
                md(two(a, b, ["label"], ["n", "rmse", "bias", "scatter_mad", "r2", "sigma_med", "calibration_ratio"]), ".4f")]
        da, db = read(base, f"metrics_distance_{s}.csv"), read(ext, f"metrics_distance_{s}.csv")
        out += [f"### APOGEE {s}: distance (parallax pull, M_G residual vs parallax S/N>10, fractional distance precision)\n",
                md(two(da, db, ["G_lo", "G_hi"], ["pull_median", "pull_rstd", "MG_resid_rmse", "MG_resid_mad", "MG_resid_median",
                                                  "frac_dist_pred_median"]), ".4f")]
        ga, gb = read(base, f"metrics_degraded_{s}.csv"), read(ext, f"metrics_degraded_{s}.csv")
        if len(ga) and len(gb):
            ga["g_prime"], gb["g_prime"] = ga.g_prime.fillna(-1), gb.g_prime.fillna(-1)
            out += [f"### APOGEE {s}: labels after noise degradation to G' (rmse)\n",
                    md(two(ga, gb, ["g_prime", "label"], ["rmse", "bias"]), ".4f")]
    return "\n".join(out)


def ood_tables(base: str, ext: str) -> str:
    sub = "ood_holdout"
    out = []
    ca, cb = read(base, f"{sub}/metrics_components.csv"), read(ext, f"{sub}/metrics_components.csv")
    cnt = json.loads((RUNS / ext / sub / "counts.json").read_text())
    out += ["### Held-out OOD test sets (identical rows for both models)\n", md(pd.DataFrame([dict(component=k, **v) for k, v in cnt.items()]), ".0f")]
    for v in ("bias", "rmse", "mad"):
        out += [f"### Per component {v} (model - external label)\n", md(two(ca, cb, ["component", "label"], [v]), ".4f")]
    out += ["### Predicted-sigma median and calibration (MAD of err/sigma)\n", md(two(ca, cb, ["component", "label"], ["sigma_med", "calib"]), ".3f")]
    ba, bb = read(base, f"{sub}/metrics_binned.csv"), read(ext, f"{sub}/metrics_binned.csv")
    if len(ba) and len(bb):
        for comp, bv, lab in (("galah", "teff_true", "teff"), ("galah", "teff_true", "logg"), ("galah", "mh_true", "m_h"),
                              ("lamost_vmp", "mh_true", "m_h"), ("cluster_gc", "G", "m_h"), ("lamost_hot", "teff_true", "teff")):
            sa, sb = ba[(ba.component == comp) & (ba.bin_var == bv) & (ba.label == lab)], bb[(bb.component == comp) & (bb.bin_var == bv) & (bb.label == lab)]
            out += [f"### {comp}: {lab} bias/RMSE by {bv}\n", md(two(sa, sb, ["bin"], ["n", "bias", "rmse"]), ".3f")]
    # ---- VMP [M/H] bias vs [Fe/H] bins
    v = pd.read_parquet(OOD / "ood_lamost_vmp.parquet", columns=["source_id", "fe_h", "m_h", "G"])
    rows = []
    pa = pd.read_parquet(RUNS / base / sub / "pred_lamost_vmp.parquet")
    pb = pd.read_parquet(RUNS / ext / sub / "pred_lamost_vmp.parquet")
    vv = v.merge(pa[["source_id", "mu_m_h"]], on="source_id").merge(pb[["source_id", "mu_m_h"]], on="source_id", suffixes=("_r3", "_ext_r1"))
    vv["bin"] = pd.cut(vv.fe_h, [-9, -3.0, -2.75, -2.5, -2.25, -2.0])
    for b, g in vv.groupby("bin", observed=True):
        rows.append(dict(**{"[Fe/H]_Yuan bin": str(b)}, n=len(g), truth_mh_mean=g.m_h.mean(),
                         bias_r3=(g.mu_m_h_r3 - g.m_h).mean(), bias_ext_r1=(g.mu_m_h_ext_r1 - g.m_h).mean(),
                         rmse_r3=np.sqrt(((g.mu_m_h_r3 - g.m_h) ** 2).mean()), rmse_ext_r1=np.sqrt(((g.mu_m_h_ext_r1 - g.m_h) ** 2).mean())))
    g = vv
    rows.append(dict(**{"[Fe/H]_Yuan bin": "all"}, n=len(g), truth_mh_mean=g.m_h.mean(), bias_r3=(g.mu_m_h_r3 - g.m_h).mean(),
                     bias_ext_r1=(g.mu_m_h_ext_r1 - g.m_h).mean(), rmse_r3=np.sqrt(((g.mu_m_h_r3 - g.m_h) ** 2).mean()),
                     rmse_ext_r1=np.sqrt(((g.mu_m_h_ext_r1 - g.m_h) ** 2).mean())))
    out += ["### Held-out LAMOST VMP: [M/H] bias vs [Fe/H] (model - truth)\n", md(pd.DataFrame(rows), ".3f")]
    # ---- GC per-cluster
    ka, kb = read(base, f"{sub}/metrics_clusters.csv"), read(ext, f"{sub}/metrics_clusters.csv")
    ga, gb = ka[ka.component == "cluster_gc"], kb[kb.component == "cluster_gc"]
    sp = pd.read_parquet(OOD / "ood_cluster_gc.parquet", columns=["cluster", "feh_harris"]).drop_duplicates("cluster")
    m = two(ga, gb, ["cluster", "n", "truth_m_h"], ["dMH_median", "MH_spread", "sig_mh_med", "dmu_median"]).merge(sp, on="cluster").sort_values("feh_harris")
    out += ["### Held-out globular clusters: per-cluster [M/H] offset (median model - truth) vs Harris [Fe/H]; dmu = median distance-modulus error (catalogue BV21)\n",
            md(m[["cluster", "feh_harris", "n", "truth_m_h", "d_dMH_median", "dMH_median_r3", "dMH_median_ext_r1", "MH_spread_r3", "MH_spread_ext_r1",
                  "dmu_median_r3", "dmu_median_ext_r1"]], ".3f")]
    m["feh_bin"] = pd.cut(m.feh_harris, [-9, -2.0, -1.5, -1.0, 0.5])
    agg = m.groupby("feh_bin", observed=True).agg(n_clusters=("cluster", "size"), dMH_r3=("dMH_median_r3", "median"), dMH_ext_r1=("dMH_median_ext_r1", "median"),
                                                  rms_r3=("dMH_median_r3", lambda x: float(np.sqrt(np.mean(x ** 2)))),
                                                  rms_ext_r1=("dMH_median_ext_r1", lambda x: float(np.sqrt(np.mean(x ** 2))))).reset_index()
    agg["feh_bin"] = agg.feh_bin.astype(str)
    out += ["### Held-out GCs: median per-cluster [M/H] offset by Harris [Fe/H] bin\n", md(agg, ".3f")]
    # ---- distance
    da, db = read(base, f"{sub}/metrics_distance.csv"), read(ext, f"{sub}/metrics_distance.csv")
    sel = lambda d: d[(d.component == "cluster_gc") & (d.reference == "catalogue distance") & (d.bin_var == "G")]
    out += ["### Held-out GCs: distance-modulus error vs BV21 catalogue distance, by G\n",
            md(two(sel(da), sel(db), ["bin"], ["n", "dmu_median", "dmu_mad", "pull_mad", "s_mg_med"]), ".3f")]
    sel = lambda d: d[(d.reference == "parallax(S/N>10)") & (d.bin_var == "all")]
    out += ["### Distance modulus vs parallax S/N>10, all G (median dmu, MAD, pull MAD)\n",
            md(two(sel(da), sel(db), ["component"], ["n", "dmu_median", "dmu_mad", "pull_mad"]), ".3f")]
    sel = lambda d: d[(d.component == "starhorse") & (d.reference.str.startswith("A0")) & (d.bin_var == "all")]
    # ---- flags
    aa, ab = read(base, f"{sub}/metrics_auc.csv"), read(ext, f"{sub}/metrics_auc.csv")
    sa = aa[aa.flag == "pred_sigma"]
    sb = ab[ab.flag == "pred_sigma"]
    out += ["### Predicted-sigma flag AUC for |err| > 3 x ID MAD\n", md(two(sa, sb, ["component", "label"], ["auc"]), ".3f")]
    fa, fb = aa[aa.flag != "pred_sigma"], ab[ab.flag != "pred_sigma"]
    out += ["### Other flags (input Mahalanobis, z-kNN, oracle label-kNN)\n", md(two(fa, fb, ["component", "label", "flag"], ["auc"]), ".3f")]
    ra, rb = read(base, f"{sub}/metrics_auc_vs_apogee_test.csv"), read(ext, f"{sub}/metrics_auc_vs_apogee_test.csv")
    out += ["### Flag AUC: component vs APOGEE iid_test stars\n", md(two(ra, rb, ["component", "flag"], ["auc_vs_apogee_test"]), ".3f")]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="aspgap_dist_r3")
    ap.add_argument("--ext", default="aspgap_dist_ext_r1")
    args = ap.parse_args()
    man = json.loads((EXT / "ext_manifest.json").read_text())
    L = [f"# aspgap_dist_ext_r1 vs aspgap_dist_r3\n",
         "d_* = ext_r1 - r3.  OOD errors are model - external label, on the held-out rows only (stars neither model trained on; "
         "`eval_ood.py --holdout`).\n",
         "## Hold-out design\n",
         f"* GALAH-only and LAMOST VMP: HEALPix nside={man['healpix_nside']} pixels hashed (seed {man['seed']}) -> {int(man['p_train']*100)}% train / rest held out.\n"
         f"* Globular clusters, held-out ({len(man['gc']['held_out_clusters'])} of {man['gc']['n_clusters']}, stratified in Harris [Fe/H]): "
         + ", ".join(man["gc"]["held_out_clusters"]) + ".\n"
         f"* Open clusters, LAMOST hot, StarHorse: never trained on.  APOGEE train/val/iid/sky splits unchanged; model selection on APOGEE val only.\n",
         "## Training-set composition\n", composition(), "\n## APOGEE tests\n", apogee_tables(args.base, args.ext), "\n## OOD held-out sets\n", ood_tables(args.base, args.ext)]
    (RUNS / args.ext / "ext_summary.md").write_text("\n".join(L))
    print("wrote", RUNS / args.ext / "ext_summary.md")


if __name__ == "__main__":
    main()
