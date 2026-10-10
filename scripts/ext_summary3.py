#!/usr/bin/env python
"""runs/<ext2>/ext_summary.md: three-column comparison r3 (APOGEE only) | ext_r1 | ext_r2 (ext with stricter GC members).

    python scripts/ext_summary3.py --runs aspgap_dist_r3 aspgap_dist_ext_r1 aspgap_dist_ext_r2 --out aspgap_dist_ext_r2

Same inputs as ext_summary.py (evaluate.py outputs, `eval_ood.py --holdout` outputs in <run>/ood_holdout).  Adds the
open-cluster [M/H] median bias by G and by reference A0 (the acceptance test), the GC cut diagnostics of the ext2 build
and the LAMOST hot [M/H] bias.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ext_summary import OOD, RUNS, md, read

EXT2 = Path(__file__).resolve().parent.parent / "data/processed/aspcap_v0.6.0_ext2"
TAGS = ["r3", "ext_r1", "ext_r2"]


def three(dfs: list[pd.DataFrame], keys: list[str], vals: list[str]) -> pd.DataFrame:
    """Outer-join the three runs' tables on keys; columns val_<tag>."""
    out = None
    for d, t in zip(dfs, TAGS):
        d = d[keys + vals].rename(columns={v: f"{v}_{t}" for v in vals})
        out = d if out is None else out.merge(d, on=keys, how="outer")
    cols = list(keys)
    for v in vals:
        cols += [f"{v}_{t}" for t in TAGS]
    return out[cols]


def gc_cut_section(man: dict) -> str:
    g = man["gc"]
    c = g["cuts"]
    o = [f"GC thresholds: {json.dumps(c['thresholds'])}\n",
         "### GC training stars after each cut (sequential; held-out clusters never used)\n",
         md(pd.DataFrame(c["steps"]).rename(columns={"cut": "step"}), ".0f"),
         "### Stars removed by the new cuts vs kept (among the ext_r1-quality stars of the training clusters, before the cap)\n",
         md(pd.DataFrame(c["groups"]), ".3f"),
         f"Clusters dropped by E(B-V) > 0.5 ({len(c['clusters_dropped_ebv'])}): " + ", ".join(c["clusters_dropped_ebv"]) + "\n",
         f"Median intrinsic PM dispersion estimated per cluster: {c['pm_sint_median']:.3f} mas/yr.\n",
         "### Contamination estimate: the ext_r1 GC training stars (n=%d) against the new tests\n" % c["r1"]["n"],
         md(pd.DataFrame([c["r1"]]), ".4f"), md(pd.DataFrame(c["r1_by_ebv_bin"]), ".4f"),
         f"ext_r2 GC training stars: {man['n_by_source'].get('gc')} rows, {g['cuts']['steps'][-1]['n_clusters']} clusters "
         f"(ext_r1: 9994 rows, 81 clusters).\n"]
    return "\n".join(o)


def apogee(runs):
    out = []
    for s in ("iid_test", "sky_test"):
        d = [read(r, f"metrics_{s}.csv") for r in runs]
        out += [f"### APOGEE {s}: label rmse / bias / scatter_mad\n", md(three(d, ["label"], ["n", "rmse", "bias", "scatter_mad"]), ".4f")]
        d = [read(r, f"metrics_distance_{s}.csv") for r in runs]
        out += [f"### APOGEE {s}: distance (parallax pull, M_G residual, fractional distance precision)\n",
                md(three(d, ["G_lo", "G_hi"], ["pull_median", "pull_rstd", "MG_resid_rmse", "MG_resid_mad", "frac_dist_pred_median"]), ".4f")]
    return "\n".join(out)


def oc_tables(runs, sub):
    oc = pd.read_parquet(OOD / "ood_cluster_oc.parquet", columns=["source_id", "G", "a0_ref", "m_h", "cluster"])
    for r, t in zip(runs, TAGS):
        p = pd.read_parquet(RUNS / r / sub / "pred_cluster_oc.parquet", columns=["source_id", "mu_m_h"])
        oc = oc.merge(p.rename(columns={"mu_m_h": f"mu_{t}"}), on="source_id")
    for t in TAGS:
        oc[f"err_{t}"] = oc[f"mu_{t}"] - oc.m_h

    def tab(col, edges, name):
        oc["bin"] = pd.cut(oc[col], edges)
        rows = []
        for b, g in list(oc.groupby("bin", observed=True)) + [("all", oc)]:
            r = {name: str(b), "n": len(g), "n_clusters": g.cluster.nunique()}
            for t in TAGS:
                r[f"med_bias_{t}"] = g[f"err_{t}"].median()
            for t in TAGS:
                r[f"mad_{t}"] = 1.4826 * np.median(np.abs(g[f"err_{t}"] - g[f"err_{t}"].median()))
            rows.append(r)
        return md(pd.DataFrame(rows), ".3f")
    o = ["### Open clusters: [M/H] median bias (model - cluster [M/H]) by G  [ACCEPTANCE TEST]\n", tab("G", [0, 13, 15, 16, 17, 30], "G bin"),
         "### Open clusters: [M/H] median bias by reference A0 (a0_ref)  [ACCEPTANCE TEST]\n", tab("a0_ref", [-1, 0.3, 1, 2, 30], "a0_ref bin")]
    return "\n".join(o)


def ood_tables(runs, sub):
    cm = [read(r, f"{sub}/metrics_components.csv") for r in runs]
    cnt = json.loads((RUNS / runs[-1] / sub / "counts.json").read_text())
    cnt1 = json.loads((RUNS / runs[1] / sub / "counts.json").read_text())
    out = ["### Held-out OOD rows (same for all runs; ext_r1 counts identical: %s)\n" % (cnt == cnt1),
           md(pd.DataFrame([dict(component=k, **v) for k, v in cnt.items()]), ".0f")]
    for v in ("bias", "rmse"):
        out += [f"### Per component {v} (model - external label); lamost_hot m_h is the LAMOST hot [M/H] bias\n", md(three(cm, ["component", "label"], [v]), ".4f")]
    out += ["### Predicted-sigma median / calibration\n", md(three(cm, ["component", "label"], ["sigma_med", "calib"]), ".3f")]
    # hot [M/H] by teff
    bs = [read(r, f"{sub}/metrics_binned.csv") for r in runs]
    def sel(d, comp, bv, lab):
        return d[(d.component == comp) & (d.bin_var == bv) & (d.label == lab)]
    for comp, bv, lab in (("lamost_hot", "teff_true", "m_h"), ("galah", "mh_true", "m_h"), ("galah", "teff_true", "teff"), ("cluster_gc", "G", "m_h")):
        out += [f"### {comp}: {lab} bias/RMSE by {bv}\n", md(three([sel(d, comp, bv, lab) for d in bs], ["bin"], ["n", "bias", "rmse"]), ".3f")]
    # VMP
    v = pd.read_parquet(OOD / "ood_lamost_vmp.parquet", columns=["source_id", "fe_h", "m_h"])
    for r, t in zip(runs, TAGS):
        p = pd.read_parquet(RUNS / r / sub / "pred_lamost_vmp.parquet", columns=["source_id", "mu_m_h"])
        v = v.merge(p.rename(columns={"mu_m_h": f"mu_{t}"}), on="source_id")
    v["bin"] = pd.cut(v.fe_h, [-9, -3.0, -2.75, -2.5, -2.25, -2.0])
    rows = []
    for b, g in list(v.groupby("bin", observed=True)) + [("all", v)]:
        r = {"[Fe/H]_Yuan bin": str(b), "n": len(g), "truth_mh_mean": g.m_h.mean()}
        for t in TAGS:
            r[f"bias_{t}"] = (g[f"mu_{t}"] - g.m_h).mean()
        for t in TAGS:
            r[f"rmse_{t}"] = np.sqrt(((g[f"mu_{t}"] - g.m_h) ** 2).mean())
        rows.append(r)
    out += ["### Held-out LAMOST VMP: [M/H] bias vs [Fe/H] (model - truth)\n", md(pd.DataFrame(rows), ".3f")]
    # GC per cluster
    ks = [read(r, f"{sub}/metrics_clusters.csv") for r in runs]
    ks = [k[k.component == "cluster_gc"] for k in ks]
    sp = pd.read_parquet(OOD / "ood_cluster_gc.parquet", columns=["cluster", "feh_harris"]).drop_duplicates("cluster")
    m = three(ks, ["cluster", "n", "truth_m_h"], ["dMH_median", "MH_spread"]).merge(sp, on="cluster").sort_values("feh_harris")
    m["feh_bin"] = pd.cut(m.feh_harris, [-9, -2.0, -1.5, -1.0, 0.5])
    rows = []
    for b, g in list(m.groupby("feh_bin", observed=True)) + [("all", m)]:
        r = {"Harris [Fe/H] bin": str(b), "n_clusters": len(g)}
        for t in TAGS:
            r[f"median_offset_{t}"] = g[f"dMH_median_{t}"].median()
        for t in TAGS:
            r[f"rms_offset_{t}"] = float(np.sqrt(np.mean(g[f"dMH_median_{t}"] ** 2)))
        for t in TAGS:
            r[f"spread_{t}"] = g[f"MH_spread_{t}"].median()
        rows.append(r)
    out += ["### Held-out GCs: per-cluster [M/H] offset (median model - truth) by Harris [Fe/H] bin\n", md(pd.DataFrame(rows), ".3f"),
            "### Held-out GCs: per cluster\n", md(m[["cluster", "feh_harris", "n", "truth_m_h"] + [f"dMH_median_{t}" for t in TAGS]], ".3f")]
    ds = [read(r, f"{sub}/metrics_distance.csv") for r in runs]
    f = lambda d: d[(d.component == "cluster_gc") & (d.reference == "catalogue distance") & (d.bin_var == "G")]
    out += ["### Held-out GCs: distance-modulus error vs BV21 distance by G (dmu = mu_pred - mu_BV21)\n",
            md(three([f(d) for d in ds], ["bin"], ["n", "dmu_median", "dmu_mad", "pull_mad"]), ".3f")]
    f = lambda d: d[(d.reference == "parallax(S/N>10)") & (d.bin_var == "all")]
    out += ["### Distance modulus vs parallax S/N>10, all G\n", md(three([f(d) for d in ds], ["component"], ["n", "dmu_median", "dmu_mad", "pull_mad"]), ".3f")]
    au = [read(r, f"{sub}/metrics_auc.csv") for r in runs]
    au = [a[a.flag == "pred_sigma"] for a in au]
    out += ["### Predicted-sigma flag AUC for |err| > 3 x ID MAD\n", md(three(au, ["component", "label"], ["auc"]), ".3f")]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=3, default=["aspgap_dist_r3", "aspgap_dist_ext_r1", "aspgap_dist_ext_r2"])
    ap.add_argument("--out", default="aspgap_dist_ext_r2")
    a = ap.parse_args()
    man = json.loads((EXT2 / "ext_manifest.json").read_text())
    sub = "ood_holdout"
    L = ["# aspgap_dist_ext_r2 vs ext_r1 vs r3\n",
         "Columns: r3 = APOGEE only; ext_r1 = + GALAH/GC(cap 300, membership prob only)/VMP; ext_r2 = same but GC members cut on E(B-V), per-star "
         "PM/parallax consistency, cap 100.  OOD errors are model - external label on the identical held-out rows "
         "(`eval_ood.py --holdout`); held-out GC list, GALAH and VMP splits identical in ext_r1 and ext_r2.\n",
         "## GC training-member selection (ext_r2)\n", gc_cut_section(man),
         "\n## APOGEE tests\n", apogee(a.runs), "\n## Open clusters (acceptance test)\n", oc_tables(a.runs, sub),
         "\n## OOD held-out sets\n", ood_tables(a.runs, sub)]
    (RUNS / a.out / "ext_summary.md").write_text("\n".join(L))
    print("wrote", RUNS / a.out / "ext_summary.md")


if __name__ == "__main__":
    main()
