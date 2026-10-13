#!/usr/bin/env python
"""Evaluate trained runs on the OOD test sets built by scripts/build_ood.py.

    python scripts/eval_ood.py --run aspgap_dist_ext2_cn [--holdout] [--summary-only]

Per run  -> <runs>/<run>/ood/{metrics_components,metrics_binned,metrics_clusters,metrics_distance,metrics_auc,
                              metrics_id}.csv, pred_<component>.parquet, report.md
Summary  -> runs/ood_summary/report.md  (all runs given, side by side)
--holdout: only the OOD stars NOT used for training (ood_split.parquet from build_ext_dataset.py) -> <run>/ood_holdout,
           runs/ood_summary_holdout.  --out-dir redirects all outputs (scratch checks); run names still resolve under runs/.

Errors are model - external label.  Label-scale offsets between the external survey and APOGEE are only partly
removed (GALAH, GC [M/H]); LAMOST labels are NOT put on the APOGEE scale.  `survey offset` and `model error`
cannot be separated for the hot / VMP components -> read those as "where does the model disagree", not as accuracy.

OOD score = mean distance (train-label-std units) of the TRUE (Teff, logg, [M/H]) to its 8 nearest APOGEE-train
stars (build_ood.py `ood_knn`); bins are quantiles of the same score for APOGEE iid_test stars.
Flags (AUC for |err| > 3 x in-distribution MAD): predicted sigma, input Mahalanobis distance to the train
coef110n distribution, kNN distance of the trunk embedding z to train stars.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
RUNS = Path("/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3/runs")
ROOT = REPO / "data/processed/aspcap_v0.6.0"
OODDIR = REPO / "data/processed/ood"
COMPS = ["galah", "cluster_oc", "cluster_gc", "lamost_hot", "lamost_vmp", "starhorse"]
LABELS = ["teff", "logg", "m_h", "alpha_m"]
SKIP = {("cluster_gc", "alpha_m")}                  # nominal GC alpha: only its spread is meaningful
EXT_G = 0.789
HOLDOUT = False                                      # --holdout: only the OOD stars NOT used for training the ext model
OUTSUB, SUMDIR = "ood", "ood_summary"
OUT_ROOT = RUNS                                       # where outputs are written (--out-dir)
HELD_COMPS = ("galah", "cluster_gc", "lamost_vmp")   # components split into train / held-out parts (build_ext_dataset.py)


def md(df: pd.DataFrame, fmt: str = ".3f") -> str:
    if df is None or len(df) == 0:
        return "(empty)\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join((f"{v:{fmt}}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                                      ("" if isinstance(v, (float, np.floating)) else str(v))) for v in r) + " |")
    return "\n".join(out) + "\n"


def mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def rstats(r) -> dict:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return dict(n=0, bias=np.nan, median=np.nan, rmse=np.nan, mad=np.nan)
    return dict(n=int(r.size), bias=float(r.mean()), median=float(np.median(r)), rmse=float(np.sqrt(np.mean(r ** 2))), mad=mad(r))


def auc(score, pos) -> float:
    ok = np.isfinite(score)
    s, p = score[ok], pos[ok]
    n1, n0 = int(p.sum()), int((~p).sum())
    if n1 < 10 or n0 < 10:
        return np.nan
    r = rankdata(s)
    return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ predictors
class NetPredictor:
    def __init__(self, run: Path, device: str):
        from aspgapv3.data.dataset import LabelScaler
        from aspgapv3.models.net import build_model
        ck = torch.load(run / "ckpt_best.pt", map_location=device, weights_only=False)
        cfg = ck["config"]
        assert cfg["data"].get("input", "coef110n") == "coef110n", "OOD sets provide the coef110n input only"
        self.labels, n_atm = ck["labels"], ck["n_atm"]
        self.net = build_model(cfg, n_atm=n_atm, n_chem=len(self.labels) - n_atm).to(device)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.root = REPO / cfg["data"]["root"] if (REPO / cfg["data"]["root"]).is_dir() else ROOT
        sc = LabelScaler.load(self.root)
        sel = [sc.labels.index(n) for n in self.labels]
        self.mean, self.std = sc.mean[sel], sc.std[sel]
        self.device, self.has_dist = device, self.net.has_distance
        self.name = run.name

    @torch.no_grad()
    def __call__(self, x: np.ndarray, batch: int = 8192) -> dict:
        mus, sgs, zs, dd = [], [], [], {"mg": [], "logvar_mg": [], "a0": []}
        for i in range(0, len(x), batch):
            out = self.net(torch.from_numpy(x[i:i + batch]).to(self.device))
            mus.append(torch.cat([out["mu_atm"], out["mu_chem"]], 1).float().cpu().numpy())
            lv = torch.cat([out["logvar_atm"], out["logvar_chem"]], 1).float().clamp(-8, 8)
            sgs.append(lv.exp().sqrt().cpu().numpy())
            zs.append(out["z"].float().cpu().numpy())
            if self.has_dist:
                for k in dd:
                    dd[k].append(out[k].float().cpu().numpy().reshape(-1))
        mu = np.concatenate(mus) * self.std + self.mean
        sg = np.concatenate(sgs) * self.std
        res = {"mu": self._to4(mu), "sigma": self._to4(sg), "z": np.concatenate(zs)}
        if self.has_dist:
            res.update(mg=np.concatenate(dd["mg"]), s_mg=np.exp(0.5 * np.clip(np.concatenate(dd["logvar_mg"]), -8, 8)),
                       a0=np.concatenate(dd["a0"]))
        return res

    def _to4(self, a):
        o = np.full((len(a), 4), np.nan)
        for j, n in enumerate(LABELS):
            if n in self.labels:
                o[:, j] = a[:, self.labels.index(n)]
        return o


def read_comp(name: str):
    t = pq.read_table(OODDIR / f"ood_{name}.parquet")
    x = np.asarray(t.column("x").combine_chunks().values, dtype=np.float32).reshape(-1, 110)
    df = t.drop(["x"]).to_pandas()
    if HOLDOUT:
        sp = pd.read_parquet(OODDIR / "ood_split.parquet")
        keep = ~df.source_id.isin(sp.source_id[sp.role == "train"])          # never a star any model variant trained on
        if name in HELD_COMPS:
            keep &= df.source_id.isin(sp.source_id[(sp.component == name) & (sp.role == "heldout")])
        keep = keep.to_numpy()
        df, x = df[keep].reset_index(drop=True), x[keep]
    return df, x


# ------------------------------------------------------------------ one run
def evaluate_run(arg: str, device: str) -> Path:
    from aspgapv3.ood import LatentKnnOOD
    out = OUT_ROOT / Path(arg).name / OUTSUB
    out.mkdir(parents=True, exist_ok=True)
    pred = NetPredictor(Path(arg) if Path(arg).is_dir() else RUNS / arg, device)
    meta = json.loads((OODDIR / "ood_meta.json").read_text())
    qs = meta["ood_knn_id_quantiles"]
    knn_edges = [0.0, qs["0.5"], qs["0.9"], qs["0.99"], qs["0.999"], np.inf]
    knn_names = ["<ID p50", "ID p50-90", "ID p90-99", "ID p99-99.9", ">ID p99.9"]

    # ---- training-set references (the model's own training set): input Mahalanobis, embedding kNN, in-distribution MAD
    ref = getattr(pred, "root", ROOT)
    cat = pd.read_parquet(ref / "catalog.parquet", columns=["row", "split"])
    X = np.load(ref / "coef110n.npy", mmap_mode="r")
    tr_rows = cat.row[cat.split == "train"].to_numpy()
    xt = np.asarray(X[np.sort(tr_rows)], np.float64)
    mu_x = xt.mean(0)
    cov = np.cov(xt, rowvar=False)
    P = np.linalg.inv(cov + 1e-6 * np.trace(cov) / 110 * np.eye(110))
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(xt), 200000, replace=False))
    z_tr = pred(xt[sub].astype(np.float32))["z"]
    knn = LatentKnnOOD(k=8, max_train=200000).fit(z_tr)
    del xt

    def flags(x, z):
        d = np.asarray(x, np.float64) - mu_x
        return np.log10(np.einsum("ij,jk,ik->i", d, P, d)), np.log10(knn.score(z, device=device) + 1e-9)

    te_rows = cat.row[cat.split == "iid_test"].to_numpy()
    z_ = np.load(ref / "labels.npz")
    names = [str(s) for s in z_["labels"]]
    y_id = np.stack([np.where(z_["mask"][te_rows, names.index(l)], z_["y"][te_rows, names.index(l)], np.nan) for l in LABELS], 1)
    p_id = pred(np.asarray(X[te_rows], np.float32))
    md_id, zk_id = flags(np.asarray(X[te_rows], np.float32), p_id["z"])
    r_id = p_id["mu"] - y_id
    mad_id = np.array([mad(r_id[:, j]) for j in range(4)])
    id_rows = [dict(label=l, **rstats(r_id[:, j]), mad_id=mad_id[j], sigma_med=float(np.nanmedian(p_id["sigma"][:, j])))
               for j, l in enumerate(LABELS)]
    pd.DataFrame(id_rows).to_csv(out / "metrics_id.csv", index=False)
    log_id = {"maha": md_id, "zknn": zk_id, "sigma": p_id["sigma"]}

    rows_c, rows_b, rows_cl, rows_d, rows_a, rows_oa = [], [], [], [], [], []
    counts = {}
    pooled = {}
    for comp in COMPS:
        if not (OODDIR / f"ood_{comp}.parquet").exists():
            continue
        df, x = read_comp(comp)
        counts[comp] = dict(n=len(df), n_clusters=int(df.cluster.nunique()) if "cluster" in df else 0)
        p = pred(x)
        mdist, zk = flags(x, p["z"])
        res = pd.DataFrame({"source_id": df.source_id})
        for j, l in enumerate(LABELS):
            res["mu_" + l], res["sig_" + l] = p["mu"][:, j], p["sigma"][:, j]
        res["maha_log"], res["zknn_log"] = mdist, zk
        if pred.has_dist:
            res["mg"], res["s_mg"], res["a0"] = p["mg"], p["s_mg"], p["a0"]
        res.to_parquet(out / f"pred_{comp}.parquet")
        n = len(df)
        err = {l: p["mu"][:, j] - df[l].to_numpy(float) for j, l in enumerate(LABELS) if (comp, l) not in SKIP}
        knn_bin = pd.cut(df.ood_knn, knn_edges, labels=knn_names, right=False)
        bins = {"ood_knn": knn_bin,
                "G": pd.cut(df.G, [0, 13, 14, 15, 16, 17, 17.7]),
                "a0_sfd": pd.cut(df.a0_sfd, [-1, 0.1, 0.3, 0.6, 1.0, 2.0, 99]),
                "teff_true": pd.cut(df.teff, [0, 4500, 5500, 6500, 8000, 12000, 1e5]),
                "mh_true": pd.cut(df.m_h, [-9, -2, -1, -0.5, 0, 9])}
        for j, l in enumerate(LABELS):
            if l not in err:
                continue
            e, s = err[l], p["sigma"][:, j]
            rows_c.append(dict(component=comp, label=l, **rstats(e), sigma_med=float(np.nanmedian(s)),
                               calib=mad(e / s), mad_id=mad_id[j], rmse_over_mad_id=rstats(e)["rmse"] / mad_id[j],
                               truth_err_med=float(np.nanmedian(df["e_" + l]))))
            for bv, bser in bins.items():
                if bser.notna().sum() == 0:
                    continue
                for b in bser.cat.categories:
                    sel = (bser == b).to_numpy()
                    if sel.sum() >= 20:
                        rows_b.append(dict(component=comp, bin_var=bv, bin=str(b), label=l, **rstats(e[sel]),
                                           sigma_med=float(np.nanmedian(s[sel])), calib=mad(e[sel] / s[sel])))
            # ---- OOD flags: |err| > 3 x ID MAD
            ok = np.isfinite(e)
            pos = np.abs(e) > 3 * mad_id[j]
            fl = {"pred_sigma": s, "input_mahalanobis": mdist, "z_knn": zk, "oracle_label_knn": df.ood_knn.to_numpy(float)}
            for fn, fv in fl.items():
                rows_a.append(dict(component=comp, label=l, flag=fn, auc=auc(fv[ok], pos[ok]), n=int(ok.sum()),
                                   frac_pos=float(pos[ok].mean()) if ok.any() else np.nan))
            pooled.setdefault(l, []).append((comp, e, pos, fl))
        # ---- detect OOD-ness itself: this component vs APOGEE iid_test
        for fn, fo, fi in (("input_mahalanobis", mdist, md_id), ("z_knn", zk, zk_id)):
            sc = np.concatenate([fo, fi])
            lab = np.concatenate([np.ones(n, bool), np.zeros(len(fi), bool)])
            rows_oa.append(dict(component=comp, flag=fn, auc_vs_apogee_test=auc(sc, lab)))
        for j, l in enumerate(LABELS):
            if np.isfinite(p["sigma"][:, j]).any():
                sc = np.concatenate([p["sigma"][:, j], p["sigma"][:, j][:0]])
                s_id = p_id["sigma"][:, j]
                rows_oa.append(dict(component=comp, flag=f"pred_sigma_{l}",
                                    auc_vs_apogee_test=auc(np.concatenate([p["sigma"][:, j], s_id]),
                                                           np.concatenate([np.ones(n, bool), np.zeros(len(s_id), bool)]))))
        # ---- distance / extinction (models with an M_G head)
        if pred.has_dist:
            mu_pred = df.G.to_numpy(float) - p["mg"] - EXT_G * p["a0"]
            plx = df.parallax.to_numpy(float) - df.zpt.to_numpy(float)
            hi = (plx / df.parallax_error.to_numpy(float) > 10) & (df.ruwe.to_numpy(float) < 1.4)
            mu_plx = np.where(hi, 5 * np.log10(np.where(hi, 1000.0 / np.maximum(plx, 1e-6), 1.0)) - 5, np.nan)
            ref = [("parallax(S/N>10)", mu_plx, np.where(hi, 5 / np.log(10) * df.parallax_error / plx, np.nan))]
            if "mu_true" in df:
                ref.append(("catalogue distance", df.mu_true.to_numpy(float), df["sigma_mu"].to_numpy(float) if "sigma_mu" in df else np.zeros(n)))
            for rn, mut, smu in ref:
                dmu = mu_pred - mut
                tot = np.sqrt(p["s_mg"] ** 2 + np.nan_to_num(smu) ** 2)
                for bv in ("all", "G", "ood_knn", "a0_sfd"):
                    cats = [("all", np.ones(n, bool))] if bv == "all" else \
                        [(str(b), (bins[bv] == b).to_numpy()) for b in bins[bv].cat.categories]
                    for b, sel in cats:
                        ok = sel & np.isfinite(dmu)
                        if ok.sum() >= 20:
                            rows_d.append(dict(component=comp, reference=rn, bin_var=bv, bin=b, n=int(ok.sum()),
                                               dmu_median=float(np.median(dmu[ok])), dmu_mad=mad(dmu[ok]),
                                               pull_mad=mad(dmu[ok] / tot[ok]), s_mg_med=float(np.median(p["s_mg"][ok])),
                                               frac_dist_median=float(np.median(np.exp(0.4605 * dmu[ok]) - 1))))
            hb = (df.b.abs() > 30).to_numpy() & np.isfinite(df.a0_sfd.to_numpy(float))
            da = p["a0"] - df.a0_sfd.to_numpy(float)
            row = dict(component=comp, reference="A0 vs SFD(x0.86) |b|>30", bin_var="all", bin="all", n=int(hb.sum()),
                       dmu_median=float(np.median(da[hb])) if hb.any() else np.nan, dmu_mad=mad(da[hb]) if hb.any() else np.nan,
                       pull_mad=np.nan, s_mg_med=np.nan, frac_dist_median=np.nan)
            rows_d.append(row)
            if "a0_ref" in df:
                for bv in ("all", "a0_sfd"):
                    cats = [("all", np.ones(n, bool))] if bv == "all" else \
                        [(str(b), (bins[bv] == b).to_numpy()) for b in bins[bv].cat.categories]
                    for b, sel in cats:
                        d_ = (p["a0"] - df.a0_ref.to_numpy(float))
                        ok = sel & np.isfinite(d_)
                        if ok.sum() >= 20:
                            rows_d.append(dict(component=comp, reference="A0 - A_V(catalogue)", bin_var=bv, bin=b, n=int(ok.sum()),
                                               dmu_median=float(np.median(d_[ok])), dmu_mad=mad(d_[ok]), pull_mad=np.nan,
                                               s_mg_med=np.nan, frac_dist_median=float(np.corrcoef(p["a0"][ok], df.a0_ref.to_numpy(float)[ok])[0, 1])))
        # ---- per-cluster table
        if "cluster" in df:
            for cl, g in df.groupby("cluster"):
                ix = g.index.to_numpy()
                if len(ix) < 10:
                    continue
                r = dict(component=comp, cluster=cl, n=len(ix), truth_m_h=float(g.m_h.iloc[0]),
                         dMH_median=float(np.median(p["mu"][ix, 2] - g.m_h)), MH_spread=mad(p["mu"][ix, 2]),
                         dAM_median=float(np.median(p["mu"][ix, 3] - g.alpha_m)) if comp != "cluster_gc" else np.nan,
                         AM_spread=mad(p["mu"][ix, 3]), sig_mh_med=float(np.median(p["sigma"][ix, 2])), G_med=float(g.G.median()))
                if pred.has_dist:
                    mup = mu_pred[ix] - g.mu_true.to_numpy()
                    r.update(dmu_median=float(np.median(mup)), dmu_spread=mad(mup), A0_median=float(np.median(p["a0"][ix])),
                             A0_spread=mad(p["a0"][ix]))
                    if "a0_ref" in g:
                        r["dA0_vs_ref"] = float(np.median(p["a0"][ix]) - g.a0_ref.iloc[0])
                rows_cl.append(r)
    # pooled flags across the non-cluster components
    for l, lst in pooled.items():
        lst = [t for t in lst if t[0] in ("galah", "lamost_hot", "lamost_vmp")]
        if not lst:
            continue
        e = np.concatenate([t[1] for t in lst])
        pos = np.concatenate([t[2] for t in lst])
        for fn in lst[0][3]:
            fv = np.concatenate([t[3][fn] for t in lst])
            ok = np.isfinite(e) & np.isfinite(fv)
            rows_a.append(dict(component="pooled(galah+hot+vmp)", label=l, flag=fn, auc=auc(fv[ok], pos[ok]), n=int(ok.sum()),
                               frac_pos=float(pos[ok].mean())))
    for nm, rws in (("components", rows_c), ("binned", rows_b), ("clusters", rows_cl), ("distance", rows_d),
                    ("auc", rows_a), ("auc_vs_apogee_test", rows_oa)):
        pd.DataFrame(rws).to_csv(out / f"metrics_{nm}.csv", index=False)
    (out / "counts.json").write_text(json.dumps(counts))
    write_run_report(out, pred.name, meta)
    return out


def load_csv(d: Path, nm: str) -> pd.DataFrame:
    f = d / f"metrics_{nm}.csv"
    try:
        return pd.read_csv(f) if f.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_run_report(out: Path, name: str, meta: dict) -> None:
    L = [f"# OOD evaluation: {name}\n", "Errors = model - external label (APOGEE scale where calibrated, see build_ood.py). "
         "`rmse_over_mad_id` = RMSE / in-distribution MAD (APOGEE iid_test).\n",
         "## In-distribution reference (APOGEE iid_test)\n", md(load_csv(out, "id"), ".4f"),
         "## Per component\n", md(load_csv(out, "components"), ".4f")]
    b = load_csv(out, "binned")
    for bv in b.bin_var.unique() if len(b) else []:
        L += [f"## Binned by {bv}\n", md(b[b.bin_var == bv].drop(columns="bin_var"), ".3f")]
    L += ["## Clusters\n", md(load_csv(out, "clusters"), ".3f"), "## Distance / extinction\n", md(load_csv(out, "distance"), ".3f"),
          "## OOD-flag AUC for |err| > 3 x ID MAD\n", md(load_csv(out, "auc"), ".3f"),
          "## Can the flag tell the component from APOGEE test stars?\n", md(load_csv(out, "auc_vs_apogee_test"), ".3f")]
    (out / "report.md").write_text("\n".join(L))


# ------------------------------------------------------------------ summary
def write_summary(runs: list[str]) -> None:
    dirs = {Path(a).name: OUT_ROOT / Path(a).name / OUTSUB for a in runs}
    meta = json.loads((OODDIR / "ood_meta.json").read_text())
    fnd = OUT_ROOT / SUMDIR / "findings.md"
    S = [fnd.read_text() if fnd.exists() else "", "# OOD summary\n", f"Models: {', '.join(dirs)}.  Error = model - external label.\n",
         "## Test-set sizes (after removing APOGEE-catalogue stars, with XP and G<17.65)\n",
         md(pd.DataFrame([dict(component=k, **v) for k, v in meta["counts"].items()]) if not HOLDOUT else
            pd.DataFrame([dict(component=k, **v) for k, v in json.loads((next(iter(dirs.values())) / "counts.json").read_text()).items()]), ".0f"),
         "## Label-scale conversions used\n", "```json\n" + json.dumps({"galah_scale": meta.get("galah_scale"), "galah_apogee_overlap": meta.get("galah_apogee_overlap"),
                                                                        "gc_scale": {k: v for k, v in meta.get("gc_scale", {}).items() if k != "table"}},
                                                                       indent=1, default=float) + "\n```\n"]

    def pivot(csvname, idx, vals, filt=None):
        fr = []
        for n, d in dirs.items():
            t = load_csv(d, csvname)
            if len(t) == 0:
                continue
            if filt is not None:
                t = t[filt(t)]
            if "n" in t:
                t = t[t.n > 0]
            t = t.assign(model=n)
            fr.append(t)
        if not fr:
            return pd.DataFrame()
        t = pd.concat(fr)
        p = t.pivot_table(index=idx, columns="model", values=vals, sort=False)
        p.columns = [f"{a}_{b}" for a, b in p.columns]
        return p.reset_index()

    S += ["## In-distribution reference (APOGEE iid_test): RMSE / MAD / bias\n", md(pivot("id", ["label"], ["rmse", "mad", "bias"]), ".4f")]
    S += ["## Per component: RMSE, MAD, bias (model - truth)\n"]
    for v in ("rmse", "mad", "bias"):
        S += [f"### {v}\n", md(pivot("components", ["component", "label"], [v]), ".4f")]
    S += ["### n and predicted-sigma / calibration (MAD of err/sigma)\n", md(pivot("components", ["component", "label"], ["n", "sigma_med", "calib"]), ".3f")]
    for l in LABELS:
        S += [f"## Degradation vs OOD score (true-label kNN distance), label {l}: RMSE and MAD\n"]
        S += [md(pivot("binned", ["component", "bin"], ["rmse", "mad", "n"],
                       lambda t, l=l: (t.bin_var == "ood_knn") & (t.label == l)), ".3f")]
    S += ["## Degradation vs G (RMSE), all labels\n"]
    for l in LABELS:
        S += [f"### {l}\n", md(pivot("binned", ["component", "bin"], ["rmse", "bias"], lambda t, l=l: (t.bin_var == "G") & (t.label == l)), ".3f")]
    S += ["## Clusters (per cluster: offset and internal spread)\n"]
    for n, d in dirs.items():
        t = load_csv(d, "clusters")
        if len(t) == 0:
            continue
        agg = t.groupby("component").agg(n_clusters=("cluster", "size"), dMH_median=("dMH_median", "median"),
                                         dMH_rms=("dMH_median", lambda x: float(np.sqrt(np.mean(x ** 2)))),
                                         MH_spread_med=("MH_spread", "median"), AM_spread_med=("AM_spread", "median"),
                                         **({"dmu_median": ("dmu_median", "median"), "dmu_spread_med": ("dmu_spread", "median"),
                                             "A0_spread_med": ("A0_spread", "median")} if "dmu_median" in t else {})).reset_index()
        S += [f"### {n}\n", md(agg, ".3f")]
    for n, d in dirs.items():
        t = load_csv(d, "clusters")
        if len(t):
            S += [f"### per-cluster, {n} (GC sorted by truth [M/H])\n", md(t[t.component == "cluster_gc"].sort_values("truth_m_h"), ".3f")]
            break
    S += ["## Distance / extinction (models with an M_G head)\n"]
    for n, d in dirs.items():
        t = load_csv(d, "distance")
        if len(t):
            S += [f"### {n}\n", md(t, ".3f")]
    S += ["## OOD-flag AUC: |err| > 3 x ID MAD vs flag\n"]
    S += [md(pivot("auc", ["component", "label", "flag"], ["auc"]), ".3f")]
    S += ["## Flag AUC: component vs APOGEE iid_test stars\n", md(pivot("auc_vs_apogee_test", ["component", "flag"], ["auc_vs_apogee_test"]), ".3f")]
    od = OUT_ROOT / SUMDIR
    od.mkdir(parents=True, exist_ok=True)
    (od / "report.md").write_text("\n".join(S))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="run dir/name (relative names resolve under runs/)")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--holdout", action="store_true",
                    help="evaluate only the held-out OOD stars (ood_split.parquet, written by build_ext_dataset.py): outputs "
                         "<run>/ood_holdout and runs/ood_summary_holdout")
    ap.add_argument("--out-dir", default=None, help="write all outputs below this directory instead of runs/")
    args = ap.parse_args()
    global HOLDOUT, OUTSUB, SUMDIR, OUT_ROOT
    if args.out_dir:
        OUT_ROOT = Path(args.out_dir)
    if args.holdout:
        HOLDOUT, OUTSUB, SUMDIR = True, "ood_holdout", "ood_summary_holdout"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.summary_only:
        for r in args.run:
            print("evaluating", r, flush=True)
            evaluate_run(r, dev)
    write_summary(args.run)


if __name__ == "__main__":
    main()
