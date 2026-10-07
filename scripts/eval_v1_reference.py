#!/usr/bin/env python
"""Evaluate the ORIGINAL AspGap v1 (xp2_lat_0_ep200 -> lat2_4lerr_0_ep200) on the
aspgap_v3 test sets with the same metrics/protocol as scripts/evaluate.py.

Inference only. v1 production preprocessing (notebook/predict_220M_v4.py):
    coef / 10^((15-G)*0.4)  -> sklearn RobustScaler (BP, RP separately) -> (n,110)
    z = MLP_upsampling(110->1024)(x);  out = MLP(1024->8)(z)
    labels = scaler_labels.inverse_transform(out[:,:4]);  err = |out[:,4:]*scale_|
    label order: teff, logg, m_h, alpha_m   (fe_h / mg_h NOT provided by v1)
The degraded eval degrades the RAW coefficients with v3's NoiseMatchAugment
(fixed seed, batch 1024, same dataset order => identical noise draws as v3's
degraded eval when run on the same device type), then re-applies v1's own
preprocessing at the new G'.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
V3 = Path("/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3_p1")
V1 = Path("/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap")
RUNS = Path("/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3/runs")
ROOT = "/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3/data/processed/aspcap_v0.6.0"
sys.path.insert(0, str(V3 / "src"))
sys.path.insert(0, str(V1))

from aspgap.xpformer import MLP, MLP_upsampling  # noqa: E402  (v1 model defs, unmodified)
from aspgapv3.augment import NoiseMatchAugment  # noqa: E402
from aspgapv3.data.dataset import APOGEEDataset  # noqa: E402
from aspgapv3.metrics import metrics_table  # noqa: E402

spec = importlib.util.spec_from_file_location("v3_evaluate", V3 / "scripts" / "evaluate.py")
v3eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3eval)          # only for domain_slices (reuse)

LABELS6 = ["teff", "logg", "m_h", "alpha_m", "fe_h", "mg_h"]
V1_LABELS = ["teff", "logg", "m_h", "alpha_m"]        # predict_220M_v4: teff, logg, moh, aom


class V1Model:
    """Thin wrapper reproducing v1 production inference."""

    def __init__(self, device):
        md = V1 / "model"
        self.sc_bp = joblib.load(md / "scaler_bp_gmagand_0330.gz")
        self.sc_rp = joblib.load(md / "scaler_rp_gmagand_0330.gz")
        self.sc_lab = joblib.load(md / "scaler_labels_0330.gz")
        self.enc = MLP_upsampling(110, 1024, hidden_size=256)
        self.dec = MLP(1024, 8)
        self.enc.load_state_dict(torch.load(md / "xp2_lat_0_ep200.pt", map_location="cpu"))
        self.dec.load_state_dict(torch.load(md / "lat2_4lerr_0_ep200.pt", map_location="cpu"))
        self.enc.eval().to(device)
        self.dec.eval().to(device)
        self.device = device

    def preprocess(self, coef110: np.ndarray, g: np.ndarray) -> np.ndarray:
        # identical to infer_pipeline(): gmag_norm = 10**((15-G)*0.4); bp/rp split 55/55
        gn = 10 ** ((15.0 - np.asarray(g, np.float64)) * 0.4)
        c = np.asarray(coef110, np.float64) / gn[:, None]
        bp = self.sc_bp.transform(c[:, :55])
        rp = self.sc_rp.transform(c[:, 55:])
        return np.c_[bp, rp].astype(np.float32)

    @torch.no_grad()
    def __call__(self, coef110, g):
        x = torch.from_numpy(self.preprocess(coef110, g)).to(self.device)
        out = self.dec(self.enc(x)).cpu().numpy()
        y = self.sc_lab.inverse_transform(out[:, :4])
        e = np.abs(out[:, 4:] * self.sc_lab.scale_)
        return y, e


def predict_v1(model, dset, device, batch=1024, aug=None, fixed_g=None, seed=1234):
    loader = torch.utils.data.DataLoader(dset, batch_size=batch, shuffle=False, num_workers=2)
    gen = torch.Generator(device=device).manual_seed(seed) if aug is not None else None
    mus, sigs, n_aug = [], [], 0
    for b in loader:
        if aug is not None:
            b = aug(b, gen=gen, fixed_g=fixed_g)
            n_aug += int(b["aug"].sum())
        coef = b["coef"].numpy() if aug is None else b["coef"].cpu().numpy()
        g = b["side"][:, 0].cpu().numpy()
        y, e = model(coef, g)
        mus.append(y)
        sigs.append(e)
    mu4, s4 = np.concatenate(mus), np.concatenate(sigs)
    mu = np.full((len(mu4), 6), np.nan)
    sg = np.full((len(mu4), 6), np.nan)
    mu[:, :4], sg[:, :4] = mu4, s4
    return mu, sg, n_aug / max(len(dset), 1)


def md(df, **kw):
    """to_markdown with fallback (tabulate is not installed in jepa311)."""
    try:
        return df.to_markdown(**kw)
    except ImportError:
        return "```\n" + df.to_string(index=kw.get("index", True), float_format="%.4f") + "\n```"


def star_in_dr17(out: Path) -> pd.DataFrame:
    """source_id -> True if ANY ASPCAP-0.6.0 row of that star has release==dr17
    (the star was observed by APOGEE DR17 => could be in v1's DR17 x XP training set).
    Cached in <out>/source_in_dr17.parquet."""
    f = out / "source_in_dr17.parquet"
    if f.is_file():
        return pd.read_parquet(f)
    from astropy.io import fits
    d = fits.open("/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3/data/raw/astraAllStarASPCAP-0.6.0.fits",
                  memmap=True)[2].data
    a = pd.DataFrame({"source_id": np.asarray(d["gaia_dr3_source_id"]).astype(np.int64),
                      "dr17": np.char.strip(np.asarray(d["release"]).astype(str)) == "dr17"})
    a = a.groupby("source_id").dr17.any().reset_index().rename(columns={"dr17": "in_dr17_any"})
    a.to_parquet(f)
    return a


def drop_missing(t: pd.DataFrame) -> pd.DataFrame:
    return t[t["label"].isin(V1_LABELS)].reset_index(drop=True)


def rmse_mad(r):
    r = r[np.isfinite(r)]
    return (float(np.sqrt(np.mean(r ** 2))), float(1.4826 * np.median(np.abs(r - np.median(r)))),
            float(np.mean(r)), len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS / "aspgap_v1_ref"))
    ap.add_argument("--splits", nargs="*", default=["iid_test", "sky_test"])
    ap.add_argument("--degrade-g", nargs="*", type=float, default=[15.0, 16.0, 17.0, 17.5])
    ap.add_argument("--degrade-seed", type=int, default=1234)
    ap.add_argument("--compare", nargs="*", default=["aspgap110_aug_r2", "pixel_fusion_r2"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="sanity: first N rows only")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device

    model = V1Model(dev)
    # v3 default for the aug runs: require_xp_valid true, noise on (for coef/side), input irrelevant
    kw = dict(input="coef110", labels=LABELS6, noise=True, require_xp_valid=True)
    aug = NoiseMatchAugment.from_root(ROOT, device=dev, p=1.0, g_max=17.65)
    summary = ["# AspGap v1 (original) reference evaluation on aspgap_v3 test sets\n",
               "v1 provides teff, logg, m_h, alpha_m only (no fe_h / mg_h). Raw v3 labels "
               "(ASPCAP astra 0.6.0) are the targets; v1 was trained on DR17 labels.\n"]

    preds = {}
    for split in args.splits:
        dset = APOGEEDataset(ROOT, split, **kw)
        if args.limit:
            dset.indices = dset.indices[:args.limit]; dset.catalog = dset.catalog.iloc[:args.limit]
            dset.y_raw, dset.mask = dset.y_raw[:args.limit], dset.mask[:args.limit]
            dset.y, dset.yerr = dset.y[:args.limit], dset.yerr[:args.limit]
        cat = dset.catalog
        mu, sig, _ = predict_v1(model, dset, dev)
        y, m = dset.y_raw, dset.mask.copy()
        m[:, 4:] = False
        table = drop_missing(metrics_table(mu, sig, y, m, LABELS6))
        table.to_csv(out / f"metrics_{split}.csv", index=False)
        summary += [f"\n## {split} per-label (raw units)\n", md(table, index=False, floatfmt=".4f")]

        rows = []
        for name, sel in v3eval.domain_slices(cat).items():
            t = metrics_table(mu, sig, y, m, LABELS6)
            for k, lab in enumerate(LABELS6[:4]):
                r = t.iloc[k].to_dict()
                s = sel & m[:, k]
                r.update({"domain": name,
                          "rmse_domain": float(np.sqrt(np.mean((mu[s, k] - y[s, k]) ** 2))) if s.sum() else np.nan,
                          "n_domain": int(s.sum())})
                rows.append(r)
        pd.DataFrame(rows).to_csv(out / f"metrics_{split}_domains.csv", index=False)
        np.savez(out / f"predictions_{split}.npz", source_id=cat["source_id"].to_numpy(), mu=mu, sigma=sig)
        preds[split] = (dset, mu, sig, y, m)

        # degraded curve (same protocol as v3 evaluate.py)
        rr = []
        for gp in [None] + list(args.degrade_g):
            if gp is None:
                mu_g, sg_g, frac = mu, sig, 0.0
            else:
                mu_g, sg_g, frac = predict_v1(model, dset, dev, aug=aug, fixed_g=gp, seed=args.degrade_seed)
            t = drop_missing(metrics_table(mu_g, sg_g, y, m, LABELS6))
            t.insert(0, "g_prime", np.nan if gp is None else gp)
            t.insert(1, "frac_degraded", frac)
            rr.append(t)
        deg = pd.concat(rr, ignore_index=True)
        deg.to_csv(out / f"metrics_degraded_{split}.csv", index=False)
        summary += [f"\n## {split} degraded to fixed G' (seed {args.degrade_seed}; NaN = native)\n",
                    md(deg[["g_prime", "frac_degraded", "label", "n", "rmse", "bias", "scatter_mad",
                         "calibration_ratio"]], index=False, floatfmt=".4f")]
        if dev.startswith("cuda"):
            summary.append(f"\n(peak torch GPU mem so far: {torch.cuda.max_memory_allocated()/2**20:.0f} MiB)")

    # ---------------- comparisons on identical rows ----------------
    rowmap = pd.read_parquet(Path(ROOT) / "apogee_rowmap.parquet")[["row", "release"]]
    dr17any = star_in_dr17(out)
    comp_rows = []
    for split in args.splits:
        dset, mu, sig, y, m = preds[split]
        cat = dset.catalog.merge(rowmap, on="row", how="left").merge(dr17any, on="source_id", how="left")
        assert len(cat) == len(dset.catalog)
        sid = cat["source_id"].to_numpy()
        others = {}
        for name in args.compare:
            z = np.load(RUNS / name / f"predictions_{split}.npz")
            if args.limit:
                continue
            assert np.array_equal(z["source_id"], sid), f"{name}: row mismatch"
            others[name] = z["mu"]
        g = cat["phot_g_mean_mag"].to_numpy()
        logg_t = y[:, 1]
        subsets = {"all": np.ones(len(cat), bool),
                   "G<16 & giants(logg<3.5)": (g < 16) & (logg_t < 3.5),
                   "G<14 & giants(logg<3.5)": (g < 14) & (logg_t < 3.5),
                   "release=sdss5 (not in DR17)": (cat["release"] == "sdss5").to_numpy(),
                   "star never observed in DR17 (any row)": ~cat["in_dr17_any"].fillna(False).to_numpy(bool),
                   "star never in DR17 & G<16 & giants": ~cat["in_dr17_any"].fillna(False).to_numpy(bool) & (g < 16) & (logg_t < 3.5),
                   "release=dr17": (cat["release"] == "dr17").to_numpy(),
                   "sdss5 & G<16 & giants": (cat["release"] == "sdss5").to_numpy() & (g < 16) & (logg_t < 3.5),
                   "dr17 & G<16 & giants": (cat["release"] == "dr17").to_numpy() & (g < 16) & (logg_t < 3.5)}
        for sname, sel in subsets.items():
            for k, lab in enumerate(V1_LABELS):
                s = sel & m[:, k]
                for mname, mm in [("v1", mu)] + list(others.items()):
                    rm, mad_, bi, n = rmse_mad(mm[s, k] - y[s, k])
                    comp_rows.append(dict(split=split, subset=sname, label=lab, model=mname,
                                          n=n, rmse=rm, scatter_mad=mad_, bias=bi))
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "compare_subsets.csv", index=False)
    summary.append("\n## Same-row comparison (v1 vs v3 models), rmse / MAD*1.4826 / bias\n")
    for (split, sname), d in comp.groupby(["split", "subset"], sort=False):
        piv = d.pivot_table(index="label", columns="model", values=["rmse", "scatter_mad"])
        summary += [f"\n### {split} | {sname} | n(teff)={int(d[d.label=='teff'].n.iloc[0])}\n",
                    md(piv, floatfmt=".4f")]

    (out / "summary.md").write_text("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
