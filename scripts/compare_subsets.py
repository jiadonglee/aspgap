#!/usr/bin/env python
"""Subset comparison from SAVED predictions (predictions_<split>.npz, raw label units).

Models (default): v1 reference, aspgap110_aug_r2, aspgap110_aug_rand_r2, aspgap110_aug_v1lat_r2.
Subsets (each also over all rows):
    G<16 & giants (true logg < 3.5)          -- bright-giant subset
    never-in-DR17                            -- star has no DR17 row in ASPCAP 0.6.0
                                               (runs/aspgap_v1_ref/source_in_dr17.parquet)
Per label: n, RMSE, 1.4826*MAD (about the median residual), bias. v1 only provides
teff/logg/m_h/alpha_m (NaN predictions elsewhere are skipped). Missing prediction
files are reported and skipped, so the script can be run before everything finished.

    python scripts/compare_subsets.py [--splits iid_test sky_test] [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3 = Path(__file__).resolve().parents[1]
MAIN = Path("/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3")
RUNS = MAIN / "runs"
ROOT = MAIN / "data/processed/aspcap_v0.6.0"
sys.path.insert(0, str(V3 / "src"))

from aspgapv3.data.dataset import APOGEEDataset  # noqa: E402

LABELS = ["teff", "logg", "m_h", "alpha_m", "fe_h", "mg_h"]
DEFAULT_MODELS = {"v1": "aspgap_v1_ref", "aug_r2": "aspgap110_aug_r2",
                  "A_rand": "aspgap110_aug_rand_r2", "B_v1lat": "aspgap110_aug_v1lat_r2"}


def stats(r):
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return dict(n=0, rmse=np.nan, mad=np.nan, bias=np.nan)
    return dict(n=len(r), rmse=float(np.sqrt(np.mean(r ** 2))),
                mad=float(1.4826 * np.median(np.abs(r - np.median(r)))), bias=float(np.mean(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["iid_test", "sky_test"])
    ap.add_argument("--models", nargs="*", default=[f"{k}={v}" for k, v in DEFAULT_MODELS.items()],
                    help="label=run_dir_name under runs/")
    ap.add_argument("--out", default=str(RUNS / "compare_subsets_rand_v1lat"))
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in args.models)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dr17 = pd.read_parquet(RUNS / "aspgap_v1_ref" / "source_in_dr17.parquet")
    rows = []
    for split in args.splits:
        d = APOGEEDataset(args.root, split, input="coef110n", labels=LABELS, noise=False,
                          require_xp_valid=True)
        cat = d.catalog.merge(dr17, on="source_id", how="left")
        assert len(cat) == len(d.catalog)
        y, m = d.y_raw, d.mask
        sid = cat["source_id"].to_numpy()
        g = cat["phot_g_mean_mag"].to_numpy()
        giants = (g < 16) & (y[:, 1] < 3.5) & m[:, 1]
        never = ~cat["in_dr17_any"].fillna(False).to_numpy(bool)
        subsets = {"all": np.ones(len(cat), bool), "G<16 & logg<3.5": giants,
                   "never-in-DR17": never, "never-in-DR17 & G<16 & logg<3.5": never & giants}
        for name, run in models.items():
            f = RUNS / run / f"predictions_{split}.npz"
            if not f.is_file():
                print(f"[skip] {name}: {f} missing")
                continue
            z = np.load(f)
            assert np.array_equal(z["source_id"], sid), f"{name}/{split}: row mismatch"
            mu = z["mu"]
            for sname, sel in subsets.items():
                for k, lab in enumerate(LABELS):
                    s = sel & m[:, k] & np.isfinite(mu[:, k])
                    rows.append(dict(split=split, subset=sname, label=lab, model=name,
                                     **stats(mu[s, k] - y[s, k])))
    df = pd.DataFrame(rows)
    df.to_csv(out / "compare_subsets.csv", index=False)
    lines = ["# Subset comparison (RMSE / 1.4826*MAD / n), raw units\n"]
    for (split, sname), t in df.groupby(["split", "subset"], sort=False):
        piv = t.pivot_table(index="label", columns="model", values=["rmse", "mad"], sort=False)
        n = t.groupby("label", sort=False)["n"].max()
        lines += [f"\n## {split} | {sname}\n", "```", f"n per label (max over models): {n.to_dict()}",
                  piv.to_string(float_format="%.4f"), "```"]
    (out / "compare_subsets.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
