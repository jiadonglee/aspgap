#!/usr/bin/env python
"""Build the data root of the final model (run aspgap_dist_ext2_cn): ext2 root + two giant-only chem labels.

    python scripts/build_cn_labels.py            # data/processed/aspcap_v0.6.0_ext2 -> ..._ext2_cn

The output root is a copy of the ext2 root (every large file symlinked) whose labels.npz / norm.npz carry two extra
labels, c_fe = [C/Fe] = c_h - fe_h and n_fe = [N/Fe] = n_h - fe_h (error: quadrature of the ASPCAP errors).
A row is valid only if it is an APOGEE giant (ASPCAP logg < 3.5 and Teff < 5500 K), C, N and Fe are valid (labels_aux
mask = unflagged and finite, catalog fe_h_ok); all external (GALAH / GC / LAMOST) rows are masked.  The label
scaler of the new labels is fitted on the train-split entries; training sigma floors (c_fe 0.03, n_fe 0.05, as for
c_h / n_h in labels_aux) are listed in `FLOORS` and enter configs/aspgap_dist_ext2_cn.yaml via the sigma floor.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

GIANT_LOGG, GIANT_TEFF = 3.5, 5500.0
FLOORS = {"c_fe": 0.03, "n_fe": 0.05}


def is_giant(logg, teff):
    return (np.asarray(logg) < GIANT_LOGG) & (np.asarray(teff) < GIANT_TEFF)


def cn_labels(root) -> dict:
    """Over the rows of `root`'s catalog: [C/Fe], [N/Fe], errors and `valid` (= giant & C, N, Fe valid)."""
    root = Path(root)
    cat = pd.read_parquet(root / "catalog.parquet")
    a = np.load(root / "labels_aux.npz", allow_pickle=True)
    names = [str(s) for s in a["labels"]]
    ic, inn = names.index("c_h"), names.index("n_h")
    y, e, m = a["y"], a["yerr"], a["mask"]
    fe, efe = cat["fe_h"].to_numpy(np.float64), cat["fe_h_err"].to_numpy(np.float64)
    ok = m[:, ic] & m[:, inn] & cat["fe_h_ok"].to_numpy(bool) & np.isfinite(fe)
    valid = ok & is_giant(cat["logg"].to_numpy(), cat["teff"].to_numpy())
    c_h, n_h = y[:, ic].astype(np.float64), y[:, inn].astype(np.float64)
    return {"c_fe": c_h - fe, "n_fe": n_h - fe, "err_c_fe": np.hypot(e[:, ic], efe),
            "err_n_fe": np.hypot(e[:, inn], efe), "valid": valid}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="data/processed/aspcap_v0.6.0", help="APOGEE root with labels_aux.npz")
    ap.add_argument("--ext", default="data/processed/aspcap_v0.6.0_ext2", help="APOGEE + external root")
    ap.add_argument("--out", default="data/processed/aspcap_v0.6.0_ext2_cn")
    a = ap.parse_args()
    base, ext, out = Path(a.base), Path(a.ext), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ext.iterdir():
        if f.name in ("labels.npz", "norm.npz"):
            continue
        dst = out / f.name
        if not dst.exists():
            os.symlink(f.resolve(), dst)
    cat_b = pd.read_parquet(base / "catalog.parquet")
    cat_e = pd.read_parquet(ext / "catalog.parquet")
    nb = len(cat_b)
    assert (cat_e["source_id"].to_numpy()[:nb] == cat_b["source_id"].to_numpy()).all()   # APOGEE rows come first
    c = cn_labels(base)
    z = np.load(ext / "labels.npz")
    n = len(cat_e)
    ynew = np.full((n, 2), np.nan, np.float32)
    enew = np.full((n, 2), np.nan, np.float32)
    mnew = np.zeros((n, 2), bool)
    v = c["valid"]
    ynew[:nb, 0], ynew[:nb, 1] = np.where(v, c["c_fe"], np.nan), np.where(v, c["n_fe"], np.nan)
    enew[:nb, 0], enew[:nb, 1] = np.where(v, c["err_c_fe"], np.nan), np.where(v, c["err_n_fe"], np.nan)
    mnew[:nb, 0] = mnew[:nb, 1] = v
    y = np.concatenate([z["y"], ynew], 1)
    e = np.concatenate([z["yerr"], enew], 1)
    m = np.concatenate([z["mask"], mnew], 1)
    labels = [str(s) for s in z["labels"]] + ["c_fe", "n_fe"]
    nz = np.load(ext / "norm.npz")
    tr = (cat_e["split"].to_numpy() == "train")
    mean, std, fl = list(nz["mean"]), list(nz["std"]), list(nz["floor_norm"])
    for k, name in ((6, "c_fe"), (7, "n_fe")):
        vals = y[tr & m[:, k], k]
        mean.append(float(vals.mean()))
        std.append(float(vals.std()))
        fl.append(FLOORS[name] / float(vals.std()))
    np.savez(out / "labels.npz", y=y, yerr=e, mask=m, labels=np.array(labels, dtype="U16"))
    np.savez(out / "norm.npz", mean=np.array(mean), std=np.array(std), floor_norm=np.array(fl),
             labels=np.array(labels, dtype="U16"))
    print("labels", labels, "n valid per split:",
          {s: int((m[:, 6] & (cat_e["split"].to_numpy() == s)).sum()) for s in ("train", "val", "iid_test", "sky_test")})
    print("mean/std c_fe, n_fe:", mean[6:], std[6:])


if __name__ == "__main__":
    main()
