#!/usr/bin/env python
"""Precompute the AspGap v1 1024-d latent (output of xp2_lat_0_ep200, v1's first
network, applied to v1-preprocessed CLEAN coefficients) for every catalog row.

Outputs next to the other processed arrays (<root>):
    v1_latent.npy        float32 [n_rows, 1024] (latent std ~0.015 => fp16 quantisation would be a few % of sigma)  row-indexed exactly like coef110.npy /
                         xp343.npy / side.npy / labels_aux.npz (row i = catalog['row'] == i)
    v1_latent_norm.npz   mean, std (float32 [1024], fit on TRAIN-split rows only),
                         source_id (int64 [n_rows], for alignment checks), valid (bool [n_rows])

Rows whose coefficients / G are not finite get a zero latent and valid=False; the
dataset masks them out of the latent loss. Preprocessing is v1 production
(V1Model.preprocess in scripts/eval_v1_reference.py, verified == v3 coef110n to 3.8e-5).

    python scripts/build_v1_latent.py [--root DATA_ROOT] [--device cuda]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT_DEFAULT = "/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3/data/processed/aspcap_v0.6.0"


def load_v1_model(device):
    spec = importlib.util.spec_from_file_location("eval_v1_reference", HERE / "eval_v1_reference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.V1Model(device)


@torch.no_grad()
def v1_latent(model, coef, g, batch=8192):
    out = np.zeros((len(coef), 1024), np.float32)
    for i in range(0, len(coef), batch):
        x = torch.from_numpy(model.preprocess(coef[i:i + batch], g[i:i + batch])).to(model.device)
        out[i:i + batch] = model.enc(x).float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="sanity: first N rows only (writes nothing)")
    args = ap.parse_args()
    root = Path(args.root)
    model = load_v1_model(args.device)

    cat = pd.read_parquet(root / "catalog.parquet").sort_values("row").reset_index(drop=True)
    coef = np.load(root / "coef110.npy", mmap_mode="r")
    side = np.load(root / "side.npy", mmap_mode="r")
    n = len(coef)
    assert len(cat) == n and np.array_equal(cat["row"].to_numpy(), np.arange(n)), "catalog rows not 0..n-1"
    if args.limit:
        n = args.limit
    g = np.asarray(side[:n, 0], np.float64)
    c = np.asarray(coef[:n], np.float64)
    valid = np.isfinite(c).all(1) & np.isfinite(g)
    lat = np.zeros((n, 1024), np.float32)
    lat[valid] = v1_latent(model, c[valid], g[valid])
    amax = float(np.abs(lat).max())
    print(f"rows {n}, valid {int(valid.sum())}, max|latent| {amax:.2f} (float16 max 65504)")
    assert amax < 6e4
    tr = (cat["split"].to_numpy()[:n] == "train") & valid
    mean = lat[tr].mean(0).astype(np.float32)
    std = lat[tr].std(0).astype(np.float32)
    print(f"train rows {int(tr.sum())}; std min/median/max {std.min():.4g}/{np.median(std):.4g}/{std.max():.4g}")
    if args.limit:
        return
    np.save(root / "v1_latent.npy", lat)
    np.savez(root / "v1_latent_norm.npz", mean=mean, std=std,
             source_id=cat["source_id"].to_numpy().astype(np.int64), valid=valid)
    print("wrote", root / "v1_latent.npy", root / "v1_latent_norm.npz")


if __name__ == "__main__":
    main()
