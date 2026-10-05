#!/usr/bin/env python
"""Add Gaia XP *noise* products to a processed dataset (+ a noise donor bank).

    python scripts/build_xp_noise.py --data data/processed/aspcap_v0.6.0_test \
        [--raw-dir /nexus/.../xp_continuous_mean_spectrum] [--nproc 12] \
        [--bank-per-bin 200] [--verify-n 20] [--limit N]

Writes next to xp343.npy (row order == catalog['row'] == xp343 row):

  xpcov_chol.npy   float32 [n, 3080]  packed lower-triangular Cholesky factors
                   of the BP and RP coefficient covariances ([BP 1540 | RP 1540],
                   absolute DR3 units; see data/xpnoise.py for the convention).
                   All-zero for stars without a usable covariance.
  xp343_sigma.npy  float32 [n, 343]   per-pixel 1-sigma = sqrt(diag(pixcov_abs))
                   / flux_550_abs, i.e. gaiaxpy's calibrate() flux_error divided
                   by the 550nm flux, in the units of xp343. The 550nm normalizer
                   is treated as noise-free (sigma_550 /= 0 here); the exact
                   first-order ratio sigma differs near 550nm only (documented in
                   PROGRESS.md; see xpnoise.normalized_sigma_ratio).
  side.npy         float32 [n, 4]  columns in xpnoise_meta.json:
                   phot_g_mean_mag (catalog), bp_rp (gxp_chunk 'bp_rp', DR3
                   BP-RP), log10_flux550 (design-matrix-absolute merged flux at
                   550nm from the RAW coefficients, before normalization),
                   log10_snr550 (= log10 flux_550 / sigma_550, propagated).
  xpnoise_meta.json  columns, sources, jitter counts, verification numbers.
  noise_bank.npz   (unless --no-bank) donors: ~200 random XP stars per 0.1 mag G
                   bin over G in [10, 17.7), Cholesky factors in ABSOLUTE
                   units, plus the G histogram of the entire XP catalog.
  [--verify-n] xpnoise_verify_raw.npz  raw rows of N stars for the gaiaxpy
                   cross-check (scripts/verify_xp_gaiaxpy.py, env `xpbuild`).

Raw DR3 files: XpContinuousMeanSpectrum_<lo>-<hi>.h5, sorted by source_id; the
name gives the inclusive range of source_id >> 43, so a star is routed to its
single file without scanning.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aspgapv3.data.dataset import load_catalog  # noqa: E402
from aspgapv3.data.xp import NORM_INDEX, absolute_flux, build_design_matrix  # noqa: E402
from aspgapv3.data.xpnoise import (N_PACK, cov_from_h5_rows, parse_h5_range,  # noqa: E402
                                   pixel_sigma_abs)

RAW_DIR = "/nexus/posix0/MIA-astro-env/hxr/shared/gaiaxp/xp_continuous_mean_spectrum"
CHUNK_DIR = "../aspgap/data/xpnorm"
SIDE_COLS = ["phot_g_mean_mag", "bp_rp", "log10_flux550", "log10_snr550"]
BANK_EDGES = np.round(np.arange(10.0, 17.7001, 0.1), 3)      # 77 bins
HIST_EDGES = np.round(np.arange(0.0, 21.0001, 0.1), 3)

_DM: dict = {}
_G: dict = {}


def log(m: str) -> None:
    print(m, flush=True)


# ------------------------------------------------------------------ raw h5

def _init_raw(dm_path: str) -> None:
    _DM.update(build_design_matrix(dm_path))


def _raw_task(args):
    path, sids, pos = args
    with h5py.File(path, "r") as h:
        ids = h["source_id"][:]
        rows = np.searchsorted(ids, sids)
        found = (rows < len(ids)) & (ids[np.minimum(rows, len(ids) - 1)] == sids)
        res = cov_from_h5_rows(h, rows[found])
    flux = absolute_flux(res["coef"], _DM) if found.any() else np.zeros((0, 343))
    sig = pixel_sigma_abs(res["chol"], _DM) if found.any() else np.zeros((0, 343))
    return pos[found], res, flux, sig


def extract_raw(sids: np.ndarray, files: list[tuple[int, int, str]], nproc: int,
                out_chol: np.ndarray, dm_path: str) -> dict:
    """Fetch covariance/coefficients for `sids` (any order). Fills out_chol
    [n, 3080] in place; returns dict of in-memory arrays aligned with sids."""
    n = len(sids)
    lo = np.array([f[0] for f in files])
    hi = np.array([f[1] for f in files])
    key = sids >> 43
    fi = np.searchsorted(lo, key, side="right") - 1
    routed = (fi >= 0) & (key <= hi[np.maximum(fi, 0)])
    order = np.argsort(fi, kind="stable")
    tasks = []
    for f in np.unique(fi[routed]):
        pos = order[(fi[order] == f) & routed[order]]
        s = sids[pos]
        o = np.argsort(s)
        tasks.append((files[f][2], s[o], pos[o]))
    log(f"  routed {int(routed.sum())}/{n} stars to {len(tasks)} raw files")
    res = {"coef": np.full((n, 110), np.nan), "found": np.zeros(n, bool),
           "ok": np.zeros(n, bool), "level": np.full((n, 2), -1, np.int8),
           "flux": np.full((n, 343), np.nan, np.float32),
           "sigma_abs": np.full((n, 343), np.nan, np.float32)}
    t0 = time.time()
    with mp.Pool(nproc, initializer=_init_raw, initargs=(dm_path,)) as pool:
        for k, (pos, r, flux, sig) in enumerate(pool.imap_unordered(_raw_task, tasks)):
            if len(pos) == 0:
                continue
            res["coef"][pos] = r["coef"]
            res["found"][pos] = True
            res["ok"][pos] = r["ok"]
            res["level"][pos] = r["level"]
            res["flux"][pos] = flux
            res["sigma_abs"][pos] = sig
            out_chol[pos] = r["chol"]
            if k % 200 == 0:
                log(f"    raw file {k + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
    return res


# ------------------------------------------------- gxp_chunk scan (G, bp_rp)

def _init_scan(wanted, frac, seed):
    _G.update(wanted=wanted, frac=frac, seed=seed)


def _scan_task(args):
    k, path = args
    with h5py.File(path, "r") as h:
        c = h["table/columns"]
        sid = c["source_id/data"][:]
        g = c["phot_g_mean_mag/data"][:].astype(np.float64)
        hist = np.histogram(g[np.isfinite(g)], HIST_EDGES)[0]
        rng = np.random.default_rng([_G["seed"], k])
        sub = np.isfinite(g) & (g >= BANK_EDGES[0]) & (g < BANK_EDGES[-1]) \
            & (rng.random(len(g)) < _G["frac"])
        w = _G["wanted"]                                   # sorted
        j = np.minimum(np.searchsorted(w, sid), len(w) - 1)
        hit = np.where(w[j] == sid)[0]
        bprp = c["bp_rp/data"][:][hit].astype(np.float64) if len(hit) else np.zeros(0)
        return hist, sid[sub], g[sub], j[hit], bprp


def scan_chunks(chunk_dir: str, wanted: np.ndarray, frac: float, nproc: int, seed: int):
    files = sorted(glob.glob(f"{chunk_dir}/gxp_chunk_*.hdf5"))
    w = np.sort(wanted)
    hist = np.zeros(len(HIST_EDGES) - 1, np.int64)
    ss, sg = [], []
    bprp_sorted = np.full(len(w), np.nan)
    t0 = time.time()
    with mp.Pool(nproc, initializer=_init_scan, initargs=(w, frac, seed)) as pool:
        for k, (h, a, b, j, bp) in enumerate(pool.imap_unordered(
                _scan_task, list(enumerate(files)))):
            hist += h
            ss.append(a)
            sg.append(b)
            bprp_sorted[j] = bp
            if k % 400 == 0:
                log(f"    scan {k + 1}/{len(files)} ({time.time() - t0:.0f}s)")
    pos = np.searchsorted(w, wanted)                         # unsort
    return hist, np.concatenate(ss), np.concatenate(sg), bprp_sorted[pos]


# ----------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--chunks", default=CHUNK_DIR)
    ap.add_argument("--nproc", type=int, default=12)
    ap.add_argument("--bank-per-bin", type=int, default=200)
    ap.add_argument("--bank-frac", type=float, default=0.003,
                    help="random fraction of the XP catalog kept as bank candidates")
    ap.add_argument("--no-bank", action="store_true")
    ap.add_argument("--verify-n", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N catalog rows")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.data)
    cat = load_catalog(root)
    assert (cat["row"].to_numpy() == np.arange(len(cat))).all(), "catalog rows != arange"
    if args.limit:
        cat = cat.iloc[:args.limit]
    n = len(cat)
    sids = cat["source_id"].to_numpy(np.int64)
    g_cat = cat["phot_g_mean_mag"].to_numpy(float)
    dm_path = str(root / "xp_design_343.npz")
    dm = build_design_matrix(dm_path)

    files = sorted((*parse_h5_range(Path(f).name), f)
                   for f in glob.glob(f"{args.raw_dir}/XpContinuousMeanSpectrum_*.h5"))
    log(f"[1/5] {n} stars, {len(files)} raw files")

    # ---- 2. scan the gxp chunks: bp_rp for our stars, G hist + bank candidates
    log("[2/5] scanning gxp_chunk files (bp_rp, G histogram, bank candidates)")
    hist, cand_sid, cand_g, bp_rp = scan_chunks(
        args.chunks, sids, 0.0 if args.no_bank else args.bank_frac, args.nproc, args.seed)

    # ---- 3. raw covariances for the dataset stars --------------------------
    log("[3/5] extracting raw coefficients + covariances")
    chol = np.lib.format.open_memmap(root / "xpcov_chol.npy", mode="w+",
                                     dtype=np.float32, shape=(n, N_PACK))
    r = extract_raw(sids, files, args.nproc, chol, dm_path)
    chol.flush()
    ok = r["ok"]
    f550 = r["flux"][:, NORM_INDEX].astype(np.float64)
    good = ok & np.isfinite(f550) & (f550 > 0)
    sigma_rel = np.full((n, 343), np.nan, np.float32)
    sigma_rel[good] = (r["sigma_abs"][good].astype(np.float64) / f550[good, None])
    sigma_rel[~good] = 0.0
    np.save(root / "xp343_sigma.npy", sigma_rel)
    side = np.full((n, 4), np.nan, np.float32)
    side[:, 0] = g_cat
    side[:, 1] = bp_rp
    side[good, 2] = np.log10(f550[good])
    side[good, 3] = np.log10(f550[good] / r["sigma_abs"][good, NORM_INDEX])
    np.save(root / "side.npy", side)
    np.save(root / "xpcov_valid.npy", good)

    # ---- 4. verification vs coef110.npy ------------------------------------
    log("[4/5] verification")
    c110 = np.load(root / "coef110.npy")[:n].astype(np.float64)
    both = r["found"] & np.isfinite(c110).all(1)
    d = np.abs(r["coef"][both] - c110[both])
    rel_el = d / np.maximum(np.abs(r["coef"][both]), 1e-12)
    rel_norm = d / np.abs(r["coef"][both]).max(1, keepdims=True)
    meta = {
        "side_columns": SIDE_COLS,
        "sources": {
            "phot_g_mean_mag": "catalog (ASPCAP/gxp G)",
            "bp_rp": "gxp_chunk 'bp_rp' (Gaia DR3), NaN if absent",
            "log10_flux550": "log10 of design-matrix-absolute merged flux at 550nm "
                             "from raw h5 coefficients (units of the gaiaxpy "
                             "design matrix, W nm-1 m-2)",
            "log10_snr550": "flux_550 / sigma_550, sigma from full BP/RP covariance",
        },
        "n": int(n), "n_found_in_raw": int(r["found"].sum()),
        "n_valid": int(good.sum()),
        "n_jitter_any": int((r["level"] > 0).any(1).sum()),
        "n_jitter_bp": int((r["level"][:, 0] > 0).sum()),
        "n_jitter_rp": int((r["level"][:, 1] > 0).sum()),
        "n_chol_failed": int((r["found"] & ~r["ok"]).sum()),
        "n_not_found": int((~r["found"]).sum()),
        "coef_vs_coef110": {
            "n_compared": int(both.sum()),
            "max_rel_elementwise": float(rel_el.max()) if both.any() else None,
            "p99.99_rel_elementwise": float(np.percentile(rel_el, 99.99)) if both.any() else None,
            "max_abs_over_star_max_coef": float(rel_norm.max()) if both.any() else None,
            "median_abs_over_star_max_coef": float(np.median(rel_norm)) if both.any() else None,
        },
    }
    log(json.dumps(meta, indent=1))

    if args.verify_n:
        rng = np.random.default_rng(args.seed)
        pick = np.sort(rng.choice(np.where(good)[0], args.verify_n, replace=False))
        _dump_verify(root / "xpnoise_verify_raw.npz", sids[pick], files, r, pick, f550,
                     np.load(root / "xp343.npy", mmap_mode="r")[pick])

    # ---- 5. donor bank ------------------------------------------------------
    if not args.no_bank:
        log("[5/5] noise donor bank")
        build_bank(root, cand_sid, cand_g, hist, files, args, dm_path)
    meta["bank"] = None if args.no_bank else "noise_bank.npz"
    (root / "xpnoise_meta.json").write_text(json.dumps(meta, indent=2))
    log("done.")


def _dump_verify(path, sids, files, r, pick, f550, xp343):
    raw = {k: [] for k in ("bp_coefficients", "bp_coefficient_errors",
                           "bp_coefficient_correlations", "bp_standard_deviation",
                           "rp_coefficients", "rp_coefficient_errors",
                           "rp_coefficient_correlations", "rp_standard_deviation")}
    lo = np.array([f[0] for f in files])
    for s in sids:
        f = files[int(np.searchsorted(lo, s >> 43, side="right") - 1)][2]
        with h5py.File(f, "r") as h:
            row = int(np.searchsorted(h["source_id"][:], s))
            for k in raw:
                raw[k].append(h[k][row])
    np.savez(path, source_id=sids, my_sigma_abs=r["sigma_abs"][pick],
             my_flux_abs=r["flux"][pick], my_f550=f550[pick], xp343=np.asarray(xp343),
             **{k: np.stack(v) for k, v in raw.items()})


def build_bank(root, cand_sid, cand_g, hist, files, args, dm_path):
    rng = np.random.default_rng(args.seed + 1)
    nb = len(BANK_EDGES) - 1
    b = np.digitize(cand_g, BANK_EDGES) - 1
    pick = []
    for k in range(nb):
        idx = np.where(b == k)[0]
        take = min(len(idx), int(args.bank_per_bin * 1.15) + 5)   # slack for invalid
        pick.append(rng.choice(idx, take, replace=False))
        if len(idx) < args.bank_per_bin:
            log(f"  WARNING bin {k} has only {len(idx)} candidates")
    pick = np.concatenate(pick)
    sid, g = cand_sid[pick], cand_g[pick]
    chol = np.zeros((len(sid), N_PACK), np.float32)
    r = extract_raw(sid, files, args.nproc, chol, dm_path)
    f550 = r["flux"][:, NORM_INDEX]
    keep = r["ok"] & np.isfinite(f550) & (f550 > 0)
    sid, g, chol, bb = sid[keep], g[keep], chol[keep], np.digitize(g[keep], BANK_EDGES) - 1
    sel = []
    for k in range(nb):                                    # trim to bank_per_bin
        idx = np.where(bb == k)[0]
        sel.append(idx[:args.bank_per_bin])
    sel = np.concatenate(sel)
    bb = bb[sel]
    counts = np.bincount(bb, minlength=nb)
    np.savez(root / "noise_bank.npz", chol=chol[sel], g=g[sel], source_id=sid[sel],
             bin=bb, bin_edges=BANK_EDGES, bin_count=counts,
             bin_start=np.concatenate([[0], np.cumsum(counts)[:-1]]),
             hist_edges=HIST_EDGES, hist_counts=hist,
             n_jitter=int((r["level"][keep][sel] > 0).any(1).sum()))
    log(f"  bank: {len(sel)} donors in {nb} bins (min/max per bin "
        f"{counts.min()}/{counts.max()}); XP catalog stars in hist: {hist.sum()}")


if __name__ == "__main__":
    main()
