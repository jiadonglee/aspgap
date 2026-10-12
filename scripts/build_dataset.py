#!/usr/bin/env python
"""Build the processed training dataset from astraAllStarASPCAP + local Gaia XP
coefficient chunks.

    python scripts/build_dataset.py \
        --aspcap <astraAllStarASPCAP-v0.6.0.fits> \
        --xp-chunks ../aspgap/data/xpnorm \
        --out data/processed/aspcap_v0.6.0 \
        [--config configs/base.yaml] [--hdu 2] [--limit 20000]

Steps: clean -> per-label masks -> dedupe by source_id -> XP crossmatch ->
343-pixel sampling (needs gaiaxpy once, then cached) -> splits + domain tags
-> LabelScaler fit on TRAIN only -> catalog/arrays/manifest.

The gaiaxpy call builds the (55,343) design matrices exactly once; later runs
reuse <out>/xp_design_343.npz.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aspgapv3.config import load_config, save_config  # noqa: E402
from aspgapv3.data.dataset import LabelScaler, save_catalog  # noqa: E402
from aspgapv3.data.splits import (assign_iid_splits, assign_sky_holdout,  # noqa: E402
                                  domain_tags)
from aspgapv3.data.xp import build_design_matrix, sample_spectra  # noqa: E402
from aspgapv3.labels.loaders import load_and_clean  # noqa: E402

# columns of labels.npz (build_ext_dataset.py relies on this order); the model trains on teff, logg, m_h, alpha_m
BUILD_LABELS = ("teff", "logg", "m_h", "alpha_m", "fe_h", "mg_h")

META = {"phot_g_mean_mag": np.nan, "ra": np.nan, "dec": np.nan,
        "snr_bp": np.nan, "snr_rp": np.nan}

# The gxp_chunk_*.hdf5 'xp' field stores the OLD ASPGAP network input:
# raw DR3 coefficients divided by 10^((15-G)*0.4), then RobustScaler-
# transformed per photometer (predict_220M_cat.ipynb). Both steps are
# affine, so the raw coefficients are recovered exactly:
#     g_norm = scaled * scale_ + center_;  raw = g_norm * 10^((15-G)*0.4)
def invert_aspgap_coef(xp_scaled: np.ndarray, g: np.ndarray,
                       scaler_bp, scaler_rp) -> np.ndarray:
    gn = np.empty_like(xp_scaled, dtype=np.float64)
    gn[:, :55] = xp_scaled[:, :55] * scaler_bp.scale_ + scaler_bp.center_
    gn[:, 55:] = xp_scaled[:, 55:] * scaler_rp.scale_ + scaler_rp.center_
    gmag_norm = 10.0 ** ((15.0 - g) * 0.4)
    return gn * gmag_norm[:, None]


def match_xp_chunks(chunk_dir: Path, source_ids: np.ndarray,
                    scaler_bp, scaler_rp, log) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Two passes over gxp_chunk_*.hdf5: locate wanted source_ids, then pull
    their coefficients + metadata. Returns (catalog_positions, coef110_raw
    [n,110], coef110_aspgap [n,110], meta dict) for the matched subset. The
    raw coefficients are the inverse of the AspGap normalization and are what
    the 343-pixel sampling consumes; coef110_aspgap is the AspGap-v1 network input verbatim (coef110n)."""
    wanted = {int(s): i for i, s in enumerate(source_ids)}
    files = sorted(chunk_dir.glob("gxp_chunk_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no gxp_chunk_*.hdf5 under {chunk_dir}")

    where: dict[int, tuple[str, int]] = {}
    t0 = time.time()
    for j, f in enumerate(files):
        with h5py.File(f, "r") as h:
            ids = h["table/columns/source_id/data"][:]
        for row in np.where(np.isin(ids, source_ids, assume_unique=False))[0]:
            sid = int(ids[row])
            if sid not in where:
                where[sid] = (str(f), int(row))
        if j % 400 == 0:
            log(f"  scan {j + 1}/{len(files)}: matched {len(where)}/{len(wanted)} "
                f"({time.time() - t0:.0f}s)")
    log(f"  crossmatch: {len(where)}/{len(wanted)} ASPCAP stars found in XP chunks")

    n = len(where)
    order = sorted(where, key=lambda s: wanted[s])          # catalog order
    dense = {s: k for k, s in enumerate(order)}             # index into arrays
    scaled = np.full((n, 110), np.nan, dtype=np.float32)
    meta = {k: np.full(n, np.nan) for k in META}
    by_file: dict[str, list[int]] = {}
    for s in order:
        f, r = where[s]
        by_file.setdefault(f, []).append(r)
    filled = 0
    for f, rws in by_file.items():
        rows = np.array(sorted(set(rws)))
        row_pos = {int(r): j for j, r in enumerate(rows)}
        with h5py.File(f, "r") as h:
            n_file = h["table/columns/xp/data"].shape[0]
            # fancy-index only the needed rows, unless we need most of the file
            if len(rows) > 0.5 * n_file:
                xp = h["table/columns/xp/data"][:]
                sids = h["table/columns/source_id/data"][:]
            else:
                xp = h["table/columns/xp/data"][rows, :]
                sids = h["table/columns/source_id/data"][rows]
            col_groups = h["table/columns"]
            cols = set(col_groups.keys())
            for r in rws:
                j = row_pos[r]
                sid = int(sids[j])
                i = dense[sid]
                scaled[i] = xp[j]
                for k in META:
                    if k in cols:
                        meta[k][i] = col_groups[f"{k}/data"][r]
                filled += 1
    log(f"  read coefficients for {filled} stars")

    g = meta["phot_g_mean_mag"]
    have = np.isfinite(scaled).all(1) & np.isfinite(g)
    raw = np.full((n, 110), np.nan, dtype=np.float64)
    if have.any():
        raw[have] = invert_aspgap_coef(scaled[have].astype(np.float64),
                                       g[have], scaler_bp, scaler_rp)
    raw[~have] = np.nan
    pos = np.array([wanted[s] for s in order], dtype=np.int64)
    return pos, raw.astype(np.float32), scaled, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aspcap", required=True,
                    help="astraAllStarASPCAP FITS (or pre-converted parquet)")
    ap.add_argument("--xp-chunks", default="../aspgap/data/xpnorm")
    ap.add_argument("--bp-scaler", default="../aspgap/model/scaler_bp_gmagand_0330.gz")
    ap.add_argument("--rp-scaler", default="../aspgap/model/scaler_rp_gmagand_0330.gz")
    ap.add_argument("--out", default="data/processed/aspcap_v0.6.0")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--hdu", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap star count")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)

    # ---- 1-2. ASPCAP: clean + per-label matrices -------------------------
    labels = list(BUILD_LABELS)
    teff_variant = cfg["data"].get("teff_variant", "raw")
    log(f"[1/6] loading ASPCAP table {args.aspcap} (labels={labels}, "
        f"teff_variant={teff_variant})")
    res = load_and_clean(args.aspcap, labels,
                         snr_min=float(cfg["data"]["snr_min"]),
                         teff_max=float(cfg["data"]["teff_max"]),
                         teff_variant=teff_variant,
                         mask_warn_bits=bool(cfg["labels"].get("mask_warn_bits", True)),
                         hdu=args.hdu)
    df, y, yerr, mask = res["df"], res["y"], res["yerr"], res["mask"]
    log(f"    clean: {res['stats']['n_input']} -> {res['stats']['n_clean']} spectra")
    log(f"    label coverage: {res['stats']['labels']}")

    # ---- 3. one row per source_id (max S/N visit) ------------------------
    sid_col = res["resolved"]["source_id"]
    snr = pd.to_numeric(df["snr"], errors="coerce").to_numpy(float)
    order = pd.DataFrame({"sid": df[sid_col].to_numpy(), "snr": snr,
                          "pos": df.index.to_numpy()}) \
        .sort_values("snr", ascending=False).drop_duplicates("sid", keep="first")
    keep = np.sort(order["pos"].to_numpy())
    df = df.loc[keep].reset_index(drop=True)
    y, yerr, mask = y[keep], yerr[keep], mask[keep]
    log(f"[2/6] deduped to {len(df)} unique gaia_dr3_source_id")
    if args.limit:
        df, y, yerr, mask = df.iloc[:args.limit].reset_index(drop=True), \
            y[:args.limit], yerr[:args.limit], mask[:args.limit]

    # ---- 4. XP crossmatch -------------------------------------------------
    import joblib
    log("[3/6] matching source_ids against Gaia XP chunks")
    sids = df[sid_col].to_numpy()
    scaler_bp = joblib.load(args.bp_scaler)
    scaler_rp = joblib.load(args.rp_scaler)
    pos, coef_raw, coef_aspgap, meta = match_xp_chunks(
        Path(args.xp_chunks), sids, scaler_bp, scaler_rp, log)
    have_coef = np.isfinite(coef_raw).all(1)
    df = df.loc[pos].reset_index(drop=True)
    y, yerr, mask = y[pos], yerr[pos], mask[pos]
    coef_raw, coef_aspgap = coef_raw[have_coef], coef_aspgap[have_coef]
    meta = {k: v[have_coef] for k, v in meta.items()}
    df, y, yerr, mask = df[have_coef].reset_index(drop=True), y[have_coef], \
        yerr[have_coef], mask[have_coef]
    log(f"    with recoverable raw coefficients: {len(df)}")

    # ---- 5. 343-pixel sampling --------------------------------------------
    log("[4/6] building/loading XP design matrix (gaiaxpy first time only)")
    t0 = time.time()
    dm = build_design_matrix(out / "xp_design_343.npz")
    log(f"    design matrix ready ({time.time() - t0:.0f}s)")
    log("    sampling 343-pixel spectra from raw coefficients")
    xp343 = sample_spectra(coef_raw, dm)
    bad = (xp343 == 0).all(1)
    log(f"    degenerate 550nm-normalization rows dropped: {int(bad.sum())}")
    xp343, coef_raw, coef_aspgap = xp343[~bad], coef_raw[~bad], coef_aspgap[~bad]
    df, y, yerr, mask = df[~bad].reset_index(drop=True), y[~bad], yerr[~bad], mask[~bad]
    np.save(out / "xp343.npy", xp343)
    np.save(out / "coef110.npy", coef_raw.astype(np.float32))     # raw DR3
    np.save(out / "coef110n.npy", coef_aspgap.astype(np.float32))  # old AspGap input

    # ---- 6. splits + tags + scaler ----------------------------------------
    log("[5/6] splits (IID + sky holdout) and domain tags")
    colmap = {n: i for i, n in enumerate(labels)}
    from aspgapv3.labels.loaders import resolve_columns
    rescol = resolve_columns(df)
    gcol = rescol.get("g_mag")
    teff_cal_col = rescol.get("teff_calibrated")
    cat = pd.DataFrame({
        "row": np.arange(len(df), dtype=np.int64),
        "source_id": df[sid_col].to_numpy(),
        "snr": df["snr"].to_numpy(float),
        "phot_g_mean_mag": df[gcol].to_numpy(float) if gcol else meta["phot_g_mean_mag"],
        "ra": df[rescol["ra"]].to_numpy(float) if rescol.get("ra") else meta["ra"],
        "dec": df[rescol["dec"]].to_numpy(float) if rescol.get("dec") else meta["dec"],
        "xp_snr_bp": meta["snr_bp"], "xp_snr_rp": meta["snr_rp"],
    })
    colmap = {n: i for i, n in enumerate(labels)}
    from aspgapv3.labels.loaders import resolve_columns
    rescol = resolve_columns(df)
    teff_cal_col = rescol.get("teff_calibrated")
    for name in set(labels) | {"teff_calibrated"}:
        if name == "teff_calibrated":
            vals = pd.to_numeric(df[teff_cal_col], errors="coerce").to_numpy(float) \
                if teff_cal_col else np.full(len(df), np.nan)
            err = np.full(len(df), np.nan)
        else:
            k = colmap[name]
            vals, err = y[:, k].astype(float), yerr[:, k].astype(float)
        cat[name] = vals
        cat[name + "_err"] = err
        cat[name + "_ok"] = mask[:, k] if name in colmap else np.isfinite(vals)

    sp = cfg["data"]["splits"]
    cat = assign_iid_splits(cat, train=sp["train"], val=sp["val"], test=sp["test"],
                            seed=int(sp["seed"]))
    cat = assign_sky_holdout(cat, n_groups=max(64, int(sp["sky_test_groups"]) * 8),
                             holdout_groups=int(sp["sky_test_groups"]),
                             seed=int(sp["seed"]))
    cat = domain_tags(cat)
    for c in ("is_metal_poor", "is_metal_poor_15", "is_mdwarf", "is_fstar",
              "is_faint_xp"):
        cat[c] = cat[c].fillna(False).astype(bool)

    train_rows = cat["row"][cat["split"] == "train"].to_numpy()
    floor = {n: float(cfg["labels"]["sigma_floor"].get(n, 0.02)) for n in labels}
    scaler = LabelScaler.fit(y[train_rows], yerr[train_rows], mask[train_rows],
                             np.array([floor[n] for n in labels]), labels)
    scaler.save(out)
    np.savez(out / "labels.npz", y=y, yerr=yerr, mask=mask,
             labels=np.array(labels, dtype="U16"))

    # calibrated teff as a separate eval-only column (leakage disclaimer)
    cat_path = save_catalog(cat, out)
    manifest = {
        "aspcap": str(args.aspcap), "hdu": args.hdu,
        "xp_chunks": str(args.xp_chunks),
        "labels": labels, "label_stats": res["stats"]["labels"],
        "teff_variant": teff_variant,
        "sigma_floor": floor,
        "cuts": res["stats"]["cuts"],
        "n_clean_spectra": res["stats"]["n_clean"],
        "n_final": int(len(cat)),
        "splits": cat["split"].value_counts().to_dict(),
        "xp_design": "xp_design_343.npz (grid np.arange(336,1021,2) nm, "
                     "normalized at 550nm = index 107)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    save_config(cfg, out / "build_config.yaml")
    log(f"[6/6] wrote {cat_path}, xp343.npy, coef110.npy, labels.npz, norm.npz, "
        f"manifest.json")
    log(f"    splits: {manifest['splits']}")
    log("done.")


if __name__ == "__main__":
    main()
