#!/usr/bin/env python
"""Extend the APOGEE training set with external (non-APOGEE) metal-poor / dwarf / hot-star labels.

    python scripts/build_ext_dataset.py [--nproc 8] [--threads 4]

Output root data/processed/aspcap_v0.6.0_ext/ = the APOGEE root (rows 0..N-1 UNCHANGED, same splits) + appended
rows (split == "train") from three external sources, in priority order GALAH > GC > LAMOST VMP:

  galah       GALAH DR4 stars not in APOGEE (ood_galah.parquet; quality flags of build_ood.py; labels on the APOGEE
              scale through the fitted galah_scale): Teff, logg, [M/H], [alpha/M]
  gc          globular-cluster members (ood_cluster_gc.parquet): [M/H] ONLY (cluster-level; Teff/logg/[alpha/M] masked).
              [M/H] = median APOGEE-member [M/H] of the cluster if it has >=5 APOGEE members, else the Harris->APOGEE fit.
  lamost_vmp  Yuan+2020 [Fe/H] < -2 stars: [M/H] ONLY (= -0.055 + 0.909 [Fe/H]; Teff/logg are on the LAMOST scale -> masked)

Hold-out (the OOD test sets stay usable):
  galah, lamost_vmp   HEALPix nside=32 pixels hashed (SEED) -> 70% train / 30% held out
  gc                  clusters sorted by Harris [Fe/H], every consecutive triple contributes one random held-out cluster
                      (omega Cen, NGC 5139, is held out: it has a real [Fe/H] spread, so a cluster-level label is wrong)
                      training members: G < 17.5, RUWE < 1.4, |C*| < 3 sigma(C*) (Riello+21), outside the crowded core
                      (r >= max(0.5', 20th percentile of the cluster's member radii) for clusters with >= 100 members),
                      at most 300 per cluster
  open clusters, LAMOST hot, StarHorse: never used
ood_split.parquet records (source_id, component, role in {train, heldout, excluded}); the held-out OOD test of a
component = rows with role == heldout whose source_id is not in ANY training set (eval_ood.py --holdout).

Photometry/astrometry for the new rows: Gaia TAP exactly as build_astrometry_phot.py; XP raw coefficients + covariance
(xp343_sigma, side, xpcov_valid for the noise augmentation) from the raw DR3 h5 files exactly as build_xp_noise.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
ROOT = REPO / "data/processed/aspcap_v0.6.0"
EXT1 = REPO / "data/processed/aspcap_v0.6.0_ext"
EXT = EXT1                                               # output root; reassigned to ..._ext2 by `--variant ext2` (see main)
EXT2 = REPO / "data/processed/aspcap_v0.6.0_ext2"
RAW = REPO / "data/raw/ood"
OODDIR = REPO / "data/processed/ood"
LABELS = ["teff", "logg", "m_h", "alpha_m", "fe_h", "mg_h"]       # labels.npz column order
NSIDE_LEVEL = 5                                                     # nside = 32
SEED = 20241009
P_TRAIN = 0.70
GC_GMAX, GC_CAP, GC_RUWE = 17.5, 300, 1.4
OMEGA_CEN = "NGC 5139"
FLOOR_EXTRA_MH_APO, E_MH_GC_FIT = 0.08, 0.10
VMP_E_MH_MAX = 0.35
# ext2: stricter GC training members (everything else, incl. held-out clusters, GALAH and VMP splits, is identical to ext)
GC2_CAP = 100                      # members per cluster (random, SEED)
GC2_EBV_MAX = 0.5                  # Harris 2010 E(B-V): drop heavily reddened clusters
GC2_PM_CHI2 = 9.0                  # per-star PM chi2 (2 dof) vs the VB21 cluster mean PM
GC2_PLX_NSIG = 3.0                 # |plx - plx_cl| < 3 sigma
GC2_PLX_SYS = 0.02                 # mas, Gaia parallax systematics added in quadrature
GC2_PM_SINT_FLOOR = 0.03           # mas/yr, floor of the per-cluster intrinsic PM dispersion


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ split helpers
def healpix32(source_id) -> np.ndarray:
    """HEALPix nested index at nside=32 from a Gaia DR3 source_id (level-12 index = source_id >> 35)."""
    return (np.asarray(source_id, np.int64) >> (35 + 2 * (12 - NSIDE_LEVEL))).astype(np.int64)


def hash_unit(pix: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Deterministic splitmix64 hash of the pixel index -> uniform [0, 1)."""
    off = np.uint64((int(seed) * 0x9E3779B97F4A7C15 + 0x9E3779B97F4A7C15) % (1 << 64))
    with np.errstate(over="ignore"):
        return _mix(pix.astype(np.uint64) + off)


def _mix(z):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def healpix_train_mask(source_id, p_train: float = P_TRAIN, seed: int = SEED) -> np.ndarray:
    return hash_unit(healpix32(source_id), seed) < p_train


def split_clusters(feh: dict, seed: int = SEED, force_heldout=(OMEGA_CEN,)) -> set:
    """Held-out cluster names: sorted by [Fe/H], each consecutive triple gives one random held-out cluster
    (a trailing partial group is held out with probability 1/3); clusters in `force_heldout` are held out
    and left out of the grouping."""
    rng = np.random.default_rng(seed)
    names = [n for n in sorted(feh, key=lambda k: (feh[k], k)) if n not in force_heldout]
    held = set(n for n in force_heldout if n in feh)
    for i in range(0, len(names), 3):
        grp = names[i:i + 3]
        if len(grp) == 3:
            held.add(grp[int(rng.integers(3))])
        elif rng.random() < 1.0 / 3.0:
            held.add(grp[int(rng.integers(len(grp)))])
    return held


def c_star(bp_rp, c_ex, g):
    """Riello+2021 corrected BP/RP flux excess C* and its sigma(G)."""
    c = np.asarray(bp_rp, float)
    f = np.where(c < 0.5, 1.154360 + 0.033772 * c + 0.032277 * c ** 2,
                 np.where(c < 4.0, 1.162004 + 0.011464 * c + 0.049255 * c ** 2 - 0.005879 * c ** 3,
                          1.057572 + 0.140537 * c))
    return np.asarray(c_ex, float) - f, 0.0059898 + 8.817481e-12 * np.asarray(g, float) ** 7.618399



# ------------------------------------------------------------------ ext2: GC astrometric consistency
def harris_ebv() -> dict:
    """Harris (2010) E(B-V) per cluster (name key -> E(B-V))."""
    from build_ood import _name_key
    import re
    lines = (RAW / "harris10_mwgc.dat").read_text().splitlines()
    i0 = next(i for i, l in enumerate(lines) if "[Fe/H]" in l and "E(B-V)" in l)
    out = {}
    for l in lines[i0 + 1:]:
        m = re.match(r"^\s*(\S.*?)\s+(-?\d+\.\d\d)\s+(\d+)\s+(\d+\.\d\d)", l)
        if m:
            out[_name_key(m.group(1))] = float(m.group(4))
        elif out and not l.strip():
            break
    return out


def vb21_members() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-star Gaia EDR3 pm/plx of the VB21 members (prob>0.7, qflag>=2), VB21 cluster mean PM table)."""
    import re
    import zipfile
    from build_ood import _name_key
    tab = pd.read_fwf(RAW / "tablea1.dat", colspecs=[(0, 12), (40, 47), (48, 55), (56, 63), (64, 71), (72, 79)],
                      names=["name", "pmra_cl", "e_pmra_cl", "pmdec_cl", "e_pmdec_cl", "corr_cl"])
    tab["name"] = tab["name"].str.strip()
    keys = {_name_key(n): n for n in tab.name}
    z = zipfile.ZipFile(RAW / "vb21_clusters.zip")
    rows = []
    for fn in z.namelist():
        if not fn.startswith("catalogues/") or not fn.endswith(".txt"):
            continue
        cands = [k for k in keys if _name_key(fn[len("catalogues/"):-4]).startswith(k)]
        if not cands:
            continue
        d = pd.read_csv(z.open(fn), sep=r"\s+", comment="#", header=None,
                        names=["source_id", "ra", "dec", "x", "y", "plx", "pmra", "pmdec", "plxe", "pmrae", "pmdece", "pmcorr",
                               "g_mag", "bp_rp", "Sigma", "qflag", "memberprob"])
        d = d[(d.memberprob > 0.7) & (d.qflag >= 2)].copy()
        d["cluster"] = keys[max(cands, key=len)]
        rows.append(d[["source_id", "cluster", "pmra", "pmdec", "pmrae", "pmdece", "pmcorr"]])
    return pd.concat(rows, ignore_index=True), tab


def gc_astrometric_consistency(gc: pd.DataFrame) -> pd.DataFrame:
    """Per star of `gc` (ood_cluster_gc rows): chi2_pm vs the VB21 cluster mean PM (star covariance + cluster-mean covariance +
    per-cluster intrinsic dispersion s_int^2 * I, s_int estimated robustly from the cluster's own members:
    s_int^2 = max(0, (1.4826 MAD)^2 - median(err^2)), averaged over the two axes, floored) and the parallax pull
    (plx - zpt - 1/d_BV21) / sqrt(plx_err^2 + sys^2)."""
    mem, cl = vb21_members()
    m = gc[["source_id", "cluster"]].merge(mem, on=["source_id", "cluster"], how="left")
    c = cl.set_index("name")
    n = len(m)
    dra = m.pmra.to_numpy() - m.cluster.map(c.pmra_cl).to_numpy()
    dde = m.pmdec.to_numpy() - m.cluster.map(c.pmdec_cl).to_numpy()
    ea, ed, rho = m.pmrae.to_numpy(), m.pmdece.to_numpy(), m.pmcorr.to_numpy()
    sint = {}
    for k, g in pd.DataFrame({"cluster": m.cluster, "dra": dra, "dde": dde, "ea": ea, "ed": ed}).dropna().groupby("cluster"):
        v = []
        for d, e in ((g.dra, g.ea), (g.dde, g.ed)):
            mad = 1.4826 * np.median(np.abs(d - np.median(d)))
            v.append(max(0.0, mad ** 2 - np.median(e ** 2)))
        sint[k] = max(float(np.sqrt(np.mean(v))), GC2_PM_SINT_FLOOR)
    s2 = m.cluster.map(sint).to_numpy() ** 2
    ca, cd, cr = m.cluster.map(c.e_pmra_cl).to_numpy(), m.cluster.map(c.e_pmdec_cl).to_numpy(), m.cluster.map(c.corr_cl).to_numpy()
    caa, cdd, cad = ea ** 2 + s2 + ca ** 2, ed ** 2 + s2 + cd ** 2, rho * ea * ed + cr * ca * cd
    det = caa * cdd - cad ** 2
    chi2 = (cdd * dra ** 2 - 2 * cad * dra * dde + caa * dde ** 2) / det
    plx_cl = 1000.0 / gc.dist_pc.to_numpy() if "dist_pc" in gc else 1000.0 / (10.0 ** (gc.mu_true.to_numpy() / 5.0 + 1.0))
    pull = (gc.parallax.to_numpy() - gc.zpt.to_numpy() - plx_cl) / np.sqrt(gc.parallax_error.to_numpy() ** 2 + GC2_PLX_SYS ** 2)
    return pd.DataFrame({"chi2_pm": chi2, "plx_pull": pull, "pm_sint": m.cluster.map(sint).to_numpy()}, index=gc.index)


# ------------------------------------------------------------------ selection
def read_ood(name: str, with_x: bool = False):
    t = pq.read_table(OODDIR / f"ood_{name}.parquet")
    df = t.drop(["x"]).to_pandas()
    if with_x:
        return df, np.asarray(t.column("x").combine_chunks().values, np.float32).reshape(-1, 110)
    return df


def select(meta: dict, v2: bool = False) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Returns (train table with columns source_id source label cols, x (n,110), split table for all components)."""
    apo = pd.read_parquet(ROOT / "catalog.parquet", columns=["source_id", "split", "teff"])
    z = np.load(ROOT / "labels.npz")
    teff_apo = z["y"][:, 0][z["mask"][:, 0]]
    t_lo = float(np.min(teff_apo))
    parts, xs, roles = [], [], []

    # ---- GALAH
    g, xg = read_ood("galah", True)
    tr = healpix_train_mask(g.source_id)
    roles.append(pd.DataFrame({"source_id": g.source_id, "component": "galah", "role": np.where(tr, "train", "heldout")}))
    ok = tr & (g.teff <= 8000) & (g.teff >= t_lo) & g.logg.between(-0.5, 5.5)
    sel = g[ok].copy()
    sel["source"] = "galah"
    parts.append(sel[["source_id", "source", "teff", "logg", "m_h", "alpha_m", "e_teff", "e_logg", "e_m_h", "e_alpha_m",
                      "snr_px_ccd3"]].rename(columns={"snr_px_ccd3": "snr"}))
    xs.append(xg[ok.to_numpy()])
    meta["galah"] = dict(n_pool=len(g), n_train_pix=int(tr.sum()), n_used=int(ok.sum()), n_teff_range_dropped=int((tr & ~ok).sum()))
    del xg

    # ---- GC
    gc, xgc = read_ood("cluster_gc", True)
    clus = gc.groupby("cluster").feh_harris.first().to_dict()
    held = split_clusters(clus)
    gold = {r["cluster"]: r["m_h_apo"] for r in json.loads((OODDIR / "ood_meta.json").read_text())["gc_scale"]["table"]}
    cs, cs_sig = c_star(gc.bp_rp, gc.phot_bp_rp_excess_factor, gc.G)
    good = (gc.G < GC_GMAX) & (gc.ruwe < GC_RUWE) & (np.abs(cs) < 3 * cs_sig) & np.isfinite(cs)
    q20 = gc.groupby("cluster").r_arcmin.transform(lambda r: np.quantile(r, 0.2))
    nmem = gc.groupby("cluster").r_arcmin.transform("size")
    rcut = np.where(nmem >= 100, np.maximum(0.5, q20), 0.0)
    good &= gc.r_arcmin >= rcut
    in_train_cl = ~gc.cluster.isin(held)
    base = good & in_train_cl                                     # ext_r1 quality + core cut (before the cap)
    if v2:
        ebv = pd.Series({k: v for k, v in harris_ebv().items()})
        from build_ood import _name_key
        ebv_c = gc.cluster.map(lambda n: ebv.get(_name_key(n), np.nan)).astype(float)
        astro = gc_astrometric_consistency(gc)
        ok_ebv = (ebv_c <= GC2_EBV_MAX).to_numpy()
        ok_pm = (astro.chi2_pm < GC2_PM_CHI2).to_numpy()
        ok_plx = (astro.plx_pull.abs() < GC2_PLX_NSIG).to_numpy()
        steps = [("train clusters, all members", in_train_cl.to_numpy()), ("+ G<17.5, RUWE, C*, core cut (= ext_r1 quality)", base.to_numpy())]
        cur = base.to_numpy()
        for nm, ok in (("+ E(B-V)_Harris <= 0.5", ok_ebv), ("+ PM chi2 < 9", ok_pm), ("+ parallax within 3 sigma", ok_plx)):
            cur = cur & ok
            steps.append((nm, cur.copy()))
        cand_pool = gc[cur].copy()
        diag = dict(steps=[dict(cut=nm, n_stars=int(k.sum()), n_clusters=int(gc.cluster[k].nunique())) for nm, k in steps])
        # kept vs dropped (among the ext_r1-quality stars)
        b = base.to_numpy()
        grp = {"kept (all cuts)": cur, "dropped by E(B-V)": b & ~ok_ebv, "dropped by PM (E(B-V) ok)": b & ok_ebv & ~ok_pm,
               "dropped by parallax (E(B-V), PM ok)": b & ok_ebv & ok_pm & ~ok_plx}
        diag["groups"] = [dict(group=k, n=int(v.sum()), G_median=float(gc.G[v].median()), bp_rp_median=float(gc.bp_rp[v].median()),
                               ebv_sfd_median=float(gc.ebv_sfd[v].median())) for k, v in grp.items()]
        diag["clusters_dropped_ebv"] = sorted(set(gc.cluster[b & ~ok_ebv]) - set(gc.cluster[b & ok_ebv]))
        diag["pm_sint_median"] = float(astro.pm_sint[b].median())
        # contamination estimate: ext_r1 GC training stars (read from the ext catalogue) failing the tests
        r1 = pd.read_parquet(EXT1 / "catalog.parquet", columns=["source_id", "source"])
        r1 = r1.source_id[r1.source == "gc"]
        i1 = gc.source_id.isin(r1).to_numpy()
        diag["r1"] = dict(n=int(i1.sum()), frac_fail_ebv=float((~ok_ebv[i1]).mean()), frac_fail_pm=float((~ok_pm[i1]).mean()),
                          frac_fail_plx=float((~ok_plx[i1]).mean()), frac_fail_any=float((~(ok_ebv & ok_pm & ok_plx)[i1]).mean()),
                          frac_fail_pm_or_plx=float((~(ok_pm & ok_plx)[i1]).mean()),
                          G_median=float(gc.G[i1].median()), bp_rp_median=float(gc.bp_rp[i1].median()),
                          frac_G_gt_16=float((gc.G[i1] > 16).mean()), ebv_sfd_median=float(gc.ebv_sfd[i1].median()))
        diag["r1_by_ebv_bin"] = [dict(bin=lab, n=int((i1 & m_).sum()), frac_fail_pm_or_plx=float((~(ok_pm & ok_plx))[i1 & m_].mean()) if (i1 & m_).any() else float("nan"))
                                 for lab, m_ in (("E(B-V)<=0.5", ok_ebv), ("E(B-V)>0.5", ~ok_ebv))]
        diag["thresholds"] = dict(ebv_max=GC2_EBV_MAX, pm_chi2=GC2_PM_CHI2, plx_nsig=GC2_PLX_NSIG, plx_sys=GC2_PLX_SYS, cap=GC2_CAP,
                                  pm_sint_floor=GC2_PM_SINT_FLOOR, ebv_source="Harris 2010 (harris10_mwgc.dat)", plx_ref="BV21 distance (1/d_BV21), plx - zpt (Lindegren+21)")
        cap = GC2_CAP
    else:
        cand_pool, cap = gc[base].copy(), GC_CAP
    cand = cand_pool
    rng = np.random.default_rng(SEED)
    cand = pd.concat([d.sample(min(len(d), cap), random_state=int(rng.integers(1 << 31))) for _, d in cand.groupby("cluster")])
    if v2:
        diag["steps"].append(dict(cut=f"+ cap {cap}/cluster (random)", n_stars=len(cand), n_clusters=int(cand.cluster.nunique())))
    role = np.where(gc.cluster.isin(held), "heldout", np.where(gc.source_id.isin(cand.source_id), "train", "excluded"))
    roles.append(pd.DataFrame({"source_id": gc.source_id, "component": "cluster_gc", "role": role, "cluster": gc.cluster}))
    cand["m_h"] = [gold.get(c, m) for c, m in zip(cand.cluster, cand.m_h)]
    cand["e_m_h"] = [FLOOR_EXTRA_MH_APO if c in gold else E_MH_GC_FIT for c in cand.cluster]
    cand["source"] = "gc"
    cand["teff"] = cand["logg"] = cand["alpha_m"] = np.nan
    cand["e_teff"] = cand["e_logg"] = cand["e_alpha_m"] = np.nan
    cand["snr"] = np.nan
    parts.append(cand[["source_id", "source", "teff", "logg", "m_h", "alpha_m", "e_teff", "e_logg", "e_m_h", "e_alpha_m", "snr"]])
    xs.append(xgc[gc.index.get_indexer(cand.index)])
    meta["gc"] = dict(n_clusters=len(clus), held_out_clusters=sorted(held, key=lambda c: clus[c]),
                      train_clusters=sorted([c for c in clus if c not in held], key=lambda c: clus[c]),
                      n_pool=len(gc), n_quality=int(good.sum()), n_train=len(cand), cuts=(diag if v2 else None),
                      n_clusters_apogee_median_label=int(sum(c in gold for c in cand.cluster.unique())))
    del xgc

    # ---- LAMOST VMP
    v, xv = read_ood("lamost_vmp", True)
    tr = healpix_train_mask(v.source_id, seed=SEED + 1) & (v.e_m_h < VMP_E_MH_MAX) & v.m_h.notna()
    roles.append(pd.DataFrame({"source_id": v.source_id, "component": "lamost_vmp",
                               "role": np.where(healpix_train_mask(v.source_id, seed=SEED + 1), np.where(tr, "train", "excluded"), "heldout")}))
    sv = v[tr].copy()
    sv["source"] = "lamost_vmp"
    for c in ("teff", "logg", "alpha_m", "e_teff", "e_logg", "e_alpha_m"):
        sv[c] = np.nan                                           # LAMOST scale != APOGEE: only [M/H]
    sv["snr"] = np.nan
    parts.append(sv[["source_id", "source", "teff", "logg", "m_h", "alpha_m", "e_teff", "e_logg", "e_m_h", "e_alpha_m", "snr"]])
    xs.append(xv[tr.to_numpy()])
    meta["lamost_vmp"] = dict(n_pool=len(v), n_train=int(tr.sum()))

    # ---- merge in priority order GALAH > GC > VMP, drop APOGEE catalogue stars
    tab = pd.concat(parts, ignore_index=True)
    X = np.concatenate(xs)
    prio = tab.source.map({"galah": 0, "gc": 1, "lamost_vmp": 2})
    order = np.argsort(prio.to_numpy(), kind="stable")
    tab, X = tab.iloc[order].reset_index(drop=True), X[order]
    keep = ~tab.source_id.duplicated().to_numpy() & ~tab.source_id.isin(apo.source_id).to_numpy()
    meta["n_dup_dropped"] = int((~keep).sum())
    tab, X = tab[keep].reset_index(drop=True), X[keep]
    split = pd.concat(roles, ignore_index=True)
    return tab, X, split


# ------------------------------------------------------------------ build
def fetch_astro(ids: np.ndarray, threads: int) -> pd.DataFrame:
    import build_astrometry_phot as bap
    cdir = EXT / "astrometry_chunks"
    cdir.mkdir(parents=True, exist_ok=True)
    ids = np.unique(ids)
    todo = [(ids[s:s + bap.CHUNK], cdir / f"{k:04d}.parquet") for k, s in enumerate(range(0, len(ids), bap.CHUNK))]
    todo = [(i, p) for i, p in todo if not p.exists()]
    log(f"Gaia/2MASS/WISE: {len(ids)} ids, {len(todo)} chunks to fetch")
    with ThreadPoolExecutor(threads) as pool:
        for k, path in enumerate(pool.map(lambda j: bap.fetch(*j) or j[1], todo)):
            if k % 10 == 0:
                log("  chunk", path.name)
    t = pd.concat([pd.read_parquet(p) for p in sorted(cdir.glob("*.parquet"))], ignore_index=True).drop_duplicates("source_id")
    miss = np.setdiff1d(ids, t.source_id.to_numpy())             # chunk files are keyed by position: a copied cache can lack ids
    if len(miss):
        log(f"  {len(miss)} ids missing from the cached chunks: fetching")
        extra = [(miss[s:s + bap.CHUNK], cdir / f"x{k:04d}.parquet") for k, s in enumerate(range(0, len(miss), bap.CHUNK))]
        for i, pth in extra:
            bap.fetch(i, pth)
        t = pd.concat([t] + [pd.read_parquet(p) for _, p in extra], ignore_index=True).drop_duplicates("source_id")
    t = bap.derive(t)
    t["zpt"], t["zpt_method"] = bap.parallax_zero_point(t)
    t["ebv_sfd"] = bap.sfd_ebv(t, str(REPO / "data/dustmaps/sfd"))
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--variant", choices=["ext", "ext2"], default="ext",
                    help="ext2: stricter GC training members (E(B-V), PM/parallax consistency, cap 100) -> data/processed/aspcap_v0.6.0_ext2; "
                         "the held-out clusters / GALAH / VMP splits are identical; ood_split_ext2.parquet is written next to ood_split.parquet")
    args = ap.parse_args()
    global EXT
    import build_xp_noise as bxn
    from aspgapv3.data.xp import NORM_INDEX, aspgap_scaler_arrays, build_design_matrix, coef_to_aspgap_input
    from aspgapv3.data.xpnoise import N_PACK, parse_h5_range
    import glob

    v2 = args.variant == "ext2"
    if v2:
        EXT = EXT2
        if not (EXT / "astrometry_chunks").exists():                 # reuse the already-downloaded Gaia chunks
            EXT.mkdir(parents=True, exist_ok=True)
            shutil.copytree(EXT1 / "astrometry_chunks", EXT / "astrometry_chunks")
    EXT.mkdir(parents=True, exist_ok=True)
    meta: dict = {"seed": SEED, "p_train": P_TRAIN, "healpix_nside": 2 ** NSIDE_LEVEL, "variant": args.variant}
    tab, X, split = select(meta, v2)
    log("selected", tab.source.value_counts().to_dict())

    # ---- Gaia / 2MASS / WISE
    astro = fetch_astro(tab.source_id.to_numpy(), args.threads)
    tab = tab[tab.source_id.isin(astro.source_id)].reset_index(drop=False)       # keep position for X
    X = X[tab["index"].to_numpy()]
    tab = tab.drop(columns="index")
    astro = tab[["source_id"]].merge(astro, on="source_id", how="left")

    # ---- raw XP coefficients + covariance -> xp343_sigma, side, coef110, valid
    dm_path = str(ROOT / "xp_design_343.npz")
    files = sorted((*parse_h5_range(Path(f).name), f) for f in glob.glob(f"{bxn.RAW_DIR}/XpContinuousMeanSpectrum_*.h5"))
    n = len(tab)
    chol = np.zeros((n, N_PACK), np.float32)
    r = bxn.extract_raw(tab.source_id.to_numpy(np.int64), files, args.nproc, chol, dm_path)
    del chol
    f550 = r["flux"][:, NORM_INDEX].astype(np.float64)
    good = r["ok"] & np.isfinite(f550) & (f550 > 0) & r["found"]
    sigma_rel = np.zeros((n, 343), np.float32)
    sigma_rel[good] = r["sigma_abs"][good].astype(np.float64) / f550[good, None]
    g_cat = astro.phot_g_mean_mag.to_numpy(float)
    side = np.full((n, 4), np.nan, np.float32)
    side[:, 0] = g_cat
    side[good, 2] = np.log10(f550[good])
    side[good, 3] = np.log10(f550[good] / r["sigma_abs"][good, NORM_INDEX])
    # bp_rp (side col 1): from the OOD parquets
    bp = pd.concat([read_ood(c)[["source_id", "bp_rp"]] for c in ("galah", "cluster_gc", "lamost_vmp")]).drop_duplicates("source_id")
    side[:, 1] = tab[["source_id"]].merge(bp, on="source_id", how="left").bp_rp.to_numpy()
    # consistency: raw coefficients reproduce the cached network input (coef110n) of the OOD parquet
    sc = aspgap_scaler_arrays(str(ROOT))
    xcheck = coef_to_aspgap_input(r["coef"][good], g_cat[good], sc)
    d = np.abs(xcheck - X[good])
    meta["coef_vs_x_check"] = dict(median_abs=float(np.median(d)), p99_9=float(np.quantile(d, 0.999)), max=float(d.max()))
    log("raw coef -> coef110n check", meta["coef_vs_x_check"])
    meta["n_dropped_no_valid_xp"] = int((~good).sum())
    sel = np.where(good)[0]
    tab, X, astro = tab.iloc[sel].reset_index(drop=True), X[sel], astro.iloc[sel].reset_index(drop=True)
    coef, sigma_rel, side, valid = r["coef"][sel].astype(np.float32), sigma_rel[sel], side[sel], np.ones(len(sel), bool)
    n = len(tab)

    # ---- assemble the extended root
    cat0 = pd.read_parquet(ROOT / "catalog.parquet")
    N = len(cat0)
    assert (cat0.row.to_numpy() == np.arange(N)).all()
    assert not tab.source_id.isin(cat0.source_id).any(), "external star in the APOGEE catalogue"
    rows = np.arange(N, N + n)
    new = pd.DataFrame({"row": rows, "source_id": tab.source_id.to_numpy(), "snr": tab.snr.to_numpy(float),
                        "phot_g_mean_mag": astro.phot_g_mean_mag.to_numpy(float),
                        "ra": astro.ra.to_numpy(float), "dec": astro.dec.to_numpy(float),
                        "split": "train", "sky_group": -1})
    ymat = np.zeros((n, 6), np.float32)
    emat = np.zeros((n, 6), np.float32)
    mmat = np.zeros((n, 6), bool)
    for j, lab in enumerate(LABELS[:4]):
        v = tab[lab].to_numpy(float)
        e = tab["e_" + lab].to_numpy(float)
        m = np.isfinite(v) & np.isfinite(e)
        ymat[m, j], emat[m, j], mmat[m, j] = v[m], e[m], True
        new[lab], new[lab + "_err"], new[lab + "_ok"] = np.where(m, v, np.nan), np.where(m, e, np.nan), m
    for lab in ("fe_h", "mg_h"):
        new[lab], new[lab + "_err"], new[lab + "_ok"] = np.nan, np.nan, False
    new["teff_calibrated"], new["teff_calibrated_err"], new["teff_calibrated_ok"] = np.nan, np.nan, False
    new["xp_snr_bp"] = new["xp_snr_rp"] = np.nan
    new["source"] = tab.source.to_numpy()
    for c in ("is_metal_poor", "is_metal_poor_15", "is_mdwarf", "is_fstar", "is_faint_xp"):
        new[c] = False
    new["is_metal_poor"] = (new.m_h < -1.0).to_numpy()
    new["is_metal_poor_15"] = (new.m_h < -1.5).to_numpy()
    cat0["source"] = "apogee"
    cat = pd.concat([cat0, new[[c for c in cat0.columns if c in new.columns]]], ignore_index=True)
    cat["sky_group"] = cat["sky_group"].astype(np.int64)
    cat.to_parquet(EXT / "catalog.parquet")
    z = np.load(ROOT / "labels.npz")
    np.savez(EXT / "labels.npz", y=np.concatenate([z["y"], ymat]), yerr=np.concatenate([z["yerr"], emat]),
             mask=np.concatenate([z["mask"], mmat]), labels=z["labels"])
    for nm, add in (("coef110n", X), ("coef110", coef)):
        out = np.lib.format.open_memmap(EXT / f"{nm}.npy", mode="w+", dtype=np.float32, shape=(N + n, 110))
        out[:N] = np.load(ROOT / f"{nm}.npy", mmap_mode="r")
        out[N:] = add
        out.flush()
    out = np.lib.format.open_memmap(EXT / "xp343_sigma.npy", mode="w+", dtype=np.float32, shape=(N + n, 343))
    old = np.load(ROOT / "xp343_sigma.npy", mmap_mode="r")
    for s in range(0, N, 100000):
        out[s:min(s + 100000, N)] = old[s:min(s + 100000, N)]
    out[N:] = sigma_rel
    out.flush()
    for nm, add in (("side", side), ("xpcov_valid", valid)):
        a = np.load(ROOT / f"{nm}.npy")
        np.save(EXT / f"{nm}.npy", np.concatenate([a, add.astype(a.dtype)]))
    # astrometry_phot.parquet: old rows + new
    a0 = pd.read_parquet(ROOT / "astrometry_phot.parquet")
    a1 = astro.copy()
    a1.insert(0, "row", rows)
    a1 = a1[[c for c in a0.columns if c in a1.columns]]
    pd.concat([a0, a1], ignore_index=True).to_parquet(EXT / "astrometry_phot.parquet", index=False)
    for f in ("norm.npz", "manifest.json", "build_config.yaml", "aspgap_scaler.npz", "xp_design_343.npz", "noise_bank.npz", "xpnoise_meta.json"):
        shutil.copy(ROOT / f, EXT / f)

    # ---- split table + manifest
    if v2:
        old = pd.read_parquet(OODDIR / "ood_split.parquet")
        h = lambda d: set(map(tuple, d[d.role == "heldout"][["source_id", "component"]].to_numpy()))
        assert h(old) == h(split), "held-out rows differ from ext_r1"
        split.to_parquet(OODDIR / "ood_split_ext2.parquet", index=False)
    else:
        split.to_parquet(OODDIR / "ood_split.parquet", index=False)
    meta["n_ext_train_rows"] = int(n)
    meta["n_by_source"] = tab.source.value_counts().to_dict()
    meta["label_availability"] = {s: {l: int(mmat[(tab.source == s).to_numpy(), j].sum()) for j, l in enumerate(LABELS)}
                                  for s in meta["n_by_source"]}
    (EXT / "ext_manifest.json").write_text(json.dumps(meta, indent=1, default=float))
    log("done", json.dumps({k: meta[k] for k in ("n_ext_train_rows", "n_by_source", "label_availability")}, default=float))


if __name__ == "__main__":
    main()
