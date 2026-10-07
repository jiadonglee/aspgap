#!/usr/bin/env python
"""Fetch Gaia DR3 astrometry/photometry + 2MASS + AllWISE for every catalog star.

    python scripts/build_astrometry_phot.py --root data/processed/aspcap_v0.6.0

No local table carries these columns (catalog.parquet has only G, ra, dec), so the
Gaia archive is queried read-only by source_id (synchronous TAP, chunks cached under
<root>/astrometry_chunks/). Cross-matches are Gaia's own best-neighbour tables:
    gaiadr3.tmass_psc_xsc_best_neighbour -> gaiadr1.tmass_original_valid (2MASS PSC)
    gaiadr3.allwise_best_neighbour       -> gaiadr1.allwise_original_valid (AllWISE)
Output <root>/astrometry_phot.parquet: one row per catalog row (same order, with `row`
and `source_id`), plus derived columns
    tmass_ok_{j,h,k}  ph_qual in {A,B};   wise_ok_{w1,w2}  ph_qual in {A,B}, cc_flags[0:2]=='00', ext_flg==0
    e_{g,bp,rp}       Gaia mag errors = 2.5/ln10 / flux_over_error
Quality flags are MASKS, never row drops. Rows with no Gaia match keep NaN.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYNC = "https://gea.esac.esa.int/tap-server/tap/sync"
CHUNK = 1500
# The 4-way server-side join is far too slow for the 60 s sync limit, so each chunk is five
# fast queries merged here.
Q_GAIA = """SELECT source_id, ra, dec, l, b, ecl_lat, parallax, parallax_error, ruwe,
       astrometric_params_solved, nu_eff_used_in_astrometry, pseudocolour, visibility_periods_used,
       phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
       phot_g_mean_flux_over_error AS g_foe, phot_bp_mean_flux_over_error AS bp_foe,
       phot_rp_mean_flux_over_error AS rp_foe
FROM gaiadr3.gaia_source WHERE source_id IN ({ids})"""
Q_TB = """SELECT source_id, original_ext_source_id, angular_distance AS tm_sep, number_of_neighbours AS tm_nn
FROM gaiadr3.tmass_psc_xsc_best_neighbour WHERE source_id IN ({ids})"""
Q_WB = """SELECT source_id, allwise_oid, number_of_neighbours AS w_nn, number_of_mates AS w_nmates
FROM gaiadr3.allwise_best_neighbour WHERE source_id IN ({ids})"""
Q_TM = """SELECT designation AS original_ext_source_id, j_m, j_msigcom, h_m, h_msigcom, ks_m, ks_msigcom,
       ph_qual AS tm_qual FROM gaiadr1.tmass_original_valid WHERE designation IN ({ids})"""
Q_W = """SELECT allwise_oid, w1mpro, w1mpro_error, w2mpro, w2mpro_error, ph_qual AS w_qual,
       cc_flags AS w_cc, ext_flag AS w_ext FROM gaiadr1.allwise_original_valid WHERE allwise_oid IN ({ids})"""


def tap(query: str) -> pd.DataFrame:
    for attempt in range(6):
        try:
            r = requests.post(SYNC, timeout=300, data=dict(
                REQUEST="doQuery", LANG="ADQL", FORMAT="csv", MAXREC=100000, QUERY=query))
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text), dtype={"tm_qual": str, "w_qual": str, "w_cc": str,
                                                           "original_ext_source_id": str})
        except (requests.RequestException, pd.errors.ParserError):
            if attempt == 5:
                raise
            time.sleep(5)


def fetch(ids, path: Path) -> None:
    t = tap(Q_GAIA.format(ids=",".join(map(str, ids))))
    if not set(ids) <= set(t.source_id):
        raise RuntimeError(f"{path.name}: {len(set(ids) - set(t.source_id))} sources missing")
    tb = tap(Q_TB.format(ids=",".join(map(str, ids)))).sort_values("tm_sep").drop_duplicates("source_id")
    wb = tap(Q_WB.format(ids=",".join(map(str, ids)))).drop_duplicates("source_id")
    des = tb["original_ext_source_id"].dropna().astype(str).unique()
    oid = wb["allwise_oid"].dropna().astype("int64").unique()
    tm = tap(Q_TM.format(ids=",".join(f"'{d}'" for d in des))).drop_duplicates("original_ext_source_id") \
        if len(des) else pd.DataFrame(columns=["original_ext_source_id"])
    w = tap(Q_W.format(ids=",".join(map(str, oid)))).drop_duplicates("allwise_oid") \
        if len(oid) else pd.DataFrame(columns=["allwise_oid"])
    tb = tb.merge(tm, on="original_ext_source_id", how="left")
    wb = wb.merge(w, on="allwise_oid", how="left")
    t = t.merge(tb, on="source_id", how="left").merge(wb, on="source_id", how="left")
    t.to_parquet(path, index=False)


def derive(t: pd.DataFrame) -> pd.DataFrame:
    """Quality masks + Gaia mag errors (pure function of the fetched columns)."""
    t = t.copy()
    q = t["tm_qual"].fillna("").astype(str).str.pad(3, fillchar="X")
    for i, b in enumerate("jhk"):
        t[f"tmass_ok_{b}"] = q.str[i].isin(list("AB")) & t[f"{'ks' if b == 'k' else b}_m"].notna()
    wq = t["w_qual"].fillna("").astype(str).str.pad(4, fillchar="X")
    cc = t["w_cc"].fillna("").astype(str).str.pad(4, fillchar="9")
    clean = cc.str[:2].eq("00") & (pd.to_numeric(t["w_ext"], errors="coerce") == 0)
    for i, b in enumerate(("w1", "w2")):
        t[f"wise_ok_{b}"] = wq.str[i].isin(list("AB")) & clean & t[f"{b}mpro"].notna()
    k = 2.5 / np.log(10.0)
    for b in ("g", "bp", "rp"):
        t[f"e_{b}"] = k / t[f"{b}_foe"].where(t[f"{b}_foe"] > 0)
    return t


def parallax_zero_point(t: pd.DataFrame, fallback: float = -0.017):
    """Lindegren+2021 DR3 zero-point [mas] (corrected parallax = parallax - zpt) from the
    `gaiadr3-zeropoint` package; rows outside its validity range (6<G<21, nu_eff 1.1-1.9 for 5-p,
    pseudocolour 1.24-1.72 for 6-p, astrometric_params_solved 31/95) get the global constant
    `fallback`.  Returns (zpt, method) with method 1 = Lindegren, 0 = constant."""
    from zero_point import zpt as zp
    zp.load_tables()
    n = len(t)
    g, nu, pc = (t[c].to_numpy(float) for c in ("phot_g_mean_mag", "nu_eff_used_in_astrometry", "pseudocolour"))
    ecl, sol = t["ecl_lat"].to_numpy(float), t["astrometric_params_solved"].to_numpy(float)
    ok = (((sol == 31) & (nu > 1.1) & (nu < 1.9)) | ((sol == 95) & (pc > 1.24) & (pc < 1.72))) \
        & (g > 6) & (g < 21) & np.isfinite(ecl)
    out = np.full(n, fallback)
    if ok.any():
        out[ok] = zp.get_zpt(g[ok], np.where(sol[ok] == 31, nu[ok], np.nan),
                             np.where(sol[ok] == 95, pc[ok], np.nan), ecl[ok], sol[ok].astype(int),
                             _warnings=False)
    return out, ok.astype(np.int8)


def sfd_ebv(t: pd.DataFrame, map_dir: str = "data/dustmaps/sfd") -> np.ndarray:
    """SFD98 E(B-V) at (ra, dec) (not recalibrated; x0.86 is applied by the users).  Needs the
    SFD_dust_4096_{ngp,sgp}.fits files in `map_dir` and the `dustmaps` package; NaN otherwise."""
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from dustmaps.sfd import SFDQuery
        q = SFDQuery(map_dir=map_dir)
        return np.asarray(q(SkyCoord(t["ra"].to_numpy(float) * u.deg, t["dec"].to_numpy(float) * u.deg)), float)
    except Exception as exc:
        print("SFD unavailable:", repr(exc))
        return np.full(len(t), np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/aspcap_v0.6.0")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.root)
    cat = pd.read_parquet(root / "catalog.parquet", columns=["row", "source_id"]).sort_values("row")
    ids = np.unique(cat["source_id"].to_numpy(np.int64))
    cdir = root / "astrometry_chunks"
    cdir.mkdir(exist_ok=True)
    todo = [(ids[s:s + CHUNK], cdir / f"{k:04d}.parquet")
            for k, s in enumerate(range(0, len(ids), CHUNK))]
    todo = [(i, p) for i, p in todo if not p.exists()]
    print(f"{len(ids)} unique source_ids, {len(todo)} chunks to fetch", flush=True)
    with ThreadPoolExecutor(args.threads) as pool:
        for k, path in enumerate(pool.map(lambda j: fetch(*j) or j[1], todo)):
            if k % 20 == 0:
                print(path.name, flush=True)
    t = pd.concat([pd.read_parquet(p) for p in sorted(cdir.glob("*.parquet"))], ignore_index=True)
    # best-neighbour tables give one row per source; keep the closest 2MASS match if duplicated
    t = t.drop_duplicates("source_id")
    t = derive(t)
    t["zpt"], t["zpt_method"] = parallax_zero_point(t)
    out = cat.merge(t, on="source_id", how="left", validate="m:1").sort_values("row").reset_index(drop=True)
    out["ebv_sfd"] = sfd_ebv(out)
    out.to_parquet(root / "astrometry_phot.parquet", index=False)
    n = len(out)
    print(f"rows {n}; parallax {out.parallax.notna().sum()}; "
          f"2MASS J/H/K ok {out.tmass_ok_j.sum()}/{out.tmass_ok_h.sum()}/{out.tmass_ok_k.sum()}; "
          f"WISE W1/W2 ok {out.wise_ok_w1.sum()}/{out.wise_ok_w2.sum()}; "
          f"Lindegren ZP {out.zpt_method.sum()}, constant {len(out) - out.zpt_method.sum()}; "
          f"nu_eff {out.nu_eff_used_in_astrometry.notna().sum()}, pseudocolour {out.pseudocolour.notna().sum()}")


if __name__ == "__main__":
    sys.exit(main())
