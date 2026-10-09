#!/usr/bin/env python
"""Build the out-of-distribution (OOD) test sets for the final AspGap-v3 model.

    python scripts/build_ood.py [--components galah cluster_oc cluster_gc lamost_hot lamost_vmp starhorse]
                                [--galah-cap 200000] [--download-only]

Output: data/processed/ood/ood_<component>.parquet (+ ood_meta.json), raw downloads in data/raw/ood/.
Every output row is a star that is NOT in the APOGEE catalog (catalog.parquet, all splits) and has Gaia XP
coefficients in the local v1 chunks; columns:
    source_id component ra dec l b G bp_rp parallax parallax_error zpt ruwe snr_bp snr_rp ebv_sfd a0_sfd
    x (110-d coef110n = the network input, FixedSizeList)
    teff logg m_h alpha_m (+ *_err)   external labels ON THE APOGEE astra-0.6.0 SCALE (NaN if not available)
    <survey raw labels>, cluster / mu_true / A0_ref ..., ood_knn (kNN distance of the TRUE labels to the APOGEE
    train labels, train-std units), flags.

Data sources (all public; URLs):
  GALAH DR4 allstar     https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR4/catalogs/galah_dr4_allstar_240705.fits
  Hunt & Reffert 2023   https://cdsarc.cds.unistra.fr/ftp/J/A+A/673/A114/{members.dat,clusters.dat.gz}  (Prob > 0.7)
  Vasiliev & Baumgardt  https://zenodo.org/records/4891252/files/clusters.zip   (member probability > 0.7, qflag >= 2)
  GC distances, [Fe/H]  https://people.smp.uq.edu.au/HolgerBaumgardt/globular/parameter.html (BV21 distances)
                        https://physwww.mcmaster.ca/~harris/mwgc.dat  (Harris 2010 [Fe/H])
  Xiang+2022 LAMOST hot https://cdsarc.cds.unistra.fr/ftp/J/A+A/662/A66/table5.dat.gz  (Teff >= 7000 K, uniqflag == 1)
  Yuan+2020 LAMOST VMP  VizieR J/ApJ/891/39/table5 (tapvizier), positional cross-match to Gaia DR3 with CDS XMatch
  StarHorse EDR3        VizieR I/354/starhorse2021 (tapvizier, queried by Source id)
  Gaia DR3 astrometry   https://gea.esac.esa.int/tap-server/tap (gaiadr3.gaia_source, read-only, by source_id)
  XP coefficients       local v1 chunks ../aspgap/data/xpnorm/gxp_chunk_*.hdf5 (= coef110n, the v1/v3 input)

Label scale (see fit_scale): GALAH -> APOGEE by a low-order polynomial fit on stars in BOTH surveys (the
overlap stars are removed from the test set).  [M/H] and [alpha/M] from GALAH [Fe/H] and [alpha/Fe] with the
Salaris+1993 relation  [M/H] = [Fe/H] + log10(0.638 f_a + 0.362), f_a = 10^[alpha/Fe]; hence
[alpha/M] = [alpha/Fe] - log10(0.638 f_a + 0.362).
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent
ROOT = REPO / "data/processed/aspcap_v0.6.0"
RAW = REPO / "data/raw/ood"
OUT = REPO / "data/processed/ood"
CACHE = RAW / "cache"
XP_CHUNKS = Path("/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap/data/xpnorm")
LABELS = ["teff", "logg", "m_h", "alpha_m"]
GMAX = 17.65
URL = {
    "galah_dr4_allstar_240705.fits": "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR4/catalogs/galah_dr4_allstar_240705.fits",
    "hr23_members.dat": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/673/A114/members.dat",
    "clusters.dat.gz": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/673/A114/clusters.dat.gz",
    "vb21_clusters.zip": "https://zenodo.org/api/records/4891252/files/clusters.zip/content",
    "bv21_parameter.html": "https://people.smp.uq.edu.au/HolgerBaumgardt/globular/parameter.html",
    "harris10_mwgc.dat": "https://physwww.mcmaster.ca/~harris/mwgc.dat",
    "xiang22_table5.dat.gz": "https://cdsarc.cds.unistra.fr/ftp/J/A+A/662/A66/table5.dat.gz",
}
VIZ_TAP = "http://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in URL.items():
        f = RAW / name
        if f.exists() and f.stat().st_size > 0:
            continue
        log("downloading", url)
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(f.with_suffix(f.suffix + ".part"), "wb") as fh:
                for blk in r.iter_content(1 << 22):
                    fh.write(blk)
        f.with_suffix(f.suffix + ".part").rename(f)


def viz_tap(q: str) -> pd.DataFrame:
    for attempt in range(6):
        try:
            r = requests.post(VIZ_TAP, data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", MAXREC=1000000, QUERY=q),
                              timeout=600)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))


# ------------------------------------------------------------------ label conversion
def mh_from_fe(feh, afe):
    """[M/H] from [Fe/H], [alpha/Fe] (Salaris+1993)."""
    return np.asarray(feh) + np.log10(0.638 * 10.0 ** np.asarray(afe) + 0.362)


def alpha_m_from_afe(afe):
    """[alpha/M] = [alpha/Fe] - (Salaris [M/H]-[Fe/H] offset)."""
    return np.asarray(afe) - np.log10(0.638 * 10.0 ** np.asarray(afe) + 0.362)


def _design(teff, logg, feh, afe):
    t = (np.asarray(teff, float) - 5000.0) / 1000.0
    return np.stack([np.ones_like(t), t, t * t, np.asarray(logg, float) - 3.0, np.asarray(feh, float),
                     np.asarray(afe, float)], 1)


def fit_scale(gal: pd.DataFrame, apo: pd.DataFrame, clip: float = 4.0) -> dict:
    """GALAH -> APOGEE.  For each label, (apo - galah-derived) = poly(Teff_g, logg_g, [Fe/H]_g, [alpha/Fe]_g)
    (1, t, t^2, logg, [Fe/H], [alpha/Fe]) fit with iterative clipping on stars in BOTH surveys.  `galah-derived` is
    the native Teff/logg and the Salaris-converted [M/H], [alpha/M]; the linear terms absorb ASPCAP's own [M/H],
    [alpha/M] convention.  `gal` needs teff logg fe_h afe + LABELS; `apo` the four APOGEE labels (aligned rows)."""
    out = {}
    feat = ("teff", "logg", "fe_h", "afe")
    lo = {c: float(np.nanpercentile(gal[c], 1)) for c in feat}
    hi = {c: float(np.nanpercentile(gal[c], 99)) for c in feat}
    X = _design(*[gal[c].clip(lo[c], hi[c]) for c in feat])
    for lab in LABELS:
        d = apo[lab].to_numpy(float) - gal[lab].to_numpy(float)
        ok = np.isfinite(d) & np.isfinite(X).all(1)
        keep = ok.copy()
        for _ in range(5):
            beta, *_ = np.linalg.lstsq(X[keep], d[keep], rcond=None)
            r = d - X @ beta
            s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
            keep = ok & (np.abs(r) < clip * s)
        mad = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v))))
        out[lab] = dict(beta=beta.tolist(), n=int(ok.sum()), median_offset=float(np.median(d[ok])),
                        mad_before=mad(d[ok]), mad_after=mad(r[ok]), rms_after=float(np.sqrt(np.mean(r[keep] ** 2))))
    out["range"] = dict(lo=lo, hi=hi)
    return out


def apply_scale(gal: pd.DataFrame, sc: dict) -> pd.DataFrame:
    """Offsets applied with every input clipped to the overlap 1-99 percentile range (no extrapolation)."""
    lo, hi = sc["range"]["lo"], sc["range"]["hi"]
    X = _design(*[gal[c].clip(lo[c], hi[c]) for c in ("teff", "logg", "fe_h", "afe")])
    return pd.DataFrame({lab: gal[lab].to_numpy(float) + X @ np.asarray(sc[lab]["beta"]) for lab in LABELS}, index=gal.index)


# ------------------------------------------------------------------ APOGEE side
def load_apogee() -> pd.DataFrame:
    cat = pd.read_parquet(ROOT / "catalog.parquet", columns=["row", "source_id", "snr", "split", "phot_g_mean_mag"])
    z = np.load(ROOT / "labels.npz")
    names = [str(s) for s in z["labels"]]
    for lab in LABELS:
        k = names.index(lab)
        y = z["y"][cat["row"].to_numpy(), k].astype(float)
        cat[lab] = np.where(z["mask"][cat["row"].to_numpy(), k], y, np.nan)
        cat["e_" + lab] = np.where(z["mask"][cat["row"].to_numpy(), k], z["yerr"][cat["row"].to_numpy(), k], np.nan)
    return cat.drop_duplicates("source_id").reset_index(drop=True)


# ------------------------------------------------------------------ components
def comp_galah(apo: pd.DataFrame, meta: dict, cap: int, seed: int = 0) -> pd.DataFrame:
    from astropy.io import fits
    cols = ["gaiadr3_source_id", "sobject_id", "flag_sp", "flag_fe_h", "snr_px_ccd3", "teff", "e_teff", "logg", "e_logg",
            "fe_h", "e_fe_h", "mg_fe", "flag_mg_fe", "si_fe", "flag_si_fe", "ca_fe", "flag_ca_fe", "ti_fe", "flag_ti_fe",
            "ruwe"]
    d = fits.open(RAW / "galah_dr4_allstar_240705.fits", memmap=True)[1].data
    g = pd.DataFrame({c: np.asarray(d[c]).astype(np.int64 if (c in ("gaiadr3_source_id", "sobject_id") or c.startswith("flag_")) else float)
                      for c in cols})
    g = g.rename(columns={"gaiadr3_source_id": "source_id"})
    n0 = len(g)
    a = []
    for x in ("mg", "si", "ca", "ti"):
        v = g[f"{x}_fe"].where(g[f"flag_{x}_fe"] == 0)
        a.append(v)
    A = pd.concat(a, axis=1)
    g["afe"] = A.mean(1).where(A.notna().sum(1) >= 2)           # mean of the unflagged Mg, Si, Ca, Ti [X/Fe]
    q = (g.flag_sp == 0) & (g.flag_fe_h == 0) & (g.snr_px_ccd3 > 30) & (g.e_fe_h < 0.2) & np.isfinite(g.afe) \
        & np.isfinite(g.teff) & np.isfinite(g.logg) & (g.source_id > 0) & (g.ruwe < 1.4)
    g = g[q].sort_values("snr_px_ccd3", ascending=False).drop_duplicates("source_id").reset_index(drop=True)
    log(f"GALAH: {n0} spectra -> {len(g)} unique good stars")
    g["m_h"] = mh_from_fe(g.fe_h, g.afe)
    g["alpha_m"] = alpha_m_from_afe(g.afe)
    ov = g.merge(apo[["source_id", "snr"] + LABELS], on="source_id", suffixes=("", "_apo"))
    ov = ov[(ov.snr > 70) & ov[[l + "_apo" for l in LABELS]].notna().all(1)]
    log(f"GALAH x APOGEE overlap for the scale fit: {len(ov)}")
    sc = fit_scale(ov[LABELS + ["fe_h", "afe"]].reset_index(drop=True),
                   ov[[l + "_apo" for l in LABELS]].rename(columns=lambda c: c[:-4]).reset_index(drop=True))
    meta["galah_scale"] = sc
    meta["galah_apogee_overlap"] = int(len(ov))
    gal = g[~g.source_id.isin(apo.source_id)].reset_index(drop=True)
    log(f"GALAH-only stars: {len(gal)}")
    cal = apply_scale(gal, sc)
    e_in = {"teff": gal.e_teff, "logg": gal.e_logg, "m_h": gal.e_fe_h, "alpha_m": 0.0}
    for lab in LABELS:
        gal["galah_" + lab] = gal[lab]
        gal[lab] = cal[lab]
        gal["e_" + lab] = np.hypot(e_in[lab], sc[lab]["rms_after"])
    if len(gal) > cap:
        gal = gal.sample(cap, random_state=seed).reset_index(drop=True)
    gal["component"] = "galah"
    return gal[["source_id", "component", "snr_px_ccd3"] + LABELS + ["e_" + l for l in LABELS]
               + ["galah_" + l for l in LABELS] + ["fe_h", "afe"]].rename(columns={"fe_h": "galah_fe_h", "afe": "galah_afe"})


def _name_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def comp_cluster_oc(apo: pd.DataFrame, meta: dict) -> pd.DataFrame:
    f = CACHE / "hr23_members.parquet"
    if not f.exists():
        m = pd.read_fwf(RAW / "hr23_members.dat", colspecs=[(0, 20), (26, 45), (52, 72)], names=["Name", "source_id", "Prob"],
                        dtype={"Name": str, "source_id": np.int64})
        m = m[m.Prob > 0.7]
        m.to_parquet(f)
    m = pd.read_parquet(f)
    cl = pd.read_fwf(RAW / "clusters.dat.gz", colspecs=[(0, 20), (273, 274), (287, 293), (592, 607), (799, 809), (832, 842)],
                     names=["Name", "Type", "N", "dist50", "logAge50", "AV50"], compression="gzip")
    cl = cl[cl.Type == "o"]
    m = m[m.Name.str.strip().isin(cl.Name.str.strip())]
    m["Name"] = m.Name.str.strip()
    cl["Name"] = cl.Name.str.strip()
    # reference [M/H], [alpha/M] of a cluster = median over its APOGEE (astra 0.6.0) members (same scale as the labels)
    am = m.merge(apo[["source_id", "snr"] + LABELS], on="source_id")
    am = am[am.snr > 50]
    g = am.groupby("Name")
    ref = pd.DataFrame({"n_apo": g.size(), "m_h_ref": g.m_h.median(), "alpha_m_ref": g.alpha_m.median(),
                        "m_h_apo_mad": g.m_h.apply(lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))),
                        "alpha_m_apo_mad": g.alpha_m.apply(lambda x: 1.4826 * np.median(np.abs(x - np.median(x))))})
    ref = ref[ref.n_apo >= 5].reset_index().merge(cl[["Name", "dist50", "logAge50", "AV50"]], on="Name")
    ref["mu_true"] = 5 * np.log10(ref.dist50 / 10.0)
    log(f"open clusters with >=5 APOGEE members: {len(ref)}")
    t = m.merge(ref, on="Name")
    t = t[~t.source_id.isin(apo.source_id)].drop_duplicates("source_id")
    t = t.rename(columns={"Name": "cluster", "m_h_ref": "m_h", "alpha_m_ref": "alpha_m", "AV50": "a0_ref"})
    t["e_m_h"], t["e_alpha_m"] = t.m_h_apo_mad / np.sqrt(t.n_apo), t.alpha_m_apo_mad / np.sqrt(t.n_apo)
    t["teff"] = t["logg"] = t["e_teff"] = t["e_logg"] = np.nan
    t["component"] = "cluster_oc"
    meta["cluster_oc"] = dict(n_clusters=int(len(ref)))
    return t[["source_id", "component", "cluster", "n_apo", "mu_true", "a0_ref", "logAge50"] + LABELS + ["e_" + l for l in LABELS]]


def comp_cluster_gc(apo: pd.DataFrame, meta: dict) -> pd.DataFrame:
    # Harris 2010 [Fe/H] (part 2 of the file), BV21 distances
    lines = (RAW / "harris10_mwgc.dat").read_text().splitlines()
    i0 = next(i for i, l in enumerate(lines) if "[Fe/H]" in l and "E(B-V)" in l)
    har = {}
    for l in lines[i0 + 1:]:
        m = re.match(r"^\s*(\S.*?)\s+(-?\d+\.\d\d)\s+(\d+)\s+\d+\.\d\d", l)
        if m:
            har[_name_key(m.group(1))] = float(m.group(2))
        elif har and not l.strip():
            break
    html = (RAW / "bv21_parameter.html").read_text()
    dist = {}
    for row in re.findall(r"<tr>(.*?)</tr>", html, flags=re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S)
        if len(tds) > 4:
            nm = re.sub(r"<[^>]+>|&nbsp;", " ", tds[0]).strip().split("  ")[0].strip()
            mm = re.search(r"([\d.]+)\s*&plusmn;", tds[3])
            if nm and mm:
                dist[_name_key(nm)] = float(mm.group(1)) * 1000.0
    z = zipfile.ZipFile(RAW / "vb21_clusters.zip")
    names = [l[:12].strip() for l in (RAW / "tablea1.dat").read_text().splitlines() if l.strip()]
    keys = {_name_key(n): n for n in names}
    rows = []
    for fn in z.namelist():
        if not fn.startswith("catalogues/") or not fn.endswith(".txt"):
            continue
        stem = fn[len("catalogues/"):-4]
        cands = [k for k in keys if _name_key(stem).startswith(k)]
        if not cands:
            continue
        nm = keys[max(cands, key=len)]
        kk = _name_key(nm)
        if kk not in har or kk not in dist:
            continue
        d = pd.read_csv(z.open(fn), sep=r"\s+", comment="#", header=None,
                        names=["source_id", "ra", "dec", "x", "y", "plx", "pmra", "pmdec", "plxe", "pmrae", "pmdece", "pmcorr",
                               "g_mag", "bp_rp", "Sigma", "qflag", "memberprob"])
        d = d[(d.memberprob > 0.7) & (d.qflag >= 2) & (d.g_mag < 17.6)]
        d["cluster"], d["feh_harris"], d["dist_pc"] = nm, har[kk], dist[kk]
        d["r_arcmin"] = 60 * np.hypot(d.x, d.y)
        rows.append(d[["source_id", "cluster", "feh_harris", "dist_pc", "r_arcmin", "Sigma"]])
    t = pd.concat(rows)
    log(f"GC: {t.cluster.nunique()} clusters, {len(t)} members (G<17.6)")
    # APOGEE-scale calibration of the Harris [Fe/H]: cluster median APOGEE [M/H] vs Harris [Fe/H]
    am = t.merge(apo[["source_id", "snr"] + LABELS], on="source_id")
    am = am[am.snr > 50]
    g = am.groupby("cluster")
    cc = pd.DataFrame({"n_apo": g.size(), "m_h_apo": g.m_h.median(), "alpha_apo": g.alpha_m.median()}).join(
        t.groupby("cluster").feh_harris.first())
    cc = cc[cc.n_apo >= 5]
    A = np.stack([np.ones(len(cc)), cc.feh_harris], 1)
    beta, *_ = np.linalg.lstsq(A, cc.m_h_apo, rcond=None)
    rms = float(np.std(cc.m_h_apo - A @ beta))
    meta["gc_scale"] = dict(beta=beta.tolist(), n_clusters_fit=int(len(cc)), rms=rms,
                            alpha_apo_median=float(cc.alpha_apo.median()),
                            table=cc.reset_index().round(3).to_dict("records"))
    log(f"GC [M/H]_APOGEE = {beta[0]:.3f} + {beta[1]:.3f}*[Fe/H]_Harris (n={len(cc)}, rms {rms:.3f})")
    t = t[~t.source_id.isin(apo.source_id)].drop_duplicates("source_id").copy()
    t["m_h"] = beta[0] + beta[1] * t.feh_harris
    t["e_m_h"] = rms
    t["alpha_m"] = meta["gc_scale"]["alpha_apo_median"]      # nominal: only the spread is meaningful
    t["e_alpha_m"] = 0.1
    t["mu_true"] = 5 * np.log10(t.dist_pc / 10.0)
    t["teff"] = t["logg"] = t["e_teff"] = t["e_logg"] = np.nan
    t["component"] = "cluster_gc"
    return t[["source_id", "component", "cluster", "feh_harris", "mu_true", "r_arcmin", "Sigma"] + LABELS + ["e_" + l for l in LABELS]]


def comp_lamost_hot(apo: pd.DataFrame, meta: dict, cap: int = 100000) -> pd.DataFrame:
    cs = [(0, 35), (57, 64), (65, 72), (73, 85), (86, 97), (98, 110), (111, 122), (123, 135), (332, 334), (357, 376)]
    nm = ["SpecID", "snrg", "teff", "e_teff", "logg", "e_logg", "fe_h", "e_fe_h", "uniqflag", "source_id"]
    h = pd.read_fwf(RAW / "xiang22_table5.dat.gz", colspecs=cs, names=nm, compression="gzip", dtype={"source_id": np.int64})
    h = h[(h.teff >= 7000) & (h.uniqflag == 1) & (h.source_id > 0) & (h.snrg >= 20) & h.logg.between(1.0, 5.5)
          & (h.e_teff < 0.15 * h.teff) & (h.e_logg < 0.5) & (h.e_fe_h < 0.5) & h.fe_h.between(-2.5, 1.0)].sort_values("snrg", ascending=False).drop_duplicates("source_id")
    h = h[~h.source_id.isin(apo.source_id)].reset_index(drop=True)
    log(f"LAMOST hot (Teff>=7000, unique): {len(h)}")
    h["m_h"], h["e_m_h"] = h.fe_h, h.e_fe_h                 # [alpha/Fe] unknown for OBA stars: [M/H] ~ [Fe/H]
    h["alpha_m"] = h["e_alpha_m"] = np.nan
    if len(h) > cap:
        h = h.sample(cap, random_state=0).reset_index(drop=True)
    h["component"] = "lamost_hot"
    return h[["source_id", "component", "snrg"] + LABELS + ["e_" + l for l in LABELS]]


def xmatch_gaia(df: pd.DataFrame, radius: float = 2.0) -> pd.DataFrame:
    buf = df[["ra", "dec"]].reset_index().rename(columns={"index": "idx"}).to_csv(index=False)
    r = requests.post("http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync",
                      data=dict(request="xmatch", distMaxArcsec=radius, selection="best", RESPONSEFORMAT="csv",
                                cat2="vizier:I/355/gaiadr3", colRA1="ra", colDec1="dec"),
                      files={"cat1": ("c.csv", buf)}, timeout=900)
    r.raise_for_status()
    x = pd.read_csv(io.StringIO(r.text), usecols=["idx", "angDist", "DR3Name", "Gmag"])
    x["source_id"] = x.DR3Name.str.replace("Gaia DR3 ", "").astype(np.int64)
    return x


def comp_lamost_vmp(apo: pd.DataFrame, meta: dict) -> pd.DataFrame:
    v = viz_tap('SELECT "LAMOST","_RA","_DE","Teff","e_Teff","logg","e_logg","[Fe/H]","e_[Fe/H]","[a/Fe]","g0mag" '
                'FROM "J/ApJ/891/39/table5"').rename(columns={"_RA": "ra", "_DE": "dec", "[Fe/H]": "fe_h", "e_[Fe/H]": "e_fe_h",
                                                              "[a/Fe]": "afe", "Teff": "teff", "e_Teff": "e_teff", "logg": "logg", "e_logg": "e_logg"})
    x = xmatch_gaia(v)
    v = v.reset_index().rename(columns={"index": "idx"}).merge(x, on="idx")
    v = v[(np.abs(v.Gmag - v.g0mag) < 1.0) | v.g0mag.isna()]       # g0 is extinction-corrected: loose sanity check only
    v = v[v.fe_h < -2.0].drop_duplicates("source_id")
    v = v[~v.source_id.isin(apo.source_id)].reset_index(drop=True)
    log(f"LAMOST VMP (Yuan+2020, [Fe/H]<-2, Gaia-matched): {len(v)}")
    b0, b1 = meta["gc_scale"]["beta"]                              # APOGEE-scale [M/H] vs [Fe/H] measured on globular clusters
    v["m_h"] = b0 + b1 * v.fe_h
    v["e_m_h"] = np.hypot(v.e_fe_h.fillna(0.2), meta["gc_scale"]["rms"])
    v["alpha_m"] = v["e_alpha_m"] = np.nan                         # [alpha/M] of VMP stars not evaluated (no APOGEE-scale truth)
    v["component"] = "lamost_vmp"
    return v[["source_id", "component", "fe_h"] + LABELS + ["e_" + l for l in LABELS]]


# ------------------------------------------------------------------ XP chunks / Gaia
def _scan_file(args):
    f, tgt = args
    import h5py
    with h5py.File(f, "r") as h:
        c = h["table/columns"]
        ids = c["source_id/data"][:]
        pos = np.searchsorted(tgt, ids)
        pos[pos >= len(tgt)] = 0
        sel = np.where(tgt[pos] == ids)[0]
        if len(sel) == 0:
            return None
        d = {"source_id": ids[sel], "x": c["xp/data"][sel, :].astype(np.float32)}
        for k in ("phot_g_mean_mag", "bp_rp", "ra", "dec", "snr_bp", "snr_rp", "ruwe", "phot_bp_rp_excess_factor"):
            d[k] = c[f"{k}/data"][sel]
        return d


def xp_lookup(ids: np.ndarray, workers: int = 16) -> dict:
    """coef110n + metadata for `ids` from the local chunks; cached (npz) and extended incrementally."""
    f = CACHE / "xp_match.npz"
    have = {}
    if f.exists():
        z = np.load(f)
        have = {k: z[k] for k in z.files}
    scanned = set(have["scanned"].tolist()) if "scanned" in have else set()
    new = np.array(sorted(set(ids.tolist()) - scanned), dtype=np.int64)
    if len(new):
        log(f"scanning 3385 XP chunks for {len(new)} new ids")
        files = sorted(XP_CHUNKS.glob("gxp_chunk_*.hdf5"))
        parts = []
        with ThreadPoolExecutor(workers) as ex:
            for j, r in enumerate(ex.map(_scan_file, [(p, new) for p in files])):
                if r is not None:
                    parts.append(r)
                if j % 500 == 0:
                    log(f"  {j}/{len(files)}")
        if parts:
            add = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            keys = list(add)
            if "source_id" in have:
                for k in keys:
                    have[k] = np.concatenate([have[k], add[k]])
            else:
                have.update(add)
        have["scanned"] = np.array(sorted(scanned | set(new.tolist())), dtype=np.int64)
        np.savez(f, **have)
    out = {k: v for k, v in have.items() if k != "scanned"}
    _, first = np.unique(out["source_id"], return_index=True)
    return {k: v[first] for k, v in out.items()}


def gaia_tap(ids: np.ndarray, threads: int = 6) -> pd.DataFrame:
    import build_astrometry_phot as bap
    d = CACHE / "gaia"
    d.mkdir(parents=True, exist_ok=True)
    ids = np.unique(ids)
    chunks = [ids[s:s + bap.CHUNK] for s in range(0, len(ids), bap.CHUNK)]

    def one(c):
        key = f"{int(c[0])}_{int(c[-1])}_{len(c)}"
        p = d / f"{key}.parquet"
        if not p.exists():
            t = bap.tap(bap.Q_GAIA.format(ids=",".join(map(str, c))))
            t.to_parquet(p, index=False)
        return pd.read_parquet(p)
    with ThreadPoolExecutor(threads) as ex:
        t = pd.concat(list(ex.map(one, chunks)), ignore_index=True)
    t = t.drop_duplicates("source_id")
    t["zpt"], _ = bap.parallax_zero_point(t)
    t["ebv_sfd"] = bap.sfd_ebv(t, str(REPO / "data/dustmaps/sfd"))
    return t[["source_id", "l", "b", "parallax", "parallax_error", "zpt", "ebv_sfd", "astrometric_params_solved"]]


def knn_score(apo: pd.DataFrame, train_ids_mask: pd.Series, y: np.ndarray, k: int = 8):
    """Mean distance to the k nearest APOGEE-train stars in (Teff, logg, [M/H]) / train-std."""
    from scipy.spatial import cKDTree
    tr = apo[train_ids_mask & apo[["teff", "logg", "m_h"]].notna().all(1)][["teff", "logg", "m_h"]].to_numpy()
    mu, sd = tr.mean(0), tr.std(0)
    tree = cKDTree((tr - mu) / sd)
    ok = np.isfinite(y).all(1)
    out = np.full(len(y), np.nan)
    if ok.any():
        out[ok] = tree.query((y[ok] - mu) / sd, k=k)[0].mean(1)
    return out, (mu, sd, tree)


def finalize(name: str, df: pd.DataFrame, xp: dict, gaia: pd.DataFrame, apo: pd.DataFrame, meta: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    xpd = pd.DataFrame({k: v for k, v in xp.items() if k != "x"})
    pos = pd.Series(np.arange(len(xpd)), index=xpd.source_id)
    df = df[df.source_id.isin(xpd.source_id)].merge(xpd, on="source_id").merge(gaia, on="source_id", how="left")
    n_before = len(df)
    df = df[(df.phot_g_mean_mag < GMAX) & np.isfinite(df.phot_g_mean_mag)].reset_index(drop=True)
    df["x_row"] = pos.loc[df.source_id].to_numpy()
    X = xp["x"][df.x_row.to_numpy()]
    df = df.drop(columns="x_row").rename(columns={"phot_g_mean_mag": "G"})
    df["a0_sfd"] = 3.1 * 0.86 * df.ebv_sfd
    y = df[["teff", "logg", "m_h"]].to_numpy(float)
    df["ood_knn"], _ = knn_score(apo, apo.split == "train", y)
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    tbl = tbl.append_column("x", pa.FixedSizeListArray.from_arrays(pa.array(X.reshape(-1).astype(np.float32)), 110))
    OUT.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, OUT / f"ood_{name}.parquet")
    meta.setdefault("counts", {})[name] = dict(after_apogee_removal=int(n_before), final_with_xp_G=int(len(df)))
    log(f"wrote ood_{name}.parquet  n={len(df)} (before G cut {n_before})")


def comp_starhorse(apo: pd.DataFrame, meta: dict, n_per_stratum: int = 50000, seed: int = 1) -> pd.DataFrame:
    """Faint stars (G 16-17.6) drawn from random XP chunks (sky patches), 3 latitude strata, with StarHorse EDR3."""
    import h5py
    rng = np.random.default_rng(seed)
    files = sorted(XP_CHUNKS.glob("gxp_chunk_*.hdf5"))
    pick = rng.choice(len(files), 400, replace=False)
    rows = []
    for j in pick:
        with h5py.File(files[j], "r") as h:
            c = h["table/columns"]
            g, b, s = c["phot_g_mean_mag/data"][:], c["b/data"][:], c["source_id/data"][:]
        ok = (g > 16) & (g < 17.6)
        rows.append(pd.DataFrame({"source_id": s[ok], "b": b[ok], "G": g[ok]}))
    d = pd.concat(rows)
    d = d[~d.source_id.isin(apo.source_id)]
    ab = d.b.abs()
    strata = [d[ab < 5], d[(ab >= 5) & (ab < 20)], d[ab >= 20]]
    d = pd.concat([s.sample(min(n_per_stratum, len(s)), random_state=seed) for s in strata])
    ids = np.sort(d.source_id.to_numpy())
    (CACHE / "sh").mkdir(exist_ok=True)

    def one(s0):
        f = CACHE / "sh" / f"{s0}.parquet"
        if not f.exists():
            q = ('SELECT "Source","dist16","dist50","dist84","AV16","AV50","AV84","teff50","logg50","met50","fidelity","FlagOut" '
                 f'FROM "I/354/starhorse2021" WHERE "Source" IN ({",".join(map(str, ids[s0:s0 + 1000]))})')
            viz_tap(q).to_parquet(f)
        log("  starhorse batch", s0)
        return pd.read_parquet(f)
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(one, range(0, len(ids), 1000)))
    sh = pd.concat(res).rename(columns={"Source": "source_id"}).drop_duplicates("source_id")
    sh = sh[(sh.fidelity >= 0.5) & (sh.FlagOut == 0)]
    log(f"StarHorse: {len(ids)} sampled -> {len(sh)} with fidelity>=0.5 & FlagOut==0")
    sh["mu_true"] = 5 * np.log10(sh.dist50 * 1000.0 / 10.0)
    sh["sigma_mu"] = 5 / np.log(10) * (sh.dist84 - sh.dist16) / (2 * sh.dist50)
    sh["a0_ref"] = sh.AV50
    for l in LABELS:
        sh[l] = sh["e_" + l] = np.nan
    sh["component"] = "starhorse"
    return sh[["source_id", "component", "mu_true", "sigma_mu", "a0_ref", "AV16", "AV84", "teff50", "logg50", "met50", "fidelity"]
              + LABELS + ["e_" + l for l in LABELS]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", nargs="*",
                    default=["galah", "cluster_oc", "cluster_gc", "lamost_hot", "lamost_vmp", "starhorse"])
    ap.add_argument("--galah-cap", type=int, default=200000)
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    download()
    if args.download_only:
        return
    apo = load_apogee()
    meta_f = OUT / "ood_meta.json"
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    fn = {"galah": lambda: comp_galah(apo, meta, args.galah_cap), "cluster_oc": lambda: comp_cluster_oc(apo, meta),
          "cluster_gc": lambda: comp_cluster_gc(apo, meta), "lamost_hot": lambda: comp_lamost_hot(apo, meta),
          "lamost_vmp": lambda: comp_lamost_vmp(apo, meta), "starhorse": lambda: comp_starhorse(apo, meta)}
    order = [c for c in ("cluster_gc", "galah", "cluster_oc", "lamost_hot", "lamost_vmp", "starhorse") if c in args.components]
    if "lamost_vmp" in order and "cluster_gc" not in order:
        comp_cluster_gc(apo, meta)                                # needs meta["gc_scale"]
    comps = {c: fn[c]() for c in order}
    ids = np.unique(np.concatenate([c.source_id.to_numpy() for c in comps.values()]))
    log(f"{len(ids)} unique OOD ids; XP lookup")
    xp = xp_lookup(ids)
    have = np.isin(ids, xp["source_id"])
    log(f"{have.sum()} with XP coefficients; Gaia TAP")
    gaia = gaia_tap(ids[have])
    # ID reference for the kNN score (APOGEE iid_test stars vs the train labels)
    te = apo[apo.split == "iid_test"]
    s_id, _ = knn_score(apo, apo.split == "train", te[["teff", "logg", "m_h"]].to_numpy())
    meta["ood_knn_id_quantiles"] = {str(q): float(np.nanquantile(s_id, q)) for q in (0.5, 0.9, 0.99, 0.999)}
    for name, df in comps.items():
        finalize(name, df, xp, gaia, apo, meta)
    meta_f.write_text(json.dumps(meta, indent=1, default=float))
    log("done", json.dumps(meta["counts"]))


if __name__ == "__main__":
    main()
