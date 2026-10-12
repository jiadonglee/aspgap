"""Table IO for the prediction CLI: parquet / csv / fits input and the Gaia-archive fetch (astroquery optional)."""

from __future__ import annotations

from pathlib import Path

import contextlib
import os
import tempfile

import numpy as np
import pandas as pd

REQUIRED = ["bp_coefficients", "rp_coefficients", "phot_g_mean_mag"]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".csv", ".gz"):
        df = pd.read_csv(path)            # array columns arrive as "[a, b, ...]" strings
    elif ext in (".fits", ".fit", ".fz"):
        try:
            from astropy.table import Table
        except ImportError as exc:
            raise ImportError("reading FITS needs astropy: pip install -e '.[predict]'") from exc
        t = Table.read(path)
        df = pd.DataFrame({n: (list(np.ma.filled(t[n].astype(float), np.nan)) if t[n].ndim > 1
                               else np.asarray(t[n].data).astype(np.asarray(t[n].data).dtype.newbyteorder("="))
                               ) for n in t.colnames})
    else:
        raise ValueError(f"unsupported input format {ext!r} (use .parquet, .csv or .fits)")
    low = {c.lower(): c for c in df.columns}
    df = df.rename(columns={low[c]: c for c in REQUIRED + ["source_id"] if c in low})
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"input is missing column(s) {miss}; found {list(df.columns)}")
    return df


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".parquet":
        df.to_parquet(path, index=False)
    elif ext == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported output format {ext!r} (use .parquet or .csv)")


def read_source_ids(spec: str) -> list[int]:
    """`spec` is a text file (one id per line, '#' comments) or a comma-separated list."""
    p = Path(spec)
    txt = p.read_text() if p.is_file() else spec
    toks = [t for line in txt.splitlines() if not line.strip().startswith("#")
            for t in line.replace(",", " ").split()]
    return [int(t) for t in toks]


@contextlib.contextmanager
def _scratch_cwd():
    """astroquery drops temp_*/download_* files in the cwd; keep them out of the user's directory."""
    old = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="aspgapv3_gaia_") as d:
        os.chdir(d)
        try:
            yield
        finally:
            os.chdir(old)


def _rows(col) -> list[np.ndarray]:
    """Per-row float arrays from a (possibly object-typed / masked) FITS/VOTable array column."""
    return [np.asarray(np.ma.filled(np.ma.asarray(v, dtype=float), np.nan), float).ravel() for v in col]


def fetch_gaia_xp(source_ids, chunk: int = 500, verbose: bool = True) -> pd.DataFrame:
    """XP continuous coefficients (DataLink) + phot_g_mean_mag (TAP) from the Gaia DR3 archive (needs astroquery + network).
    Returns one row per requested source_id (in the requested order); stars without a published XP
    spectrum have NaN coefficients."""
    try:
        from astroquery.gaia import Gaia
    except ImportError as exc:
        raise ImportError("fetching from the Gaia archive needs astroquery: pip install -e '.[fetch]' "
                          "(or pip install astroquery)") from exc
    ids = [int(i) for i in source_ids]
    parts = []
    for i in range(0, len(ids), chunk):
        sub = ids[i:i + chunk]
        # XP continuous coefficients are served by the DataLink service (not a TAP table)
        with _scratch_cwd():
            res = Gaia.load_data(ids=sub, data_release="Gaia DR3", retrieval_type="XP_CONTINUOUS",
                                 data_structure="RAW", format="votable", valid_data=False, verbose=False)
            xp = next(iter(res.values()))[0].to_table()
        gt = Gaia.launch_job("SELECT source_id, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE source_id IN "
                             f"({','.join(map(str, sub))})").get_results()
        g = pd.DataFrame({"source_id": np.asarray(gt["source_id"]).astype(np.int64),
                          "phot_g_mean_mag": np.ma.filled(gt["phot_g_mean_mag"].astype(float), np.nan)})
        x = pd.DataFrame({
            "source_id": np.asarray(xp["source_id"]).astype(np.int64),
            "bp_coefficients": _rows(xp["bp_coefficients"]),
            "rp_coefficients": _rows(xp["rp_coefficients"])})
        parts.append(x.merge(g, on="source_id", how="inner"))
        if verbose:
            print(f"fetched {min(i + chunk, len(ids))}/{len(ids)} requested ({sum(len(q) for q in parts)} with XP)")
    got = pd.concat(parts, ignore_index=True).drop_duplicates("source_id") if parts else pd.DataFrame(
        columns=["source_id", "bp_coefficients", "rp_coefficients", "phot_g_mean_mag"])
    out = pd.DataFrame({"source_id": np.array(ids, np.int64)}).merge(got, on="source_id", how="left")
    nan55 = np.full(55, np.nan)
    for c in ("bp_coefficients", "rp_coefficients"):
        out[c] = [v if isinstance(v, np.ndarray) else nan55 for v in out[c]]
    n_miss = int(out["phot_g_mean_mag"].isna().sum())
    if n_miss and verbose:
        print(f"warning: {n_miss} source_id(s) have no XP continuous spectrum in Gaia DR3 (output rows are NaN)")
    return out
