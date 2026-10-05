"""Load + clean astraAllStarASPCAP into the training catalog tables.

Deliberately the ONLY module that knows about the ASPCAP table layout. DR21:
swap the table, re-run `scripts/build_dataset.py`, retrain. Nothing else
changes.

Outputs of `load_and_clean`:
    df     one row per APOGEE spectrum (pre-dedup), global quality cuts applied
    y      float32 [n, K]  raw-unit labels (NaN where masked)
    yerr   float32 [n, K]  raw-unit 1-sigma errors (NaN where masked)
    mask   bool     [n, K] True where the label is usable
    meta   resolved column names + counts, for the manifest

Masking rule (never a global dropna):
    mask[i,k] = finite(y) & finite(yerr)
                & (y within valid_range)
                & (per-element flag bits 0..3 clear, when a flag column exists)
Stars enter the sample if they pass the global cuts; which *labels* they carry
is then purely per-label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
    FLAG_BITS,
    GLOBAL_COLUMNS,
    LABELS,
    LabelSpec,
    spec,
    specs,
)


class ColumnResolutionError(RuntimeError):
    pass


def _resolve(df: pd.DataFrame, candidates: tuple[str, ...], what: str,
             required: bool = True) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    if required:
        raise ColumnResolutionError(
            f"cannot find {what}: tried {candidates}; table has {list(df.columns)[:40]}..."
        )
    return None


def resolve_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Map every logical name to an actual column (or None if optional-missing)."""
    out: dict[str, str | None] = {}
    for logical, cands in GLOBAL_COLUMNS.items():
        out[logical] = _resolve(df, cands, logical, required=logical in ("source_id", "snr"))
    for s in LABELS:
        out[f"label:{s.name}"] = _resolve(df, s.column, f"label {s.name}",
                                          required=s.tier == 1)
        out[f"err:{s.name}"] = _resolve(df, s.error_column, f"error for {s.name}",
                                        required=False)
        out[f"flag:{s.name}"] = _resolve(df, s.flag_column, f"flags for {s.name}",
                                         required=False)
    return out


def wanted_columns() -> list[str]:
    """Union of every column this pipeline might consume (candidates only;
    resolution happens at load time)."""
    cols: set[str] = set()
    for cands in GLOBAL_COLUMNS.values():
        cols.update(cands)
    for s in LABELS:
        cols.update(s.column)
        cols.update(s.error_column)
        cols.update(s.flag_column)
    return sorted(cols)


def _match_columns(available: list[str], wanted: list[str]) -> list[str]:
    """Case-insensitive matching of wanted candidate names against the table."""
    low = {c.lower(): c for c in available}
    return [low[w.lower()] for w in wanted if w.lower() in low]


def read_aspcap_table(path: str, hdu: int = 2,
                      columns: list[str] | None = None) -> pd.DataFrame:
    """Read astraAllStarASPCAP (FITS HDU2 by default) or a pre-converted
    parquet/csv. Uncompressed FITS is read through numpy memmap, so passing
    `columns` makes this cheap; .fits.gz works too but decompresses the whole
    file into memory on first access. `columns` entries are matched
    case-insensitively; missing ones are silently skipped (resolution happens
    downstream)."""
    if str(path).endswith((".fits", ".fits.gz")):
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise ImportError(
                "reading the ASPCAP FITS needs astropy: pip install astropy "
                "(or pre-convert to parquet)"
            ) from exc
        with fits.open(path, memmap=not str(path).endswith(".gz")) as hdul:
            data = hdul[hdu].data
            if columns is None:
                names = list(data.names)
            else:
                names = _match_columns(list(data.names), columns)
            out = {}
            for name in names:
                col = np.asarray(data[name])
                if col.dtype.byteorder not in ("=", "|"):
                    col = col.astype(col.dtype.newbyteorder("="))
                out[name] = col
            return pd.DataFrame(out)
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def global_clean(df: pd.DataFrame, snr_min: float = 50.0, teff_max: float = 8000.0,
                 teff_variant: str = "raw") -> tuple[pd.DataFrame, dict]:
    """Cuts that define *the sample* (not the labels):
    valid gaia_dr3_source_id, ASPCAP flag_bad False, S/N > snr_min,
    Teff < teff_max (the MARCS cool-star validation domain)."""
    cols = resolve_columns(df)
    keep = np.ones(len(df), dtype=bool)
    stats = {"n_input": int(len(df))}

    sid = df[cols["source_id"]]
    keep &= np.isfinite(pd.to_numeric(sid, errors="coerce").to_numpy(dtype=float))
    keep &= pd.to_numeric(sid, errors="coerce").to_numpy(dtype=float) > 0
    stats["n_bad_source_id"] = int((~keep).sum())

    snr = pd.to_numeric(df[cols["snr"]], errors="coerce").to_numpy(dtype=float)
    keep &= np.isfinite(snr) & (snr > snr_min)

    teff = pd.to_numeric(df[cols["teff_for_range"]], errors="coerce").to_numpy(dtype=float)
    keep &= np.isfinite(teff) & (teff < teff_max)

    if cols["flag_bad"] is not None:
        # flag_bad may arrive as bool or as an int bitmask; treat nonzero as bad
        bad = df[cols["flag_bad"]]
        bad_num = pd.to_numeric(bad, errors="coerce").to_numpy(dtype=float)
        bad_mask = (bad_num != 0) if np.issubdtype(np.asarray(bad_num).dtype, np.number) \
            else bad.astype(bool).to_numpy()
        keep &= ~np.nan_to_num(bad_mask, nan=True).astype(bool)

    out = df.loc[keep].reset_index(drop=True)
    stats["n_clean"] = int(len(out))
    stats["cuts"] = {"snr_min": snr_min, "teff_max": teff_max,
                     "flag_bad_excluded": cols["flag_bad"] is not None}
    return out, stats


def _flag_mask(flags: np.ndarray, include_warn: bool = True) -> np.ndarray:
    """True where the element-level bitmask has masking bits set.

    Hard bits (5-8: censored x3 + bad grid edge) always mask; warn bits
    (9-11: warn grid edge / teff / m_h) mask when include_warn. Missing flag
    entries (NaN) are treated conservatively as bad."""
    f = np.nan_to_num(np.asarray(flags, dtype=float), nan=-1.0)
    bad = ~np.isfinite(np.asarray(flags, dtype=float)) | (f < 0)
    fi = np.floor(f).astype(np.int64)
    m = np.zeros(f.shape, dtype=bool)
    for bit in FLAG_BITS["hard"]:
        m |= (fi >> bit) & 1 == 1
    if include_warn:
        for bit in FLAG_BITS["warn"]:
            m |= (fi >> bit) & 1 == 1
    return m | bad


def build_label_matrix(df: pd.DataFrame, label_names: list[str],
                       teff_variant: str = "raw",
                       mask_warn_bits: bool = True):
    """Per-label y / yerr / mask for the requested labels, in the given order."""
    cols = resolve_columns(df)
    n = len(df)
    K = len(label_names)
    y = np.full((n, K), np.nan, dtype=np.float32)
    yerr = np.full((n, K), np.nan, dtype=np.float32)
    mask = np.zeros((n, K), dtype=bool)
    resolved: dict[str, str | None] = {}

    for k, name in enumerate(label_names):
        s: LabelSpec = spec(name)
        if name == "teff" and teff_variant == "calibrated":
            col = cols["label:teff_calibrated"]
        else:
            col = cols[f"label:{name}"]
        resolved[name] = col
        if col is None:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        y[:, k] = vals

        ecol = cols.get(f"err:{name}")
        if ecol is not None:
            yerr[:, k] = pd.to_numeric(df[ecol], errors="coerce").to_numpy(dtype=float)
        else:
            yerr[:, k] = 0.0    # no error column: rely on sigma_floor alone

        ok = np.isfinite(vals) & np.isfinite(yerr[:, k])
        lo, hi = s.valid_range
        ok &= (vals >= lo) & (vals <= hi)

        fcol = cols.get(f"flag:{name}")
        if fcol is not None:
            ok &= ~_flag_mask(df[fcol].to_numpy(), include_warn=mask_warn_bits)

        y[:, k] = np.where(ok, vals, np.nan)
        yerr[:, k] = np.where(ok, yerr[:, k], np.nan)
        mask[:, k] = ok

    return y, yerr, mask, resolved


def load_and_clean(path: str, label_names: list[str], snr_min: float = 50.0,
                   teff_max: float = 8000.0, teff_variant: str = "raw",
                   mask_warn_bits: bool = True, hdu: int = 2) -> dict:
    """One call: read -> global clean -> per-label matrices. Returns a dict with
    df (cleaned), y, yerr, mask, resolved columns (global + label), and stats."""
    df = read_aspcap_table(path, hdu=hdu, columns=wanted_columns())
    cols = resolve_columns(df)
    df, stats = global_clean(df, snr_min=snr_min, teff_max=teff_max,
                             teff_variant=teff_variant)
    y, yerr, mask, label_resolved = build_label_matrix(
        df, label_names, teff_variant=teff_variant, mask_warn_bits=mask_warn_bits)
    resolved = {**{k: v for k, v in cols.items()}, **label_resolved}
    stats["labels"] = {k: int(mask[:, i].sum()) for i, k in enumerate(label_names)}
    stats["label_order"] = list(label_names)
    stats["teff_variant"] = teff_variant
    return {"df": df, "y": y, "yerr": yerr, "mask": mask,
            "resolved": resolved, "stats": stats}


def dedupe_max_snr(df: pd.DataFrame, sid_col: str) -> np.ndarray:
    """Positions (sorted) of the max-S/N visit per source_id. Mirrors the
    dedupe in scripts/build_dataset.py exactly (same sort, same tie-break) so
    that independently rebuilt label tables line up with the processed
    catalog; `scripts/build_aux_labels.py` verifies this against labels.npz."""
    snr = pd.to_numeric(df["snr"], errors="coerce").to_numpy(float)
    order = pd.DataFrame({"sid": df[sid_col].to_numpy(), "snr": snr,
                          "pos": df.index.to_numpy()}) \
        .sort_values("snr", ascending=False).drop_duplicates("sid", keep="first")
    return np.sort(order["pos"].to_numpy())
