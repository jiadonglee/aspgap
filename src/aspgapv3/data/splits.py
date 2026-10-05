"""Splits that a referee cannot argue with.

- one row per gaia_dr3_source_id (dedupe by max APOGEE S/N) -- grouping is then
  automatic, but we still split on the id so a future multi-visit catalog
  cannot leak;
- an IID split (train / val / iid_test);
- a sky-held-out test: the sky is partitioned into spherical-kmeans groups and
  whole groups are removed from training, so the model cannot exploit
  extinction / field-level population correlations;
- domain tags that stay in whatever split they landed in but get reported
  separately at evaluation: metal-poor ([M/H] < -1 / < -1.5), M dwarf
  (Teff < 4000 K & logg > 4), F star (Teff > 6000 K), faint-XP (top-quartile G).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dedupe_max_snr(df: pd.DataFrame, snr_col: str = "snr",
                   id_col: str = "source_id") -> pd.DataFrame:
    """One row per source_id, keeping the highest-S/N APOGEE spectrum."""
    d = df.sort_values(snr_col, ascending=False)
    d = d.drop_duplicates(subset=[id_col], keep="first")
    return d.sort_index().reset_index(drop=True)


def assign_iid_splits(df: pd.DataFrame, train: float = 0.90, val: float = 0.05,
                      test: float = 0.05, seed: int = 42,
                      id_col: str = "source_id") -> pd.DataFrame:
    """Group-safe random split on source ids."""
    assert abs(train + val + test - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    ids = df[id_col].unique()
    rng.shuffle(ids)
    n_tr = int(round(train * len(ids)))
    n_va = int(round(val * len(ids)))
    assignment = {}
    for i, sid in enumerate(ids):
        assignment[sid] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "iid_test")
    out = df.copy()
    out["split"] = [assignment[s] for s in df[id_col]]
    return out


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], 1)


def spherical_kmeans(xy: np.ndarray, k: int, seed: int = 42,
                     iters: int = 50) -> np.ndarray:
    """Tiny numpy k-means on the unit sphere; k-means++ init. Returns labels."""
    rng = np.random.default_rng(seed)
    n = len(xy)
    centers = [xy[rng.integers(n)]]
    d2 = np.sum((xy - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        p = d2 / max(d2.sum(), 1e-12)
        centers.append(xy[rng.choice(n, p=p)])
        d2 = np.minimum(d2, np.sum((xy - centers[-1]) ** 2, axis=1))
    C = np.stack(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = ((xy[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = dist.argmin(1)
        if (new == labels).all() and _ > 0:
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = xy[m].mean(0)
                C[j] /= max(np.linalg.norm(C[j]), 1e-12)
    return labels


def assign_sky_holdout(df: pd.DataFrame, n_groups: int = 64,
                       holdout_groups: int = 8, seed: int = 42) -> pd.DataFrame:
    """Hold out `holdout_groups` whole sky groups as split='sky_test'; those
    stars are removed from train/val/iid_test entirely."""
    out = df.copy()
    xy = _unit_vectors(df["ra"].to_numpy(float), df["dec"].to_numpy(float))
    out["sky_group"] = spherical_kmeans(xy, n_groups, seed=seed)
    rng = np.random.default_rng(seed + 1)
    held = rng.choice(n_groups, size=min(holdout_groups, n_groups), replace=False)
    is_holdout = out["sky_group"].isin(set(held.tolist()))
    # keep existing iid split; holdout stars move to sky_test and out of the rest
    out.loc[is_holdout, "split"] = "sky_test"
    return out


def domain_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean applicability slices (reported at eval; NOT removed from train)."""
    out = df.copy()
    mh = out["m_h"].to_numpy(float) if "m_h" in out else np.full(len(out), np.nan)
    teff = out["teff"].to_numpy(float) if "teff" in out else np.full(len(out), np.nan)
    logg = out["logg"].to_numpy(float) if "logg" in out else np.full(len(out), np.nan)
    out["is_metal_poor"] = mh < -1.0
    out["is_metal_poor_15"] = mh < -1.5
    out["is_mdwarf"] = (teff < 4000.0) & (logg > 4.0)
    out["is_fstar"] = teff > 6000.0
    g = out["phot_g_mean_mag"].to_numpy(float) if "phot_g_mean_mag" in out \
        else np.full(len(out), np.nan)
    thr = np.nanquantile(g, 0.75) if np.isfinite(g).any() else np.inf
    out["is_faint_xp"] = g > thr
    return out
