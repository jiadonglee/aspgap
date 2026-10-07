"""Catalog IO + the training dataset.

Processed layout (written by scripts/build_dataset.py):

    <root>/catalog.parquet (or .csv.gz)   labels, errors, split, tags
    <root>/xp343.npy    float32 [n, 343]  memmap-able
    <root>/coef110.npy  float32 [n, 110]
    <root>/norm.npz     LabelScaler stats + label order

APOGEEDataset.__getitem__ returns a dict:
    x      float32 [343] or [110]
    y      float32 [K]        normalized labels (0 where masked)
    yerr   float32 [K]        normalized 1-sigma errors
    mask   bool    [K]
    index  int
  with `aux=[...]` (labels_aux.npz + norm_aux.npz, see build_aux_labels.py):
    y_aux, yerr_aux  float32 [A] normalized; mask_aux bool [A]
  and, with `noise=True` (needs scripts/build_xp_noise.py outputs):
    sigma  float32 [343]      xp343_sigma (per-pixel 1-sigma, xp343 units)
    side   float32 [4]        phot_g_mean_mag, bp_rp, log10_flux550, log10_snr550
    coef   float32 [110]      raw DR3 BP|RP coefficients (augmentation input)
    valid  bool               star has a usable coefficient covariance
Everything the loss needs is per-label; a star with only [Fe/H] carries only
[Fe/H]. No star was dropped for missing labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------- catalog IO

def save_catalog(df: pd.DataFrame, root: str | Path) -> Path:
    root = Path(root)
    try:
        import pyarrow  # noqa: F401
        path = root / "catalog.parquet"
        df.to_parquet(path)
        return path
    except ImportError:
        path = root / "catalog.csv.gz"
        df.to_csv(path, index=False)
        return path


def load_catalog(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    pq = root / "catalog.parquet"
    if pq.is_file():
        return pd.read_parquet(pq)
    return pd.read_csv(root / "catalog.csv.gz")


# ---------------------------------------------------------------- normalizer

class LabelScaler:
    """Per-label affine scaling fit on TRAIN entries only (masked).

    y_norm = (y - mean) / std;  err_norm = err / std;  floor_norm = floor / std.
    Defined in normalized space so the NLL is unit-free; metrics denormalize.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray,
                 floor_norm: np.ndarray, labels: list[str]):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.floor_norm = np.asarray(floor_norm, dtype=np.float64)
        self.labels = list(labels)

    @classmethod
    def fit(cls, y: np.ndarray, yerr: np.ndarray, mask: np.ndarray,
            sigma_floor_raw: np.ndarray, labels: list[str]) -> "LabelScaler":
        mean = np.zeros(y.shape[1])
        std = np.ones(y.shape[1])
        for k in range(y.shape[1]):
            v = y[mask[:, k], k]
            if v.size > 1:
                mean[k] = v.mean()
                std[k] = max(v.std(), 1e-4)
            elif v.size == 1:
                mean[k] = v[0]
        floor_norm = np.asarray(sigma_floor_raw, dtype=np.float64) / std
        return cls(mean, std, floor_norm, labels)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def transform_err(self, yerr: np.ndarray) -> np.ndarray:
        return yerr / self.std

    def inverse(self, mu: np.ndarray, sigma: np.ndarray):
        return mu * self.std + self.mean, sigma * self.std

    def save(self, root: str | Path, name: str = "norm.npz") -> None:
        np.savez(Path(root) / name, mean=self.mean, std=self.std,
                 floor_norm=self.floor_norm,
                 labels=np.array(self.labels, dtype="U16"))

    @classmethod
    def load(cls, root: str | Path, name: str = "norm.npz") -> "LabelScaler":
        z = np.load(Path(root) / name, allow_pickle=False)
        return cls(z["mean"], z["std"], z["floor_norm"],
                   [str(s) for s in z["labels"]])


# ------------------------------------------------------------------ dataset

class APOGEEDataset(Dataset):
    """One split of the processed ASPCAP x Gaia-XP sample."""

    def __init__(self, root: str | Path, split: str = "train",
                 input: str = "xp343", labels: list[str] | None = None,
                 noise: bool = False, require_xp_valid: bool = False,
                 aux: list[str] | None = None, aux_ref: str = "h",
                 latent: str | None = None, astro: bool = False):
        if aux_ref not in ("h", "fe"):
            raise ValueError(f"labels.aux_ref must be h|fe, got {aux_ref!r}")
        self.aux_ref = aux_ref
        self.root = Path(root)
        self.split = split
        self.input = input

        cat = load_catalog(root)
        self.catalog = cat[cat["split"] == split].reset_index(drop=True)
        self.n_dropped_invalid = 0
        if require_xp_valid:
            # drop stars without a usable XP covariance / positive 550nm flux
            # (xpcov_valid.npy from build_xp_noise.py) so every model, whatever
            # its input type, sees the identical star set
            vf = self.root / "xpcov_valid.npy"
            if not vf.is_file():
                raise FileNotFoundError(f"{vf} missing; run scripts/build_xp_noise.py "
                                        "or set data.require_xp_valid: false")
            ok = np.load(vf)[self.catalog["row"].to_numpy(np.int64)]
            self.n_dropped_invalid = int((~ok).sum())
            self.catalog = self.catalog[ok].reset_index(drop=True)
        self.indices = self.catalog["row"].to_numpy(np.int64)   # rows into the arrays

        arr_file = self.root / f"{input}.npy"
        if not arr_file.is_file():
            raise FileNotFoundError(f"{arr_file} missing; rerun build_dataset.py")
        self.x = np.load(arr_file, mmap_mode="r")
        self.noise = noise
        if noise:
            for f in ("xp343_sigma", "side", "coef110", "xpcov_valid"):
                if not (self.root / f"{f}.npy").is_file():
                    raise FileNotFoundError(f"{f}.npy missing; run scripts/build_xp_noise.py")
            self.sigma = np.load(self.root / "xp343_sigma.npy", mmap_mode="r")
            self.side = np.load(self.root / "side.npy", mmap_mode="r")
            self.coef = np.load(self.root / "coef110.npy", mmap_mode="r")
            self.valid = np.load(self.root / "xpcov_valid.npy", mmap_mode="r")

        z = np.load(self.root / "labels.npz")
        scaler = LabelScaler.load(root)
        self.labels = scaler.labels if labels is None else list(labels)
        missing = [n for n in self.labels if n not in scaler.labels]
        if missing:
            raise ValueError(f"labels {missing} not in {scaler.labels}")
        col = {name: i for i, name in enumerate(scaler.labels)}
        if self.labels != scaler.labels:
            # a subset / reordering of the stored labels: restrict the scaler so that
            # y, floors, inverse() and metrics all refer to exactly these labels
            sel = [col[n] for n in self.labels]
            scaler = LabelScaler(scaler.mean[sel], scaler.std[sel], scaler.floor_norm[sel], self.labels)
            col = {name: i for i, name in enumerate(self.labels)}
            zcol = {n: i for i, n in enumerate(LabelScaler.load(root).labels)}
        else:
            zcol = col

        zc = [zcol[n] for n in self.labels]
        y = z["y"][self.indices][:, zc]
        yerr = z["yerr"][self.indices][:, zc]
        mask = z["mask"][self.indices][:, zc].astype(bool)

        self.y = scaler.transform_y(y).astype(np.float32)
        self.yerr = scaler.transform_err(yerr).astype(np.float32)
        self.mask = mask
        self.y_raw = y
        self.yerr_raw = yerr
        self.scaler = scaler
        self.floor_norm = scaler.floor_norm.astype(np.float32)

        # auxiliary labels (P1): separate files, never part of `labels` / metrics
        self.aux_labels: list[str] = list(aux or [])
        if self.aux_labels:
            self._load_aux(cat)

        # teacher latent target (v1_latent.npy + v1_latent_norm.npz, scripts/build_v1_latent.py):
        # row-indexed like the other arrays, standardized with TRAIN-split stats. Always the
        # latent of the CLEAN input (the augmenter never touches it).
        self.latent = latent
        if latent:
            self._load_latent(cat, latent)

        # Gaia parallax + 2MASS/WISE colours (astrometry_phot.parquet, build_astrometry_phot.py)
        self.astro = bool(astro)
        if self.astro:
            self._load_astro(cat)

        # per-label sample weights can be attached later by the trainer
        self.label_weights = np.ones(len(self.labels), dtype=np.float32)

    def _load_aux(self, full_cat: pd.DataFrame) -> None:
        """labels_aux.npz + norm_aux.npz (scripts/build_aux_labels.py), aligned
        to the catalog row order and checked against it via source_id."""
        fy, fn = self.root / "labels_aux.npz", self.root / "norm_aux.npz"
        for f in (fy, fn):
            if not f.is_file():
                raise FileNotFoundError(
                    f"labels.aux={self.aux_labels} configured but {f} is missing; "
                    f"run scripts/build_aux_labels.py --root {self.root}")
        z = np.load(fy, allow_pickle=False)
        sc = LabelScaler.load(self.root, "norm_aux.npz")
        have = [str(s) for s in z["labels"]]
        missing = [n for n in self.aux_labels if n not in have]
        if missing:
            raise ValueError(f"aux labels {missing} not in {fy} (available: {have})")
        full_sid = full_cat.sort_values("row")["source_id"].to_numpy()
        if len(full_sid) != len(z["y"]) or not np.array_equal(
                z["source_id"].astype(np.int64), full_sid.astype(np.int64)):
            raise ValueError(f"{fy} is not aligned with catalog row order (source_id mismatch)")
        col = [have.index(n) for n in self.aux_labels]
        if self.aux_ref == "h":                                  # [X/H] as stored
            y = z["y"][self.indices][:, col]
            yerr = z["yerr"][self.indices][:, col]
            self.mask_aux = z["mask"][self.indices][:, col].astype(bool)
            self.aux_scaler = LabelScaler(sc.mean[col], sc.std[col], sc.floor_norm[col],
                                          self.aux_labels)
        else:                                                    # [X/Fe] = X_H - fe_h
            y_all, e_all, m_all, floor_raw = self._aux_over_fe(z, sc, col, full_cat)
            tr = (full_cat.sort_values("row")["split"] == "train").to_numpy()
            self.aux_scaler = LabelScaler.fit(y_all[tr], e_all[tr], m_all[tr], floor_raw,
                                              self.aux_labels)
            y, yerr, self.mask_aux = y_all[self.indices], e_all[self.indices], m_all[self.indices]
        self.y_aux_raw, self.yerr_aux_raw = y, yerr
        self.y_aux = self.aux_scaler.transform_y(y).astype(np.float32)
        self.yerr_aux = self.aux_scaler.transform_err(yerr).astype(np.float32)
        self.floor_aux = self.aux_scaler.floor_norm.astype(np.float32)

    def _load_astro(self, full_cat: pd.DataFrame) -> None:
        """Per-star parallax likelihood inputs and observed colours for the distance model.
        Everything refers to the ORIGINAL star (never touched by the noise augmentation).
        plx_ok = finite parallax and error > 0 (no S/N or sign cut)."""
        from ..models.distance import BANDS, COLOURS, PLX_ZP_FALLBACK
        f = self.root / "astrometry_phot.parquet"
        if not f.is_file():
            raise FileNotFoundError(f"{f} missing; run scripts/build_astrometry_phot.py")
        t = pd.read_parquet(f).sort_values("row").reset_index(drop=True)
        full_sid = full_cat.sort_values("row")["source_id"].to_numpy().astype(np.int64)
        if len(t) != len(full_sid) or not np.array_equal(t["source_id"].to_numpy(np.int64), full_sid):
            raise ValueError(f"{f} is not aligned with catalog row order (source_id mismatch)")
        t = t.iloc[self.indices].reset_index(drop=True)
        self.astro_table = t                           # l, b, ra, dec, raw photometry for evaluation
        g = t["phot_g_mean_mag"].to_numpy(np.float64)       # Gaia G of the original star
        plx, e = t["parallax"].to_numpy(np.float64), t["parallax_error"].to_numpy(np.float64)
        zp = t["zpt"].to_numpy(np.float64) if "zpt" in t else np.full(len(t), PLX_ZP_FALLBACK)
        zp = np.where(np.isfinite(zp), zp, PLX_ZP_FALLBACK)
        ok = np.isfinite(plx) & np.isfinite(e) & (e > 0) & np.isfinite(g)
        mag = {"G": g, "BP": t["phot_bp_mean_mag"], "RP": t["phot_rp_mean_mag"], "J": t["j_m"],
               "H": t["h_m"], "K": t["ks_m"], "W1": t["w1mpro"], "W2": t["w2mpro"]}
        err = {"G": t["e_g"], "BP": t["e_bp"], "RP": t["e_rp"], "J": t["j_msigcom"],
               "H": t["h_msigcom"], "K": t["ks_msigcom"], "W1": t["w1mpro_error"], "W2": t["w2mpro_error"]}
        good = {"G": np.ones(len(t), bool), "BP": np.ones(len(t), bool), "RP": np.ones(len(t), bool),
                "J": t["tmass_ok_j"], "H": t["tmass_ok_h"], "K": t["tmass_ok_k"],
                "W1": t["wise_ok_w1"], "W2": t["wise_ok_w2"]}
        mag = {b: np.asarray(mag[b], np.float64) for b in BANDS}
        err = {b: np.asarray(err[b], np.float64) for b in BANDS}
        good = {b: np.asarray(good[b], bool) & np.isfinite(mag[b]) & np.isfinite(err[b]) & (err[b] > 0)
                for b in BANDS}
        col = np.stack([mag[a] - mag[b] for a, b in COLOURS], 1)
        cerr = np.stack([np.hypot(err[a], err[b]) for a, b in COLOURS], 1)
        cok = np.stack([good[a] & good[b] for a, b in COLOURS], 1)
        # A0 anchors, TRAIN split only (test stars are never anchors -> fair evaluation)
        from ..models.distance import anchor_targets
        ebv = t["ebv_sfd"].to_numpy(float) if "ebv_sfd" in t else np.full(len(t), np.nan)
        tgt, asig, aset = anchor_targets(plx - zp, e, t["b"].to_numpy(float), ebv)
        aset = np.where(ok & (self.split == "train"), aset, 0).astype(np.int8)
        self.anchor_set = aset
        self.astro_arr = {
            "anc_t": tgt.astype(np.float32), "anc_s": asig.astype(np.float32), "anc_ok": aset > 0,
            "g": g.astype(np.float32), "plx": np.nan_to_num(plx).astype(np.float32),
            "plx_err": np.where(ok, e, 1.0).astype(np.float32), "zp": zp.astype(np.float32),
            "plx_ok": ok, "col": np.nan_to_num(col).astype(np.float32),
            "col_err": np.where(cok, cerr, 1.0).astype(np.float32), "col_ok": cok,
            "mag": np.nan_to_num(np.stack([mag[b] for b in BANDS], 1)).astype(np.float32),
            "mag_ok": np.stack([good[b] for b in BANDS], 1),
        }

    def _load_latent(self, full_cat: pd.DataFrame, source: str) -> None:
        if source != "v1_latent":
            raise ValueError(f"model.latent_head.source must be v1_latent, got {source!r}")
        fl, fn = self.root / "v1_latent.npy", self.root / "v1_latent_norm.npz"
        for f in (fl, fn):
            if not f.is_file():
                raise FileNotFoundError(f"latent_head.source=v1_latent but {f} is missing; "
                                        f"run scripts/build_v1_latent.py --root {self.root}")
        z = np.load(fn, allow_pickle=False)
        full_sid = full_cat.sort_values("row")["source_id"].to_numpy().astype(np.int64)
        self.lat_arr = np.load(fl, mmap_mode="r")
        if len(full_sid) != len(self.lat_arr) or not np.array_equal(z["source_id"], full_sid):
            raise ValueError(f"{fl} is not aligned with catalog row order (source_id mismatch)")
        self.lat_mean = z["mean"].astype(np.float32)
        self.lat_std = np.maximum(z["std"].astype(np.float32), 1e-4)
        self.lat_valid = z["valid"].astype(bool)
        self.latent_dim = int(self.lat_arr.shape[1])

    def _aux_over_fe(self, z, sc, col, full_cat):
        """[X/Fe] targets for ALL catalog rows: y = X_H - fe_h (main label),
        err = sqrt(e_X^2 + e_Fe^2) -- the correlated part of the two ASPCAP
        errors is ignored (conservative), floor = sqrt(floor_X^2 + floor_Fe^2)
        in raw units, mask = mask_X & mask_Fe. Normalisation stats are fit on the
        TRAIN split (all train rows of the catalog, independent of which split
        this dataset instance holds) at load time, so every instance agrees."""
        zm = np.load(self.root / "labels.npz")
        main = [str(s) for s in zm["labels"]]
        if "fe_h" not in main:
            raise ValueError("labels.aux_ref=fe needs fe_h among the main labels")
        k = main.index("fe_h")
        msc = LabelScaler.load(self.root)
        fe, efe, mfe = zm["y"][:, k], zm["yerr"][:, k], zm["mask"][:, k].astype(bool)
        m_all = z["mask"][:, col].astype(bool) & mfe[:, None]
        y_all = np.where(m_all, z["y"][:, col] - fe[:, None], np.nan)
        e_all = np.where(m_all, np.sqrt(z["yerr"][:, col] ** 2 + efe[:, None] ** 2), np.nan)
        floor_fe = msc.floor_norm[k] * msc.std[k]
        floor_raw = np.sqrt((sc.floor_norm[col] * sc.std[col]) ** 2 + floor_fe ** 2)
        return y_all.astype(np.float32), e_all.astype(np.float32), m_all, floor_raw

    def __len__(self) -> int:
        return len(self.catalog)

    def __getitem__(self, i: int) -> dict:
        xi = np.asarray(self.x[self.indices[i]], dtype=np.float32)
        row = self.indices[i]
        extra = {}
        if self.noise:
            extra = {"sigma": torch.from_numpy(np.array(self.sigma[row], np.float32)),
                     "side": torch.from_numpy(np.array(self.side[row], np.float32)),
                     "coef": torch.from_numpy(np.array(self.coef[row], np.float32)),
                     "valid": torch.tensor(bool(self.valid[row]))}
        if self.latent:
            lat = (np.asarray(self.lat_arr[row], np.float32) - self.lat_mean) / self.lat_std
            extra = {**extra, "lat": torch.from_numpy(lat),
                     "lat_ok": torch.tensor(bool(self.lat_valid[row]))}
        if self.aux_labels:
            extra = {**extra,
                     "y_aux": torch.from_numpy(np.nan_to_num(self.y_aux[i])),
                     "yerr_aux": torch.from_numpy(np.nan_to_num(self.yerr_aux[i])),
                     "mask_aux": torch.from_numpy(self.mask_aux[i])}
        if self.astro:
            extra = {**extra, "astro": {k: torch.as_tensor(v[i]) for k, v in self.astro_arr.items()}}
        return {
            **extra,
            "x": torch.from_numpy(np.array(xi, dtype=np.float32, copy=True)),
            "y": torch.from_numpy(np.nan_to_num(self.y[i])),
            "yerr": torch.from_numpy(np.nan_to_num(self.yerr[i])),
            "mask": torch.from_numpy(self.mask[i]),
            "index": self.indices[i],
        }


def input_dim(input: str) -> int:
    if input == "xp343":
        return 343
    if input in ("coef110", "coef110n"):
        return 110
    raise ValueError(f"unknown input {input!r}")


def load_manifest(root: str | Path) -> dict:
    return json.loads((Path(root) / "manifest.json").read_text())
