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

    def save(self, root: str | Path) -> None:
        np.savez(Path(root) / "norm.npz", mean=self.mean, std=self.std,
                 floor_norm=self.floor_norm,
                 labels=np.array(self.labels, dtype="U16"))

    @classmethod
    def load(cls, root: str | Path) -> "LabelScaler":
        z = np.load(Path(root) / "norm.npz", allow_pickle=False)
        return cls(z["mean"], z["std"], z["floor_norm"],
                   [str(s) for s in z["labels"]])


# ------------------------------------------------------------------ dataset

class APOGEEDataset(Dataset):
    """One split of the processed ASPCAP x Gaia-XP sample."""

    def __init__(self, root: str | Path, split: str = "train",
                 input: str = "xp343", labels: list[str] | None = None,
                 noise: bool = False, require_xp_valid: bool = False):
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
        assert self.labels == scaler.labels, (
            f"label order mismatch: {self.labels} vs {scaler.labels}"
        )
        col = {name: i for i, name in enumerate(scaler.labels)}

        y = z["y"][self.indices][:, [col[n] for n in self.labels]]
        yerr = z["yerr"][self.indices][:, [col[n] for n in self.labels]]
        mask = z["mask"][self.indices][:, [col[n] for n in self.labels]].astype(bool)

        self.y = scaler.transform_y(y).astype(np.float32)
        self.yerr = scaler.transform_err(yerr).astype(np.float32)
        self.mask = mask
        self.y_raw = y
        self.yerr_raw = yerr
        self.scaler = scaler
        self.floor_norm = scaler.floor_norm[[col[n] for n in self.labels]] \
            .astype(np.float32)

        # per-label sample weights can be attached later by the trainer
        self.label_weights = np.ones(len(self.labels), dtype=np.float32)

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
