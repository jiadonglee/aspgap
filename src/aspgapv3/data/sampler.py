"""Binned weighted sampler: keep RGB/solar-metallicity from dominating.

p(bin) ∝ 1 / N_bin^power  (power=0.5 default: oversample rare bins without the
pathologies of full 1/N), uniform within a bin. Bins are (teff, logg, [M/H])
in raw units from the catalog.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler

BIN_DEFAULTS = {"teff": 300.0, "logg": 0.5, "m_h": 0.25}


class BinnedWeightedSampler(Sampler[int]):
    def __init__(self, bin_values: dict[str, np.ndarray], power: float = 0.5,
                 bins: dict[str, float] | None = None, seed: int = 42,
                 num_samples: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.power = float(power)
        widths = {**BIN_DEFAULTS, **(bins or {})}
        keys = list(widths)
        V = np.stack([np.asarray(bin_values[k], float) for k in keys], 1)
        ok = np.all(np.isfinite(V), 1)
        edges = [np.arange(np.nanmin(V[:, j]), np.nanmax(V[:, j]) + widths[k],
                           widths[k]) for j, k in enumerate(keys)]
        idx_ok = np.where(ok)[0]
        hist, _ = np.histogramdd(V[idx_ok], bins=edges)
        cell = np.ravel_multi_index(
            tuple(np.clip(np.searchsorted(e, V[:, j], "right") - 1, 0, len(e) - 2)
                  for j, e in enumerate(edges)), tuple(len(e) - 1 for e in edges))
        flat = hist.reshape(-1)
        cell_count = np.maximum(flat, 1)
        p_cell = 1.0 / np.power(cell_count, self.power)
        p_cell[flat == 0] = 0.0
        w = np.zeros(len(V), dtype=np.float64)
        w[idx_ok] = p_cell[cell[idx_ok]] / cell_count[cell[idx_ok]]
        if w.sum() <= 0:
            w = np.ones(len(V)) / len(V)
        self.weights = w / w.sum()
        self.num_samples = int(num_samples if num_samples is not None else len(V))

    def __iter__(self):
        draws = self.rng.choice(len(self.weights), size=self.num_samples,
                                replace=True, p=self.weights)
        return iter(draws.tolist())

    def __len__(self) -> int:
        return self.num_samples


def make_sampler(dataset, cfg_sampler: dict) -> Sampler[int] | None:
    if cfg_sampler.get("type", "binned") == "random":
        return None
    cat = dataset.catalog
    vals = {k: (cat[k].to_numpy(float) if k in cat else np.full(len(cat), np.nan))
            for k in BIN_DEFAULTS}
    return BinnedWeightedSampler(
        vals, power=cfg_sampler.get("power", 0.5),
        bins=cfg_sampler.get("bins", None), seed=cfg_sampler.get("seed", 42),
        num_samples=len(dataset),
    )
