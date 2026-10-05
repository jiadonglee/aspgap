"""Out-of-domain score: distance from a star's z_256 to the APOGEE training
manifold (mean distance to its k nearest training latents).

Shipped in the final catalog so users know when a prediction sits where APOGEE
never supported the model. Pure torch; chunked matmul, no sklearn dependency.
"""

from __future__ import annotations

import numpy as np
import torch


class LatentKnnOOD:
    def __init__(self, k: int = 8, chunk: int = 4096, max_train: int = 200000,
                 seed: int = 42):
        self.k, self.chunk, self.max_train = k, chunk, max_train
        self.seed = seed
        self._ref: torch.Tensor | None = None

    def fit(self, z_train: np.ndarray) -> "LatentKnnOOD":
        z = np.asarray(z_train, dtype=np.float32)
        if len(z) > self.max_train:
            rng = np.random.default_rng(self.seed)
            z = z[rng.choice(len(z), self.max_train, replace=False)]
        self._ref = torch.from_numpy(z)
        return self

    def score(self, z: np.ndarray, device: str = "cpu") -> np.ndarray:
        if self._ref is None:
            raise RuntimeError("fit on TRAIN latents first")
        ref = self._ref.to(device)
        out = np.empty(len(z), dtype=np.float32)
        with torch.no_grad():
            for lo in range(0, len(z), self.chunk):
                q = torch.from_numpy(np.asarray(z[lo:lo + self.chunk],
                                                dtype=np.float32)).to(device)
                # mean distance to the k nearest reference points
                d = torch.cdist(q, ref)                       # (b, n_ref)
                kk = min(self.k, ref.shape[0])
                out[lo:lo + len(q)] = d.topk(kk, largest=False).values.mean(1).cpu().numpy()
        return out
