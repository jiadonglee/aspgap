"""Shared toy fixtures: a tiny processed root and a tiny model config."""
import numpy as np
import pandas as pd

from aspgapv3.data.dataset import LabelScaler, save_catalog

MAIN6 = ["teff", "logg", "m_h", "alpha_m", "fe_h", "mg_h"]


def tiny_cfg(distance=False):
    """Small coefficient-MLP model config (4 labels: 3 atm + 1 chem)."""
    c = {"model": {"encoder": {"embed_dim": 64}, "trunk": {"dim": 64, "dropout": 0.0}}}
    if distance:
        c["model"]["distance"] = {"enabled": True, "colour_hidden": 16}
    return c


def toy_root(tmp_path, n=60, seed=0):
    """labels (6 stored), xp343 + coef110n arrays, 40 train / 10 val / 10 iid_test rows."""
    rng = np.random.default_rng(seed)
    K = len(MAIN6)
    y = rng.normal(size=(n, K)).astype(np.float32)
    mask = np.ones((n, K), bool)
    sc = LabelScaler.fit(y, np.full_like(y, 0.05), mask, np.full(K, 0.02), MAIN6)
    sc.save(tmp_path)
    np.savez(tmp_path / "labels.npz", y=y, yerr=np.full_like(y, .05), mask=mask,
             labels=np.array(MAIN6, dtype="U16"))
    np.save(tmp_path / "xp343.npy", rng.normal(1, .05, (n, 343)).astype(np.float32))
    np.save(tmp_path / "coef110n.npy", rng.normal(0, 1, (n, 110)).astype(np.float32))
    splits = np.array(["train"] * 40 + ["val"] * 10 + ["iid_test"] * 10)
    cat = pd.DataFrame({"row": np.arange(n), "source_id": np.arange(100, 100 + n), "split": splits})
    save_catalog(cat, tmp_path)
