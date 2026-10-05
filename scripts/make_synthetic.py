#!/usr/bin/env python
"""Synthetic processed-dataset generator for smoke tests (NO science content):
labels correlated with the spectrum through a planted linear relation, so a
working trainer must recover it."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aspgapv3.data.dataset import LabelScaler, save_catalog  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/aspgap_smoke/data")
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
n = 900
labels = ["teff", "logg", "m_h", "alpha_m", "fe_h", "mg_h"]

# latent stellar params
teff = rng.normal(5200, 600, n)
logg = rng.normal(2.8, 1.0, n)
mh = rng.normal(-0.3, 0.4, n)
alpha = 0.15 * mh + 0.05 * rng.normal(size=n)          # planted chem correlation
fe = mh + 0.03 * rng.normal(size=n)
mg = alpha + fe + 0.03 * rng.normal(size=n)

# spectrum: smooth function of params + noise
wav = np.linspace(336, 1020, 343)
base = 1 + 0.1 * np.sin(wav / 60.0)
x = np.empty((n, 343), np.float32)
for i in range(n):
    slope = 1e-4 * (teff[i] - 5200) / 600 + 0.02 * mh[i]
    line = -0.03 * alpha[i] * np.exp(-((wav - 650) / 25) ** 2) \
         - 0.05 * np.clip(logg[i] - 2.5, -2, 2) * np.exp(-((wav - 850) / 40) ** 2)
    x[i] = base * (1 + slope) + line + 0.01 * rng.normal(size=343)
x = (x / x[:, [107]]).astype(np.float32)

y = np.stack([teff, logg, mh, alpha, fe, mg], 1).astype(np.float32)
yerr = np.stack([np.full(n, 60.0), np.full(n, 0.08), np.full(n, 0.04),
                 np.full(n, 0.03), np.full(n, 0.03), np.full(n, 0.03)], 1).astype(np.float32)
mask = rng.random((n, 6)) > 0.15                        # per-label missingness
mask[:, 0] = True                                       # teff always there (domain cut)
y[~mask] = np.nan
yerr[~mask] = np.nan

splits = rng.choice(["train", "val", "iid_test", "sky_test"], n,
                    p=[0.8, 0.07, 0.07, 0.06])
cat = pd.DataFrame({
    "row": np.arange(n),
    "source_id": np.arange(5000, 5000 + n),
    "snr": rng.uniform(50, 300, n),
    "phot_g_mean_mag": rng.uniform(10, 17, n),
    "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-70, 70, n),
    "split": splits,
    "teff": y[:, 0], "teff_err": yerr[:, 0], "teff_ok": mask[:, 0],
    "logg": y[:, 1], "logg_err": yerr[:, 1], "logg_ok": mask[:, 1],
    "m_h": y[:, 2], "m_h_err": yerr[:, 2], "m_h_ok": mask[:, 2],
    "alpha_m": y[:, 3], "alpha_m_err": yerr[:, 3], "alpha_m_ok": mask[:, 3],
    "fe_h": y[:, 4], "fe_h_err": yerr[:, 4], "fe_h_ok": mask[:, 4],
    "mg_h": y[:, 5], "mg_h_err": yerr[:, 5], "mg_h_ok": mask[:, 5],
})

floor = np.array([50.0, 0.05, 0.02, 0.02, 0.02, 0.02])
tr = cat["split"] == "train"
scaler = LabelScaler.fit(y[tr.to_numpy()], yerr[tr.to_numpy()],
                         mask[tr.to_numpy()], floor, labels)
scaler.save(root)
np.savez(root / "labels.npz", y=y, yerr=yerr, mask=mask,
         labels=np.array(labels, dtype="U16"))
np.save(root / "xp343.npy", x)
np.save(root / "coef110.npy", rng.normal(size=(n, 110)).astype(np.float32))
np.save(root / "coef110n.npy", rng.normal(size=(n, 110)).astype(np.float32))
save_catalog(cat, root)
print(f"synthetic dataset -> {root} ({n} stars, splits={cat['split'].value_counts().to_dict()})")
