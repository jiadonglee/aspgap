#!/usr/bin/env python
"""Cross-check our per-pixel sigma / flux against gaiaxpy's own calibrate().

Run in the isolated `xpbuild` env (gaiaxpy 2.1.4, numpy 2, no h5py):

    source ~/miniforge3/bin/activate xpbuild
    python scripts/verify_xp_gaiaxpy.py --data data/processed/aspcap_v0.6.0_test

Input: <data>/xpnoise_verify_raw.npz written by build_xp_noise.py (raw DR3 rows
of N stars + our sigma_abs / flux_abs). gaiaxpy calibrate() is run on the same
rows with the same 343-point grid; we compare flux_error and flux.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
args = ap.parse_args()
root = Path(args.data)
z = np.load(root / "xpnoise_verify_raw.npz")
n = len(z["source_id"])

df = pd.DataFrame({"source_id": z["source_id"]})
for b in ("bp", "rp"):
    df[f"{b}_n_parameters"] = 55
    df[f"{b}_basis_function_id"] = 56 if b == "bp" else 57
    df[f"{b}_degrees_of_freedom"] = 100
    df[f"{b}_n_measurements"] = 100
    df[f"{b}_n_relevant_bases"] = 55
    df[f"{b}_n_rejected_measurements"] = 0
    df[f"{b}_chi_squared"] = 1.0
    df[f"{b}_relative_shrinking"] = 0.0
    df[f"{b}_standard_deviation"] = z[f"{b}_standard_deviation"]
    df[f"{b}_coefficients"] = list(z[f"{b}_coefficients"])
    df[f"{b}_coefficient_errors"] = list(z[f"{b}_coefficient_errors"])
    df[f"{b}_coefficient_correlations"] = list(z[f"{b}_coefficient_correlations"])
df["solution_id"] = 0

from gaiaxpy import calibrate  # noqa: E402

grid = np.arange(336.0, 1021.0, 2.0)
out, _ = calibrate(df, sampling=grid, truncation=False, save_file=False)
g_flux = np.stack(out["flux"].to_numpy())
g_err = np.stack(out["flux_error"].to_numpy())
my_err, my_flux = z["my_sigma_abs"].astype(np.float64), z["my_flux_abs"].astype(np.float64)
rel_err = np.abs(my_err - g_err) / g_err
rel_flux = np.abs(my_flux - g_flux) / np.abs(g_flux).max(1, keepdims=True)
res = {"n_stars": int(n),
       "sigma_max_rel_diff": float(rel_err.max()),
       "sigma_median_rel_diff": float(np.median(rel_err)),
       "flux_max_diff_over_star_max": float(rel_flux.max()),
       "gaiaxpy_flux_550_ratio_to_ours": float(np.median(g_flux[:, 107] / my_flux[:, 107]))}
print(json.dumps(res, indent=1))
(root / "xpnoise_verify_gaiaxpy.json").write_text(json.dumps(res, indent=1))
