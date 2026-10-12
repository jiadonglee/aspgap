#!/usr/bin/env python
"""Export a trained run to the single self-contained inference file weights/aspgap_v3_ext_r2.pt.

    python scripts/export_weights.py --run runs/aspgap_dist_ext_r2 --out weights/aspgap_v3_ext_r2.pt

Needs the (private) training data root for the input scaler, label statistics and the sigma thresholds.
Collaborators do NOT need this script: the exported file is committed to the repository.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aspgapv3.models.distance import EXT_K  # noqa: E402

LABELS = ["teff", "logg", "m_h", "alpha_m"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/aspgap_dist_ext_r2")
    ap.add_argument("--out", default="weights/aspgap_v3_ext_r2.pt")
    ap.add_argument("--sigma-quantile", type=float, default=0.95)
    args = ap.parse_args()
    run = Path(args.run)

    ck = torch.load(run / "ckpt_best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    assert ck["labels"] == LABELS and ck["n_atm"] == 3
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root

    norm = np.load(root / "norm.npz")
    idx = [list(norm["labels"]).index(n) for n in LABELS]
    sc = np.load(root / "aspgap_scaler.npz")

    # training domain
    import pandas as pd
    cat = pd.read_parquet(root / "catalog.parquet", columns=["row", "split", "phot_g_mean_mag"])
    lab = np.load(root / "labels.npz")
    tr = cat[cat["split"] == "train"]["row"].to_numpy()
    domain = {"g_max": float(cfg["train"]["augment"]["noise_match"]["g_max"]),
              "g_min_train": float(np.nanmin(cat.loc[cat["split"] == "train", "phot_g_mean_mag"]))}
    for k, n in enumerate(LABELS):
        j = list(lab["labels"]).index(n)
        v = lab["y"][tr, j][lab["mask"][tr, j].astype(bool)]
        domain[n] = {"min": float(v.min()), "max": float(v.max()),
                     "p01": float(np.percentile(v, 1)), "p99": float(np.percentile(v, 99)), "n": int(v.size)}
    domain["teff_edge"] = 7500.0
    domain["notes"] = ("Trained on APOGEE/ASPCAP (astra 0.6.0) (Teff<8000 K, SNR>50) + GALAH DR4, GC members, "
                       "LAMOST VMP labels; G <= 17.65; Teff saturates near 8000 K.")

    # sigma thresholds: quantile of predicted sigma on the APOGEE iid_test set
    p = np.load(run / "predictions_iid_test.npz")
    pd_ = np.load(run / "predictions_dist_iid_test.npz")
    sig = {n: float(np.quantile(p["sigma"][:, k], args.sigma_quantile))
           for k, n in enumerate(LABELS)}   # predictions npz already holds raw-unit sigma
    sig["mg"] = float(np.quantile(np.exp(0.5 * np.clip(pd_["logvar_mg"], -8, 8)), args.sigma_quantile))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

    out = {
        "format_version": 1,
        "version": "aspgap_v3_ext_r2",
        "source": {"run": run.name, "git_commit": commit, "checkpoint": "ckpt_best.pt",
                   "epoch": int(ck["epoch"]), "val_loss": float(ck["val_loss"]),
                   "train_config": "configs/aspgap_dist_ext2.yaml"},
        "state_dict": {k: v.detach().float().contiguous() for k, v in ck["state_dict"].items()},
        "model_config": {"model": cfg["model"], "n_atm": ck["n_atm"], "n_chem": len(LABELS) - ck["n_atm"]},
        "input": {"n_coef": 110, "g_ref": 15.0, "g_exponent": 0.4,
                  "center": torch.tensor(sc["center"], dtype=torch.float64),
                  "scale": torch.tensor(sc["scale"], dtype=torch.float64),
                  "description": "x = (coef / 10**(0.4*(15-G)) - center) / scale; coef = [BP(55), RP(55)]"},
        "labels": LABELS,
        "label_units": {"teff": "K", "logg": "dex (cgs)", "m_h": "dex", "alpha_m": "dex", "mg": "mag", "a0": "mag"},
        "label_norm": {"mean": torch.tensor(norm["mean"][idx], dtype=torch.float64),
                       "std": torch.tensor(norm["std"][idx], dtype=torch.float64)},
        "logvar_clamp": [float(v) for v in cfg["train"]["logvar_clamp"]],
        "distance": {"k_g": EXT_K["G"], "mg_mean": float(ck["state_dict"]["head_dist.mg_mean"]),
                     "mg_scale": float(ck["state_dict"]["head_dist.mg_scale"]),
                     "formula": "dist_mod = G - M_G - k_g*A0; distance_pc = 10**(dist_mod/5 + 1)"},
        "domain": domain,
        "sigma_thresholds": {"quantile": args.sigma_quantile, "reference": "APOGEE iid_test", **sig},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e6:.2f} MB)")
    print("domain", domain)
    print("sigma thresholds", out["sigma_thresholds"])


if __name__ == "__main__":
    main()
