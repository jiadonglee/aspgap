#!/usr/bin/env python
"""Evaluate a trained run: per-label metrics per split and domain slice, OOD
scores, and the conditional chemical-information experiment (XP ->
Delta[alpha/M], Delta[Mg/Fe] beyond Teff/logg/[M/H] correlations).

    python scripts/evaluate.py --run runs/specclip_ft_last2 \
        [--splits iid_test sky_test] [--which best]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aspgapv3.data.dataset import APOGEEDataset, load_catalog  # noqa: E402
from aspgapv3.losses import MaskedHeteroscedasticNLL  # noqa: E402
from aspgapv3.metrics import (derived_mgfe, fit_conditional_mean,  # noqa: E402
                              metrics_table, residual_recovery_score)
from aspgapv3.models.net import build_model  # noqa: E402
from aspgapv3.ood import LatentKnnOOD  # noqa: E402


@torch.no_grad()
def predict(net, dset: APOGEEDataset, device: str, batch: int = 1024):
    net.eval()
    loader = torch.utils.data.DataLoader(dset, batch_size=batch, shuffle=False,
                                         num_workers=2)
    mus, sigs, zs = [], [], []
    for b in loader:
        out = net(b["x"].to(device))
        mu = torch.cat([out["mu_atm"], out["mu_chem"]], 1).float().cpu().numpy()
        lv = torch.cat([out["logvar_atm"], out["logvar_chem"]], 1).clamp(-8, 8)
        sig = lv.exp().sqrt().float().cpu().numpy()
        mus.append(mu)
        sigs.append(sig)
        zs.append(out["z"].float().cpu().numpy())
    mu = np.concatenate(mus)
    sig = np.concatenate(sigs)
    z = np.concatenate(zs)
    mu_r, sig_r = dset.scaler.inverse(mu, sig)
    return mu_r, sig_r, z


def domain_slices(cat: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {"all": np.ones(len(cat), bool)}
    for c in ("is_metal_poor", "is_metal_poor_15", "is_mdwarf", "is_fstar",
              "is_faint_xp"):
        if c in cat:
            out[c.replace("is_", "")] = cat[c].to_numpy(bool)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", nargs="*", default=["iid_test", "sky_test"])
    ap.add_argument("--which", default="best", choices=["best", "last"])
    args = ap.parse_args()

    run = Path(args.run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(run / f"ckpt_{args.which}.pt", map_location=device,
                      weights_only=False)
    cfg = ckpt["config"]
    labels = ckpt["labels"]
    n_atm = ckpt["n_atm"]

    net = build_model(cfg, n_atm=n_atm, n_chem=len(labels) - n_atm).to(device)
    net.load_state_dict(ckpt["state_dict"])

    root = cfg["data"]["root"]
    cat_all = load_catalog(root)
    inp = cfg["data"].get("input", "xp343")

    # latent space for OOD, fit on TRAIN z
    train_dset = APOGEEDataset(root, "train", input=inp, labels=labels)
    _, _, z_train = predict(net, train_dset, device)
    ood = LatentKnnOOD(k=int(cfg["eval"].get("ood_k", 8)),
                       max_train=int(cfg["eval"].get("ood_max_train", 200000)))
    ood.fit(z_train)

    summary: list[str] = [f"# evaluation — {run.name} (ckpt_{args.which}, "
                          f"epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})\n"]

    for split in args.splits:
        dset = APOGEEDataset(root, split, input=inp, labels=labels)
        cat = cat_all[cat_all["split"] == split].reset_index(drop=True)
        mu, sig, z = predict(net, dset, device)
        y, m = dset.y_raw, dset.mask

        table = metrics_table(mu, sig, y, m, labels)
        table.to_csv(run / f"metrics_{split}.csv", index=False)
        summary.append(f"\n## {split} — per-label (raw units)\n")
        try:
            summary.append(table.to_markdown(index=False, floatfmt=".4f"))
        except ImportError:                       # tabulate missing
            summary.append(table.to_string(index=False, float_format="%.4f"))

        # domain slices for the chem labels
        rows = []
        for name, sel in domain_slices(cat).items():
            t = metrics_table(mu, sig, y, m, labels)
            for k, lab in enumerate(labels):
                r = t.iloc[k].to_dict()
                r.update({"domain": name,
                          "rmse_domain": float(np.sqrt(np.mean(
                              (mu[sel & m[:, k], k] - y[sel & m[:, k], k]) ** 2))
                              if (sel & m[:, k]).sum() else np.nan),
                          "n_domain": int((sel & m[:, k]).sum())})
                rows.append(r)
        pd.DataFrame(rows).to_csv(run / f"metrics_{split}_domains.csv", index=False)

        # OOD score for every star in this split
        np.savez(run / f"predictions_{split}.npz",
                 source_id=cat["source_id"].to_numpy(),
                 mu=mu, sigma=sig, z256=z,
                 ood_score=ood.score(z, device=device))

    # ---- conditional chemical-information experiment (pooled test sets) ----
    summary.append("\n## Conditional chemical information (beyond "
                   "Teff/logg/[M/H] correlations)\n")
    frames = []
    for split in args.splits:
        dset = APOGEEDataset(root, split, input=inp, labels=labels)
        cat = cat_all[cat_all["split"] == split].reset_index(drop=True)
        mu, sig, _ = predict(net, dset, device)
        frames.append((cat, mu, sig, dset.y_raw, dset.mask))
    cat = pd.concat([f[0] for f in frames], ignore_index=True)
    mu = np.concatenate([f[1] for f in frames])
    sig = np.concatenate([f[2] for f in frames])
    y = np.concatenate([f[3] for f in frames])
    m = np.concatenate([f[4] for f in frames])
    ix = {lab: i for i, lab in enumerate(labels)}

    # fit the conditional mean on TRAIN stars only (no leakage into the test)
    tr = train_dset
    tr_cat = cat_all[cat_all["split"] == "train"]
    lab = {n: i for i, n in enumerate(labels)}

    def train_cols(name):
        if name == "mg_fe":                  # derived label: [Mg/H] - [Fe/H]
            okm = tr.mask[:, lab["mg_h"]] & tr.mask[:, lab["fe_h"]]
            v = np.full(len(tr.y_raw), np.nan)
            v[okm] = tr.y_raw[okm, lab["mg_h"]] - tr.y_raw[okm, lab["fe_h"]]
            return v
        k = lab[name]
        ok = tr.mask[:, k]
        v = np.full(len(tr.y_raw), np.nan)
        v[ok] = tr.y_raw[ok, k]
        return v

    chem_targets = {}
    if "alpha_m" in ix:
        chem_targets["alpha_m"] = (mu[:, ix["alpha_m"]], y[:, ix["alpha_m"]],
                                   m[:, ix["alpha_m"]])
    if "mg_h" in ix and "fe_h" in ix:
        mgfe_mu, _ = derived_mgfe(mu[:, ix["mg_h"]], sig[:, ix["mg_h"]],
                                  mu[:, ix["fe_h"]], sig[:, ix["fe_h"]])
        mgfe_y = np.where(m[:, ix["mg_h"]] & m[:, ix["fe_h"]],
                          y[:, ix["mg_h"]] - y[:, ix["fe_h"]], np.nan)
        chem_targets["mg_fe"] = (mgfe_mu, mgfe_y,
                                 m[:, ix["mg_h"]] & m[:, ix["fe_h"]])

    for name, (pred, truth, valid) in chem_targets.items():
        # conditional mean fit on train, applied to test
        f, _, info = fit_conditional_mean(train_cols("teff"), train_cols("logg"),
                                          train_cols("m_h"), train_cols(name))
        ok_tr = np.isfinite(tr_cat["teff"].to_numpy(float))
        base = f(tr_cat["teff"].to_numpy(float)[ok_tr],
                 tr_cat["logg"].to_numpy(float)[ok_tr],
                 tr_cat["m_h"].to_numpy(float)[ok_tr])
        # residuals on the test sets
        teff_t = y[:, lab["teff"]] if "teff" in lab else cat["teff"].to_numpy(float)
        logg_t = y[:, lab["logg"]] if "logg" in lab else cat["logg"].to_numpy(float)
        mh_t = y[:, lab["m_h"]] if "m_h" in lab else cat["m_h"].to_numpy(float)
        base_t = f(teff_t, logg_t, mh_t)
        delta_true = truth - base_t
        delta_pred = pred - base_t
        res = residual_recovery_score(delta_true, delta_pred)
        # restrict to rows where the label itself is valid
        res_valid = residual_recovery_score(np.where(valid, delta_true, np.nan),
                                            np.where(valid, delta_pred, np.nan))
        summary.append(f"\n**{name}** (conditional-mean R² on train = "
                       f"{info['r2_conditional_mean']:.3f})\n")
        summary.append(f"- residual recovery (all rows): R² = "
                       f"{res.get('r2_delta', np.nan):.4f}, n = {res.get('n')}\n")
        summary.append(f"- residual recovery (label-valid rows): R² = "
                       f"{res_valid.get('r2_delta', np.nan):.4f}, "
                       f"pearson = {res_valid.get('pearson', np.nan):.4f}, "
                       f"n = {res_valid.get('n')}\n")

    (run / "summary.md").write_text("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
