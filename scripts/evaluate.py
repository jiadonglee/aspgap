#!/usr/bin/env python
"""Evaluate a trained run: per-label metrics per split and domain slice, OOD
scores, and the conditional chemical-information experiment (XP ->
Delta[alpha/M], Delta[Mg/Fe] beyond Teff/logg/[M/H] correlations).

    python scripts/evaluate.py --run runs/specclip_ft_last2 \
        [--splits iid_test sky_test] [--which best] [--degrade-g 15 16 17 17.5]

Degraded mode (needs build_xp_noise.py outputs, xp343 input): every star is
re-run after noise-matched degradation to a fixed G' (stars already fainter
than G' are left untouched), fixed seed, per-label RMSE / bias /
calibration_ratio vs G' -> metrics_degraded_<split>.csv. This is the DR4
validation curve: how the label error grows as the XP noise approaches the
faint end. Disable with --degrade-g (no values).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aspgapv3.amp import amp_settings, autocast_ctx, float_outputs  # noqa: E402
from aspgapv3.augment import NoiseMatchAugment, apply_aspgap_input  # noqa: E402
from aspgapv3.data.xp import aspgap_scaler_arrays  # noqa: E402
from aspgapv3.data.dataset import APOGEEDataset  # noqa: E402
from aspgapv3.dist_eval import distance_tables, extinction_report, red_clump  # noqa: E402
from aspgapv3.losses import MaskedHeteroscedasticNLL  # noqa: E402
from aspgapv3.metrics import (derived_mgfe, fit_conditional_mean,  # noqa: E402
                              metrics_table, residual_recovery_score)
from aspgapv3.models.net import build_model  # noqa: E402
from aspgapv3.models.side import build_side  # noqa: E402
from aspgapv3.ood import LatentKnnOOD  # noqa: E402


@torch.no_grad()
def predict(net, dset: APOGEEDataset, device: str, batch: int = 1024, amp=(False, torch.bfloat16),
            aug: NoiseMatchAugment | None = None, fixed_g: float | None = None,
            seed: int = 0, aspgap_sc: dict | None = None, dist_out: dict | None = None):
    """Returns (mu_raw, sigma_raw, z256, frac_degraded). With `aug`, every
    batch is degraded to fixed_g (see augment.py) using a seeded generator."""
    net.eval()
    loader = torch.utils.data.DataLoader(dset, batch_size=batch, shuffle=False,
                                         num_workers=2)
    gen = torch.Generator(device=device).manual_seed(seed) if aug is not None else None
    mus, sigs, zs, n_aug = [], [], [], 0
    dd = {"mg": [], "logvar_mg": [], "a0": []}
    for b in loader:
        if aug is not None:
            b = aug(b, gen=gen, fixed_g=fixed_g)
            n_aug += int(b["aug"].sum())
        x = b["x"].to(device)
        if aug is not None and aspgap_sc is not None:
            # coef110n baseline: re-apply the AspGap input transform to the
            # degraded raw coefficients at their new G'
            x = apply_aspgap_input(b, aspgap_sc, device)
        with autocast_ctx(*amp, device=device):
            out = net(x, build_side(net, b))
        out = float_outputs(out)
        mu = torch.cat([out["mu_atm"], out["mu_chem"]], 1).cpu().numpy()
        lv = torch.cat([out["logvar_atm"], out["logvar_chem"]], 1).clamp(-8, 8)
        mus.append(mu)
        sigs.append(lv.exp().sqrt().cpu().numpy())
        zs.append(out["z"].cpu().numpy())
        if dist_out is not None and "mg" in out:
            for k in dd:
                dd[k].append(out[k].cpu().numpy())
    if dist_out is not None and dd["mg"]:
        dist_out.update({k: np.concatenate(v) for k, v in dd.items()})
    mu = np.concatenate(mus)
    sig = np.concatenate(sigs)
    z = np.concatenate(zs)
    mu_r, sig_r = dset.scaler.inverse(mu, sig)
    return mu_r, sig_r, z, n_aug / max(len(dset), 1)


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        return df.to_string(index=False, float_format="%.4f")


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
    ap.add_argument("--degrade-g", nargs="*", type=float, default=[15.0, 16.0, 17.0, 17.5])
    ap.add_argument("--degrade-seed", type=int, default=1234)
    ap.add_argument("--ref", default="aspgap110_aug_rand_r2",
                    help="reference run (dir name beside --run, or a path) for the label-metric deltas; '' = off")
    args = ap.parse_args()

    run = Path(args.run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(run / f"ckpt_{args.which}.pt", map_location=device,
                      weights_only=False)
    cfg = ckpt["config"]
    labels = ckpt["labels"]
    n_atm = ckpt["n_atm"]

    use_dist = bool((cfg["model"].get("distance") or {}).get("enabled", False))
    net = build_model(cfg, n_atm=n_atm, n_chem=len(labels) - n_atm,
                      n_aux=int(ckpt.get("n_aux", 0))).to(device)
    net.load_state_dict(ckpt["state_dict"])

    root = cfg["data"]["root"]
    inp = cfg["data"].get("input", "xp343")
    side_cfg = cfg["model"].get("side") or {}
    use_amp, amp_dtype = amp_settings(cfg, device)
    amp = (use_amp and bool(cfg.get("eval", {}).get("amp", False)), amp_dtype)   # fp32 by default
    root_p = Path(root)
    have_noise = all((root_p / f).is_file() for f in
                     ("xp343_sigma.npy", "side.npy", "xpcov_valid.npy", "noise_bank.npz"))
    degrade = bool(args.degrade_g) and inp in ("xp343", "coef110n") and have_noise
    aspgap_sc = aspgap_scaler_arrays(root) if (degrade and inp == "coef110n") else None
    need_noise = degrade or bool(side_cfg.get("scalars") or side_cfg.get("noise_channel"))
    kw = dict(input=inp, labels=labels, noise=need_noise, astro=use_dist,
              require_xp_valid=bool(cfg["data"].get("require_xp_valid", False)))
    aug = NoiseMatchAugment.from_root(root, device=device, p=1.0,
                                      g_max=float(((cfg["train"].get("augment") or {})
                                                   .get("noise_match") or {}).get("g_max", 17.65))) \
        if degrade else None

    # latent space for OOD, fit on TRAIN z
    train_dset = APOGEEDataset(root, "train", **kw)
    _, _, z_train, _ = predict(net, train_dset, device, amp=amp)
    ood = LatentKnnOOD(k=int(cfg["eval"].get("ood_k", 8)),
                       max_train=int(cfg["eval"].get("ood_max_train", 200000)))
    ood.fit(z_train)

    summary: list[str] = [f"# evaluation — {run.name} (ckpt_{args.which}, "
                          f"epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})\n"]

    for split in args.splits:
        dset = APOGEEDataset(root, split, **kw)
        cat = dset.catalog      # already filtered (require_xp_valid) like the predictions
        dout: dict = {}
        mu, sig, z, _ = predict(net, dset, device, amp=amp, dist_out=dout)
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

        if use_dist:
            dtab, _ = distance_tables(dout["mg"], dout["logvar_mg"], dout["a0"], dset.astro_arr)
            dtab.to_csv(run / f"metrics_distance_{split}.csv", index=False)
            summary.append(f"\n## {split} — distance: parallax pull / M_G residual (parallax S/N>10) / "
                           "fractional distance precision, by G (first row = all)\n")
            summary.append(_md(dtab))
            rc = red_clump(y, m, labels, dset.astro_arr, dout["mg"], dout["a0"],
                           dset.astro_arr["mag_ok"][:, 5])
            ext = extinction_report(dout["a0"], dset.astro_table, y[:, labels.index("teff")], mu[:, labels.index("teff")],
                                    nearby=(dset.astro_arr["plx"] - dset.astro_arr["zp"] > 10)
                                    & ((dset.astro_arr["plx"] - dset.astro_arr["zp"]) > 10 * dset.astro_arr["plx_err"]))
            summary.append(f"\n**{split} red clump** (2.3<logg<2.6, 4500<Teff<5100, [M/H]>-0.6): {rc}\n")
            summary.append(f"\n**{split} extinction**: {ext}\n")
            np.savez(run / f"predictions_dist_{split}.npz", source_id=cat["source_id"].to_numpy(), **dout)

        # OOD score for every star in this split
        np.savez(run / f"predictions_{split}.npz",
                 source_id=cat["source_id"].to_numpy(),
                 mu=mu, sigma=sig, z256=z,
                 ood_score=ood.score(z, device=device))

    # ---- degraded-G' evaluation (DR4 validation curve) ------------------------
    if args.degrade_g and not degrade:
        summary.append("\n## Degraded-G' evaluation skipped (needs xp343 input + "
                       "build_xp_noise.py outputs incl. noise_bank.npz)\n")
    if degrade:
        for split in args.splits:
            dset = APOGEEDataset(root, split, **kw)
            rows, drows = [], []
            for gp in [None] + list(args.degrade_g):
                dout = {}
                mu, sig, _, frac = predict(net, dset, device, amp=amp, aug=aug, dist_out=dout,
                                           fixed_g=gp, seed=args.degrade_seed, aspgap_sc=aspgap_sc) \
                    if gp is not None else (*predict(net, dset, device, amp=amp, dist_out=dout)[:3], 0.0)
                if use_dist:
                    dt, _ = distance_tables(dout["mg"], dout["logvar_mg"], dout["a0"], dset.astro_arr)
                    dt = dt.iloc[[0]].copy()          # all-G row
                    dt.insert(0, "g_prime", np.nan if gp is None else gp)
                    drows.append(dt)
                t = metrics_table(mu, sig, dset.y_raw, dset.mask, labels)
                t.insert(0, "g_prime", np.nan if gp is None else gp)
                t.insert(1, "frac_degraded", frac)
                rows.append(t)
            deg = pd.concat(rows, ignore_index=True)
            deg.to_csv(run / f"metrics_degraded_{split}.csv", index=False)
            if use_dist:
                dd_ = pd.concat(drows, ignore_index=True)
                dd_.to_csv(run / f"metrics_distance_degraded_{split}.csv", index=False)
                summary.append(f"\n## {split} — M_G / parallax pull vs degraded G' (M_G residual: parallax S/N>10 "
                               "stars, original-star G and parallax; all-G row)\n")
                summary.append(_md(dd_[["g_prime", "n", "pull_median", "pull_rstd", "n_hi_snr", "MG_resid_rmse",
                                        "MG_resid_mad", "MG_resid_median", "frac_dist_pred_median"]]))
            summary.append(f"\n## {split} — degraded to fixed G' (seed {args.degrade_seed}; "
                           "g_prime NaN = native)\n")
            cols = ["g_prime", "frac_degraded", "label", "n", "rmse", "bias",
                    "calibration_ratio"]
            try:
                summary.append(deg[cols].to_markdown(index=False, floatfmt=".4f"))
            except ImportError:
                summary.append(deg[cols].to_string(index=False, float_format="%.4f"))

    if args.ref:
        refd = run.parent / args.ref if not Path(args.ref).is_dir() else Path(args.ref)
        summary.append(f"\n## Label metrics vs reference run {refd.name} (negative delta = better)\n")
        for split in args.splits:
            f0 = refd / f"metrics_{split}.csv"
            f1 = run / f"metrics_{split}.csv"
            if f0.is_file() and f1.is_file():
                a, b = pd.read_csv(f1).set_index("label"), pd.read_csv(f0).set_index("label")
                cmp = pd.DataFrame({"rmse": a["rmse"], "rmse_ref": b["rmse"].reindex(a.index)})
                cmp["delta"] = cmp["rmse"] - cmp["rmse_ref"]
                cmp["delta_pct"] = 100 * cmp["delta"] / cmp["rmse_ref"]
                summary.append(f"\n**{split}**\n")
                summary.append(_md(cmp.reset_index()))
            g0, g1 = refd / f"metrics_degraded_{split}.csv", run / f"metrics_degraded_{split}.csv"
            if g0.is_file() and g1.is_file():
                a = pd.read_csv(g1).merge(pd.read_csv(g0), on=["g_prime", "label"], suffixes=("", "_ref"))
                a = a.dropna(subset=["rmse_ref"])
                a["delta_pct"] = 100 * (a["rmse"] - a["rmse_ref"]) / a["rmse_ref"]
                summary.append(f"\n**{split} degraded**\n")
                summary.append(_md(a[["g_prime", "label", "rmse", "rmse_ref", "delta_pct"]]))

    # ---- conditional chemical-information experiment (pooled test sets) ----
    summary.append("\n## Conditional chemical information (beyond "
                   "Teff/logg/[M/H] correlations)\n")
    frames = []
    for split in args.splits:
        dset = APOGEEDataset(root, split, **kw)
        cat = dset.catalog      # already filtered (require_xp_valid) like the predictions
        mu, sig, _, _ = predict(net, dset, device, amp=amp)
        frames.append((cat, mu, sig, dset.y_raw, dset.mask))
    cat = pd.concat([f[0] for f in frames], ignore_index=True)
    mu = np.concatenate([f[1] for f in frames])
    sig = np.concatenate([f[2] for f in frames])
    y = np.concatenate([f[3] for f in frames])
    m = np.concatenate([f[4] for f in frames])
    ix = {lab: i for i, lab in enumerate(labels)}

    # fit the conditional mean on TRAIN stars only (no leakage into the test)
    tr = train_dset
    tr_cat = train_dset.catalog     # same filtered rows as train_dset.y_raw
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
