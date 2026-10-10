#!/usr/bin/env python
"""Train one benchmark model.

    python scripts/train.py --config configs/specclip_frozen.yaml
    python scripts/train.py --config configs/specclip_ft.yaml --set run.name=b2_lr1e-5

Mixed precision: bf16 autocast on CUDA (`train.amp_dtype: bf16|fp16`); both NLL
losses are computed in fp32 outside autocast; non-finite batches are skipped
and counted per epoch (`n_nonfinite`).

Stage A/B/C differ only in `model.encoder.unfreeze_last_k` and the two LRs:
    A  frozen            lr_enc=0
    B  last-2 blocks     lr_enc=1e-5,  lr_head=3e-4
    C  full (rare)       unfreeze_last_k=-1, lr_enc=1e-6..3e-6
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aspgapv3.amp import amp_settings, autocast_ctx, float_outputs  # noqa: E402
from aspgapv3.augment import NoiseMatchAugment, apply_aspgap_input  # noqa: E402
from aspgapv3.config import load_config, resolve_ckpt, save_config  # noqa: E402
from aspgapv3.data.dataset import APOGEEDataset, input_dim  # noqa: E402
from aspgapv3.data.sampler import make_sampler  # noqa: E402
from aspgapv3.data.xp import aspgap_scaler_arrays  # noqa: E402
from aspgapv3.losses import MaskedHeteroscedasticNLL, combine_losses, masked_latent_mse  # noqa: E402
from aspgapv3.metrics import metrics_table  # noqa: E402
from aspgapv3.models.distance import distance_losses  # noqa: E402
from aspgapv3.models.net import build_model  # noqa: E402
from aspgapv3.models.side import build_side  # noqa: E402


def lr_at(epoch: int, cfg: dict) -> float:
    """Linear warmup then cosine decay to 3%."""
    warm = int(cfg["train"].get("warmup_epochs", 3))
    total = int(cfg["train"]["epochs"])
    if epoch < warm:
        return (epoch + 1) / max(warm, 1)
    t = (epoch - warm) / max(total - warm, 1)
    return 0.03 + 0.5 * (1 - 0.03) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(net, loader, device, loss_atm, loss_chem,
             amp: tuple[bool, torch.dtype] = (False, torch.bfloat16),
             loss_aux=None, extra: dict | None = None, dist: dict | None = None) -> tuple[float, float, dict]:
    """Returns (total NLL = atm + chem, chem NLL, per-label raw-unit metrics df).
    Same precision policy as training: forward under autocast, losses in fp32.
    The auxiliary-head NLL (if `loss_aux` and the model has an aux head) is NOT
    part of the returned total (checkpoint selection stays chem/atm-only); it is
    written to extra["aux_nll"].
    With `dist` ({"w_plx", "w_phot"}; net has the distance head) the returned total also contains
    w_plx * parallax NLL + w_phot * colour NLL (so checkpoint selection scores the distance model
    too); the unweighted terms go to extra["plx_nll"] / extra["phot_nll"]."""
    net.eval()
    tot, tot_c, tot_x, tot_l, n = 0.0, 0.0, 0.0, 0.0, 0
    tot_p, tot_ph = 0.0, 0.0
    mus, sigmas, ys, ms = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device)
        yerr = batch["yerr"].to(device)
        m = batch["mask"].to(device)
        with autocast_ctx(*amp, device=device):
            out = net(x, build_side(net, batch))
        out = float_outputs(out)
        l_all = loss_atm(out["mu_atm"], out["logvar_atm"], y[:, :net.n_atm],
                         yerr[:, :net.n_atm], m[:, :net.n_atm]) if net.n_atm else 0.0
        l_chem = loss_chem(out["mu_chem"], out["logvar_chem"],
                           y[:, net.n_atm:], yerr[:, net.n_atm:],
                           m[:, net.n_atm:]) if net.n_chem else 0.0
        bs = len(x)
        if loss_aux is not None and net.n_aux:
            tot_x += float(loss_aux(out["mu_aux"], out["logvar_aux"], batch["y_aux"].to(device),
                                    batch["yerr_aux"].to(device), batch["mask_aux"].to(device))) * bs
        if "lat" in out and "lat" in batch:      # latent distillation head (monitoring only)
            tot_l += float(masked_latent_mse(out["lat"], batch["lat"].to(device),
                                             batch["lat_ok"].to(device))) * bs
        if dist is not None:
            dl = distance_losses(net, out, batch, device)
            tot_p += float(dl["plx"]) * bs
            tot_ph += float(dl["phot"]) * bs
        tot += float(l_all) * bs
        tot_c += float(l_chem) * bs
        n += bs
        mu = torch.cat([out["mu_atm"], out["mu_chem"]], 1).float().cpu().numpy()
        sig = torch.cat([(out["logvar_atm"].clamp(-8, 8).exp().sqrt()),
                         out["logvar_chem"].clamp(-8, 8).exp().sqrt()], 1) \
            .float().cpu().numpy()
        mus.append(mu)
        sigmas.append(sig)
        ys.append(y.cpu().numpy())
        ms.append(m.cpu().numpy())
    mu = np.concatenate(mus)
    sig = np.concatenate(sigmas)
    y = np.concatenate(ys)
    m = np.concatenate(ms)
    scaler = loader.dataset.scaler
    mu_r, sig_r = scaler.inverse(mu, sig)
    y_r, m_r = loader.dataset.y_raw, loader.dataset.mask
    table = metrics_table(mu_r, sig_r, y_r, m_r, scaler.labels)
    if extra is not None:
        extra["aux_nll"] = tot_x / max(n, 1)
        extra["lat_mse"] = tot_l / max(n, 1)
        extra["plx_nll"], extra["phot_nll"] = tot_p / max(n, 1), tot_ph / max(n, 1)
    if dist is not None:
        tot += dist["w_plx"] * tot_p + dist["w_phot"] * tot_ph
    return tot / max(n, 1), tot_c / max(n, 1), table


class CachedLoader:
    """Pre-computed batches (e.g. the degraded val set) with the loader
    interface evaluate() needs."""

    def __init__(self, batches: list, dataset):
        self.batches, self.dataset = batches, dataset

    def __iter__(self):
        return iter(self.batches)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = ap.parse_args()

    cfg = resolve_ckpt(load_config(args.config, args.overrides))
    seed = int(cfg["run"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = cfg["run"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    run_dir = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    logf = open(run_dir / "log.jsonl", "a")

    def log(row: dict):
        logf.write(json.dumps(row) + "\n")
        logf.flush()
        print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in row.items()), flush=True)

    # ---- data --------------------------------------------------------------
    root = cfg["data"]["root"]
    inp = cfg["data"].get("input", "xp343")
    side_cfg = cfg["model"].get("side") or {}
    aug_cfg = (cfg["train"].get("augment") or {}).get("noise_match") or {}
    p_aug = float(aug_cfg.get("p", 0.0))
    select_on = cfg["train"].get("select_on", "mean" if p_aug > 0 else "clean")
    if select_on not in ("clean", "degraded", "mean"):
        raise ValueError(f"train.select_on must be clean|degraded|mean, got {select_on!r}")
    # the degraded val set (select_on != clean) needs the augmenter even when p == 0;
    # training batches are only augmented if p > 0 (so a no-aug ablation can be
    # selected with the same criterion as the augmented model)
    need_aug = p_aug > 0 or select_on != "clean"
    use_side = bool(side_cfg.get("scalars") or side_cfg.get("noise_channel"))
    need_noise = use_side or need_aug
    if use_side and inp != "xp343":
        raise ValueError("side conditioning needs data.input=xp343")
    if need_aug and inp not in ("xp343", "coef110n"):
        raise ValueError("noise augmentation needs data.input=xp343 or coef110n")
    # coef110n: the augmenter degrades the raw coef110; the network input is
    # rebuilt with the cached AspGap forward transform (aspgap_scaler.npz)
    aspgap_sc = aspgap_scaler_arrays(root) if (need_aug and inp == "coef110n") else None
    req_valid = bool(cfg["data"].get("require_xp_valid", False))
    aux_names = list(cfg["labels"].get("aux") or [])      # P1: [] => old behaviour
    lat_cfg = cfg["model"].get("latent_head") or {}       # latent distillation: absent => off
    lat_src = lat_cfg.get("source") or None
    w_lat = float(lat_cfg.get("weight", 0.0)) if lat_src else 0.0
    dist_cfg = cfg["model"].get("distance") or {}          # M_G / A0 + photometric forward model
    use_dist = bool(dist_cfg.get("enabled", False))
    w_plx, w_phot = (float(dist_cfg.get("weight_plx", 1.0)), float(dist_cfg.get("weight_phot", 1.0))) \
        if use_dist else (0.0, 0.0)
    w_anc = float(dist_cfg.get("weight_anchor", 0.0)) if use_dist else 0.0
    rc_cfg = dist_cfg.get("rc_anchor") or {}                # red-clump A0 anchors (rc_anchor.py); absent => off
    use_rc = use_dist and bool(rc_cfg.get("enabled", False))
    w_rc = float(rc_cfg.get("weight", w_anc)) if use_rc else 0.0
    dist_w = {"w_plx": w_plx, "w_phot": w_phot} if use_dist else None
    label_names = list(cfg["labels"]["atm"]) + list(cfg["labels"].get("chem") or [])
    dsets = {s: APOGEEDataset(root, split=s, input=inp, noise=need_noise, labels=label_names, astro=use_dist,
                              require_xp_valid=req_valid, aux=aux_names,
                              aux_ref=str(cfg["labels"].get("aux_ref", "h")),
                              latent=lat_src, rc_anchor=rc_cfg if use_rc else None) for s in
             ("train", "val", "iid_test", "sky_test")}
    n_labels = len(dsets["train"].labels)
    n_atm = len(cfg["labels"]["atm"])
    log({"event": "data", "train": len(dsets["train"]), "val": len(dsets["val"]),
         "iid_test": len(dsets["iid_test"]), "sky_test": len(dsets["sky_test"]),
         "input": inp, "require_xp_valid": req_valid,
         "n_dropped_invalid": {s: d.n_dropped_invalid for s, d in dsets.items()},
         "distance": use_dist, "n_labels": n_labels, "labels": dsets["train"].labels, "aux": aux_names,
         **({"latent_head": {**lat_cfg, "dim_data": dsets["train"].latent_dim}} if lat_src else {})})
    if lat_src and int(lat_cfg.get("dim", 0)) != dsets["train"].latent_dim:
        raise ValueError(f"model.latent_head.dim={lat_cfg.get('dim')} != latent file dim "
                         f"{dsets['train'].latent_dim}")

    sampler = make_sampler(dsets["train"], cfg["train"].get("sampler", {}))
    train_loader = DataLoader(
        dsets["train"], batch_size=cfg["train"]["batch_size"],
        sampler=sampler, shuffle=sampler is None,
        num_workers=cfg["train"]["num_workers"], pin_memory=(device == "cuda"),
        drop_last=True, persistent_workers=cfg["train"]["num_workers"] > 0,
    )
    eval_loader = lambda s: DataLoader(dsets[s], batch_size=1024, shuffle=False,
                                       num_workers=2, pin_memory=(device == "cuda"))

    # ---- model / loss ------------------------------------------------------
    net = build_model(cfg, n_atm=n_atm, n_chem=n_labels - n_atm, n_aux=len(aux_names)).to(device)
    n_enc = sum(p.numel() for p in net.encoder_parameters())
    log({"event": "model", "encoder_trainable_params": n_enc,
         "total_params": sum(p.numel() for p in net.parameters())})

    dset = dsets["train"]
    clamp = tuple(cfg["train"].get("logvar_clamp", [-8.0, 8.0]))
    floors = torch.as_tensor(dset.floor_norm)   # normalized space, per label
    w_atm = torch.full((n_atm,), float(cfg["labels"].get("weight_atm", 0.5)))
    w_chem = torch.ones(n_labels - n_atm)
    loss_atm = MaskedHeteroscedasticNLL(w_atm, floors[:n_atm], clamp).to(device)
    loss_chem = MaskedHeteroscedasticNLL(w_chem, floors[n_atm:], clamp).to(device)
    loss_mode = str(cfg["labels"].get("loss_weighting", "per_group"))
    w_atm_glob = float(cfg["labels"].get("weight_atm", 0.5))   # used only if loss_weighting=global
    combine_losses(0.0, 0.0, None, loss_mode, w_atm_glob, 0.0)  # validates the mode early
    loss_aux, w_aux = None, float(cfg["labels"].get("weight_aux", 0.3))
    if aux_names:
        # NLL is normalized by sum(mask*w), so a uniform per-label weight cancels;
        # weight_aux is therefore applied as a scalar on the aux-loss term below.
        loss_aux = MaskedHeteroscedasticNLL(torch.ones(len(aux_names)),
                                            torch.as_tensor(dset.floor_aux), clamp).to(device)

    if net.input_norm:                                       # RobustScaler-like, clean TRAIN set
        rng = np.random.default_rng(seed)
        ix = dset.indices if len(dset.indices) <= 200_000 else \
            np.sort(rng.choice(dset.indices, 200_000, replace=False))
        xs = np.asarray(dset.x[ix], np.float64)
        q25, med, q75 = np.nanpercentile(xs, [25, 50, 75], axis=0)
        net.set_input_stats(med, q75 - q25)
        log({"event": "input_norm", "n": int(len(ix)), "iqr_min": float((q75 - q25).min()),
             "iqr_median": float(np.median(q75 - q25))})
        del xs
    if net.side is not None and net.side.scalars:           # train-split stats
        cols = net.side.cols
        sd = np.asarray(dset.side[dset.indices][:, cols], np.float64)
        net.side.set_scalar_stats(np.nanmean(sd, 0), np.nanstd(sd, 0))
    if use_dist:
        # data-driven init: M_G head centred on the parallax-based M_G of well-measured stars,
        # colour net output bias at the mean observed colours (losses start O(1), not O(10^2))
        from aspgapv3.models.distance import EXT_K
        ar = dset.astro_arr
        zc = ar["plx_ok"] & (ar["plx"] - ar["zp"] > 10 * ar["plx_err"])
        mg_obs = ar["g"][zc] + 5 * np.log10(ar["plx"][zc] - ar["zp"][zc]) - 10.0
        net.head_dist.set_mg_stats(float(np.median(mg_obs)), float(mg_obs.std()))
        cw = ar["col_ok"].astype(np.float64)
        net.colour_net.set_output_bias((ar["col"] * cw).sum(0) / np.maximum(cw.sum(0), 1))
        log({"event": "distance_init", "mg_mean": float(np.median(mg_obs)), "mg_std": float(mg_obs.std()),
             "n_mg": int(zc.sum()), "n_anchor_nearby": int((dset.anchor_set == 1).sum()),
             "n_anchor_highlat": int((dset.anchor_set == 2).sum()), "n_anchor_rc": int((dset.anchor_set == 3).sum()),
             "rc_calib": dset.rc_calib, "weight_anchor_rc": w_rc, "weight_anchor": w_anc, "plx_ok": float(ar["plx_ok"].mean()),
             "col_ok_frac": [round(float(v), 3) for v in ar["col_ok"].mean(0)],
             "weight_plx": w_plx, "weight_phot": w_phot})
    augmenter = None
    if need_aug:
        augmenter = NoiseMatchAugment.from_root(
            root, device=device, p=p_aug, g_target=aug_cfg.get("g_target", "catalog_hist"),
            g_max=float(aug_cfg.get("g_max", 17.65)))
        log({"event": "augment", **aug_cfg, "bank": int(augmenter.L.shape[0])})
    use_amp, amp_dtype = amp_settings(cfg, device)
    # eval runs in fp32 unless eval.amp: true (comparable to the *_v1 runs)
    amp = (use_amp and bool(cfg.get("eval", {}).get("amp", False)), amp_dtype)
    val_deg_loader = None
    if augmenter is not None:
        # fixed-seed degraded val set (G' from the catalog histogram, p=1), built
        # once so every epoch scores the identical noisy stars
        gen = torch.Generator(device=device).manual_seed(seed + 1)
        cached = []
        for b in eval_loader("val"):
            b = augmenter(b, gen=gen, p=1.0)
            if aspgap_sc is not None:
                b["x"] = apply_aspgap_input(b, aspgap_sc, device)
            cached.append(b)
        val_deg_loader = CachedLoader(cached, dsets["val"])
        log({"event": "val_degraded", "frac_degraded":
             float(sum(int(b["aug"].sum()) for b in cached)) / len(dsets["val"]),
             "select_on": select_on})
    groups = net.optimizer_param_groups(float(cfg["train"]["lr_head"]),
                                        float(cfg["train"]["lr_enc"]),
                                        float(cfg["train"]["weight_decay"]))
    if not any(g["name"] == "encoder" for g in groups):
        log({"event": "note", "msg": "encoder frozen (lr_enc=0 or no grads)"})
    opt = torch.optim.AdamW(groups)
    base_lrs = [g["lr"] for g in groups]

    if lat_src:
        # weight-choice diagnostic: label vs latent loss of the UNTRAINED net (eval mode) on
        # a few clean val batches, so the latent weight can be set to a comparable magnitude
        net.eval()
        acc = np.zeros(3)
        with torch.no_grad():
            for i, b in enumerate(eval_loader("val")):
                if i >= 4:
                    break
                o = float_outputs(net(b["x"].to(device), build_side(net, b)))
                yb, eb, mb = (b[k].to(device) for k in ("y", "yerr", "mask"))
                acc += [float(loss_atm(o["mu_atm"], o["logvar_atm"], yb[:, :n_atm], eb[:, :n_atm], mb[:, :n_atm])),
                        float(loss_chem(o["mu_chem"], o["logvar_chem"], yb[:, n_atm:], eb[:, n_atm:], mb[:, n_atm:])),
                        float(masked_latent_mse(o["lat"], b["lat"].to(device), b["lat_ok"].to(device)))]
        acc /= 4
        log({"event": "init_losses", "l_atm": acc[0], "l_chem": acc[1], "l_lat_mse": acc[2],
             "label_total": acc[0] + acc[1], "weight_lat": w_lat,
             "weighted_lat": w_lat * acc[2]})

    if use_dist:
        net.eval()
        acc = np.zeros(5)
        with torch.no_grad():
            for i, b in enumerate(eval_loader("train")):
                if i >= 4:
                    break
                o = float_outputs(net(b["x"].to(device), build_side(net, b)))
                yb, eb, mb = (b[k].to(device) for k in ("y", "yerr", "mask"))
                dl = distance_losses(net, o, b, device)
                acc += [float(loss_atm(o["mu_atm"], o["logvar_atm"], yb[:, :n_atm], eb[:, :n_atm], mb[:, :n_atm]))
                        + float(loss_chem(o["mu_chem"], o["logvar_chem"], yb[:, n_atm:], eb[:, n_atm:], mb[:, n_atm:])),
                        float(dl["plx"]), float(dl["phot"]), float(dl["pull"].abs().median()), float(dl["anchor"])]
        acc /= 4
        log({"event": "init_losses", "label_nll": acc[0], "plx_nll": acc[1], "phot_nll": acc[2],
             "plx_median_abs_pull": acc[3], "anchor_nll": acc[4], "weighted_plx": w_plx * acc[1], "weighted_phot": w_phot * acc[2]})

    # ---- loop --------------------------------------------------------------
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    clip = float(cfg["train"].get("grad_clip", 1.0))
    best_val = math.inf
    patience = int(cfg["train"].get("early_stop_patience", 12))
    bad_epochs = 0

    for epoch in range(int(cfg["train"]["epochs"])):
        r = lr_at(epoch, cfg)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * r
        net.train()
        t0 = time.time()
        tr_loss, nb, n_nonfinite, n_aug, n_seen = 0.0, 0, 0, 0, 0
        tr_lab, tr_lat = 0.0, 0.0
        tr_plx, tr_phot = 0.0, 0.0
        for batch in train_loader:
            if p_aug > 0:
                batch = augmenter(batch)
                n_aug += int(batch["aug"].sum())
                n_seen += len(batch["aug"])
                if aspgap_sc is not None:
                    batch["x"] = apply_aspgap_input(batch, aspgap_sc, device)
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            yerr = batch["yerr"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(use_amp, amp_dtype, device):
                out = net(x, build_side(net, batch))
            out = float_outputs(out)                   # losses in fp32, outside autocast
            l_atm = loss_atm(out["mu_atm"], out["logvar_atm"],
                             y[:, :n_atm], yerr[:, :n_atm], m[:, :n_atm]) \
                if n_atm else torch.zeros((), device=device)
            l_chem = loss_chem(out["mu_chem"], out["logvar_chem"],
                               y[:, n_atm:], yerr[:, n_atm:], m[:, n_atm:]) \
                if n_labels > n_atm else torch.zeros((), device=device)
            l_x = loss_aux(out["mu_aux"], out["logvar_aux"],
                           batch["y_aux"].to(device, non_blocking=True),
                           batch["yerr_aux"].to(device, non_blocking=True),
                           batch["mask_aux"].to(device, non_blocking=True)) \
                if loss_aux is not None else None
            l_lat = masked_latent_mse(out["lat"], batch["lat"].to(device, non_blocking=True),
                                      batch["lat_ok"].to(device, non_blocking=True)) \
                if lat_src else None             # target = latent of the CLEAN input (never augmented)
            loss = combine_losses(l_atm, l_chem, l_x, loss_mode, w_atm_glob, w_aux,
                                  l_lat=l_lat, weight_lat=w_lat)
            if use_dist:          # astro columns are the ORIGINAL star's (augmentation never touches them)
                dl = distance_losses(net, out, batch, device)
                loss = loss + w_plx * dl["plx"] + w_phot * dl["phot"] + w_anc * dl["anchor"] + w_rc * dl["anchor_rc"]
            if not torch.isfinite(loss):
                n_nonfinite += 1
                continue
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            if not torch.isfinite(gnorm):
                n_nonfinite += 1
                scaler_amp.update()                    # no-op unless fp16 scaler
                opt.zero_grad(set_to_none=True)
                continue
            scaler_amp.step(opt)
            scaler_amp.update()
            tr_loss += float(loss)
            if use_dist:
                tr_plx += float(dl["plx"])
                tr_phot += float(dl["phot"])
            if lat_src:
                tr_lat += float(l_lat)
                tr_lab += float(loss) - w_lat * float(l_lat)
            nb += 1
        ex = {}
        val, val_chem, table = evaluate(net, eval_loader("val"), device, loss_atm, loss_chem, amp,
                                        loss_aux=loss_aux, extra=ex, dist=dist_w)
        sel = val
        extra = {"val_aux_nll": ex["aux_nll"]} if aux_names else {}
        if use_dist:
            extra = {**extra, "train_plx_nll": tr_plx / max(nb, 1), "train_phot_nll": tr_phot / max(nb, 1),
                     "val_plx_nll": ex["plx_nll"], "val_phot_nll": ex["phot_nll"]}
        if lat_src:
            extra = {**extra, "train_label_loss": tr_lab / max(nb, 1), "train_lat_mse": tr_lat / max(nb, 1),
                     "train_lat_weighted": w_lat * tr_lat / max(nb, 1), "val_lat_mse": ex["lat_mse"]}
        if val_deg_loader is not None:
            ex_d = {}
            val_d, _, table_d = evaluate(net, val_deg_loader, device, loss_atm, loss_chem, amp, extra=ex_d,
                                         dist=dist_w)
            extra = {**extra, "val_loss_degraded": val_d,
                     **({"val_plx_nll_degraded": ex_d["plx_nll"]} if use_dist else {}),
                     **({"val_lat_mse_degraded": ex_d["lat_mse"]} if lat_src else {}),
                     "val_rmse_degraded": {r0["label"]: round(r0["rmse"], 5)
                                           for _, r0 in table_d.iterrows() if r0.get("n", 0) > 0}}
            sel = {"clean": val, "degraded": val_d, "mean": 0.5 * (val + val_d)}[select_on]
        row = {"event": "epoch", "epoch": epoch, "lr_scale": round(r, 4),
               "train_loss": tr_loss / max(nb, 1), "n_nonfinite": n_nonfinite,
               "aug_frac": n_aug / max(n_seen, 1), "val_loss": val, **extra, "val_select": sel,
               "val_chem_nll": val_chem, "sec": round(time.time() - t0, 1),
               "val_rmse": {r0["label"]: round(r0["rmse"], 5)
                            for _, r0 in table.iterrows() if r0.get("n", 0) > 0}}
        log(row)
        if sel < best_val - 1e-5:
            best_val, bad_epochs = sel, 0
            torch.save({"state_dict": net.state_dict(), "config": cfg,
                        "labels": dsets["train"].labels, "n_atm": n_atm,
                        "n_aux": len(aux_names), "aux_labels": aux_names,
                        "aux_ref": dset.aux_ref,
                        "aux_norm": ({"mean": dset.aux_scaler.mean, "std": dset.aux_scaler.std}
                                     if aux_names else None),
                        "epoch": epoch, "val_loss": sel},
                       run_dir / "ckpt_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log({"event": "early_stop", "epoch": epoch, "best_val": best_val})
                break

    torch.save({"state_dict": net.state_dict(), "config": cfg,
                "labels": dsets["train"].labels, "n_atm": n_atm,
                "n_aux": len(aux_names), "aux_labels": aux_names,
                        "aux_ref": dset.aux_ref,
                        "aux_norm": ({"mean": dset.aux_scaler.mean, "std": dset.aux_scaler.std}
                                     if aux_names else None),
                "epoch": epoch, "val_loss": sel},
               run_dir / "ckpt_last.pt")

    # ---- final quick report on the held-out sets ----------------------------
    ckpt = torch.load(run_dir / "ckpt_best.pt", map_location=device,
                      weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    for s in ("iid_test", "sky_test"):
        _, _, table = evaluate(net, eval_loader(s), device, loss_atm, loss_chem, amp, dist=dist_w)
        table.to_csv(run_dir / f"metrics_{s}.csv", index=False)
        log({"event": "final", "split": s,
             "rmse": {r["label"]: round(r["rmse"], 5)
                      for _, r in table.iterrows() if r.get("n", 0) > 0}})
    log({"event": "done", "best_val": best_val,
         "ckpt": str(run_dir / "ckpt_best.pt")})


if __name__ == "__main__":
    main()
