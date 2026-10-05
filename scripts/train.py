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
from aspgapv3.losses import MaskedHeteroscedasticNLL, combine_losses  # noqa: E402
from aspgapv3.metrics import metrics_table  # noqa: E402
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
             loss_aux=None, extra: dict | None = None) -> tuple[float, float, dict]:
    """Returns (total NLL = atm + chem, chem NLL, per-label raw-unit metrics df).
    Same precision policy as training: forward under autocast, losses in fp32.
    The auxiliary-head NLL (if `loss_aux` and the model has an aux head) is NOT
    part of the returned total (checkpoint selection stays chem/atm-only); it is
    written to extra["aux_nll"]."""
    net.eval()
    tot, tot_c, tot_x, n = 0.0, 0.0, 0.0, 0
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
    use_side = bool(side_cfg.get("scalars") or side_cfg.get("noise_channel"))
    need_noise = use_side or p_aug > 0
    if use_side and inp != "xp343":
        raise ValueError("side conditioning needs data.input=xp343")
    if p_aug > 0 and inp not in ("xp343", "coef110n"):
        raise ValueError("noise augmentation needs data.input=xp343 or coef110n")
    # coef110n: the augmenter degrades the raw coef110; the network input is
    # rebuilt with the cached AspGap forward transform (aspgap_scaler.npz)
    aspgap_sc = aspgap_scaler_arrays(root) if (p_aug > 0 and inp == "coef110n") else None
    req_valid = bool(cfg["data"].get("require_xp_valid", False))
    aux_names = list(cfg["labels"].get("aux") or [])      # P1: [] => old behaviour
    dsets = {s: APOGEEDataset(root, split=s, input=inp, noise=need_noise,
                              require_xp_valid=req_valid, aux=aux_names,
                              aux_ref=str(cfg["labels"].get("aux_ref", "h"))) for s in
             ("train", "val", "iid_test", "sky_test")}
    n_labels = len(dsets["train"].labels)
    n_atm = len(cfg["labels"]["atm"])
    log({"event": "data", "train": len(dsets["train"]), "val": len(dsets["val"]),
         "iid_test": len(dsets["iid_test"]), "sky_test": len(dsets["sky_test"]),
         "input": inp, "require_xp_valid": req_valid,
         "n_dropped_invalid": {s: d.n_dropped_invalid for s, d in dsets.items()},
         "n_labels": n_labels, "labels": dsets["train"].labels, "aux": aux_names})

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

    if net.side is not None and net.side.scalars:           # train-split stats
        cols = net.side.cols
        sd = np.asarray(dset.side[dset.indices][:, cols], np.float64)
        net.side.set_scalar_stats(np.nanmean(sd, 0), np.nanstd(sd, 0))
    augmenter = None
    if p_aug > 0:
        augmenter = NoiseMatchAugment.from_root(
            root, device=device, p=p_aug, g_target=aug_cfg.get("g_target", "catalog_hist"),
            g_max=float(aug_cfg.get("g_max", 17.65)))
        log({"event": "augment", **aug_cfg, "bank": int(augmenter.L.shape[0])})
    use_amp, amp_dtype = amp_settings(cfg, device)
    # eval runs in fp32 unless eval.amp: true (comparable to the *_v1 runs)
    amp = (use_amp and bool(cfg.get("eval", {}).get("amp", False)), amp_dtype)
    select_on = cfg["train"].get("select_on", "mean" if augmenter is not None else "clean")
    if select_on not in ("clean", "degraded", "mean"):
        raise ValueError(f"train.select_on must be clean|degraded|mean, got {select_on!r}")
    if select_on != "clean" and augmenter is None:
        raise ValueError("train.select_on != clean needs train.augment.noise_match.p > 0")
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
        for batch in train_loader:
            if augmenter is not None:
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
            loss = combine_losses(l_atm, l_chem, l_x, loss_mode, w_atm_glob, w_aux)
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
            nb += 1
        ex = {}
        val, val_chem, table = evaluate(net, eval_loader("val"), device, loss_atm, loss_chem, amp,
                                        loss_aux=loss_aux, extra=ex)
        sel = val
        extra = {"val_aux_nll": ex["aux_nll"]} if aux_names else {}
        if val_deg_loader is not None:
            val_d, _, table_d = evaluate(net, val_deg_loader, device, loss_atm, loss_chem, amp)
            extra = {**extra, "val_loss_degraded": val_d,
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
        _, _, table = evaluate(net, eval_loader(s), device, loss_atm, loss_chem, amp)
        table.to_csv(run_dir / f"metrics_{s}.csv", index=False)
        log({"event": "final", "split": s,
             "rmse": {r["label"]: round(r["rmse"], 5)
                      for _, r in table.iterrows() if r.get("n", 0) > 0}})
    log({"event": "done", "best_val": best_val,
         "ckpt": str(run_dir / "ckpt_best.pt")})


if __name__ == "__main__":
    main()
