#!/usr/bin/env python
"""Train one benchmark model.

    python scripts/train.py --config configs/specclip_frozen.yaml
    python scripts/train.py --config configs/specclip_ft.yaml --set run.name=b2_lr1e-5

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

from aspgapv3.config import load_config, resolve_ckpt, save_config  # noqa: E402
from aspgapv3.data.dataset import APOGEEDataset, input_dim  # noqa: E402
from aspgapv3.data.sampler import make_sampler  # noqa: E402
from aspgapv3.losses import MaskedHeteroscedasticNLL  # noqa: E402
from aspgapv3.metrics import metrics_table  # noqa: E402
from aspgapv3.models.net import build_model  # noqa: E402


def lr_at(epoch: int, cfg: dict) -> float:
    """Linear warmup then cosine decay to 3%."""
    warm = int(cfg["train"].get("warmup_epochs", 3))
    total = int(cfg["train"]["epochs"])
    if epoch < warm:
        return (epoch + 1) / max(warm, 1)
    t = (epoch - warm) / max(total - warm, 1)
    return 0.03 + 0.5 * (1 - 0.03) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(net, loader, device, loss_atm, loss_chem) -> tuple[float, float, dict]:
    """Returns (total NLL, chem NLL, per-label raw-unit metrics df)."""
    net.eval()
    tot, tot_c, n = 0.0, 0.0, 0
    mus, sigmas, ys, ms = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device)
        yerr = batch["yerr"].to(device)
        m = batch["mask"].to(device)
        out = net(x)
        with torch.autocast(device_type=device, enabled=False):
            l_all = loss_atm(out["mu_atm"], out["logvar_atm"], y[:, :net.n_atm],
                             yerr[:, :net.n_atm], m[:, :net.n_atm]) if net.n_atm else 0.0
            l_chem = loss_chem(out["mu_chem"], out["logvar_chem"],
                               y[:, net.n_atm:], yerr[:, net.n_atm:],
                               m[:, net.n_atm:]) if net.n_chem else 0.0
            bs = len(x)
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
    return tot / max(n, 1), tot_c / max(n, 1), table


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
    dsets = {s: APOGEEDataset(root, split=s, input=inp) for s in
             ("train", "val", "iid_test", "sky_test")}
    n_labels = len(dsets["train"].labels)
    n_atm = len(cfg["labels"]["atm"])
    log({"event": "data", "train": len(dsets["train"]), "val": len(dsets["val"]),
         "iid_test": len(dsets["iid_test"]), "sky_test": len(dsets["sky_test"]),
         "input": inp, "n_labels": n_labels, "labels": dsets["train"].labels})

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
    net = build_model(cfg, n_atm=n_atm, n_chem=n_labels - n_atm).to(device)
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

    groups = net.optimizer_param_groups(float(cfg["train"]["lr_head"]),
                                        float(cfg["train"]["lr_enc"]),
                                        float(cfg["train"]["weight_decay"]))
    if not any(g["name"] == "encoder" for g in groups):
        log({"event": "note", "msg": "encoder frozen (lr_enc=0 or no grads)"})
    opt = torch.optim.AdamW(groups)
    base_lrs = [g["lr"] for g in groups]

    # ---- loop --------------------------------------------------------------
    use_amp = bool(cfg["train"].get("amp", True)) and device == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val = math.inf
    patience = int(cfg["train"].get("early_stop_patience", 12))
    bad_epochs = 0

    for epoch in range(int(cfg["train"]["epochs"])):
        r = lr_at(epoch, cfg)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * r
        net.train()
        t0 = time.time()
        tr_loss, nb = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            yerr = batch["yerr"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                out = net(x)
                l_atm = loss_atm(out["mu_atm"], out["logvar_atm"],
                                 y[:, :n_atm], yerr[:, :n_atm], m[:, :n_atm]) \
                    if n_atm else torch.zeros((), device=device)
                l_chem = loss_chem(out["mu_chem"], out["logvar_chem"],
                                   y[:, n_atm:], yerr[:, n_atm:], m[:, n_atm:]) \
                    if n_labels > n_atm else torch.zeros((), device=device)
                loss = l_atm + l_chem
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(),
                                           float(cfg["train"].get("grad_clip", 1.0)))
            scaler_amp.step(opt)
            scaler_amp.update()
            tr_loss += float(loss)
            nb += 1
        val, val_chem, table = evaluate(net, eval_loader("val"), device, loss_atm, loss_chem)
        row = {"event": "epoch", "epoch": epoch, "lr_scale": round(r, 4),
               "train_loss": tr_loss / max(nb, 1), "val_loss": val,
               "val_chem_nll": val_chem, "sec": round(time.time() - t0, 1),
               "val_rmse": {r0["label"]: round(r0["rmse"], 5)
                            for _, r0 in table.iterrows() if r0.get("n", 0) > 0}}
        log(row)
        if val < best_val - 1e-5:
            best_val, bad_epochs = val, 0
            torch.save({"state_dict": net.state_dict(), "config": cfg,
                        "labels": dsets["train"].labels, "n_atm": n_atm,
                        "epoch": epoch, "val_loss": val},
                       run_dir / "ckpt_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log({"event": "early_stop", "epoch": epoch, "best_val": best_val})
                break

    torch.save({"state_dict": net.state_dict(), "config": cfg,
                "labels": dsets["train"].labels, "n_atm": n_atm,
                "epoch": epoch, "val_loss": val},
               run_dir / "ckpt_last.pt")

    # ---- final quick report on the held-out sets ----------------------------
    ckpt = torch.load(run_dir / "ckpt_best.pt", map_location=device,
                      weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    for s in ("iid_test", "sky_test"):
        _, _, table = evaluate(net, eval_loader(s), device, loss_atm, loss_chem)
        table.to_csv(run_dir / f"metrics_{s}.csv", index=False)
        log({"event": "final", "split": s,
             "rmse": {r["label"]: round(r["rmse"], 5)
                      for _, r in table.iterrows() if r.get("n", 0) > 0}})
    log({"event": "done", "best_val": best_val,
         "ckpt": str(run_dir / "ckpt_best.pt")})


if __name__ == "__main__":
    main()
