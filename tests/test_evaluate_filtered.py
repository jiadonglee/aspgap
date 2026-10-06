"""Regression: evaluate.py must index catalog rows exactly like APOGEEDataset
(require_xp_valid drops rows in every split); previously the domain tables
crashed with 'operands could not be broadcast (26201,) (26112,)'."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aspgapv3.data.dataset import LabelScaler, save_catalog
from aspgapv3.models.net import build_model

from test_augment import make_bank, make_batch
from test_xpnoise import synth_dm

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["teff", "logg", "m_h", "alpha_m", "fe_h", "mg_h"]


def make_root(tmp_path, n=400, seed=0):
    rng = np.random.default_rng(seed)
    dm = synth_dm()
    np.savez(tmp_path / "xp_design_343.npz", **dm)
    np.savez(tmp_path / "noise_bank.npz", **make_bank())
    batch, coef, _ = make_batch(dm, n=n, g=13.0)
    np.save(tmp_path / "xp343.npy", batch["x"].numpy())
    np.save(tmp_path / "coef110.npy", batch["coef"].numpy())
    np.save(tmp_path / "xp343_sigma.npy", batch["sigma"].numpy())
    np.save(tmp_path / "side.npy", batch["side"].numpy())
    split = np.array(["train"] * 240 + ["val"] * 40 + ["iid_test"] * 60 + ["sky_test"] * 60)
    valid = np.ones(n, bool)
    for s in ("train", "val", "iid_test", "sky_test"):          # drop rows in EVERY split
        valid[np.where(split == s)[0][3::7]] = False
    np.save(tmp_path / "xpcov_valid.npy", valid)
    y = np.stack([rng.normal(5000, 500, n), rng.normal(3, 1, n), rng.normal(-.3, .3, n),
                  rng.normal(.1, .05, n), rng.normal(-.3, .3, n), rng.normal(-.2, .3, n)], 1).astype(np.float32)
    yerr, mask = np.full_like(y, .05), np.ones(y.shape, bool)
    LabelScaler.fit(y, yerr, mask, np.array([50, .05, .02, .02, .02, .02]), LABELS).save(tmp_path)
    np.savez(tmp_path / "labels.npz", y=y, yerr=yerr, mask=mask, labels=np.array(LABELS, dtype="U16"))
    cat = pd.DataFrame({"row": np.arange(n), "source_id": np.arange(900, 900 + n), "split": split,
                        "teff": y[:, 0], "logg": y[:, 1], "m_h": y[:, 2], "alpha_m": y[:, 3],
                        "fe_h": y[:, 4], "mg_h": y[:, 5],
                        "is_metal_poor": rng.random(n) > .8, "is_mdwarf": rng.random(n) > .8,
                        "is_fstar": rng.random(n) > .8, "is_faint_xp": rng.random(n) > .8})
    save_catalog(cat, tmp_path)
    return cat, valid


def test_evaluate_all_sections_with_xp_valid_filtering(tmp_path, monkeypatch):
    cat, valid = make_root(tmp_path)
    cfg = {"run": {"device": "cpu"},
           "model": {"encoder": {"type": "scratch", "d_model": 32, "n_layers": 1, "n_heads": 4,
                                 "embed_dim": 768}, "trunk": {"dim": 64, "dropout": 0.0}},
           "data": {"root": str(tmp_path), "input": "xp343", "require_xp_valid": True},
           "train": {"amp": False, "augment": {"noise_match": {"p": 0.5, "g_max": 17.65}}},
           "eval": {"ood_k": 4, "ood_max_train": 1000}}
    net = build_model(cfg, n_atm=3, n_chem=3)
    run = tmp_path / "run"
    run.mkdir()
    torch.save({"state_dict": net.state_dict(), "config": cfg, "labels": LABELS, "n_atm": 3,
                "epoch": 0, "val_loss": 0.0}, run / "ckpt_best.pt")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    sp = importlib.util.spec_from_file_location("evalmod", ROOT / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--run", str(run), "--splits", "iid_test",
                                      "sky_test", "--degrade-g", "15", "17"])
    mod.main()

    for s in ("iid_test", "sky_test"):
        n_ok = int((valid & (cat["split"] == s).to_numpy()).sum())
        assert n_ok < int((cat["split"] == s).sum())               # filtering is active
        p = np.load(run / f"predictions_{s}.npz")
        assert len(p["mu"]) == len(p["source_id"]) == n_ok
        dom = pd.read_csv(run / f"metrics_{s}_domains.csv")
        assert set(dom["domain"]) >= {"all", "metal_poor", "mdwarf", "fstar", "faint_xp"}
        assert (dom[dom["domain"] == "all"]["n_domain"] > 0).all()
        deg = pd.read_csv(run / f"metrics_degraded_{s}.csv")
        assert set(deg["g_prime"].dropna()) == {15.0, 17.0}
    summary = (run / "summary.md").read_text()
    assert "Conditional chemical information" in summary and "residual recovery" in summary
