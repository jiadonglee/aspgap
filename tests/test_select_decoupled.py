"""train.select_on != clean must work with p == 0 (no-aug ablation selected by
the same clean+degraded criterion): degraded val computed, training not augmented."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from test_evaluate_filtered import make_root

ROOT = Path(__file__).resolve().parents[1]


def run_train(tmp_path, monkeypatch, train_extra, model=None, ckpt=False):
    (tmp_path / "data_root").mkdir()
    make_root(tmp_path / "data_root")
    cfg = {"inherit": str(ROOT / "configs" / "base.yaml"),
           "run": {"name": "t", "out_dir": str(tmp_path / "runs"), "device": "cpu"},
           "data": {"root": str(tmp_path / "data_root"), "input": "xp343", "require_xp_valid": True},
           "model": model or {"encoder": {"type": "scratch", "d_model": 32, "n_layers": 1, "n_heads": 4,
                                          "embed_dim": 768, "unfreeze_last_k": -1},
                              "trunk": {"dim": 64, "dropout": 0.0}},
           "train": {"epochs": 1, "batch_size": 16, "num_workers": 0, "lr_enc": 1e-4, "amp": False,
                     "sampler": {"type": "random"}, **train_extra}}
    f = tmp_path / "cfg.yaml"
    f.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    sp = importlib.util.spec_from_file_location("trainmod", ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(f)])
    mod.main()
    rows = [json.loads(l) for l in (tmp_path / "runs" / "t" / "log.jsonl").read_text().splitlines()]
    return (rows, tmp_path / "runs" / "t") if ckpt else rows


def test_select_mean_with_p0_computes_degraded_val_without_training_aug(tmp_path, monkeypatch):
    rows = run_train(tmp_path, monkeypatch, {"select_on": "mean",
                                             "augment": {"noise_match": {"p": 0.0}}})
    ep = [r for r in rows if r["event"] == "epoch"][0]
    assert ep["aug_frac"] == 0
    assert "val_loss_degraded" in ep and ep["val_select"] == pytest.approx(0.5 * (ep["val_loss"] + ep["val_loss_degraded"]))
    vd = [r for r in rows if r["event"] == "val_degraded"][0]
    assert vd["frac_degraded"] > 0 and vd["select_on"] == "mean"


def test_default_old_behaviour_no_augmenter(tmp_path, monkeypatch):
    rows = run_train(tmp_path, monkeypatch, {})                  # select_on default clean, p default 0
    ep = [r for r in rows if r["event"] == "epoch"][0]
    assert "val_loss_degraded" not in ep and ep["aug_frac"] == 0
    assert not any(r["event"] in ("augment", "val_degraded") for r in rows)
    assert ep["val_select"] == ep["val_loss"]


def test_augmented_training_still_augments(tmp_path, monkeypatch):
    rows = run_train(tmp_path, monkeypatch, {"augment": {"noise_match": {"p": 1.0, "g_max": 17.65}}})
    ep = [r for r in rows if r["event"] == "epoch"][0]
    assert ep["aug_frac"] > 0 and "val_loss_degraded" in ep
