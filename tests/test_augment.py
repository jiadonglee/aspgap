"""Noise-matched augmentation on synthetic data (CPU, no real data needed)."""
import numpy as np
import pytest
import torch

from aspgapv3.augment import NoiseMatchAugment
from aspgapv3.data.xp import N_COEF, NORM_INDEX, absolute_flux, sample_spectra
from aspgapv3.data.xpnoise import N_TRI, pack_star, pixel_sigma_abs

from test_xpnoise import synth_dm


EDGES = np.round(np.arange(10.0, 17.7001, 0.1), 3)
NB = len(EDGES) - 1


def make_bank(scale=1.0, per_bin=1, seed=0):
    rng = np.random.default_rng(seed)
    n = NB * per_bin
    L = np.tril(rng.normal(size=(n, N_COEF, N_COEF))) * 0.02 * scale
    chol = pack_star(L, L * 0.7).astype(np.float32)
    counts = np.full(NB, per_bin)
    return {"chol": chol, "bin_edges": EDGES, "bin_count": counts,
            "bin_start": np.concatenate([[0], np.cumsum(counts)[:-1]]),
            "hist_edges": np.round(np.arange(0.0, 21.0001, 0.1), 3),
            "hist_counts": np.exp(-0.5 * ((np.arange(210) * 0.1 - 16) / 1.5) ** 2) * 1e4}


def make_batch(dm, n=5, g=13.0, seed=2):
    rng = np.random.default_rng(seed)
    coef = np.abs(rng.normal(size=(n, 110))) * 2e3 + 1e3
    x = sample_spectra(coef, dm)
    flux550 = (coef[:, :N_COEF] @ dm["dm_bp"] * dm["w_bp"]
               + coef[:, N_COEF:] @ dm["dm_rp"] * dm["w_rp"])[:, NORM_INDEX]
    sigma = np.full((n, 343), 0.01, np.float32)
    side = np.stack([np.full(n, g), np.full(n, 1.0), np.log10(flux550),
                     np.full(n, 2.5)], 1).astype(np.float32)
    return {"x": torch.from_numpy(x), "coef": torch.from_numpy(coef.astype(np.float32)),
            "sigma": torch.from_numpy(sigma), "side": torch.from_numpy(side),
            "valid": torch.ones(n, dtype=torch.bool)}, coef, flux550


def test_identity_when_gprime_equals_g_and_donor_noise_zero():
    dm = synth_dm()
    aug = NoiseMatchAugment(dm, make_bank(scale=0.0), device="cpu", p=1.0)
    batch, coef, f550 = make_batch(dm)
    g = batch["side"][:, 0].double()
    gen = torch.Generator().manual_seed(0)
    out = aug(batch, gen=gen, fixed_g=g + 1e-9)       # s = 1 - O(1e-9)
    assert out["aug"].all()                           # the augmentation path ran
    # coef is stored as float32 in the batch, hence the loose tolerance
    np.testing.assert_allclose(out["coef"].numpy(), batch["coef"].numpy(), rtol=1e-5)
    assert out["x"] is batch["x"]                     # the network input is rebuilt by the caller, never here
    np.testing.assert_allclose(out["sigma"].numpy(), batch["sigma"].numpy(), rtol=1e-5)
    np.testing.assert_allclose(out["side"][:, :3].numpy(), batch["side"][:, :3].numpy(), atol=1e-5)


def test_noise_std_matches_propagated_sigma():
    dm = synth_dm()
    bank = make_bank(scale=1.0, per_bin=1)
    aug = NoiseMatchAugment(dm, bank, device="cpu", p=1.0)
    n = 20000
    batch, coef, f550 = make_batch(dm, n=n, g=13.0)
    for k in ("x", "coef", "sigma", "side"):
        batch[k] = batch[k][:1].expand(n, *batch[k].shape[1:]).clone()
    batch["valid"] = torch.ones(n, dtype=torch.bool)
    batch["sigma"][:] = 0.0                           # isolate the donor term
    gp = 16.0
    out = aug(batch, gen=torch.Generator().manual_seed(1), fixed_g=gp)
    assert out["aug"].all()
    s = 10 ** (-0.4 * (gp - 13.0))
    f550_new = 10.0 ** out["side"][:, 2].double()
    flux = torch.as_tensor(absolute_flux(out["coef"].double().numpy(), dm))      # absolute units
    donor = int(((gp - 10.0) / 0.1) // 1)
    sig_d = pixel_sigma_abs(bank["chol"][donor:donor + 1], dm)[0]
    emp = flux.std(0).numpy()
    np.testing.assert_allclose(emp, sig_d, rtol=0.05)
    # the sigma the model sees is the propagated one (own term zero here)
    pred = (out["sigma"].double() * f550_new[:, None]).mean(0).numpy()
    np.testing.assert_allclose(pred, sig_d, rtol=0.02)
    # mean flux is the dimmed star
    base = (batch["coef"][0].double().numpy())
    mean_flux = ((base[:N_COEF] * s) @ dm["dm_bp"]) * dm["w_bp"] \
        + ((base[N_COEF:] * s) @ dm["dm_rp"]) * dm["w_rp"]
    assert abs(flux.mean(0).numpy()[200] / mean_flux[200] - 1) < 0.01
    assert abs(out["side"][0, 0] - gp) < 1e-5


def test_own_noise_scales_and_adds_in_quadrature():
    dm = synth_dm()
    aug = NoiseMatchAugment(dm, make_bank(scale=0.0), device="cpu", p=1.0)  # donor 0
    batch, coef, f550 = make_batch(dm, n=3)
    out = aug(batch, gen=torch.Generator().manual_seed(0), fixed_g=15.0)
    s = 10 ** (-0.4 * 2.0)
    # relative own sigma is scale-invariant when the donor adds nothing
    np.testing.assert_allclose(out["sigma"].numpy(), batch["sigma"].numpy(), rtol=1e-4)
    np.testing.assert_allclose(out["side"][:, 2].numpy(),
                               batch["side"][:, 2].numpy() + np.log10(s), atol=1e-4)


def test_stars_fainter_than_target_untouched_and_p_zero():
    dm = synth_dm()
    aug = NoiseMatchAugment(dm, make_bank(), device="cpu", p=0.0)
    batch, *_ = make_batch(dm, g=16.5)
    out = aug(batch, gen=torch.Generator().manual_seed(0))
    assert not out["aug"].any()
    aug.p = 1.0
    out = aug(batch, gen=torch.Generator().manual_seed(0), fixed_g=15.0)
    assert not out["aug"].any()
    out = aug(batch, gen=torch.Generator().manual_seed(0), fixed_g=17.0)
    assert out["aug"].all()
    assert (out["side"][:, 0] > batch["side"][:, 0]).all()


def test_random_target_in_range_and_labels_untouched():
    dm = synth_dm()
    aug = NoiseMatchAugment(dm, make_bank(), device="cpu", p=1.0, g_max=17.65)
    g = torch.full((2000,), 14.0, dtype=torch.float64)
    gp, b, can = aug.draw_target(g, torch.Generator().manual_seed(0))
    assert can.all() and (gp >= 14.0).all() and (gp <= 17.65).all()
    assert (gp.mean() > 15.0)
    gu = NoiseMatchAugment(dm, make_bank(), device="cpu", g_target="uniform")
    gp2, *_ = gu.draw_target(g, torch.Generator().manual_seed(0))
    assert abs(gp2.mean() - (14.0 + 17.65) / 2) < 0.2
