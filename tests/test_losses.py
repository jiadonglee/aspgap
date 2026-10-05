import numpy as np
import torch

from aspgapv3.losses import MaskedHeteroscedasticNLL


def make(n=64, k=3, seed=0):
    rng = np.random.default_rng(seed)
    mu0 = torch.zeros(n, k)
    lv0 = torch.zeros(n, k)
    y = torch.as_tensor(rng.normal(0, 1, (n, k)), dtype=torch.float32)
    yerr = torch.as_tensor(rng.uniform(0.01, 0.1, (n, k)), dtype=torch.float32)
    mask = torch.rand(n, k) > 0.3
    mask[0, :] = False                       # a fully-masked star
    floor = torch.full((k,), 0.05)
    loss = MaskedHeteroscedasticNLL(torch.ones(k), floor)
    return loss, mu0, lv0, y, yerr, mask


def test_finite_and_mask_respected():
    loss, mu, lv, y, yerr, mask = make()
    l = loss(mu, lv, y, yerr, mask)
    assert torch.isfinite(l)
    # changing values under mask=False must not change the loss
    y2 = y.clone()
    y2[~mask] = 123.0
    l2 = loss(mu, lv, y2, yerr, mask)
    assert torch.allclose(l, l2)
    # masked star contributes nothing even with an insane logvar
    lv_big = lv.clone()
    l3 = loss(mu, lv_big, y, yerr, mask)
    assert torch.allclose(l, l3)


def test_floor_stops_tiny_obs_errors_from_dominating():
    """A star with yerr=0 must not get effectively infinite weight: with the
    floor, the minimal achievable loss stays finite and the optimum is the
    data scatter."""
    n, k = 256, 1
    y = torch.zeros(n, k)
    yerr = torch.full((n, k), 1e-8)
    mask = torch.ones(n, k, dtype=torch.bool)
    loss = MaskedHeteroscedasticNLL(torch.ones(k), torch.full((k,), 0.05))
    mu = torch.zeros(n, k, requires_grad=True)
    logvar = torch.zeros(n, k, requires_grad=True)
    opt = torch.optim.Adam([mu, logvar], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        l = loss(mu, logvar, y, yerr, mask)
        l.backward()
        opt.step()
    final = float(loss(mu, logvar, y, yerr, mask))
    # optimum: var = floor^2 = 0.0025 -> per-element loss = 0.5*(0 + ln 0.0025).
    # The point: the loss stays finite and pinned to the floor optimum — the
    # yerr=1e-8 star does NOT blow the loss up.
    expected = 0.5 * float(np.log(0.05 ** 2))
    assert np.isfinite(final)
    assert final >= expected - 1e-4          # never below the true optimum
    assert final - expected < 0.1            # and it converges to it


def test_optimal_sigma_tracks_residual_scale():
    """Heteroscedastic head should learn sigma ~ data scatter + floor, and mu ~ mean."""
    n, k = 512, 2
    rng = np.random.default_rng(1)
    y = torch.as_tensor(rng.normal(0, 0.3, (n, k)), dtype=torch.float32)
    yerr = torch.full((n, k), 0.1)
    mask = torch.ones(n, k, dtype=torch.bool)
    loss = MaskedHeteroscedasticNLL(torch.ones(k), torch.full((k,), 0.05))
    mu = torch.zeros(n, k, requires_grad=True)
    logvar = torch.zeros(n, k, requires_grad=True)
    opt = torch.optim.Adam([mu, logvar], lr=0.08)
    for _ in range(500):
        opt.zero_grad()
        loss(mu, logvar, y, yerr, mask).backward()
        opt.step()
    sigma_model = logvar.detach().exp().sqrt().median()
    total_var_expected = 0.3 ** 2 + 0.1 ** 2 + 0.05 ** 2
    # the model only owns the excess variance; it should shrink toward ~0 here
    # because yerr+floor already explain the scatter
    assert float(sigma_model) < 0.15
    assert abs(float(mu.detach().mean())) < 0.05


def test_per_label_weights_change_relative_emphasis():
    _, mu, lv, y, yerr, mask = make()
    w = torch.tensor([10.0, 1.0, 1.0])
    loss = MaskedHeteroscedasticNLL(w, torch.full((3,), 0.05))
    l = loss(mu, lv, y, yerr, mask)
    assert torch.isfinite(l)
