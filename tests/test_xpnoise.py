"""XP noise: covariance convention, packed Cholesky round trip, pixel sigma."""
import numpy as np
import pytest

from aspgapv3.data.xp import N_COEF
from aspgapv3.data.xpnoise import (N_PACK, N_TRI, cholesky_jitter, corr_to_cov_dr3,
                                   normalized_sigma_ratio, pack_star, pack_tril,
                                   pixel_sigma_abs, unpack_star, unpack_tril)


def synth_inputs(n=6, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, N_COEF, 80))
    S = A @ np.swapaxes(A, 1, 2) / 80                       # random SPD
    d = np.sqrt(np.diagonal(S, axis1=1, axis2=2))
    corr = S / d[:, :, None] / d[:, None, :]
    tri = np.tril_indices(N_COEF, k=-1)
    err = rng.uniform(1, 30, size=(n, N_COEF))
    std = rng.uniform(0.8, 1.4, size=n)
    return corr[:, tri[0], tri[1]], err, std, corr


def synth_dm(seed=1):
    rng = np.random.default_rng(seed)
    dm = {"dm_bp": np.abs(rng.normal(size=(N_COEF, 343))) * 1e-19,
          "dm_rp": np.abs(rng.normal(size=(N_COEF, 343))) * 1e-19}
    w = np.linspace(1, 0, 343)
    dm.update(w_bp=w, w_rp=1 - w, grid=np.arange(336.0, 1021.0, 2.0))
    return dm


def test_cov_matches_gaiaxpy_convention():
    flat, err, std, corr = synth_inputs()
    C = corr_to_cov_dr3(flat, err, std)
    for i in range(len(err)):
        d = np.diag(err[i]) / std[i]
        cov_gx = d @ corr[i] @ d                            # gaiaxpy dr3int5
        np.testing.assert_allclose(C[i], std[i] ** 2 * cov_gx, rtol=1e-10)
        np.testing.assert_allclose(C[i], np.diag(err[i]) @ corr[i] @ np.diag(err[i]),
                                   rtol=1e-10)


def test_pack_roundtrip_and_pd():
    flat, err, std, _ = synth_inputs()
    C = corr_to_cov_dr3(flat, err, std)
    L, lvl, ok = cholesky_jitter(C)
    assert ok.all() and (lvl == 0).all()
    assert N_TRI == 1540 and N_PACK == 3080
    p = pack_tril(L)
    assert p.shape == (len(C), N_TRI)
    np.testing.assert_array_equal(unpack_tril(p), L)
    star = pack_star(L, L[::-1])
    a, b = unpack_star(star)
    np.testing.assert_array_equal(a, L)
    np.testing.assert_array_equal(b, L[::-1])
    # float32 storage still reproduces the covariance
    L32 = unpack_tril(p.astype(np.float32)).astype(np.float64)
    np.testing.assert_allclose(L32 @ np.swapaxes(L32, 1, 2), C, rtol=1e-4,
                               atol=1e-4 * np.abs(C).max())


def test_jitter_rescues_semidefinite_and_flags_garbage():
    flat, err, std, _ = synth_inputs(n=3)
    C = corr_to_cov_dr3(flat, err, std)
    v = np.random.default_rng(3).normal(size=N_COEF)
    C[1] = np.outer(v, v)                                   # rank 1, singular
    C[2] = np.nan
    L, lvl, ok = cholesky_jitter(C)
    assert lvl[0] == 0 and lvl[1] > 0 and ok[1]
    assert lvl[2] == -1 and not ok[2] and (L[2] == 0).all()


def test_pixel_sigma_matches_full_covariance():
    flat, err, std, _ = synth_inputs(n=4)
    dm = synth_dm()
    C = corr_to_cov_dr3(flat, err, std)
    L, _, _ = cholesky_jitter(C)
    packed = pack_star(L, L).astype(np.float64)
    sig = pixel_sigma_abs(packed, dm)
    pix = (dm["dm_bp"].T @ C[0] @ dm["dm_bp"]) * np.outer(dm["w_bp"], dm["w_bp"]) \
        + (dm["dm_rp"].T @ C[0] @ dm["dm_rp"]) * np.outer(dm["w_rp"], dm["w_rp"])
    np.testing.assert_allclose(sig[0], np.sqrt(np.diag(pix)), rtol=1e-8)


def test_ratio_sigma_vs_monte_carlo():
    flat, err, std, _ = synth_inputs(n=1)
    dm = synth_dm()
    L, _, _ = cholesky_jitter(corr_to_cov_dr3(flat, err, std))
    packed = pack_star(L, L)
    rng = np.random.default_rng(5)
    c0 = np.abs(rng.normal(size=(1, 110))) * 1e3
    z = rng.normal(size=(20000, 2, N_COEF))
    cb = c0[:, :N_COEF] + np.einsum("ij,nj->ni", L[0], z[:, 0])
    cr = c0[:, N_COEF:] + np.einsum("ij,nj->ni", L[0], z[:, 1])
    f = (cb @ dm["dm_bp"]) * dm["w_bp"] + (cr @ dm["dm_rp"]) * dm["w_rp"]
    r = f / f[:, 107:108]
    emp = r.std(0)
    pred = normalized_sigma_ratio(packed, c0, dm)[0]
    sel = np.arange(343) != 107
    np.testing.assert_allclose(emp[sel], pred[sel], rtol=0.06)
    assert pred[107] < 1e-6 * pred[sel].max() or abs(pred[107]) < 1e-12 + 1e-3 * pred[sel].mean()
