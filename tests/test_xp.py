import numpy as np
import torch

from aspgapv3.data.xp import (NORM_INDEX, XP_GRID_NM, sample_spectra,
                              build_design_matrix)


def test_grid_and_normalization_convention():
    assert XP_GRID_NM.size == 343
    assert XP_GRID_NM[0] == 336.0 and XP_GRID_NM[-1] == 1020.0
    assert XP_GRID_NM[NORM_INDEX] == 550.0


def test_sample_spectra_constant_flux_normalizes_to_one():
    # mock design matrices: only the bias-like coefficient contributes
    dm = {"dm_bp": np.zeros((55, 343)), "dm_rp": np.zeros((55, 343)),
          "w_bp": np.ones(343), "w_rp": np.zeros(343)}
    dm["dm_bp"][0, :] = 1.0
    coef = np.zeros((2, 110))
    coef[:, 0] = 5.0                          # BP bias coefficient
    out = sample_spectra(coef, dm)
    assert out.shape == (2, 343)
    assert np.allclose(out, 1.0)              # constant flux -> 550nm-normalized 1s


def test_sample_spectra_handles_degenerate_norm():
    dm = {"dm_bp": np.zeros((55, 343)), "dm_rp": np.zeros((55, 343)),
          "w_bp": np.ones(343), "w_rp": np.ones(343)}
    coef = np.zeros((2, 110))                 # all-zero coefficients -> zero norm
    out = sample_spectra(coef, dm)
    assert np.isfinite(out).all()
    assert (out == 0).all()


def test_design_matrix_cached(tmp_path):
    cache = tmp_path / "xp_design_343.npz"
    grid = np.arange(336.0, 1021.0, 2.0)
    np.savez(cache, dm_bp=np.ones((55, 343)), dm_rp=np.ones((55, 343)),
             w_bp=np.ones(343), w_rp=np.ones(343), grid=grid)
    dm = build_design_matrix(cache)           # must read the cache, not gaiaxpy
    assert dm["dm_bp"].shape == (55, 343)
