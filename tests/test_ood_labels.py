"""Sanity test of the GALAH -> APOGEE label-scale conversion in scripts/build_ood.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("requests")      # build-only dependency
import build_ood as b  # noqa: E402


def test_salaris_roundtrip():
    # solar-scaled: no shift; alpha-enhanced +0.4: [M/H] - [Fe/H] ~ +0.293, [alpha/M] ~ +0.107
    assert abs(b.mh_from_fe(-1.0, 0.0) + 1.0) < 1e-12 and abs(b.alpha_m_from_afe(0.0)) < 1e-12
    assert abs(b.mh_from_fe(-1.0, 0.4) - (-1.0 + 0.2933)) < 2e-3
    afe = np.linspace(-0.2, 0.6, 9)
    assert np.allclose(b.alpha_m_from_afe(afe) + (b.mh_from_fe(0.0, afe)), afe)   # [alpha/M] + ([M/H]-[Fe/H]) = [alpha/Fe]


def test_fit_scale_recovers_injected_offsets():
    rng = np.random.default_rng(0)
    n = 5000
    g = pd.DataFrame({"teff": rng.uniform(4000, 6500, n), "logg": rng.uniform(1, 4.5, n), "fe_h": rng.uniform(-1.5, 0.4, n),
                      "afe": rng.uniform(-0.1, 0.4, n)})
    g["m_h"], g["alpha_m"] = b.mh_from_fe(g.fe_h, g.afe), b.alpha_m_from_afe(g.afe)
    a = pd.DataFrame({"teff": g.teff - 60 + 20 * (g.teff - 5000) / 1000, "logg": g.logg - 0.1, "m_h": g.m_h + 0.03 + 0.05 * g.fe_h,
                      "alpha_m": g.alpha_m + 0.02}) + rng.normal(0, 1e-3, (n, 4)) * np.array([1e3, 1, 1, 1])
    sc = b.fit_scale(g, a)
    cal = b.apply_scale(g, sc)
    for lab, tol in (("teff", 3.0), ("logg", 0.01), ("m_h", 0.01), ("alpha_m", 0.01)):
        assert np.abs(cal[lab] - a[lab]).median() < tol, lab
