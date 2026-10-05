import numpy as np
import pandas as pd
import pytest

from aspgapv3.labels.loaders import (build_label_matrix, global_clean,
                                     resolve_columns)


@pytest.fixture
def aspcap_like():
    n = 50
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "gaia_dr3_source_id": np.arange(1, n + 1),
        "snr": rng.uniform(10, 200, n),
        "flag_bad": np.zeros(n),
        "raw_teff": rng.uniform(3200, 9000, n),
        "raw_e_teff": np.full(n, 50.0),
        "teff": rng.uniform(3200, 9000, n),          # calibrated analog
        "e_teff": np.full(n, 50.0),
        "logg": rng.uniform(0.3, 5.2, n),
        "logg_err": np.full(n, 0.1),
        "m_h": rng.uniform(-2.5, 0.5, n),
        "m_h_err": np.full(n, 0.03),
        "alpha_m": rng.uniform(-0.2, 0.4, n),
        "alpha_m_err": np.full(n, 0.02),
        "alpha_m_flag": np.zeros(n),
        "fe_h": rng.uniform(-2.5, 0.5, n),
        "fe_h_err": np.full(n, 0.02),
        "mg_h": rng.uniform(-1.5, 0.8, n),
        "mg_h_err": np.full(n, 0.02),
    })
    # inject pathologies
    df.loc[3, "raw_teff"] = np.nan          # missing teff
    df.loc[4, "alpha_m"] = 5.0              # out of physical range
    df.loc[5, "alpha_m_flag"] = 0b100001100  # bit 2 (upper limit: keep) +
                                            # bits 5,7 (censored: mask)
    df.loc[6, "fe_h"] = np.nan              # missing abundance
    df.loc[7, "snr"] = 30.0                 # low S/N
    df.loc[8, "flag_bad"] = 1               # global bad flag
    return df


def test_resolve_columns(aspcap_like):
    cols = resolve_columns(aspcap_like)
    assert cols["source_id"] == "gaia_dr3_source_id"
    assert cols["label:teff"] == "raw_teff"          # first candidate wins
    assert cols["label:alpha_m"] == "alpha_m"
    assert cols["flag:mg_h"] is None                 # optional flags may be absent


def test_global_clean_cuts(aspcap_like):
    df, stats = global_clean(aspcap_like, snr_min=50.0, teff_max=8000.0)
    assert stats["n_clean"] < stats["n_input"]
    assert (df["snr"] > 50.0).all()
    assert (df["raw_teff"] < 8000.0).all()
    assert not (pd.to_numeric(df["flag_bad"]) != 0).any()
    sid = df["gaia_dr3_source_id"]
    assert 4 not in set(sid)          # missing teff dropped by the teff cut
    assert 8 not in set(sid)          # low snr
    assert 9 not in set(sid)          # flag_bad


def test_per_label_masks_never_dropna_globally(aspcap_like):
    labels = ["teff", "logg", "m_h", "alpha_m", "fe_h", "mg_h"]
    df, _ = global_clean(aspcap_like)
    y, yerr, mask, _ = build_label_matrix(df, labels)
    # star with NaN fe_h keeps its other labels
    i6 = df.index[df["gaia_dr3_source_id"] == 7][0]
    assert not mask[i6, labels.index("fe_h")]
    assert mask[i6, labels.index("teff")]
    # out-of-range alpha_m masked, not dropped from the sample
    i4 = df.index[df["gaia_dr3_source_id"] == 5][0]
    assert not mask[i4, labels.index("alpha_m")]
    assert mask[i4, labels.index("logg")]
    # flagged alpha_m (bits 0|1) masked
    i5 = df.index[df["gaia_dr3_source_id"] == 6][0]
    assert not mask[i5, labels.index("alpha_m")]
    # coverage: no label has zero stars, no star has zero labels
    assert (mask.sum(0) > 0).all()
    assert (mask.sum(1) > 0).all()


def test_teff_variant_selects_column(aspcap_like):
    df, _ = global_clean(aspcap_like)
    y_raw, _, m_raw, _ = build_label_matrix(df, ["teff"], teff_variant="raw")
    y_cal, _, m_cal, _ = build_label_matrix(df, ["teff"], teff_variant="calibrated")
    k = 0
    ok = m_raw[:, k] & m_cal[:, k]
    assert not np.allclose(y_raw[ok, k], y_cal[ok, k])   # different columns used
