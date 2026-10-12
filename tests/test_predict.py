from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aspgapv3.io import read_table, write_table
from aspgapv3.predict import FLAG_COLUMNS, OUTPUT_COLUMNS, load_model, predict

EX = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def example():
    return pd.read_parquet(EX / "example_input.parquet")


def _run(df, model, **kw):
    return predict(df["bp_coefficients"], df["rp_coefficients"], df["phot_g_mean_mag"], model=model, **kw)


def test_reproduces_example_output(model, example):
    ref = pd.read_csv(EX / "example_output.csv")
    out = _run(example, model, batch_size=7)
    assert list(out.columns) == OUTPUT_COLUMNS + FLAG_COLUMNS
    for c in OUTPUT_COLUMNS:
        np.testing.assert_allclose(out[c], ref[c], rtol=1e-4, atol=1e-3, err_msg=c)
    for c in FLAG_COLUMNS:
        assert (out[c].to_numpy() == ref[c].to_numpy()).all(), c


def test_input_formats_agree(model, example, tmp_path):
    base = _run(example, model)
    bp = np.stack(example["bp_coefficients"]); rp = np.stack(example["rp_coefficients"])
    arr = predict(bp, rp, example["phot_g_mean_mag"].to_numpy(), model=model)
    pd.testing.assert_frame_equal(base, arr)
    strs = predict([str([float(v) for v in r]) for r in bp], ["{" + ",".join(repr(float(v)) for v in r) + "}" for r in rp],
                   example["phot_g_mean_mag"], model=model)
    pd.testing.assert_frame_equal(base, strs)
    # csv round trip with "[..]" strings
    ex = example.copy()
    for c in ("bp_coefficients", "rp_coefficients"):
        ex[c] = [str([float(v) for v in r]) for r in ex[c]]
    ex.to_csv(tmp_path / "in.csv", index=False)
    pd.testing.assert_frame_equal(base, _run(read_table(tmp_path / "in.csv"), model))
    write_table(base, tmp_path / "o.csv")


def test_bad_rows_and_flags(model, example):
    bp = np.stack(example["bp_coefficients"]); rp = np.stack(example["rp_coefficients"])
    g = example["phot_g_mean_mag"].to_numpy().copy()
    bp[0, 3] = np.nan
    g[1] = 18.0
    out = predict(bp, rp, g, model=model)
    assert out.loc[0, "flag_bad_input"] and out.loc[0, OUTPUT_COLUMNS].isna().all()
    assert out.loc[1, "flag_g_outside"] and not out.loc[2, "flag_g_outside"]
    assert not out.loc[2:, "flag_bad_input"].any()


def test_shape_error(model):
    with pytest.raises(ValueError):
        predict(np.zeros((3, 54)), np.zeros((3, 55)), np.ones(3), model=model)


def test_inference_import_is_light():
    import os, subprocess, sys
    code = ("import sys, aspgapv3.predict; "
            "bad=[m for m in ('yaml','aspgapv3.data','astropy','sklearn','joblib') if m in sys.modules]; "
            "sys.exit(len(bad))")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0
