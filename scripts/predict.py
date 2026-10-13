#!/usr/bin/env python
"""Predict Teff, logg, [M/H], [alpha/M], [C/Fe], [N/Fe], [C/N] (giants), M_G, A0 and distance from Gaia DR3 XP coefficients.

    python scripts/predict.py --input stars.parquet --output out.csv
    python scripts/predict.py --source-ids ids.txt --output out.parquet     # fetch from the Gaia archive
    python scripts/predict.py --source-ids 4295806720,4295806336 --output out.csv

Input columns: bp_coefficients (55), rp_coefficients (55), phot_g_mean_mag [, source_id]; parquet / fits array
columns or csv "[...]" strings.  Fetching needs astroquery (pip install -e ".[fetch]").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="parquet / csv / fits table with the XP coefficients")
    src.add_argument("--source-ids", help="text file (one id per line) or comma-separated Gaia DR3 source_ids")
    ap.add_argument("--output", required=True, help="output .parquet or .csv")
    ap.add_argument("--device", default="auto", help="auto (cuda if available), cpu or cuda")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=500, help="source_ids per Gaia archive query")
    ap.add_argument("--weights", default=None, help="custom weights file (default: packaged)")
    ap.add_argument("--keep-input", action="store_true",
                    help="also write the input columns (coefficients are large; default: source_id and G only)")
    args = ap.parse_args()

    import numpy as np  # noqa: F401
    import pandas as pd
    import torch
    from aspgapv3.io import fetch_gaia_xp, read_source_ids, read_table, write_table
    from aspgapv3.predict import load_model, predict

    if args.input:
        df = read_table(args.input)
    else:
        df = fetch_gaia_xp(read_source_ids(args.source_ids), chunk=args.chunk)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = load_model(args.weights, device)
    res = predict(df["bp_coefficients"], df["rp_coefficients"], df["phot_g_mean_mag"],
                  batch_size=args.batch_size, model=model)
    head = df if args.keep_input else df[[c for c in ("source_id", "phot_g_mean_mag") if c in df.columns]]
    out = pd.concat([head.reset_index(drop=True), res], axis=1)
    write_table(out, args.output)
    print(f"{model} -> wrote {len(out)} rows to {args.output}; "
          f"flagged: g_outside={int(res.flag_g_outside.sum())} teff_edge={int(res.flag_teff_edge.sum())} "
          f"bad_input={int(res.flag_bad_input.sum())} cn_not_giant={int(res.flag_cn_not_giant.sum())}")


if __name__ == "__main__":
    main()
