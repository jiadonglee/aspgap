#!/usr/bin/env python
"""First contact with astraAllStarASPCAP: dump HDU structure, columns, and
which of our schema's column candidates actually match — BEFORE any column
mapping is trusted.

    python scripts/inspect_aspcap.py --fits <file> [--hdu 2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aspgapv3.labels.loaders import (resolve_columns, read_aspcap_table,  # noqa: E402
                                     wanted_columns)
from aspgapv3.labels.schema import GLOBAL_COLUMNS, LABELS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True)
    ap.add_argument("--hdu", type=int, default=2)
    args = ap.parse_args()

    from astropy.io import fits
    with fits.open(args.fits, memmap=not args.fits.endswith(".gz")) as hdul:
        print("HDUs:")
        for i, h in enumerate(hdul):
            ncols = len(h.columns) if hasattr(h, "columns") and \
                h.columns is not None else "-"
            print(f"  [{i}] {h.name}: ({ncols} cols)")
        hdr = hdul[args.hdu].header
        coldefs = hdul[args.hdu].columns
        print(f"\nHDU{args.hdu}: NAXIS2 = {hdr.get('NAXIS2')} rows")
        for k in ("EXTNAME", "V ASTRAVER", "ASTRAVER", "V JDCALIBR", "RUNSD",
                  "TELESCOP"):
            if k in hdr:
                print(f"  {k} = {hdr[k]}")
        print("\nColumns:")
        for col in coldefs:
            unit = f" [{col.unit}]" if col.unit else ""
            print(f"  {col.name:28s} {str(col.format):>8s}{unit}")
        n_rows = hdr.get("NAXIS2", 0)

    # resolution on a candidate-column slice only (memmap-safe)
    df = read_aspcap_table(args.fits, hdu=args.hdu,
                           columns=wanted_columns()).head(5000)
    try:
        resolved = resolve_columns(df)
    except Exception as exc:
        print(f"\ncolumn resolution failed: {exc}")
        return
    print("\nResolved against schema:")
    for logical, cands in GLOBAL_COLUMNS.items():
        got = resolved.get(logical)
        print(f"  {logical:20s} -> {got or 'MISSING (optional)'}")
    for s in LABELS:
        got = resolved.get(f"label:{s.name}")
        err = resolved.get(f"err:{s.name}")
        flag = resolved.get(f"flag:{s.name}")
        print(f"  {s.name:20s} -> {got or 'MISSING'} | err: {err or '-'} | "
              f"flag: {flag or '-'} | tier {s.tier} floor {s.sigma_floor}")


if __name__ == "__main__":
    main()
