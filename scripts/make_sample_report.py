#!/usr/bin/env python
"""Write a clearly labelled SYNTHETIC sample patient report PDF (no real data, no real model).

    python scripts/make_sample_report.py --out sample_report.pdf
"""
import argparse
from pathlib import Path

from _common import ROOT  # noqa: F401  (puts the project root on sys.path)

from app.report.builder import build_report_pdf
from app.report.sample import sample_data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sample_report.pdf")
    args = ap.parse_args()
    Path(args.out).write_bytes(build_report_pdf(sample_data()))
    print(f"Wrote {args.out} (synthetic sample)")


if __name__ == "__main__":
    main()
