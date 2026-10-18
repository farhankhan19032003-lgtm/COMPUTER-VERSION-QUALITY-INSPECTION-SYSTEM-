#!/usr/bin/env python
"""Generate synthetic product photos so you can try the whole flow without real data.

    python scripts/make_demo_data.py            # writes data/demo/good and data/demo/defective
    python scripts/build_memory_bank.py --images data/demo/good --product phone --no-crop
    python scripts/benchmark.py --images data/demo/defective

These are drawings, not real products: good enough to test the plumbing, not to judge accuracy.
"""
import argparse
from pathlib import Path

import cv2

from _common import ROOT

from tests.synth import jitter, make_phone, missing_part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-good", type=int, default=40)
    ap.add_argument("--n-bad", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "data" / "demo"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "good").mkdir(parents=True, exist_ok=True)
    (out / "defective").mkdir(parents=True, exist_ok=True)

    for i in range(args.n_good):
        cv2.imwrite(str(out / "good" / f"good_{i:03d}.jpg"), jitter(make_phone(1000 + i), i))
    for i in range(args.n_bad):
        kind = ("crack", "scratch", "missing")[i % 3]
        img = make_phone(5000 + i, crack=kind == "crack", scratch=kind == "scratch")
        if kind == "missing":
            img = missing_part(img)
        cv2.imwrite(str(out / "defective" / f"{kind}_{i:03d}.jpg"), jitter(img, 900 + i))
    print(f"Wrote {args.n_good} good and {args.n_bad} defective images to {out}")


if __name__ == "__main__":
    main()
