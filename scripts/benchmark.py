#!/usr/bin/env python
"""Measure inference latency and verdict distribution on a folder of images.

    python scripts/benchmark.py --images data/demo/defective --repeat 3
"""
import argparse
import time

import cv2
import numpy as np

from _common import ROOT, list_images  # noqa: F401

from app.config import Settings
from app.inspection.pipeline import InspectionPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--product", default=None)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    pipe = InspectionPipeline(Settings())
    imgs = [cv2.imread(str(p)) for p in list_images(args.images)]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        raise SystemExit("no images")
    pipe.inspect(imgs[0], args.product, annotate=False)  # warm-up
    lat, verdicts = [], {}
    for _ in range(args.repeat):
        for im in imgs:
            t = time.perf_counter()
            r = pipe.inspect(im, args.product, annotate=False)
            lat.append((time.perf_counter() - t) * 1000)
            verdicts[r.verdict] = verdicts.get(r.verdict, 0) + 1
    lat = np.array(lat)
    print(f"{len(lat)} inferences | detector={pipe.detector.backend} banks={pipe.anomaly.available()}")
    print(f"latency ms: mean {lat.mean():.1f}  p50 {np.percentile(lat, 50):.1f}  p95 {np.percentile(lat, 95):.1f}  "
          f"max {lat.max():.1f}  -> {1000 / lat.mean():.1f} FPS")
    print("verdicts:", verdicts)


if __name__ == "__main__":
    main()
