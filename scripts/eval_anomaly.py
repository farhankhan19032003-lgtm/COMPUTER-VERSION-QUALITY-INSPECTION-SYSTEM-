#!/usr/bin/env python
"""Measure how well the anomaly model separates good from defective units.

MVTec AD / VisA style folders (research use; MVTec AD is CC BY-NC-SA, not for commercial use):
    python scripts/eval_anomaly.py --mvtec path/to/mvtec_ad/bottle

Your own data:
    python scripts/eval_anomaly.py --train-good data/good --test-good data/test_good --test-bad data/test_bad

Reports image-level AUROC (1.0 = perfect, 0.5 = coin flip) and, at the bank's own calibrated
threshold, the share of good units wrongly flagged and defective units caught. Run it on
images the bank never saw. Use it before trusting the system on a new product.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from _common import list_images
from _metrics import auc_binary, bootstrap_ci, fmt

from app.inspection.anomaly import AnomalyModel


def load(paths, limit=None):
    out = []
    for p in list(paths)[:limit]:
        img = cv2.imread(str(p))
        if img is not None:
            out.append(img)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mvtec", help="a category folder with train/good, test/good, test/<defect>/")
    ap.add_argument("--train-good")
    ap.add_argument("--test-good")
    ap.add_argument("--test-bad")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    bad_by_type = defaultdict(list)
    if args.mvtec:
        root = Path(args.mvtec)
        train = list_images(root / "train" / "good")
        good = list_images(root / "test" / "good")
        for d in sorted((root / "test").iterdir()):
            if d.is_dir() and d.name != "good":
                bad_by_type[d.name] = list_images(d)
    elif args.train_good and args.test_good and args.test_bad:
        train, good = list_images(args.train_good), list_images(args.test_good)
        bad_by_type["defective"] = list_images(args.test_bad)
    else:
        ap.error("give --mvtec, or all of --train-good --test-good --test-bad")
    if not train or not good or not any(bad_by_type.values()):
        raise SystemExit("Missing images: need train/good, test/good and at least one defect folder")

    print(f"Building bank from {len(train)} good images...")
    model = AnomalyModel.build(load(train), "eval", image_size=args.image_size, device=args.device,
                               pretrained=not args.no_pretrained)
    if not model.meta["pretrained"]:
        print("NOTE: no ImageNet weights available: using a random backbone, results will be weaker.")

    sg = np.array([model.score(i).ratio for i in load(good)])
    scores, labels, per_type = [sg], [np.zeros(len(sg))], {}
    for t, paths in bad_by_type.items():
        sb = np.array([model.score(i).ratio for i in load(paths)])
        per_type[t] = sb
        scores.append(sb)
        labels.append(np.ones(len(sb)))
    s, y = np.concatenate(scores), np.concatenate(labels)

    auc = auc_binary(y, s)
    lo, hi = bootstrap_ci(y, s, auc_binary, 500)
    print(f"\nImage-level AUROC: {fmt(auc)}  95% CI [{fmt(lo)}, {fmt(hi)}]   "
          f"({len(sg)} good, {int(y.sum())} defective test images)")
    print(f"At the calibrated threshold (ratio >= 1.0):")
    print(f"  good units wrongly flagged FAIL : {(sg >= 1.0).mean() * 100:5.1f}%")
    print(f"  good units sent to REVIEW (>=0.8): {(sg >= 0.8).mean() * 100:5.1f}%")
    print(f"  defective units caught (FAIL)   : {(s[y == 1] >= 1.0).mean() * 100:5.1f}%")
    if len(per_type) > 1:
        print("\nPer defect type, share caught at ratio >= 1.0:")
        for t, sb in sorted(per_type.items()):
            print(f"  {t:24s} {(sb >= 1.0).mean() * 100:5.1f}%  (n={len(sb)})")


if __name__ == "__main__":
    main()
