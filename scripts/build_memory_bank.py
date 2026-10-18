#!/usr/bin/env python
"""Build an anomaly memory bank from photos of GOOD units (no defect labels needed).

    python scripts/build_memory_bank.py --images data/good_phones --product phone

Tips for a bank that works in production:
  * 30-200 photos of defect-free units, same camera, lighting and distance as production
  * include normal variation (colours, small position shifts, lighting)
  * one bank per product type; name it 'default' for a catch-all
"""
import argparse
import logging

import cv2

from _common import ROOT, list_images  # noqa: F401  (also fixes sys.path)

from app.config import Settings
from app.inspection.anomaly import AnomalyModel
from app.inspection.pipeline import InspectionPipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of good-unit photos")
    ap.add_argument("--product", required=True, help="product type name, e.g. phone, laptop, watch, default")
    ap.add_argument("--out", default=str(ROOT / "models" / "anomaly"))
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-crop", action="store_true", help="use whole photos instead of detector crops")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline / tests only)")
    ap.add_argument("--coreset-ratio", type=float, default=0.1)
    args = ap.parse_args()

    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    paths = list_images(args.images)
    if not paths:
        raise SystemExit(f"No images found in {args.images}")

    pipe = None if args.no_crop else InspectionPipeline(Settings(_env_file=None, device=args.device))
    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"skip unreadable: {p}")
            continue
        if pipe is not None:  # crop exactly like inspection does, so train == inference
            img, _ = pipe.crop_product(img, hint=args.product if args.product != "default" else None)
        images.append(img)
    print(f"Building '{args.product}' bank from {len(images)} images...")

    model = AnomalyModel.build(
        images, args.product, image_size=args.image_size, device=args.device,
        coreset_ratio=args.coreset_ratio, pretrained=not args.no_pretrained,
    )
    out = f"{args.out}/{args.product}.pt"
    model.save(out)
    print(f"Saved {out}\n  patches in bank: {model.bank.shape[0]}\n  decision threshold: {model.threshold:.3f}")
    print(f"  calibration scores of held-out good images: {model.meta['cal_scores']}")
    if not model.meta["pretrained"]:
        print("  NOTE: ImageNet weights were unavailable; a random backbone was used and saved with the bank. "
              "Accuracy will be much better with pretrained weights (needs internet once).")


if __name__ == "__main__":
    main()
