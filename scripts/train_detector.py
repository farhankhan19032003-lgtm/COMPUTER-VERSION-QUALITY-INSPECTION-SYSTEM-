#!/usr/bin/env python
"""Fine-tune a YOLO detector on your own labelled product photos (needs `pip install ultralytics`).

Dataset layout (YOLO format) described by a data.yaml, see data/products.example.yaml:
    images/train, images/val  +  labels/train, labels/val   (one .txt per image)

Classes can be product types (phone, laptop, watch ...) and/or defect types
(crack, scratch, dent, missing_screw ...). Label tools: Roboflow, CVAT, Label Studio.

    python scripts/train_detector.py --data data/products.yaml --epochs 80 --name products

Then set YOLO_WEIGHTS=models/yolo/products.pt in .env.
"""
import argparse
import shutil

from _common import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n.pt", help="start point: yolov8n/s/m.pt (n = fastest, edge friendly)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 for first GPU, cpu for CPU")
    ap.add_argument("--name", default="products")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics is not installed: pip install -r requirements-extra.txt")

    model = YOLO(args.model)
    results = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=str(ROOT / "runs"), name=args.name, patience=20,
        # augmentation tuned for product photos: lighting/angle change, no vertical flips
        hsv_v=0.4, degrees=10, translate=0.1, scale=0.4, fliplr=0.5, flipud=0.0, mosaic=1.0,
    )
    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    dest = ROOT / "models" / "yolo" / f"{args.name}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, dest)
    metrics = model.val(data=args.data)
    print(f"\nSaved {dest}\nmAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")
    print(f"Use it: YOLO_WEIGHTS={dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
