#!/usr/bin/env python
"""Export models for edge deployment (smaller, faster runtimes: ONNX Runtime, TensorRT, OpenVINO).

    python scripts/export_onnx.py --yolo models/yolo/products.pt
    python scripts/export_onnx.py --anomaly models/anomaly/phone.pt

The YOLO export uses Ultralytics (also supports --format engine for TensorRT).
The anomaly export writes the feature extractor to ONNX; the memory bank (.pt) stays
as is, and nearest-neighbour search is a plain matrix operation in any runtime.
"""
import argparse
from pathlib import Path

import torch

from _common import ROOT  # noqa: F401

from app.inspection.anomaly import AnomalyModel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo")
    ap.add_argument("--anomaly")
    ap.add_argument("--format", default="onnx", help="ultralytics export format: onnx, engine, openvino, ...")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 (GPU / TensorRT)")
    args = ap.parse_args()
    if not (args.yolo or args.anomaly):
        ap.error("give --yolo and/or --anomaly")

    if args.yolo:
        from ultralytics import YOLO

        path = YOLO(args.yolo).export(format=args.format, imgsz=args.imgsz, half=args.half, simplify=True)
        print("YOLO exported:", path)

    if args.anomaly:
        model = AnomalyModel.load(Path(args.anomaly), "cpu")
        size = model.image_size
        dummy = torch.zeros(1, 3, size, size)
        out = Path(args.anomaly).with_suffix(".onnx")
        torch.onnx.export(
            model.backbone, dummy, str(out), input_names=["image"], output_names=["features"],
            dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}}, opset_version=17,
        )
        print("Anomaly backbone exported:", out)


if __name__ == "__main__":
    main()
