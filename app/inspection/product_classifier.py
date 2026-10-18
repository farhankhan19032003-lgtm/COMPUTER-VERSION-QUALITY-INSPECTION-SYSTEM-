"""Name the product in a photo when YOLO cannot: headphones, tablets, wrist watches, and anything else outside COCO.

COCO (YOLO's default classes) has no headphones, tablet or wrist-watch class, and it misses products that fill the
frame. A frozen ImageNet backbone (ResNet50 by default; the head file records which) turns the whole photo into a
feature vector and a tiny linear head, trained by scripts/train_product_classifier.py, picks the product type. If the head file or the
backbone weights are missing this class switches itself off and detection works exactly as before.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
OTHER = "other"          # "not one of the products we know": never reported as a product


DEFAULT_BACKBONE = "resnet50"        # measured best on the product photos: 50% named correctly vs 26% for MobileNetV3-small
LEGACY_BACKBONE = "mobilenet_v3_small"   # heads saved before the backbone was recorded used this one


def load_backbone(name: str = DEFAULT_BACKBONE):
    """Frozen ImageNet backbone up to the global average pool. Downloads the weights once (resnet50 about 100 MB)."""
    import torch.nn as nn
    import torchvision.models as tv

    if name == "resnet50":
        net = tv.resnet50(weights=tv.ResNet50_Weights.DEFAULT)
        return nn.Sequential(*list(net.children())[:-1], nn.Flatten()).eval()      # 2048-d
    if name == LEGACY_BACKBONE:
        net = tv.mobilenet_v3_small(weights=tv.MobileNet_V3_Small_Weights.DEFAULT)
        return nn.Sequential(net.features, net.avgpool, nn.Flatten()).eval()      # 576-d
    raise ValueError(f"unknown product backbone '{name}'")


def embed(backbone, images: Sequence[np.ndarray], batch: int = 16) -> np.ndarray:
    """BGR images -> L2-normalised features, shape (n, d). The same function is used for training and inference."""
    import torch

    out: List[np.ndarray] = []
    for i in range(0, len(images), batch):
        chunk = []
        for im in images[i:i + batch]:
            rgb = cv2.cvtColor(cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            chunk.append(((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))
        with torch.inference_mode():
            f = backbone(torch.from_numpy(np.stack(chunk))).numpy()
        out.append(f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8))
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


class ProductClassifier:
    """Whole-photo product type. `ok` is False when the head file is missing, so callers can simply skip it."""

    def __init__(self, head_path: str, min_conf: float = 0.55) -> None:
        self.ok = False
        self.min_conf = min_conf
        self.classes: List[str] = []
        self._lock = threading.Lock()
        self._backbone = None
        p = Path(head_path) if head_path else None
        if p is None or not p.is_file():
            return
        try:
            d = np.load(p, allow_pickle=False)
            self.classes = [str(c) for c in d["classes"]]
            self.W, self.b = d["W"].astype(np.float32), d["b"].astype(np.float32)
            self.backbone_name = str(d["backbone"]) if "backbone" in d.files else LEGACY_BACKBONE
            self.ok = True                               # the backbone is loaded on first use (keeps start-up light)
        except Exception as exc:
            log.warning("Product classifier head unreadable (%s): auto-detect uses YOLO only.", exc)

    def _net(self):
        if self._backbone is None:
            self._backbone = load_backbone(self.backbone_name)
        return self._backbone

    def predict(self, img_bgr: np.ndarray, min_conf: Optional[float] = None) -> Optional[Tuple[str, float]]:
        """(product_type, probability) when confident that the photo shows one product we know, else None.
        `min_conf` overrides the head's cut-off for one call."""
        if not self.ok:
            return None
        try:
            with self._lock:
                f = embed(self._net(), [img_bgr])[0]
        except Exception as exc:                          # offline first run, out of memory, ...
            log.warning("Product classifier failed (%s); turning it off for this run.", exc)
            self.ok = False
            return None
        z = f @ self.W.T + self.b
        p = np.exp(z - z.max())
        p /= p.sum()
        i = int(p.argmax())
        if self.classes[i] == OTHER or float(p[i]) < (self.min_conf if min_conf is None else min_conf):
            return None
        return self.classes[i], float(p[i])
