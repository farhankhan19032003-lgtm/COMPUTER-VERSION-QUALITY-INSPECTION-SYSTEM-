"""Product localisation.

Primary: Ultralytics YOLO (COCO weights out of the box, or your fine-tuned weights).
Fallback: classical foreground segmentation, so the service still works without YOLO.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..config import Settings
from ..utils import resolve_device
from .cv_defects import foreground_mask, mask_bbox
from .product_classifier import ProductClassifier
from .types import Detection

PERSON_ONLY_MIN_CONF = 0.40    # classifier cut-off for worn products when YOLO found only a person (ProductDetector.detect)
WORN_PRODUCTS = {"watch", "headphones"}
# What YOLO may label when it sees only a piece of the whole product named by the classifier.
PARTS_OF = {"laptop": {"keyboard", "monitor"}, "monitor": {"keyboard", "mouse"}}
PART_NAMES = {p for parts in PARTS_OF.values() for p in parts}
PART_OF_WHOLE_MIN_CONF = 0.45

log = logging.getLogger(__name__)

# Normalise detector labels to product categories. Keys are matched as substrings so
# custom classes like "smartphone_front" or "wrist_watch" still resolve.
PRODUCT_KEYWORDS = {
    "phone": "phone",
    "smartphone": "phone",
    "laptop": "laptop",
    "notebook": "laptop",
    "macbook": "laptop",
    "watch": "watch",
    "clock": "watch",       # COCO has no 'watch'; fine-tune for real watch inspection
    "tablet": "tablet",
    "ipad": "tablet",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "tv": "monitor",
    "monitor": "monitor",
    "remote": "remote",
    "headphone": "headphones",
    "earbud": "earbuds",
    "camera": "camera",
    "bottle": "bottle",
    "person": "person",     # not a product: reported as a person, never defect-checked (see pipeline)
}


def normalize_product(label: str) -> Optional[str]:
    low = label.lower().replace("_", " ")
    for key, val in PRODUCT_KEYWORDS.items():
        if key in low:
            return val
    return None


class YoloDetector:
    def __init__(self, settings: Settings) -> None:
        self.conf = settings.yolo_conf
        self.device = resolve_device(settings.device)
        self.model = None
        self.accept_all = False
        self.error: Optional[str] = None
        self._lock = threading.Lock()
        try:
            from ultralytics import YOLO  # optional dependency

            self.model = YOLO(settings.yolo_weights)
            names = getattr(self.model, "names", {}) or {}
            # COCO has 80 classes; anything else is assumed to be a custom product model.
            self.accept_all = len(names) != 80
            log.info("YOLO loaded: %s (%d classes, device=%s)", settings.yolo_weights, len(names), self.device)
        except Exception as exc:  # ImportError, missing weights, offline download...
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("YOLO unavailable (%s). Using classical fallback detector.", self.error)

    @property
    def available(self) -> bool:
        return self.model is not None

    def detect(self, img: np.ndarray) -> List[Detection]:
        if not self.available:
            return []
        with self._lock:  # ultralytics predictors are not thread-safe
            res = self.model.predict(
                img, conf=self.conf, verbose=False, device=0 if self.device == "cuda" else "cpu"
            )[0]
        out: List[Detection] = []
        for b in res.boxes:
            label = str(res.names[int(b.cls)])
            ptype = normalize_product(label) or (label if self.accept_all else None)
            if ptype is None:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in b.xyxy[0].tolist())
            out.append(Detection(label=label, product_type=ptype, confidence=float(b.conf), bbox=(x1, y1, x2, y2)))
        out.sort(key=lambda d: d.confidence * (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
        return out


_yunet = {}
_yunet_lock = threading.Lock()


def find_faces(img: np.ndarray, model_path: str) -> List[Detection]:
    """Faces that YOLO's whole-body 'person' class misses (close-ups, portraits), using OpenCV's YuNet model.

    Returns [] if the model file or OpenCV's face detector is missing, so inspection still works without it.
    """
    import cv2

    if not model_path or not hasattr(cv2, "FaceDetectorYN") or not Path(model_path).is_file():
        return []
    h, w = img.shape[:2]
    s = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    with _yunet_lock:  # the detector keeps its input size as state
        det = _yunet.get(model_path)
        if det is None:
            det = _yunet[model_path] = cv2.FaceDetectorYN.create(model_path, "", (320, 320), 0.7, 0.3, 5000)
        det.setInputSize((small.shape[1], small.shape[0]))
        _, found = det.detect(small)
    out = []
    for f in (found if found is not None else []):
        x, y, fw, fh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])
        px, py = 0.15 * fw, 0.2 * fh      # the model's box is tight on the face; widen it a little
        box = (int(max(0, (x - px) / s)), int(max(0, (y - py) / s)), int(min(w, (x + fw + px) / s)), int(min(h, (y + fh + py) / s)))
        out.append(Detection(label="face", product_type="person", confidence=score, bbox=box, source="face"))
    out.sort(key=lambda d: d.confidence, reverse=True)
    return out

def merge_overlapping(dets: List[Detection], nested: float = 0.8, iou: float = 0.6) -> List[Detection]:
    """One product seen twice: YOLO can return a big box, a shifted copy and one around the screen for the SAME phone.
    Keep the most confident box of each product type and drop others that sit mostly inside it or overlap it a lot.
    Boxes of the same type that do not overlap (three laptops in a row) are all kept."""
    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    kept: List[Detection] = []
    for d in sorted(dets, key=lambda x: x.confidence, reverse=True):
        dup = False
        for k in kept:
            if k.product_type != d.product_type:
                continue
            ix = max(0, min(d.bbox[2], k.bbox[2]) - max(d.bbox[0], k.bbox[0]))
            iy = max(0, min(d.bbox[3], k.bbox[3]) - max(d.bbox[1], k.bbox[1]))
            inter = ix * iy
            small = min(area(d.bbox), area(k.bbox)) or 1
            union = (area(d.bbox) + area(k.bbox) - inter) or 1
            if inter / small >= nested or inter / union >= iou:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


class ProductDetector:
    """YOLO if available, otherwise the largest foreground object."""

    def __init__(self, settings: Settings) -> None:
        self.yolo = YoloDetector(settings)
        self.face_model = settings.face_model
        self.classifier = ProductClassifier(settings.product_head, settings.product_min_conf)

    @property
    def backend(self) -> str:
        return "yolo" if self.yolo.available else "classical"

    def detect(self, img: np.ndarray, hint: Optional[str] = None, max_items: int = 3) -> List[Detection]:
        dets = self.yolo.detect(img) if self.yolo.available else []
        if hint:
            hinted = [d for d in dets if d.product_type == hint]
            # The user said what to inspect: a person in the frame (say, a hand holding the phone) must not
            # replace it. With no match we fall through to the classical object finder below.
            dets = hinted or ([d for d in dets if d.product_type != "person"] if hint != "person" else dets)
        # A person never outranks a product: a phone held in a hand is still inspected as a phone.
        dets = merge_overlapping(dets)
        dets = [d for d in dets if d.product_type != "person"] + [d for d in dets if d.product_type == "person"]
        if not hint and not any(d.product_type != "person" for d in dets):
            # YOLO named no product (COCO has no headphones, tablet or wrist watch, and it can miss a product that
            # fills the frame): ask the whole-photo classifier before giving up.
            guess = self.classifier.predict(img)
            if not guess and dets:
                # YOLO saw only a person: a wrist with a watch or headphones on a head is a person to YOLO too, so
                # for those worn products accept a lower confidence. Plain portraits come back as "other".
                low = self.classifier.predict(img, min_conf=PERSON_ONLY_MIN_CONF)
                guess = low if low and low[0] in WORN_PRODUCTS else None
            if guess:
                box = mask_bbox(foreground_mask(img)) or (0, 0, img.shape[1], img.shape[0])
                return [Detection(label=guess[0], product_type=guess[0], confidence=guess[1], bbox=box, source="classifier")]
        if not hint and dets and dets[0].product_type != "person":
            # YOLO named a product, but COCO's classes are coarse (headphones become "phone", a laptop becomes
            # "keyboard"). When the whole-photo classifier is confident and about as sure as YOLO, trust its name.
            guess = self.classifier.predict(img)
            top = dets[0]
            if not guess and top.product_type in PART_NAMES:
                # YOLO often sees just a part of the product (the keyboard of a laptop, a desk setup's keyboard). If the
                # whole-photo classifier names the whole thing, its answer is worth a lower confidence than usual.
                low = self.classifier.predict(img, min_conf=PART_OF_WHOLE_MIN_CONF)
                if low and top.product_type in PARTS_OF.get(low[0], ()):
                    guess = low
            if guess and guess[0] != top.product_type and (
                guess[1] >= top.confidence - 0.15 or top.product_type in PARTS_OF.get(guess[0], ())
            ):
                parts = PARTS_OF.get(guess[0], ())
                gone = [top] + [d for d in dets[1:] if d.product_type in parts]
                box = (min(d.bbox[0] for d in gone), min(d.bbox[1] for d in gone), max(d.bbox[2] for d in gone), max(d.bbox[3] for d in gone))
                # the parts YOLO saw (keyboard + screen) are the one product the classifier named: do not list them again
                dets = [Detection(label=guess[0], product_type=guess[0], confidence=guess[1], bbox=box, source="classifier")] + [d for d in dets if d not in gone]
        if dets:
            return dets[:max_items]

        if not hint:  # nothing from YOLO: a face close-up is still a person, not an unknown object
            faces = find_faces(img, self.face_model)
            if faces:
                return faces[:max_items]

        # Fallback: treat the dominant object as the product.
        mask = foreground_mask(img)
        box = mask_bbox(mask)
        h, w = img.shape[:2]
        if box is None:
            box = (0, 0, w, h)
        return [Detection(label="object", product_type=hint or "unknown", confidence=0.5, bbox=box,
                          source="selected" if hint else "none")]
