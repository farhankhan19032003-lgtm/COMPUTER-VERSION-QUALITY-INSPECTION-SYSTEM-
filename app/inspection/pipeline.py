"""Inspection pipeline: detect product -> classical checks + anomaly model -> verdict."""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import Settings
from ..repair.advice import build_advice
from ..utils import INFERENCE_GATE, clip_box, encode_jpeg
from .anomaly import AnomalyRegistry, AnomalyResult
from .cv_defects import assess_quality, find_defects, foreground_mask
from .detector import ProductDetector
from .types import Defect, Detection, InspectionResult, ItemResult, VERDICT_ORDER

log = logging.getLogger(__name__)

VERDICT_COLOR = {"PASS": (60, 180, 60), "REVIEW": (0, 165, 255), "FAIL": (50, 50, 230)}  # BGR
DEFECT_COLOR = (200, 40, 200)
PERSON_COLOR = (200, 130, 40)   # BGR: blue, neutral (a person is not PASS/REVIEW/FAIL)
UNKNOWN_NOTE = "Couldn't tell what this is. Choose the product type above (phone, laptop, headphones...) and try again."
PERSON_NOTE = (
    "A person was detected, not a product. Product inspection doesn't apply to people: "
    "use Health screening or Live vitals instead."
)


class InspectionPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.detector = ProductDetector(settings)
        self.anomaly = AnomalyRegistry(settings.anomaly_bank_dir, settings.device)

    # ------------------------------------------------------------------ public API
    def status(self) -> dict:
        return {
            "detector": self.detector.backend,
            "detector_error": self.detector.yolo.error,
            "anomaly_banks": self.anomaly.available(),
        }

    def crop_product(self, img: np.ndarray, hint: Optional[str] = None) -> Tuple[np.ndarray, Detection]:
        """Crop exactly as inspection does. Use this when building anomaly banks."""
        det = self.detector.detect(img, hint=hint, max_items=1)[0]
        x1, y1, x2, y2 = self._padded(det.bbox, img.shape)
        return img[y1:y2, x1:x2], det

    def inspect(self, img: np.ndarray, product_hint: Optional[str] = None, annotate: bool = True) -> InspectionResult:
        with INFERENCE_GATE:                       # at most a few photos analysed at once (small hosts)
            return self._inspect(img, product_hint, annotate)

    def _inspect(
        self,
        img: np.ndarray,
        product_hint: Optional[str] = None,
        annotate: bool = True,
    ) -> InspectionResult:
        t0 = time.perf_counter()
        quality = assess_quality(img)
        detections = self.detector.detect(img, hint=product_hint)

        people = [d for d in detections if d.product_type == "person"]
        items: List[ItemResult] = []
        heats: List[Tuple[Tuple[int, int, int, int], AnomalyResult]] = []
        for det in (d for d in detections if d.product_type != "person"):
            item, heat = self._inspect_item(img, det)
            if item is not None:
                items.append(item)
                if heat is not None:
                    heats.append((item.bbox, heat))

        if not items and people:  # only people in the photo: say so instead of "unknown"
            p = max(people, key=lambda d: d.confidence * (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
            items.append(ItemResult("person", p.bbox, p.confidence, "REVIEW", 0.0, None, [], [PERSON_NOTE], source=p.source))

        if not items:  # nothing usable found
            h, w = img.shape[:2]
            items.append(
                ItemResult("unknown", (0, 0, w, h), 0.0, "REVIEW", 0.0, None, [], [UNKNOWN_NOTE], source="none")
            )

        verdict = max((it.verdict for it in items), key=lambda v: VERDICT_ORDER[v])
        reasons = [r for it in items for r in it.reasons]
        if not quality["ok"] and verdict == "PASS":
            verdict = "REVIEW"
            reasons.append("Image quality too low to certify a PASS: " + " ".join(quality["warnings"]))
        reasons.extend(w for w in quality["warnings"] if w not in " ".join(reasons))

        primary = max(items, key=lambda it: it.confidence)
        ratios = [it.anomaly_ratio for it in items if it.anomaly_ratio is not None]
        result = InspectionResult(
            verdict=verdict,
            score=max(it.score for it in items),
            product_type=primary.product_type,
            anomaly_ratio=max(ratios) if ratios else None,
            items=items,
            detections=detections,
            quality=quality,
            reasons=reasons,
            latency_ms=0.0,
            backends={"detector": self.detector.backend, "anomaly": "patchcore" if heats else "none"},
            people=len(people),
        )
        if primary.product_type != "person":
            result.advice = build_advice(result.defects, verdict, primary.product_type)
        if annotate:
            result.annotated = encode_jpeg(self._annotate(img, result, heats))
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # --------------------------------------------------------------------- internals
    @staticmethod
    def _padded(box, shape, pad: float = 0.04) -> Tuple[int, int, int, int]:
        h, w = shape[:2]
        x1, y1, x2, y2 = box
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        return clip_box((x1 - px, y1 - py, x2 + px, y2 + py), w, h)

    def _inspect_item(self, img: np.ndarray, det: Detection):
        x1, y1, x2, y2 = self._padded(det.bbox, img.shape)
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None, None
        roi = img[y1:y2, x1:x2]
        defects: List[Defect] = []

        # 1) classical checks
        mask = foreground_mask(roi)
        for d in find_defects(roi, mask):
            d.bbox = (d.bbox[0] + x1, d.bbox[1] + y1, d.bbox[2] + x1, d.bbox[3] + y1)
            defects.append(d)

        # 2) learned anomaly model (if a bank exists for this product)
        ratio: Optional[float] = None
        heat: Optional[AnomalyResult] = None
        model = self.anomaly.get(det.product_type)
        if model is not None:
            heat = model.score(roi)
            ratio = heat.ratio
            defects.extend(self._anomaly_defects(heat, roi.shape, (x1, y1)))

        item = self._verdict(det, (x1, y1, x2, y2), defects, ratio, model is not None)
        return item, heat

    def _anomaly_defects(self, res: AnomalyResult, roi_shape, offset) -> List[Defect]:
        s = self.settings
        rh, rw = roi_shape[:2]
        sx, sy = rw / res.size, rh / res.size
        ox, oy = offset
        sev = (
            "high" if res.ratio >= s.anomaly_fail_ratio * 1.5
            else "medium" if res.ratio >= s.anomaly_fail_ratio
            else "low"
        )
        regions = res.regions
        if not regions and res.ratio >= s.anomaly_review_ratio:
            py, px = np.unravel_index(int(res.heat.argmax()), res.heat.shape)
            regions = [(max(0, px - 16), max(0, py - 16), min(res.size, px + 16), min(res.size, py + 16))]
        out = []
        for (a, b, c, d) in regions:
            peak = float(res.heat[b:d, a:c].max()) if d > b and c > a else res.ratio
            out.append(
                Defect(
                    type="anomaly",
                    severity=sev,
                    score=min(1.0, peak / (2 * s.anomaly_fail_ratio)),
                    bbox=(int(a * sx) + ox, int(b * sy) + oy, int(c * sx) + ox, int(d * sy) + oy),
                    source="anomaly",
                    detail=f"{peak:.2f}x the normal threshold",
                )
            )
        return out

    def _verdict(self, det, box, defects, ratio, has_model) -> ItemResult:
        s = self.settings
        reasons: List[str] = []
        high = [d for d in defects if d.severity == "high"]
        med = [d for d in defects if d.severity == "medium"]
        low = [d for d in defects if d.severity == "low"]

        if high:
            verdict = "FAIL"
            reasons.append("Severe visible damage: " + ", ".join(sorted({d.type for d in high})) + ".")
        elif ratio is not None and ratio >= s.anomaly_fail_ratio:
            verdict = "FAIL"
            reasons.append(f"Differs strongly from known-good {det.product_type} units ({ratio:.2f}x threshold).")
        elif med:
            verdict = "REVIEW"
            reasons.append("Possible damage: " + ", ".join(sorted({d.type for d in med})) + ".")
        elif ratio is not None and ratio >= s.anomaly_review_ratio:
            verdict = "REVIEW"
            reasons.append(f"Slightly unusual appearance ({ratio:.2f}x threshold).")
        elif len([d for d in low if d.type != "spot"]) >= 3:
            # "spot" findings are shown but never change the verdict on their own: logos,
            # camera lenses and UI elements look like spots, so they are weak evidence.
            verdict = "REVIEW"
            reasons.append("Several minor marks detected.")
        else:
            verdict = "PASS"
        if verdict == "FAIL":
            # Only fail on evidence we trust; otherwise show the marks but leave the call to a person.
            why = None
            if det.source in ("classifier", "none"):
                what = f"the {det.product_type}" if det.source == "classifier" else "the product"
                why = (f"{what} was not located in the photo, so the marks may belong to the background. "
                       "Choose the product type or retake with the product alone in frame")
            elif ratio is not None and ratio >= s.anomaly_trust_ratio:
                why = (f"the photo looks very different from the reference photos of {det.product_type} units "
                       f"({ratio:.1f}x the threshold), so the automatic check cannot judge it. Use the same background, "
                       "light and distance as the reference photos")
            elif s.require_bank_for_fail and not has_model:
                why = (f"there is no trained reference for '{det.product_type}', and plain photo checks alone fire on "
                       "edges, keys and backgrounds. Build a reference bank (scripts/build_memory_bank.py) to enable FAIL")
            if why:
                verdict = "REVIEW"
                reasons[:] = [f"Possible damage marks found, but {why}, or have a person check."]
        if det.product_type == "unknown":
            reasons.append("Product not recognised, so only generic damage checks ran. Choose the product type for a better result.")
        elif not has_model and verdict == "PASS":
            reasons.append(
                f"No trained anomaly model for '{det.product_type}': only visible damage checks were run."
            )

        score = max([d.score for d in defects] + [0.0])
        if ratio is not None:
            score = max(score, min(1.0, ratio / (2 * s.anomaly_fail_ratio)))
        return ItemResult(det.product_type, box, det.confidence, verdict, score, ratio, defects, reasons, source=det.source)

    # -------------------------------------------------------------------- drawing
    def _annotate(self, img: np.ndarray, res: InspectionResult, heats) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        t = max(2, int(round(max(h, w) / 400)))
        fs = max(0.5, max(h, w) / 1200)

        for (x1, y1, x2, y2), heat in heats:  # anomaly heatmap overlay
            if x2 <= x1 or y2 <= y1:
                continue
            m = cv2.resize(np.clip(heat.heat / 2.0, 0, 1), (x2 - x1, y2 - y1))
            colored = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
            alpha = (np.clip((m - 0.35) / 0.65, 0, 1) * 0.55)[..., None]
            roi = out[y1:y2, x1:x2].astype(np.float32)
            out[y1:y2, x1:x2] = (roi * (1 - alpha) + colored.astype(np.float32) * alpha).astype(np.uint8)

        if res.product_type != "person":   # people next to a product: thin box for context only
            for d in res.detections:
                if d.product_type == "person":
                    px1, py1, px2, py2 = d.bbox
                    cv2.rectangle(out, (px1, py1), (px2, py2), PERSON_COLOR, max(1, t - 1))
                    cv2.putText(out, "person", (px1 + 4, max(14, py1 + 16)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7,
                                PERSON_COLOR, 2, cv2.LINE_AA)

        for it in res.items:
            person = it.product_type == "person"
            color = PERSON_COLOR if person else VERDICT_COLOR[it.verdict]
            x1, y1, x2, y2 = it.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
            label = f"person {it.confidence:.2f}" if person else f"{it.product_type} {it.verdict} {it.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            ty = max(th + 6, y1)
            cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty + 2), color, cv2.FILLED)
            cv2.putText(out, label, (x1 + 4, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
            for d in it.defects:
                dx1, dy1, dx2, dy2 = d.bbox
                cv2.rectangle(out, (dx1, dy1), (dx2, dy2), DEFECT_COLOR, max(1, t - 1))
                cv2.putText(out, d.type, (dx1, max(12, dy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7,
                            DEFECT_COLOR, 2, cv2.LINE_AA)
        return out
