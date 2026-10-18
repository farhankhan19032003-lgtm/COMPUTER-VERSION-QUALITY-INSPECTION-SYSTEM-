"""Plain data types shared by the inspection engine and the API."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2

SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}
VERDICT_ORDER = {"PASS": 0, "REVIEW": 1, "FAIL": 2}


@dataclass
class Detection:
    label: str            # raw detector label
    product_type: str     # normalized: phone | laptop | watch | ...
    confidence: float
    bbox: Box
    source: str = "yolo"  # who named it: yolo | classifier | face | selected (the user chose) | none

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "product_type": self.product_type,
            "confidence": round(float(self.confidence), 3),
            "bbox": [int(v) for v in self.bbox],
            "source": self.source,
        }


@dataclass
class Defect:
    type: str             # crack | scratch | spot | anomaly
    severity: str         # low | medium | high
    score: float          # 0..1
    bbox: Box             # full-image coordinates
    source: str           # cv | anomaly
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "score": round(float(self.score), 3),
            "bbox": [int(v) for v in self.bbox],
            "source": self.source,
            "detail": self.detail,
        }


@dataclass
class ItemResult:
    """Result for one product found in the image."""
    product_type: str
    bbox: Box
    confidence: float
    verdict: str
    score: float
    anomaly_ratio: Optional[float]
    defects: List[Defect] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_type": self.product_type,
            "source": self.source,
            "bbox": [int(v) for v in self.bbox],
            "confidence": round(float(self.confidence), 3),
            "verdict": self.verdict,
            "score": round(float(self.score), 3),
            "anomaly_ratio": None if self.anomaly_ratio is None else round(float(self.anomaly_ratio), 3),
            "defects": [d.to_dict() for d in self.defects],
            "reasons": self.reasons,
        }


@dataclass
class InspectionResult:
    verdict: str
    score: float
    product_type: str
    anomaly_ratio: Optional[float]
    items: List[ItemResult]
    detections: List[Detection]
    quality: Dict[str, Any]
    reasons: List[str]
    latency_ms: float
    backends: Dict[str, str]
    annotated: Optional[bytes] = None  # JPEG bytes, not serialised by to_dict
    advice: Dict[str, Any] = field(default_factory=dict)
    people: int = 0                    # people found in the frame (never defect-checked)

    DISCLAIMER = (
        "Visual inspection only. A photo cannot reveal internal faults, battery health, "
        "software problems or whether a product is genuine."
    )

    @property
    def defects(self) -> List[Defect]:
        return [d for it in self.items for d in it.defects]

    @property
    def detected_as(self) -> Dict[str, Any]:
        """What the main item was taken to be, how sure, and who decided (yolo, classifier, selected, face, none)."""
        it = max(self.items, key=lambda i: i.confidence)
        return {"product_type": it.product_type, "confidence": round(float(it.confidence), 3), "source": it.source or "none"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": round(float(self.score), 3),
            "product_type": self.product_type,
            "anomaly_ratio": None if self.anomaly_ratio is None else round(float(self.anomaly_ratio), 3),
            "reasons": self.reasons,
            "items": [it.to_dict() for it in self.items],
            "defects": [d.to_dict() for d in self.defects],
            "detections": [d.to_dict() for d in self.detections],
            "quality": self.quality,
            "latency_ms": round(float(self.latency_ms), 1),
            "backends": self.backends,
            "people": self.people,
            "detected_as": self.detected_as,
            "advice": self.advice,
            "disclaimer": self.DISCLAIMER,
        }
