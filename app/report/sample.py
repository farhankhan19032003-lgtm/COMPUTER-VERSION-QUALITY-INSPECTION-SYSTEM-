"""A clearly SYNTHETIC example report: invented patient, invented numbers, drawn images.

Used by GET /api/v1/report/sample and scripts/make_sample_report.py so you can see the layout
before any real model exists. Nothing here comes from a real person or a trained model.
"""
from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from ..utils import encode_jpeg


def _fake_heatmap(seed: int, kind: str) -> bytes:
    rng = np.random.default_rng(seed)
    size = 320
    img = np.zeros((size, size, 3), np.uint8)
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(xx - size / 2, yy - size / 2)
    base = {"retina": (40, 70, 150), "skin": (120, 150, 200), "eye": (190, 200, 215)}[kind]
    for c in range(3):
        img[..., c] = np.clip(base[c] * (1.0 - 0.45 * r / (size / 2)) + rng.normal(0, 4, (size, size)), 0, 255)
    img[r > size / 2 - 4] = 0
    cx, cy = int(size * rng.uniform(0.35, 0.65)), int(size * rng.uniform(0.35, 0.65))
    heat = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 38.0 ** 2)))
    hm = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return encode_jpeg(cv2.addWeighted(img, 0.6, hm, 0.4, 0))


def _prov(title: str, arch: str) -> Dict[str, Any]:
    return {"title": title, "arch": arch, "training_data": "SYNTHETIC SAMPLE: no real training data",
            "validation": {"n_images": 0, "accuracy": None, "macro_auc": None,
                           "note": "Invented example. No model was trained or validated."},
            "calibrated": False, "temperature": 1.0,
            "thresholds": {"moderate": 0.30, "high": 0.60, "source": "example defaults"}, "ood_guard": False}


def sample_data() -> Dict[str, Any]:
    ok_q = {"ok": True, "warnings": []}
    return {
        "synthetic": True,
        "operator": "Example Clinic (sample)",
        "patient": {"name": "SAMPLE PATIENT (synthetic)", "age_years": 54, "sex": "female", "external_ref": "SAMPLE-0001",
                    "skin_type": "IV", "symptoms": ["blurred vision in the evening", "tiredness"], "consent_given": True},
        "vitals": {"bpm": 76, "confidence": 0.81, "status": "ok"},
        "findings": [
            {"module": "retina_dr", "triage": "HIGH", "top_label": "moderate", "concern_probability": 0.78,
             "probabilities": {"moderate": 0.62, "proliferative": 0.10, "severe": 0.06, "no_dr": 0.14, "mild": 0.08},
             "quality": ok_q, "heatmap_jpeg": _fake_heatmap(1, "retina")},
            {"module": "anemia_conjunctiva", "triage": "MODERATE", "top_label": "anemic", "concern_probability": 0.41,
             "probabilities": {"anemic": 0.41, "not_anemic": 0.59}, "quality": ok_q, "heatmap_jpeg": _fake_heatmap(2, "eye")},
            {"module": "skin_lesion", "triage": "LOW", "top_label": "benign", "concern_probability": 0.06,
             "probabilities": {"benign": 0.94, "suspicious": 0.06}, "quality": ok_q, "heatmap_jpeg": _fake_heatmap(3, "skin")},
            {"module": "jaundice", "triage": "INCONCLUSIVE", "top_label": "normal", "concern_probability": 0.0, "probabilities": {},
             "quality": {"ok": False, "warnings": ["Image is too dark."]}, "heatmap_jpeg": None},
        ],
        "provenance": {"retina_dr": _prov("Diabetic retinopathy screen", "efficientnet_b0"),
                       "anemia_conjunctiva": _prov("Anaemia screen", "resnet18"),
                       "skin_lesion": _prov("Skin lesion triage", "efficientnet_b0")},
    }
