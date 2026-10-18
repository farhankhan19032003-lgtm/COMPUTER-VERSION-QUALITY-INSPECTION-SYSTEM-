"""Small shared helpers."""
from __future__ import annotations

import io
from typing import Optional, Tuple

import cv2
import numpy as np


def resolve_device(setting: str = "auto") -> str:
    """Return 'cuda' or 'cpu' (torch is imported lazily)."""
    import torch

    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if setting.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return setting


def decode_image(data: bytes, max_side: int = 1920) -> np.ndarray:
    """Decode bytes -> BGR uint8 image. Raises ValueError on garbage input."""
    if not data:
        raise ValueError("Empty file")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image (supported: JPEG, PNG, WebP, BMP)")
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def encode_jpeg(img: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def clip_box(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        int(max(0, min(w - 1, round(x1)))),
        int(max(0, min(h - 1, round(y1)))),
        int(max(0, min(w, round(x2)))),
        int(max(0, min(h, round(y2)))),
    )


def percentile(values, q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))
