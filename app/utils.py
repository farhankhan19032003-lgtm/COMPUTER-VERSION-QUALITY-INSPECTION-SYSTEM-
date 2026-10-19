"""Small shared helpers."""
from __future__ import annotations

import io
import os
import threading
from typing import Optional, Tuple

import cv2
import numpy as np


class InferenceGate:
    """Lets only a few photos be analysed at once.

    The routes already run the heavy work in a thread pool (up to 40 threads). Without a limit a live camera or a burst
    of uploads runs dozens of PyTorch inferences at the same time: memory grows, a small CPU is shared out so thinly that
    even /health answers late, and the host (Render) restarts the service, which the browser sees as a 502."""

    def __init__(self, n: int = 2) -> None:
        self.n = max(1, n)
        self._sem = threading.BoundedSemaphore(self.n)

    def resize(self, n: int) -> None:
        """Only call at start-up, before any request is running."""
        self.n = max(1, n)
        self._sem = threading.BoundedSemaphore(self.n)

    def __enter__(self):
        self._sem.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self._sem.release()


INFERENCE_GATE = InferenceGate()


def configure_runtime(torch_threads: int = 0, max_concurrent: int = 2) -> dict:
    """Cap the CPU threads PyTorch / OpenCV use and how many inferences run together. 0 threads = 2 (or fewer cores).

    On a shared host os.cpu_count() reports the whole machine, not the slice you pay for, so PyTorch would start far more
    threads than the CPU quota can run."""
    import cv2
    import torch

    n = torch_threads if torch_threads > 0 else min(2, os.cpu_count() or 1)
    torch.set_num_threads(n)
    cv2.setNumThreads(n)
    INFERENCE_GATE.resize(max_concurrent)
    return {"torch_threads": n, "max_concurrent_inference": INFERENCE_GATE.n}


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
