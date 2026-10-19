"""Small shared helpers."""
from __future__ import annotations

import io
import os
import threading
from typing import Optional, Tuple

import cv2
import numpy as np


class ServerBusy(RuntimeError):
    """No analysis slot became free in time. Routes turn this into a clear 503 instead of a proxy timeout (502)."""


class InferenceGate:
    """Lets only a few photos be analysed at once, and makes the rest wait for a limited time.

    The routes run the heavy work in a thread pool (up to 40 threads). Without a limit a live camera or a burst of
    uploads runs dozens of PyTorch inferences together: memory grows, a small CPU is shared out so thinly that even
    /health answers late, and the host (Render) restarts the service, which the browser sees as a 502. Waiting at most
    `timeout` seconds and then answering "busy" also keeps every request well under the proxy's own time limit."""

    def __init__(self, n: int = 1, timeout: Optional[float] = None) -> None:
        self.n = max(1, n)
        self.timeout = timeout
        self._sem = threading.BoundedSemaphore(self.n)

    def resize(self, n: int, timeout: Optional[float] = None) -> None:
        """Only call at start-up, before any request is running."""
        self.n = max(1, n)
        self.timeout = timeout
        self._sem = threading.BoundedSemaphore(self.n)

    def __enter__(self):
        if not self._sem.acquire(timeout=self.timeout):
            raise ServerBusy(f"no analysis slot free after {self.timeout}s")
        return self

    def __exit__(self, *exc) -> None:
        self._sem.release()


INFERENCE_GATE = InferenceGate()


def cgroup_cpus() -> Optional[float]:
    """CPUs this container may really use (Docker/Render limit), or None when unlimited / unknown. os.cpu_count() only
    reports the whole host, so PyTorch would start far more threads than the quota can run and get throttled."""
    try:
        txt = open("/sys/fs/cgroup/cpu.max").read().split()          # cgroup v2: "<quota> <period>" or "max <period>"
        if txt[0] != "max":
            return int(txt[0]) / int(txt[1])
    except Exception:
        pass
    try:                                                              # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        per = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0 and per > 0:
            return q / per
    except Exception:
        pass
    return None


def configure_runtime(torch_threads: int = 0, max_concurrent: int = 0, wait_seconds: float = 45.0) -> dict:
    """Size the CPU use to the CPUs this container really has.

    torch_threads / max_concurrent of 0 mean automatic: one PyTorch thread and one analysis at a time when the container
    has one CPU or less (a second thread would only be throttled), otherwise up to 2 threads."""
    import cv2
    import torch

    quota = cgroup_cpus()
    cpus = quota if quota else float(os.cpu_count() or 1)
    n = torch_threads if torch_threads > 0 else max(1, min(2, int(cpus)))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass                                                          # already used in this process (tests)
    cv2.setNumThreads(n)
    slots = max_concurrent if max_concurrent > 0 else (1 if cpus <= 1.5 else 2)
    INFERENCE_GATE.resize(slots, wait_seconds)
    return {"torch_threads": n, "max_concurrent_inference": slots, "cpu_count": os.cpu_count(),
            "cpu_quota": round(quota, 2) if quota else None}


def resolve_device(setting: str = "auto") -> str:
    """Return 'cuda' or 'cpu' (torch is imported lazily)."""
    import torch

    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if setting.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return setting


def decode_image(data: bytes, max_side: int = 1280) -> np.ndarray:
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
