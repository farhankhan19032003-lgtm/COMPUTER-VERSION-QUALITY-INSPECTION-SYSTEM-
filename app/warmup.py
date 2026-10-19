"""Load every model once at start-up, so the first real photo is not the one that pays for it (and any failure shows up
in /ready instead of as a 502 on someone's phone).

On a small host loading YOLO, the product classifier (ResNet50) and the anomaly backbone on the first request can take
30 to 60 seconds; a phone on a mobile network gives up or the proxy answers 502. Running one dummy inspection in a
background thread right after start-up moves that cost to the deploy.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict

import numpy as np

log = logging.getLogger("visioninspect.warmup")


def rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / 1048576)
    except Exception:
        return -1.0


def dummy_photo() -> np.ndarray:
    """A grey photo with a dark rounded rectangle: enough to run every stage once."""
    img = np.full((720, 540, 3), 200, np.uint8)
    img[120:600, 150:400] = (40, 40, 40)
    img[160:560, 170:380] = (90, 90, 90)
    return img


def run(pipeline, state: Dict[str, Any]) -> None:
    """Runs in a thread. Fills `state` (shown by /ready); never raises."""
    t0 = time.perf_counter()
    state.update(status="running", rss_mb=rss_mb())
    stages: Dict[str, float] = {}
    try:
        img = dummy_photo()
        t = time.perf_counter()
        pipeline.inspect(img, None, False)                       # YOLO + quality + OpenCV checks
        stages["detect_and_checks_s"] = round(time.perf_counter() - t, 1)
        t = time.perf_counter()
        pipeline.detector.classifier.predict(img)                # loads the ResNet50 product backbone
        stages["product_classifier_s"] = round(time.perf_counter() - t, 1)
        t = time.perf_counter()
        pipeline.inspect(img, "phone", False)                    # loads the anomaly backbone and the phone bank
        stages["anomaly_s"] = round(time.perf_counter() - t, 1)
        state.update(status="ok", stages=stages)
        log.info("Warm-up done in %.1fs %s, memory %s MB", time.perf_counter() - t0, stages, rss_mb())
    except Exception as exc:                                     # report it, keep the service up
        log.exception("Warm-up failed")
        state.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300], stages=stages)
    finally:
        state.update(seconds=round(time.perf_counter() - t0, 1), rss_mb=rss_mb())


def start(pipeline, state: Dict[str, Any]) -> threading.Thread:
    th = threading.Thread(target=run, args=(pipeline, state), name="warmup", daemon=True)
    th.start()
    return th
