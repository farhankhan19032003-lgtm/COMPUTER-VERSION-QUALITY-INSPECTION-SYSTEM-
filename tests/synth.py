"""Synthetic 'product photo' generator for tests and quick demos (no real data needed)."""
from __future__ import annotations

import cv2
import numpy as np


def make_phone(seed: int, crack: bool = False, scratch: bool = False, W: int = 640, H: int = 800) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (205, 200, 195), np.uint8)
    img = (img.astype(np.int16) + rng.normal(0, 4, img.shape)).clip(0, 255).astype(np.uint8)
    cv2.rectangle(img, (160, 100), (480, 700), (35, 35, 40), -1)      # body
    cv2.rectangle(img, (175, 130), (465, 670), (90, 70, 60), -1)      # screen
    for k in range(6):                                                # fake UI rows
        y = 160 + k * 80
        cv2.rectangle(img, (200, y), (440, y + 45), (150 + k * 10, 120, 100), -1)
        cv2.circle(img, (225, y + 22), 14, (230, 230, 240), -1)
    img = (img.astype(np.int16) + rng.normal(0, 3, img.shape)).clip(0, 255).astype(np.uint8)
    if crack:
        pts = np.array([[190, 180], [260, 300], [310, 330], [380, 470], [440, 560]], np.int32)
        cv2.polylines(img, [pts], False, (15, 15, 15), 2, cv2.LINE_AA)
    if scratch:
        cv2.line(img, (200, 620), (430, 600), (235, 235, 235), 1, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (3, 3), 0.6)


def jitter(img: np.ndarray, seed: int) -> np.ndarray:
    """Small brightness change and shift, like photographing the same product twice."""
    rng = np.random.default_rng(seed)
    out = np.clip(img.astype(np.float32) * rng.uniform(0.92, 1.08) + rng.uniform(-8, 8), 0, 255).astype(np.uint8)
    dx, dy = rng.integers(-4, 5, 2)
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(out, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def missing_part(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (200, 320), (440, 365), (90, 70, 60), -1)
    return out


def jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buf.tobytes()
