"""Classical OpenCV checks: image quality gate, foreground mask, scratch/crack/spot finder.

These heuristics need no training data and run in a few milliseconds. They catch obvious
linear damage (cracks, scratches) and isolated spots. Anything subtler (dents, misaligned
parts, screen burn-in, missing screws) is the job of the learned anomaly model or a
custom-trained detector.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .types import Defect

WORK_SIZE = 640  # analysis resolution (longest side); keeps thresholds scale-invariant


# ----------------------------------------------------------------------------- quality
def assess_quality(
    img: np.ndarray, min_sharpness: float = 25.0, min_side: int = 160, min_brightness: float = 40.0
) -> Dict[str, Any]:
    """Decide whether the photo is good enough to trust a verdict.

    Defaults suit product photos. Screening modules pass limits calibrated to their training
    data (X-rays are dark, cell crops are small), see scripts/train_classifier.py.
    """
    h, w = img.shape[:2]
    s = WORK_SIZE / max(h, w)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    over = float((gray >= 250).mean())
    under = float((gray <= 5).mean())

    warnings: List[str] = []
    ok = True
    if min(h, w) < min_side:
        warnings.append("Resolution is very low; move closer or use a higher resolution camera.")
        ok = False
    if sharpness < min_sharpness:
        warnings.append("Image looks blurry; hold the camera steady or refocus.")
        ok = False
    if brightness < min_brightness:
        warnings.append("Image is too dark; add light.")
        ok = False
    if over > 0.15:
        warnings.append("Strong glare / overexposure may hide defects; change the angle or lighting.")
    if under > 0.35:
        warnings.append("Large underexposed areas; add light.")

    return {
        "ok": ok,
        "sharpness": round(sharpness, 1),
        "brightness": round(brightness, 1),
        "overexposed_pct": round(over * 100, 1),
        "underexposed_pct": round(under * 100, 1),
        "width": int(w),
        "height": int(h),
        "warnings": warnings,
    }


# ----------------------------------------------------------------------------- mask
def foreground_mask(img: np.ndarray) -> np.ndarray:
    """Rough object mask (uint8 0/255) using colour distance from the border background."""
    h, w = img.shape[:2]
    s = 320.0 / max(h, w)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    sh, sw = small.shape[:2]

    lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(2, int(0.06 * min(sh, sw)))
    border = np.concatenate(
        [lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)]
    )
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if contours:
        big = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big) >= 0.04 * sh * sw:
            cv2.drawContours(out, [big], -1, 255, thickness=cv2.FILLED)
    if out.sum() == 0 or out.sum() > 0.985 * 255 * sh * sw:
        out[:] = 255  # no clear object: inspect everything
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ----------------------------------------------------------------------------- defects
def _robust_threshold(vals: np.ndarray, k: float, floor: float) -> float:
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1e-6
    return max(med + k * mad, floor)


def _components(binary: np.ndarray, min_area: int, limit: int = 60):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    order = sorted(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    for i in order[:limit]:
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            yield i, labels, stats[i]


def find_defects(
    roi: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_defects: int = 12,
) -> List[Defect]:
    """Find cracks, scratches and spots inside a product crop.

    Returned boxes are in the coordinate system of `roi`.
    """
    h0, w0 = roi.shape[:2]
    scale = min(1.0, WORK_SIZE / max(h0, w0))
    img = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else roi
    h, w = img.shape[:2]
    M = max(h, w)

    if mask is None:
        mask = foreground_mask(img)
    elif mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Ignore a band along the object outline: edges would otherwise look like scratches.
    k = max(5, int(0.04 * M)) | 1
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if inner.sum() < 0.05 * 255 * h * w:
        inner = mask
    valid = inner > 0
    if valid.sum() < 400:
        return []

    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    defects: List[Defect] = []

    # --- thin linear structures: dark (cracks) and bright (scratches) -------------------
    kb = max(7, int(round(0.015 * M))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))
    responses = {
        "dark": cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ker),
        "bright": cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ker),
    }
    for kind, resp in responses.items():
        vals = resp[valid].astype(np.float32)
        thr = _robust_threshold(vals, k=6.0, floor=22.0)
        binary = ((resp > thr) & valid).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        for i, labels, st in _components(binary, min_area=12):
            pts = cv2.findNonZero((labels == i).astype(np.uint8))
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            length, width = max(rw, rh), min(rw, rh) + 1.0
            if length < 0.07 * M or length / width < 4.0:
                continue
            contrast = float(resp[labels == i].mean())
            L = length / M
            is_crack = kind == "dark" and (L >= 0.25 or contrast >= 60)
            dtype = "crack" if is_crack else "scratch"
            if is_crack:
                severity = "high" if L >= 0.2 else "medium"
            else:
                severity = "medium" if L >= 0.2 else "low"
            score = min(1.0, 0.4 * min(L / 0.4, 1) + 0.4 * min(contrast / 80, 1) + (0.2 if is_crack else 0.0))
            x, y, bw, bh = st[cv2.CC_STAT_LEFT], st[cv2.CC_STAT_TOP], st[cv2.CC_STAT_WIDTH], st[cv2.CC_STAT_HEIGHT]
            defects.append(
                Defect(
                    type=dtype,
                    severity=severity,
                    score=score,
                    bbox=(int(x), int(y), int(x + bw), int(y + bh)),
                    source="cv",
                    detail=f"{kind} line, length {L * 100:.0f}% of object, contrast {contrast:.0f}",
                )
            )

    # --- isolated spots (stains, dead-pixel clusters, debris) --------------------------
    g1 = cv2.GaussianBlur(gray, (0, 0), 1.5).astype(np.float32)
    g2 = cv2.GaussianBlur(gray, (0, 0), 9.0).astype(np.float32)
    dog = np.abs(g1 - g2)
    vals = dog[valid]
    thr = _robust_threshold(vals, k=8.0, floor=30.0)
    binary = ((dog > thr) & valid).astype(np.uint8) * 255
    spots: List[Defect] = []
    for i, labels, st in _components(binary, min_area=max(6, int(0.0002 * h * w))):
        area = int(st[cv2.CC_STAT_AREA])
        if area > 0.01 * h * w:
            continue
        bw, bh = int(st[cv2.CC_STAT_WIDTH]), int(st[cv2.CC_STAT_HEIGHT])
        if max(bw, bh) / max(1, min(bw, bh)) >= 3 or area / float(bw * bh) < 0.4:
            continue
        contrast = float(dog[labels == i].mean())
        x, y = int(st[cv2.CC_STAT_LEFT]), int(st[cv2.CC_STAT_TOP])
        spots.append(
            Defect(
                type="spot",
                severity="low",
                score=min(0.35, 0.1 + contrast / 400),
                bbox=(x, y, x + bw, y + bh),
                source="cv",
                detail=f"isolated spot, contrast {contrast:.0f}",
            )
        )
    if len(spots) <= 8:  # more than that means texture / UI content, not damage
        defects.extend(spots)

    defects.sort(key=lambda d: d.score, reverse=True)
    defects = defects[:max_defects]

    inv = 1.0 / scale
    out: List[Defect] = []
    for d in defects:
        x1, y1, x2, y2 = d.bbox
        d.bbox = (
            int(max(0, x1 * inv)), int(max(0, y1 * inv)),
            int(min(w0, x2 * inv)), int(min(h0, y2 * inv)),
        )
        out.append(d)
    return out
