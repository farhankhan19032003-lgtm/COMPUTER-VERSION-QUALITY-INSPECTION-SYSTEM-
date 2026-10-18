"""Contactless pulse estimate from a camera (remote photoplethysmography, rPPG).

Each heartbeat changes how much light the skin of the face absorbs, by a fraction of a
percent. We average the skin pixels' colour per frame, run the POS algorithm
(Wang et al., 2017) over the colour traces, and find the dominant frequency in the
42-180 bpm band.

What this is NOT: a medical device, and it does not "see inside the body". It estimates
pulse only, and only when lighting is steady, the person is still and the face is in view.
Published results are good in controlled settings and much worse in uncontrolled ones, and
skin-tone effects are under-studied, so a low-confidence result returns no number at all.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

FS = 30.0               # resampling rate (Hz)
BAND = (0.7, 3.0)       # 42 .. 180 bpm
MIN_SECONDS = 8.0
WINDOW_SECONDS = 15.0
MIN_CONFIDENCE = 0.40


def skin_mean(frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """Mean (r, g, b) of skin-coloured pixels inside a central oval, plus the skin fraction.

    The user is asked to fit their face in an on-screen oval, which is the region used.
    """
    h, w = frame_bgr.shape[:2]
    oval = np.zeros((h, w), np.uint8)
    cv2.ellipse(oval, (w // 2, h // 2), (int(w * 0.22), int(h * 0.36)), 0, 0, 360, 255, -1)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (40, 133, 77), (255, 173, 127))
    mask = cv2.bitwise_and(skin, oval)
    n_oval = int((oval > 0).sum())
    n_skin = int((mask > 0).sum())
    if n_skin < 50:
        return None
    b, g, r = (float(frame_bgr[..., c][mask > 0].mean()) for c in range(3))
    return r, g, b, n_skin / max(1, n_oval)


def pos(rgb: np.ndarray, fs: float, win_s: float = 1.6) -> np.ndarray:
    """POS pulse signal from an (N, 3) RGB trace."""
    n = rgb.shape[0]
    L = max(8, int(round(win_s * fs)))
    h = np.zeros(n)
    for t in range(0, n - L + 1):
        c = rgb[t:t + L].T                                   # 3 x L
        cn = c / (c.mean(axis=1, keepdims=True) + 1e-9)
        s1 = cn[1] - cn[2]
        s2 = -2 * cn[0] + cn[1] + cn[2]
        seg = s1 + (s1.std() / (s2.std() + 1e-9)) * s2
        h[t:t + L] += seg - seg.mean()
    return h


def dominant_frequency(sig: np.ndarray, fs: float) -> Tuple[float, float]:
    """Return (frequency Hz, confidence 0..1) of the strongest in-band peak.

    Confidence is the share of in-band power sitting at the peak and its first harmonic.
    """
    sig = sig - sig.mean()
    win = np.hanning(len(sig))
    nfft = 8192
    spec = np.abs(np.fft.rfft(sig * win, nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    if not band.any() or spec[band].sum() <= 0:
        return 0.0, 0.0
    f0 = float(freqs[band][np.argmax(spec[band])])
    near = band & (np.abs(freqs - f0) <= 0.12)
    harm = (np.abs(freqs - 2 * f0) <= 0.12) & (freqs <= 2 * BAND[1])
    conf = float((spec[near].sum() + spec[harm].sum()) / (spec[band].sum() + spec[harm & ~band].sum()))
    return f0, min(1.0, conf)


class RPPGEstimator:
    """Collects colour samples and produces a pulse estimate. One instance per session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buf: Deque[Tuple[float, float, float, float, float]] = deque()
        self._recent: Deque[float] = deque(maxlen=5)

    def add(self, t: float, r: float, g: float, b: float, skin_frac: float = 1.0) -> None:
        """t in seconds (any monotonic clock)."""
        with self._lock:
            if self._buf and t <= self._buf[-1][0]:
                return
            self._buf.append((t, r, g, b, skin_frac))
            while self._buf and t - self._buf[0][0] > WINDOW_SECONDS:
                self._buf.popleft()

    def add_frame(self, frame_bgr: np.ndarray, t: float) -> bool:
        m = skin_mean(frame_bgr)
        if m is None:
            self.add(t, 0.0, 0.0, 0.0, 0.0)
            return False
        self.add(t, *m)
        return True

    def reset(self) -> None:
        with self._lock:
            self._buf.clear()
            self._recent.clear()

    def estimate(self) -> Dict[str, Any]:
        with self._lock:
            data = np.array(self._buf, dtype=np.float64) if self._buf else np.empty((0, 5))
        warnings: List[str] = []
        base = {"status": "collecting", "bpm": None, "confidence": 0.0, "seconds": 0.0,
                "needed_seconds": MIN_SECONDS, "warnings": warnings}
        if len(data) < 10:
            return base
        t, rgb, skin = data[:, 0], data[:, 1:4], data[:, 4]
        dur = float(t[-1] - t[0])
        base["seconds"] = round(dur, 1)
        if dur < MIN_SECONDS:
            return base

        face_ok = skin > 0.12
        if face_ok.mean() < 0.8:
            warnings.append("Face not found in the oval most of the time: centre your face and add front light.")
            return {**base, "status": "no_face"}
        fs_est = (len(t) - 1) / dur
        if fs_est < 10:
            warnings.append(f"Camera delivers only {fs_est:.0f} frames/s; need at least 10.")
            return {**base, "status": "low_quality"}

        # Drop frames where no face was found; they are bridged by interpolation below.
        t, rgb, skin = t[face_ok], rgb[face_ok], skin[face_ok]
        grid = np.arange(t[0], t[-1], 1 / FS)
        sig_rgb = np.stack([np.interp(grid, t, rgb[:, i]) for i in range(3)], axis=1)
        if sig_rgb.min() <= 0:
            return {**base, "status": "low_quality", "warnings": ["Bad colour data."]}

        if np.diff(t).max() > 0.6:
            warnings.append("Frames were dropped for over half a second: the estimate may be off.")
        if skin.std() > 0.12:
            warnings.append("The face moved a lot: sit still and look at the camera.")
        brightness = sig_rgb.mean(axis=1)
        if brightness.std() / brightness.mean() > 0.08:
            warnings.append("Lighting is changing: use steady light and avoid screens flickering on your face.")

        h = pos(sig_rgb, FS)
        f0, conf = dominant_frequency(h, FS)
        bpm = f0 * 60
        out = {**base, "confidence": round(conf, 2), "seconds": round(dur, 1), "warnings": warnings}
        if conf < MIN_CONFIDENCE:
            out.update(status="uncertain", tentative_bpm=round(bpm))
            warnings.append("Signal too weak for a reliable number. Hold still with steady, bright, even light.")
            return out

        self._recent.append(bpm)
        smooth = float(np.median(self._recent))
        stable = len(self._recent) >= 3 and (max(self._recent) - min(self._recent)) < 8
        out.update(status="ok", bpm=round(smooth), stable=stable, interpretation=interpret(smooth), **classify(smooth))
        return out


def classify(bpm: float) -> Dict[str, str]:
    """Where a resting adult's pulse falls: normal, caution (low / elevated / high) or danger (very low / very high).

    Rule-of-thumb bands for adults at rest, not a diagnosis. Children, athletes, pregnancy and medicines change what
    is normal, and the camera estimate itself can be off by several bpm.
    """
    if bpm < 40:
        return {"zone": "danger_low", "zone_label": "Danger: very low", "level": "danger"}
    if bpm < 60:
        return {"zone": "low", "zone_label": "Low", "level": "caution"}
    if bpm <= 100:
        return {"zone": "normal", "zone_label": "Normal", "level": "normal"}
    if bpm <= 120:
        return {"zone": "elevated", "zone_label": "Elevated", "level": "caution"}
    if bpm <= 150:
        return {"zone": "high", "zone_label": "High", "level": "caution"}
    return {"zone": "danger_high", "zone_label": "Danger: very high", "level": "danger"}


def interpret(bpm: float) -> str:
    zone = classify(bpm)["zone"]
    return {
        "danger_low": "Very low for a resting adult. If you feel dizzy, faint or short of breath, get medical help now.",
        "low": "Below the usual resting range (60 to 100). Athletes often run lower.",
        "normal": "Within the usual resting range (60 to 100).",
        "elevated": "Above the usual resting range. Exercise, caffeine, stress or fever can raise it.",
        "high": "Well above the usual resting range. Rest and re-check; seek care if it stays high or you feel unwell.",
        "danger_high": "Very high for a resting adult. If you have chest pain, dizziness, fainting or breathlessness, get medical help now.",
    }[zone]
