"""Temporal smoothing for video streams so a single noisy frame doesn't flip the verdict."""
from __future__ import annotations

from collections import deque
from typing import Deque


class TemporalSmoother:
    """Sliding-window vote over the latest verdicts.

    FAIL needs `fail_votes` FAIL frames in the window, REVIEW needs `review_votes`
    non-PASS frames. Until the window fills, the latest raw verdict is reported.
    """

    def __init__(self, window: int = 5, fail_votes: int = 2, review_votes: int = 3) -> None:
        self.window = window
        self.fail_votes = fail_votes
        self.review_votes = review_votes
        self._buf: Deque[str] = deque(maxlen=window)

    def update(self, verdict: str) -> str:
        self._buf.append(verdict)
        if len(self._buf) < 3:
            return verdict
        fails = sum(v == "FAIL" for v in self._buf)
        non_pass = sum(v != "PASS" for v in self._buf)
        if fails >= self.fail_votes:
            return "FAIL"
        if non_pass >= self.review_votes:
            return "REVIEW"
        return "PASS"

    def reset(self) -> None:
        self._buf.clear()
