"""Shared FastAPI dependencies and request helpers."""
from __future__ import annotations

import base64
import re
from typing import Optional

from fastapi import HTTPException, Request, UploadFile, status

from .config import Settings
from .db import Store
from .inspection.pipeline import InspectionPipeline
from .metrics import Metrics
from .screening.classifier import ScreeningRegistry

_PRODUCT_RE = re.compile(r"^[a-z0-9_\-]{1,32}$")


class AppState:
    """Everything long-lived, built once at startup and stored on app.state."""

    def __init__(
        self,
        settings: Settings,
        store: Store,
        pipeline: InspectionPipeline,
        screening: ScreeningRegistry,
        metrics: Metrics,
    ) -> None:
        self.settings = settings
        self.store = store
        self.pipeline = pipeline
        self.screening = screening
        self.metrics = metrics


def get_state(request: Request) -> AppState:
    return request.app.state.ctx


def clean_product(value: Optional[str]) -> Optional[str]:
    if value is None or not value.strip():
        return None
    v = value.strip().lower()
    if not _PRODUCT_RE.match(v):
        raise HTTPException(422, "product_type must be 1-32 chars: a-z, 0-9, _ or -")
    return v


async def read_upload(file: UploadFile, max_mb: int) -> bytes:
    data = await file.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File larger than {max_mb} MB")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
    return data


def data_uri(jpeg: Optional[bytes]) -> Optional[str]:
    if not jpeg:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
