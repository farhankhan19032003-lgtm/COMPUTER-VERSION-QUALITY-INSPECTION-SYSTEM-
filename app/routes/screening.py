"""Health screening endpoints (decision support only, never diagnosis)."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from ..auth import User, current_user
from ..deps import AppState, data_uri, get_state, read_upload
from ..screening.catalog import DISCLAIMER
from .patients import require_patient_storage
from .. import warmup
from ..utils import ServerBusy, decode_image

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/screen", tags=["screening"])


@router.get("/modules")
async def modules(ctx: AppState = Depends(get_state), user: User = Depends(current_user)):
    return {"disclaimer": DISCLAIMER, "modules": ctx.screening.describe()}


@router.post("/{module}")
async def screen(
    module: str,
    file: UploadFile = File(...),
    explain: bool = Form(False, description="Return a Grad-CAM heatmap showing what the model looked at"),
    save: bool = Form(True),
    patient_id: Optional[str] = Form(None, description="Attach this screening to a stored patient (needs consent on file)"),
    ctx: AppState = Depends(get_state),
    user: User = Depends(current_user),
):
    if patient_id:
        require_patient_storage(ctx, user)
        patient = ctx.store.get_patient(user.id, patient_id)
        if patient is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Patient not found")
        if not patient.get("consent_given"):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "No consent on file for this patient")
    warmup.require_ready()
    model = ctx.screening.get(module)
    if model is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Screening module '{module}' is not installed. Train one with scripts/train_classifier.py "
            f"and place <name>.pt and <name>.json in {ctx.settings.screening_model_dir}. "
            "See GET /api/v1/screen/modules for suggested datasets.",
        )
    data = await read_upload(file, ctx.settings.max_upload_mb)
    try:
        img = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))

    try:
        res = await run_in_threadpool(model.predict, img, explain)
    except ServerBusy:
        raise warmup.busy()
    except Exception:
        ctx.metrics.record_error()
        log.exception("Screening failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Screening failed")

    ctx.metrics.record_screening()
    body = res.to_dict()
    if save:
        try:
            path = ctx.store.upload_image(user.id, data)
            row = ctx.store.insert_screening(
                {
                    "user_id": user.id,
                    "module": module,
                    "top_label": res.top_label,
                    "probabilities": body["probabilities"],
                    "triage": res.triage,
                    "quality": res.quality,
                    "image_path": path,
                    "latency_ms": res.latency_ms,
                    **({"patient_id": patient_id} if patient_id else {}),
                }
            )
            body["id"] = row.get("id")
            if patient_id:
                ctx.store.audit(user.id, "screening.create", patient_id, {"module": module, "triage": res.triage})
        except Exception:
            log.exception("Could not persist screening")
            ctx.store.log_event("ERROR", "Failed to persist screening", {"user": user.id})
    body["heatmap_image"] = data_uri(res.heatmap_jpeg)
    return body


@router.get("")
async def list_screenings(
    limit: int = Query(50, ge=1, le=500),
    ctx: AppState = Depends(get_state),
    user: User = Depends(current_user),
):
    rows = ctx.store.list_screenings(user.id, limit)
    return {"count": len(rows), "items": rows}
