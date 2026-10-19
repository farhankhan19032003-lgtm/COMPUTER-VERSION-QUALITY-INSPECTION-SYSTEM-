"""Product inspection endpoints."""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from ..auth import User, current_user
from ..deps import AppState, clean_product, data_uri, get_state, read_upload
from ..inspection.types import InspectionResult
from .. import warmup
from ..utils import ServerBusy, decode_image
from fastapi import Request

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["inspection"])

MAX_BATCH = 20


def persist_inspection(
    ctx: AppState, user: User, result: InspectionResult, original: Optional[bytes], source: str
) -> dict:
    """Store images + a row in Supabase (or memory). Never raises into the request path.

    Photos of people are not product inspections: they are neither stored nor counted.
    """
    if result.product_type == "person":
        return {"id": None, "annotated_url": None}
    try:
        orig_path = ctx.store.upload_image(user.id, original) if original else None
        ann_path = ctx.store.upload_image(user.id, result.annotated) if result.annotated else None
        row = ctx.store.insert_inspection(
            {
                "user_id": user.id,
                "source": source,
                "product_type": result.product_type,
                "verdict": result.verdict,
                "score": result.score,
                "anomaly_ratio": result.anomaly_ratio,
                "defects": [d.to_dict() for d in result.defects],
                "detections": [d.to_dict() for d in result.detections],
                "quality": result.quality,
                "image_path": orig_path,
                "annotated_path": ann_path,
                "latency_ms": result.latency_ms,
            }
        )
        return {"id": row.get("id"), "annotated_url": ctx.store.signed_url(ann_path)}
    except Exception:
        log.exception("Could not persist inspection")
        ctx.store.log_event("ERROR", "Failed to persist inspection", {"user": user.id})
        return {"id": None, "annotated_url": None}


def record(ctx: AppState, result: InspectionResult, source: str) -> None:
    if result.product_type == "person":
        return
    ctx.metrics.record_inspection(
        result.verdict, result.latency_ms, result.quality["ok"], result.product_type,
        [d.type for d in result.defects], source,
    )


@router.post("/inspect")
async def inspect_image(
    file: UploadFile = File(..., description="JPEG/PNG photo of the product"),
    product_type: Optional[str] = Form(None, description="Optional hint: phone, laptop, watch..."),
    save: bool = Form(True),
    annotate: bool = Form(True),
    ctx: AppState = Depends(get_state),
    user: User = Depends(current_user),
):
    warmup.require_ready()
    hint = clean_product(product_type)
    data = await read_upload(file, ctx.settings.max_upload_mb)
    try:
        img = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))

    try:
        result = await run_in_threadpool(ctx.pipeline.inspect, img, hint, annotate)
    except ServerBusy:
        raise warmup.busy()
    except Exception:
        ctx.metrics.record_error()
        log.exception("Inspection failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Inspection failed")

    record(ctx, result, "upload")
    saved = persist_inspection(ctx, user, result, data, "upload") if save else {}
    body = result.to_dict()
    body.update(saved)
    body["annotated_image"] = data_uri(result.annotated)
    return body


@router.post("/inspect/batch")
async def inspect_batch(
    files: List[UploadFile] = File(...),
    product_type: Optional[str] = Form(None),
    save: bool = Form(True),
    ctx: AppState = Depends(get_state),
    user: User = Depends(current_user),
):
    """Inspect up to 20 images in one call (no annotated images returned, to keep it light)."""
    if len(files) > MAX_BATCH:
        raise HTTPException(422, f"At most {MAX_BATCH} files per batch")
    warmup.require_ready()
    hint = clean_product(product_type)
    out = []
    for f in files:
        entry = {"filename": f.filename}
        try:
            data = await read_upload(f, ctx.settings.max_upload_mb)
            img = decode_image(data)
            result = await run_in_threadpool(ctx.pipeline.inspect, img, hint, save)
            record(ctx, result, "batch")
            if save:
                entry.update(persist_inspection(ctx, user, result, data, "batch"))
            entry.update(result.to_dict())
        except HTTPException as exc:
            entry["error"] = exc.detail
        except ValueError as exc:
            entry["error"] = str(exc)
        except ServerBusy:
            entry["error"] = warmup.BUSY_MSG
        except Exception:
            ctx.metrics.record_error()
            log.exception("Batch item failed")
            entry["error"] = "Inspection failed"
        out.append(entry)
    verdicts = [e["verdict"] for e in out if "verdict" in e]
    return {
        "count": len(out),
        "summary": {v: verdicts.count(v) for v in ("PASS", "REVIEW", "FAIL")},
        "results": out,
    }


@router.get("/inspections")
async def list_inspections(
    limit: int = Query(50, ge=1, le=500),
    ctx: AppState = Depends(get_state),
    user: User = Depends(current_user),
):
    rows = ctx.store.list_inspections(user.id, limit)
    try:
        reviewed = {f["inspection_id"]: f for f in ctx.store.list_feedback(user.id)}
    except Exception:  # history must still load if the feedback table is missing
        log.exception("Could not load reviewer feedback")
        reviewed = {}
    for r in rows:
        r["annotated_url"] = ctx.store.signed_url(r.get("annotated_path"))
        f = reviewed.get(r.get("id"))
        r["feedback"] = {"verdict": f["reviewer_verdict"], "note": f.get("note")} if f else None
    return {"count": len(rows), "items": rows}
