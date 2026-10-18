"""Phone health check (internal): battery, storage, memory and symptoms."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from ..auth import User, current_user
from ..deps import AppState, get_state
from ..diagnostics.phone import PhoneDiagnostics, evaluate

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/diagnostics", tags=["diagnostics"])


@router.post("/phone")
async def phone_health(
    body: PhoneDiagnostics,
    save: bool = Query(True),
    ctx: AppState = Depends(get_state),
    user: User = Depends(current_user),
):
    result = evaluate(body)
    if save:
        try:
            row = ctx.store.insert_diagnostic(
                {
                    "user_id": user.id,
                    "platform": body.platform,
                    "model": body.model,
                    "health_score": result["health_score"],
                    "verdict": result["verdict"],
                    "findings": result["findings"],
                    "input": body.model_dump(mode="json", exclude_none=True),
                }
            )
            result["id"] = row.get("id")
        except Exception:
            log.exception("Could not persist diagnostics")
            ctx.store.log_event("ERROR", "Failed to persist phone diagnostics", {"user": user.id})
    return result


@router.get("/phone")
async def phone_history(
    limit: int = Query(20, ge=1, le=200),
    ctx: AppState = Depends(get_state),
    user: User = Depends(current_user),
):
    rows = ctx.store.list_diagnostics(user.id, limit)
    return {"count": len(rows), "items": rows}
