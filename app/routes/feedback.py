"""Reviewer feedback: a human confirms or overrides the model's verdict on a stored inspection.

Each reviewer has one verdict per inspection (sending again replaces it). The rows are labelled
examples: they show how often the model is right, and feed retraining.
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from ..auth import User, current_user
from ..deps import AppState, get_state

router = APIRouter(prefix="/api/v1", tags=["feedback"])
Verdict = Literal["PASS", "REVIEW", "FAIL"]


class FeedbackIn(BaseModel):
    verdict: Verdict = Field(..., description="The reviewer's own verdict for this unit")
    note: Optional[str] = Field(None, max_length=500)

    @field_validator("note")
    @classmethod
    def _strip(cls, v):
        return v.strip() or None if isinstance(v, str) else v


def summarize(rows: List[Dict]) -> Dict:
    """Agreement between model and reviewers. A false alarm = model flagged, reviewer says fine;
    a miss = model passed, reviewer says it is not fine."""
    n = len(rows)
    agree = sum(r["model_verdict"] == r["reviewer_verdict"] for r in rows)
    false_alarms = sum(r["model_verdict"] != "PASS" and r["reviewer_verdict"] == "PASS" for r in rows)
    misses = sum(r["model_verdict"] == "PASS" and r["reviewer_verdict"] != "PASS" for r in rows)
    return {
        "reviewed": n,
        "agree": agree,
        "overridden": n - agree,
        "agreement_rate": round(agree / n, 3) if n else None,
        "false_alarms": false_alarms,
        "misses": misses,
    }


@router.post("/inspections/{inspection_id}/feedback")
async def submit_feedback(inspection_id: str, body: FeedbackIn, ctx: AppState = Depends(get_state),
                          user: User = Depends(current_user)):
    try:
        uuid.UUID(inspection_id)
    except ValueError:
        raise HTTPException(404, "Inspection not found")
    insp = ctx.store.get_inspection(user.id, inspection_id)
    if insp is None:
        raise HTTPException(404, "Inspection not found")
    return ctx.store.upsert_feedback(user.id, inspection_id, insp["verdict"], body.verdict, body.note)


@router.get("/feedback")
async def list_feedback(limit: int = Query(200, ge=1, le=1000), ctx: AppState = Depends(get_state),
                        user: User = Depends(current_user)):
    rows = ctx.store.list_feedback(user.id, 1000)
    return {"summary": summarize(rows), "count": min(len(rows), limit), "items": rows[:limit]}
