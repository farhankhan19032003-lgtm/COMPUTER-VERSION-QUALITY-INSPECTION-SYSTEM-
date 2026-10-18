"""Patient records: created with consent, owned by the signed-in operator, every access audited.

These rows are personal health data. They are only stored for signed-in users (or when
ALLOW_ANONYMOUS_PATIENTS=true for a local demo). Deleting a patient also deletes their screenings.
Audit entries hold ids and counts only, never names.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

from ..auth import User, current_user
from ..deps import AppState, get_state

router = APIRouter(prefix="/api/v1/patients", tags=["patients"])
CONSENT_VERSION = "v1"


class PatientIn(BaseModel):
    full_name: Optional[str] = Field(None, max_length=120)
    external_ref: Optional[str] = Field(None, max_length=60)
    age_years: Optional[int] = Field(None, ge=0, le=130)
    sex: Literal["female", "male", "other", "unknown"] = "unknown"
    skin_type: Literal["I", "II", "III", "IV", "V", "VI", "unknown"] = "unknown"
    notes: Optional[str] = Field(None, max_length=1000)
    consent_given: bool = Field(..., description="The patient (or guardian) agreed to this screening and to storing these records")

    @field_validator("full_name", "external_ref", "notes")
    @classmethod
    def _strip(cls, v):
        return v.strip() or None if isinstance(v, str) else v


def require_patient_storage(ctx: AppState, user: User) -> None:
    if user.is_anonymous and not ctx.settings.allow_anonymous_patients:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Saving patient records needs a signed-in user. You can still screen and download a report without saving.",
        )


@router.post("", status_code=201)
async def create_patient(body: PatientIn, ctx: AppState = Depends(get_state), user: User = Depends(current_user)):
    require_patient_storage(ctx, user)
    if not body.consent_given:
        raise HTTPException(422, "Consent is required before a patient record can be created")
    row = ctx.store.insert_patient({**body.model_dump(), "user_id": user.id, "consent_version": CONSENT_VERSION})
    ctx.store.audit(user.id, "patient.create", row.get("id"))
    return row


@router.get("")
async def list_patients(limit: int = Query(100, ge=1, le=500), ctx: AppState = Depends(get_state),
                        user: User = Depends(current_user)):
    require_patient_storage(ctx, user)
    rows = ctx.store.list_patients(user.id, limit)
    ctx.store.audit(user.id, "patient.list", None, {"count": len(rows)})
    return {"count": len(rows), "items": rows}


def _owned(ctx: AppState, user: User, patient_id: str) -> dict:
    row = ctx.store.get_patient(user.id, patient_id)
    if row is None:                                        # same answer for "missing" and "someone else's"
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Patient not found")
    return row


@router.get("/{patient_id}")
async def get_patient(patient_id: str, ctx: AppState = Depends(get_state), user: User = Depends(current_user)):
    require_patient_storage(ctx, user)
    row = _owned(ctx, user, patient_id)
    ctx.store.audit(user.id, "patient.read", patient_id)
    return row


@router.get("/{patient_id}/screenings")
async def patient_screenings(patient_id: str, ctx: AppState = Depends(get_state), user: User = Depends(current_user)):
    require_patient_storage(ctx, user)
    _owned(ctx, user, patient_id)
    rows = ctx.store.list_patient_screenings(user.id, patient_id)
    ctx.store.audit(user.id, "patient.screenings.read", patient_id, {"count": len(rows)})
    return {"count": len(rows), "items": rows}


@router.delete("/{patient_id}")
async def delete_patient(patient_id: str, ctx: AppState = Depends(get_state), user: User = Depends(current_user)):
    """Right to erasure: removes the patient and all their screening rows. Stored images need your retention job."""
    require_patient_storage(ctx, user)
    _owned(ctx, user, patient_id)
    n = len(ctx.store.list_patient_screenings(user.id, patient_id, 10000))
    ctx.store.delete_patient(user.id, patient_id)
    ctx.store.audit(user.id, "patient.delete", patient_id, {"screenings_deleted": n})
    return {"deleted": True, "screenings_deleted": n,
            "note": "Uploaded images in the storage bucket are not removed by this call; delete them with your retention job."}


@router.get("/audit/log", include_in_schema=True)
async def audit_log(limit: int = Query(100, ge=1, le=500), ctx: AppState = Depends(get_state),
                    user: User = Depends(current_user)):
    require_patient_storage(ctx, user)
    return {"items": ctx.store.list_audit(user.id, limit)}
