"""Patient screening report (PDF). Stateless: results in, PDF out; nothing about the patient is stored here."""
from __future__ import annotations

import base64
import binascii
import logging
import re
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..auth import User, current_user
from ..deps import AppState, get_state
from ..report.builder import build_report_pdf
from ..report.sample import sample_data

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/report", tags=["report"])
_MODULE_RE = re.compile(r"^[a-z0-9_\-]{1,40}$")
_MAX_IMG_BYTES = 2 * 1024 * 1024


class FindingIn(BaseModel):
    module: str = Field(..., pattern=_MODULE_RE.pattern)
    triage: Literal["LOW", "MODERATE", "HIGH", "INCONCLUSIVE"]
    top_label: str = Field("", max_length=60)
    concern_probability: Optional[float] = Field(None, ge=0, le=1)
    probabilities: Dict[str, float] = Field(default_factory=dict, max_length=20)
    quality: Dict[str, object] = Field(default_factory=dict)
    message: Optional[str] = Field(None, max_length=500)
    heatmap_image: Optional[str] = Field(None, description="data:image/jpeg;base64,... as returned by the screening API")


class PatientInline(BaseModel):
    name: Optional[str] = Field(None, max_length=120)
    age_years: Optional[int] = Field(None, ge=0, le=130)
    sex: Literal["female", "male", "other", "unknown"] = "unknown"
    external_ref: Optional[str] = Field(None, max_length=60)
    skin_type: Literal["I", "II", "III", "IV", "V", "VI", "unknown"] = "unknown"
    symptoms: List[str] = Field(default_factory=list, max_length=15)
    consent_given: bool = False


class VitalsIn(BaseModel):
    bpm: Optional[float] = Field(None, ge=20, le=250)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    status: str = Field("unknown", max_length=40)


class ReportIn(BaseModel):
    patient_id: Optional[str] = Field(None, description="use a stored patient (needs sign-in) instead of 'patient'")
    patient: Optional[PatientInline] = None
    findings: List[FindingIn] = Field(..., min_length=1, max_length=12)
    vitals: Optional[VitalsIn] = None
    operator: Optional[str] = Field(None, max_length=120)
    symptoms: List[str] = Field(default_factory=list, max_length=15)


def _decode_heatmap(uri: Optional[str]) -> Optional[bytes]:
    if not uri:
        return None
    m = re.match(r"^data:image/(jpeg|png);base64,(.+)$", uri, re.S)
    if not m or len(m.group(2)) > _MAX_IMG_BYTES * 2:
        return None
    try:
        raw = base64.b64decode(m.group(2), validate=False)
    except (binascii.Error, ValueError):
        return None
    return raw if len(raw) <= _MAX_IMG_BYTES else None


def _pdf_response(pdf: bytes, name: str) -> Response:
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="{name}"', "Cache-Control": "no-store"})


@router.post("")
async def make_report(body: ReportIn, ctx: AppState = Depends(get_state), user: User = Depends(current_user)):
    patient: Dict = {}
    if body.patient_id:
        if user.is_anonymous and not ctx.settings.allow_anonymous_patients:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Stored patients need a signed-in user")
        row = ctx.store.get_patient(user.id, body.patient_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Patient not found")
        patient = {"name": row.get("full_name"), "age_years": row.get("age_years"), "sex": row.get("sex"),
                   "external_ref": row.get("external_ref"), "skin_type": row.get("skin_type"),
                   "consent_given": row.get("consent_given"), "symptoms": body.symptoms}
    elif body.patient:
        patient = {**body.patient.model_dump(), "symptoms": body.patient.symptoms or body.symptoms}

    installed = set(ctx.screening.installed())
    provenance = {}
    for f in body.findings:
        if f.module in installed:
            m = ctx.screening.get(f.module)
            if m is not None:
                provenance[f.module] = m.provenance()
    data = {
        "patient": patient,
        "operator": body.operator,
        "vitals": body.vitals.model_dump() if body.vitals else None,
        "provenance": provenance,                      # taken from the server's own models, never from the client
        "findings": [{**f.model_dump(exclude={"heatmap_image"}), "heatmap_jpeg": _decode_heatmap(f.heatmap_image)}
                     for f in body.findings],
    }
    try:
        pdf = await run_in_threadpool(build_report_pdf, data)
    except Exception:
        log.exception("Report generation failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not build the report")
    ctx.store.audit(user.id, "report.create", body.patient_id, {"modules": [f.module for f in body.findings]})
    return _pdf_response(pdf, "screening-report.pdf")


@router.get("/sample")
async def sample_report(user: User = Depends(current_user)):
    """A clearly labelled SYNTHETIC example, to preview the layout."""
    pdf = await run_in_threadpool(build_report_pdf, sample_data())
    return _pdf_response(pdf, "sample-report-synthetic.pdf")
