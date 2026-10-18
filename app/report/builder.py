"""Build the patient screening report as a PDF (reportlab).

build_report_pdf(data) -> bytes. `data` is plain dicts so it is easy to test:
  patient     {name, age_years, sex, external_ref, skin_type, symptoms[list], consent_given}
  findings    [{module, triage, top_label, concern_probability, probabilities, quality, message, heatmap_jpeg(bytes|None)}]
  provenance  {module: ScreeningModel.provenance()}   (model card: training data, validation, calibration)
  vitals      {bpm, confidence, status} | None
  operator    clinic or clinician name (optional)
  synthetic   True for the sample report: adds a loud "SYNTHETIC SAMPLE" mark everywhere

The report never says "diagnosis" except to say it is not one, and puts the referral and the
red-flag list next to every flagged area.
"""
from __future__ import annotations

import hashlib
import io
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

from ..screening.catalog import DISCLAIMER
from .summary import summarise

LEVEL_COLOR = {
    "HIGH": colors.HexColor("#B42318"), "MODERATE": colors.HexColor("#B54708"),
    "LOW": colors.HexColor("#067647"), "INCOMPLETE": colors.HexColor("#475467"),
    "INCONCLUSIVE": colors.HexColor("#475467"),
}
LEVEL_BG = {
    "HIGH": colors.HexColor("#FEF3F2"), "MODERATE": colors.HexColor("#FFFAEB"),
    "LOW": colors.HexColor("#ECFDF3"), "INCOMPLETE": colors.HexColor("#F2F4F7"), "INCONCLUSIVE": colors.HexColor("#F2F4F7"),
}
INK, MUTED, RULE = colors.HexColor("#101828"), colors.HexColor("#667085"), colors.HexColor("#D0D5DD")
PAGE_W, PAGE_H = A4
MARGIN = 16 * mm


def _t(value: Any, limit: int = 400) -> str:
    """Escape for reportlab markup and make safe for the built-in Helvetica font (Latin text only)."""
    s = "" if value is None else str(value)
    s = s.encode("cp1252", "replace").decode("cp1252")        # Devanagari/CJK names become '?', never broken boxes
    return escape(s[:limit])


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    mk = lambda name, **kw: ParagraphStyle(name, parent=base, **kw)  # noqa: E731
    return {
        "body": mk("body", fontName="Helvetica", fontSize=9.2, leading=12.6, textColor=INK),
        "small": mk("small", fontName="Helvetica", fontSize=7.8, leading=10.4, textColor=MUTED),
        "hi": mk("hi", fontName="Helvetica-Oblique", fontSize=8.8, leading=12, textColor=colors.HexColor("#344054")),
        "h1": mk("h1", fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=INK),
        "h2": mk("h2", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=INK, spaceBefore=9, spaceAfter=4),
        "label": mk("label", fontName="Helvetica-Bold", fontSize=7.4, leading=9.6, textColor=MUTED),
        "banner_t": mk("banner_t", fontName="Helvetica-Bold", fontSize=13, leading=16),
        "cell_b": mk("cell_b", fontName="Helvetica-Bold", fontSize=9, leading=11.5, textColor=INK),
        "center": mk("center", fontName="Helvetica", fontSize=8, alignment=TA_CENTER, textColor=MUTED),
    }


def _chip(text: str, level: str, st) -> Paragraph:
    col = LEVEL_COLOR.get(level, MUTED)
    return Paragraph(f'<font color="{col.hexval().replace("0x", "#")}"><b>{_t(text)}</b></font>', st["body"])


def _bullets(items: List[str], st, style="body") -> List[Paragraph]:
    return [Paragraph(f"&bull;&nbsp; {_t(i, 600)}", st[style]) for i in items]


def _rule() -> Table:
    t = Table([[""]], colWidths=[PAGE_W - 2 * MARGIN], rowHeights=[1])
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.6, RULE)]))
    return t


def _digest(data: Dict[str, Any]) -> str:
    """Short fingerprint of the inputs (images excluded) so a printed report can be matched to its data."""
    slim = {
        "patient": data.get("patient"), "vitals": data.get("vitals"),
        "findings": [{k: v for k, v in f.items() if k != "heatmap_jpeg"} for f in data.get("findings", [])],
    }
    return hashlib.sha256(json.dumps(slim, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _image_flowable(jpeg: Optional[bytes], width_mm: float) -> Optional[Image]:
    if not jpeg:
        return None
    try:
        from PIL import Image as PILImage

        w, h = PILImage.open(io.BytesIO(jpeg)).size
        img = Image(io.BytesIO(jpeg), width=width_mm * mm, height=width_mm * mm * h / max(1, w))
        return img
    except Exception:
        return None


def _model_card(module: str, prov: Optional[Dict[str, Any]], st) -> List[Any]:
    from ..screening.catalog import module_info

    info = module_info(module)
    lines = [f"<b>{_t(info['title'])}</b>  ({_t(info['evidence'].replace('_', ' '))}; needs: {_t(info.get('device', ''))})"]
    if not prov:
        lines.append("No model details were available for this module. Treat the result as unvalidated.")
        return [Paragraph(l, st["small"]) for l in lines]
    val = prov.get("validation") or {}
    lines.append(f"Model: {_t(prov.get('arch'))}. Training data: {_t(prov.get('training_data') or 'not recorded')}.")
    if val:
        lines.append(f"Internal validation: {val.get('n_images', '?')} images, accuracy {_pct(val.get('accuracy'))}, "
                     f"macro AUC {'n/a' if val.get('macro_auc') is None else round(val['macro_auc'], 3)}. "
                     f"{_t(val.get('note', ''), 300)}")
        lt = val.get("locked_test")
        if lt:
            ci = lt.get("auc_ci95") or [None, None]
            f = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
            lines.append(f"Held-out test: AUC {f(lt.get('auc'))} (95% CI {f(ci[0])} to {f(ci[1])}) on {lt.get('n_images', '?')} images "
                         f"from the same public dataset. {_t(lt.get('note', ''), 260)}")
        else:
            lines.append("No held-out test numbers are recorded for this model: do not rely on it.")
    else:
        lines.append("No validation numbers are recorded for this model: it must NOT be relied on.")
    th = prov.get("thresholds", {})
    lines.append(f"Calibrated probabilities: {'yes (temperature %.2f)' % prov['temperature'] if prov.get('calibrated') else 'no'}. "
                 f"Triage thresholds: moderate {th.get('moderate')}, high {th.get('high')} ({_t(th.get('source', ''), 160)}). "
                 f"Out-of-distribution guard: {'on' if prov.get('ood_guard') else 'off'}.")
    return [Paragraph(l, st["small"]) for l in lines]


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


# ------------------------------------------------------------------------------ page chrome
def _chrome(report_id: str, digest: str, synthetic: bool, generated: str):
    def draw(canvas, doc):
        canvas.saveState()
        # header band
        canvas.setFillColor(colors.HexColor("#101828"))
        canvas.rect(0, PAGE_H - 13 * mm, PAGE_W, 13 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(MARGIN, PAGE_H - 8.4 * mm, "VisionInspect  |  Health Screening Report")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 8.4 * mm, f"Report {report_id}   {generated}")
        # watermark
        canvas.translate(PAGE_W / 2, PAGE_H / 2)
        canvas.rotate(35)
        canvas.setFillColor(colors.Color(0.55, 0.1, 0.1, alpha=0.09) if synthetic else colors.Color(0.2, 0.2, 0.3, alpha=0.07))
        canvas.setFont("Helvetica-Bold", 46 if not synthetic else 40)
        canvas.drawCentredString(0, 14, "SYNTHETIC SAMPLE" if synthetic else "SCREENING")
        canvas.setFont("Helvetica-Bold", 30)
        canvas.drawCentredString(0, -26, "NOT REAL DATA - NOT A DIAGNOSIS" if synthetic else "NOT A DIAGNOSIS")
        canvas.restoreState()
        # footer
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.line(MARGIN, 15 * mm, PAGE_W - MARGIN, 15 * mm)
        canvas.setFont("Helvetica", 6.8)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 11.2 * mm, "Screening aid only. Not a medical diagnosis. A clinician must interpret every result.")
        canvas.drawString(MARGIN, 8 * mm, f"Data fingerprint {digest}. Page {doc.page}.")
        canvas.restoreState()
    return draw


# ------------------------------------------------------------------------------------ main
def build_report_pdf(data: Dict[str, Any], now: Optional[datetime] = None, report_id: Optional[str] = None) -> bytes:
    st = _styles()
    now = now or datetime.now(timezone.utc)
    generated = now.strftime("%d %b %Y %H:%M UTC")
    report_id = report_id or uuid.uuid4().hex[:10].upper()
    synthetic = bool(data.get("synthetic"))
    patient = data.get("patient") or {}
    findings = data.get("findings") or []
    provenance = data.get("provenance") or {}
    digest = _digest(data)
    summ = summarise(findings, data.get("vitals"))
    width = PAGE_W - 2 * MARGIN

    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=19 * mm, bottomMargin=20 * mm,
                          title="Health Screening Report", author="VisionInspect", subject="Screening aid, not a diagnosis")
    frame = Frame(MARGIN, 20 * mm, width, PAGE_H - 39 * mm, id="main", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=_chrome(report_id, digest, synthetic, generated))])
    S: List[Any] = []

    if synthetic:
        t = Table([[Paragraph("<b>SYNTHETIC SAMPLE.</b> Every name, number and image in this report is invented to show the layout. "
                              "No real patient, no real model.", st["body"])]], colWidths=[width])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3F2")), ("BOX", (0, 0), (-1, -1), 0.8, LEVEL_COLOR["HIGH"]),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        S += [t, Spacer(1, 6)]

    # ---- patient block
    S.append(Paragraph("Patient screening report", st["h1"]))
    S.append(Spacer(1, 4))
    cells = [
        ("PATIENT", patient.get("name") or "Not recorded"),
        ("AGE / SEX", f"{patient.get('age_years') if patient.get('age_years') is not None else '-'} / {patient.get('sex') or 'unknown'}"),
        ("PATIENT ID", patient.get("external_ref") or "-"),
        ("SKIN TYPE (FITZPATRICK)", patient.get("skin_type") or "unknown"),
        ("SCREENED ON", generated),
        ("CLINIC / OPERATOR", data.get("operator") or "-"),
    ]
    rows = [[Paragraph(f"<font size=6.8 color='#667085'><b>{k}</b></font><br/>{_t(v, 80)}", st["body"]) for k, v in cells[i:i + 3]] for i in (0, 3)]
    pt = Table(rows, colWidths=[width / 3] * 3)
    pt.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, RULE), ("INNERGRID", (0, 0), (-1, -1), 0.3, RULE),
                            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    S.append(pt)
    symptoms = [s for s in (patient.get("symptoms") or []) if s]
    if symptoms:
        S += [Spacer(1, 4), Paragraph("<b>Symptoms reported:</b> " + _t(", ".join(symptoms[:10]), 500), st["body"])]
    consent = patient.get("consent_given")
    S += [Spacer(1, 3), Paragraph(f"Consent for this screening: <b>{'recorded' if consent else 'NOT recorded'}</b>.", st["small"])]

    # ---- overall banner
    lvl = summ["level"]
    banner = Table([[Paragraph(f'<font color="{LEVEL_COLOR[lvl].hexval().replace("0x", "#")}">{_t(summ["title"])}</font>', st["banner_t"])],
                    [Paragraph(_t(summ["english"], 700), st["body"])], [Paragraph(_t(summ["hinglish"], 700), st["hi"])]], colWidths=[width])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LEVEL_BG[lvl]), ("BOX", (0, 0), (-1, -1), 1, LEVEL_COLOR[lvl]),
                                ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    S += [Spacer(1, 9), Paragraph("Overall result", st["h2"]), banner]

    # ---- findings
    S.append(Paragraph("What was checked, by body area", st["h2"]))
    by_module = {f["module"]: f for f in findings}
    if not summ["rows"]:
        S.append(Paragraph("No image checks were included in this report.", st["body"]))
    for row in summ["rows"]:
        f = by_module.get(row["module"], {})
        left = [Paragraph(f"<b>{_t(row['body_area'].upper())}</b>", st["label"]), Paragraph(_t(row["title"]), st["cell_b"]),
                Spacer(1, 3), _chip(row["headline"], row["triage"], st)]
        score = row["concern_probability"]
        if row["triage"] != "INCONCLUSIVE" and score is not None:
            calibrated = (provenance.get(row["module"]) or {}).get("calibrated")
            left.append(Paragraph(f"Model concern score {_pct(score)}" + ("" if calibrated else " (uncalibrated)"), st["small"]))
        mid = [Paragraph(_t(row["english"], 700), st["body"]), Spacer(1, 3), Paragraph(_t(row["hinglish"], 700), st["hi"])]
        if row["top_probs"] and row["triage"] != "INCONCLUSIVE":
            mid += [Spacer(1, 3), Paragraph("Model scores: " + _t(", ".join(f"{k} {v * 100:.0f}%" for k, v in row["top_probs"])), st["small"])]
        q = f.get("quality") or {}
        if q:
            mid.append(Paragraph(f"Image quality: {'adequate' if q.get('ok') else 'poor'}"
                                 + (f" ({_t(' '.join(q.get('warnings', [])), 200)})" if q.get("warnings") else ""), st["small"]))
        img = _image_flowable(f.get("heatmap_jpeg"), 38)
        right: List[Any] = [img, Paragraph("Highlighted areas influenced the model. It is not an outline of a disease.", st["small"])] if img else \
            [Paragraph("No heat-map for this check.", st["small"])]
        t = Table([[left, mid, right]], colWidths=[width * 0.24, width * 0.50, width * 0.26])
        t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, RULE), ("BACKGROUND", (0, 0), (0, 0), LEVEL_BG.get(row["triage"], colors.white)),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                               ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        S += [KeepTogether([t]), Spacer(1, 5)]

    # ---- vitals
    if summ["vitals"]:
        S += [Paragraph("Camera pulse", st["h2"]), Paragraph(_t(summ["vitals"]["text"], 700), st["body"])]

    # ---- next steps
    steps = summ["steps"]
    S.append(Paragraph("Recommended next steps", st["h2"]))
    if steps["referrals"]:
        S.append(Paragraph("<b>See, in this order:</b>", st["body"]))
        S += _bullets(steps["referrals"], st)
    else:
        S.append(Paragraph("No specialist visit is suggested by these checks. Keep any routine check-ups, and see a doctor if symptoms appear or persist.", st["body"]))
    S += [Spacer(1, 4), Paragraph("<b>See a doctor promptly, or go to an emergency department for severe symptoms, if:</b>", st["body"])]
    S += _bullets(steps["red_flags"], st)

    # ---- not checked
    S.append(Paragraph("What this screening did NOT check", st["h2"]))
    S += _bullets(summ["not_checked"], st, "small")

    # ---- technical block
    S.append(Paragraph("For the clinician: model details and limits", st["h2"]))
    for row in summ["rows"]:
        S += _model_card(row["module"], provenance.get(row["module"]), st) + [Spacer(1, 4)]
    S.append(Paragraph("Sensitivity and specificity from internal validation are optimistic and say nothing about this patient's "
                       "camera, lighting or skin tone unless a test on that population is cited. Results are unvalidated for clinical use "
                       "unless an independent clinical study is attached.", st["small"]))

    # ---- sign-off
    sign = Table([[Paragraph("<b>Reviewed by (name, registration no.)</b><br/><br/><br/>______________________________", st["body"]),
                   Paragraph("<b>Signature</b><br/><br/><br/>______________________", st["body"]),
                   Paragraph("<b>Date</b><br/><br/><br/>______________", st["body"])]], colWidths=[width * 0.5, width * 0.3, width * 0.2])
    sign.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, RULE), ("TOPPADDING", (0, 0), (-1, -1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    S += [Spacer(1, 10), KeepTogether([Paragraph("Clinician review", st["h2"]), sign, Spacer(1, 6),
                                       Paragraph(_t(DISCLAIMER, 500), st["small"])])]

    doc.build(S)
    return buf.getvalue()
