"""Turn raw screening outputs into plain-language statements for a patient report.

No diagnosis is ever stated. Wording is "the model found a pattern that may go with ..." and every
statement is tied to a referral. English comes first; each statement also has a Roman-Hindi
(Hinglish) line because many patients read that more easily. (Devanagari is avoided on purpose:
the PDF engine cannot shape conjunct letters reliably, and a broken script would be worse than none.)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..screening.catalog import NOT_DETECTABLE, URGENCY_RANK, module_info

EMERGENCY_GENERAL = [
    "Chest pain, trouble breathing, fainting, sudden weakness or numbness, sudden severe headache, or a sudden change in vision: "
    "go to an emergency department or call your local emergency number now. Do not wait for an appointment.",
]

OVERALL = {
    "HIGH": ("Professional evaluation recommended soon",
             "One or more checks found a pattern that may need medical attention. Please arrange to see the suggested specialist soon.",
             "Kuch jaanch me aisa pattern mila hai jise doctor ko dikhana zaroori hai. Kripya jaldi suggested specialist se milein."),
    "MODERATE": ("Professional check advisable",
                 "Some checks found features that may deserve attention. A check-up with the suggested professional is advisable.",
                 "Kuch jaanch me aise nishaan mile hain jinhe doctor se check karwana behtar hai."),
    "INCOMPLETE": ("Incomplete: some images could not be assessed",
                   "At least one image was not clear enough, or did not look like the images the model was built for. "
                   "Nothing can be said about those areas. Please retake them.",
                   "Kuch photo saaf nahi thi, isliye un hisson ke baare me kuch nahi kaha ja sakta. Kripya dobara photo lein."),
    "LOW": ("No concerning pattern found by the checks done",
            "This is NOT a clearance. The model only looked at the areas listed below and can miss disease. "
            "If you have symptoms, see a doctor regardless of this result.",
            "Jo jaanch ki gayi usme koi chinta wala pattern nahi mila. Yeh 'sab theek hai' ka certificate nahi hai. "
            "Takleef ho to doctor se zaroor milein."),
}


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def overall_level(findings: List[Dict[str, Any]]) -> str:
    triages = [f.get("triage", "INCONCLUSIVE") for f in findings]
    if not triages:
        return "INCOMPLETE"
    if "HIGH" in triages:
        return "HIGH"
    if "MODERATE" in triages:
        return "MODERATE"
    if "INCONCLUSIVE" in triages:
        return "INCOMPLETE"
    return "LOW"


def describe_finding(f: Dict[str, Any]) -> Dict[str, Any]:
    """Plain-language row for one module result."""
    info = module_info(f["module"])
    triage = f.get("triage", "INCONCLUSIVE")
    area, finding, ref = info["body_area"], info.get("finding", "a possible finding"), info.get("referral", "a qualified clinician")
    if triage == "HIGH":
        en = f"The model found a pattern that may go with {finding}. Please see: {ref}, soon."
        hi = f"{area} me aisa pattern mila hai jiske liye {ref} se jaldi milna chahiye. Yeh diagnosis nahi hai."
        headline = "Needs professional evaluation"
    elif triage == "MODERATE":
        en = f"The model saw some features that may relate to {finding}. A check with: {ref} is advisable."
        hi = f"{area} me kuch nishaan dikhe hain. {ref} se check karwana behtar hai. Yeh diagnosis nahi hai."
        headline = "Check-up advisable"
    elif triage == "LOW":
        en = (f"No pattern of concern for {finding} was found by this model. This does not rule out disease; "
              "see a doctor if you have symptoms.")
        hi = f"{area} me is model ko koi chinta wala pattern nahi mila. Phir bhi takleef ho to doctor se milein."
        headline = "No concerning pattern found"
    else:
        reasons = " ".join(f.get("quality", {}).get("warnings", []) or []) or f.get("message", "")
        en = f"Could not assess this area. {reasons} Please retake the image as instructed: {info.get('input', 'a clear photo')}."
        hi = f"{area} ki photo se result nahi nikal paya (photo saaf nahi thi ya sahi tarah se nahi li gayi). Dobara photo lein."
        headline = "Could not be assessed"
    probs = f.get("probabilities") or {}
    top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return {
        "module": f["module"], "title": info["title"], "body_area": area, "triage": triage, "headline": headline,
        "english": en, "hinglish": hi, "referral": ref,
        "concern_probability": f.get("concern_probability"), "top_probs": [(k, v) for k, v in top],
        "device": info.get("device", ""), "evidence": info.get("evidence", ""),
        "red_flags": info.get("red_flags", []),
    }


def vitals_row(v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not v:
        return None
    bpm, conf, status = v.get("bpm"), v.get("confidence"), v.get("status", "unknown")
    if bpm is None:
        return {"measured": False, "text": f"Pulse was not measured ({status}). The camera method needs about 15 seconds of "
                                           "steady face video in good light, and gives no number when unsure."}
    note = ""
    if bpm < 50 or bpm > 120:
        note = " This is outside the usual resting range (about 60 to 100 per minute); if you feel unwell, see a doctor."
    return {"measured": True, "bpm": round(float(bpm)), "confidence": conf,
            "text": f"Camera pulse estimate: about {round(float(bpm))} beats per minute (signal confidence {_pct(conf)}). "
                    "This is a rough estimate from video, not a medical measurement." + note}


def next_steps(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Referrals (most urgent first, no repeats) and the red-flag list for areas that were flagged or incomplete."""
    ranked = sorted(rows, key=lambda r: URGENCY_RANK.get(r["triage"], 0), reverse=True)
    refs, flags = [], []
    for r in ranked:
        if r["triage"] in ("HIGH", "MODERATE"):
            line = f"{r['referral']}  (for: {r['body_area']})"
            if line not in refs:
                refs.append(line)
        if r["triage"] in ("HIGH", "MODERATE", "INCONCLUSIVE"):
            for fl in r["red_flags"]:
                if fl not in flags:
                    flags.append(fl)
    return {"referrals": refs, "red_flags": EMERGENCY_GENERAL + flags}


def summarise(findings: List[Dict[str, Any]], vitals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rows = [describe_finding(f) for f in findings]
    level = overall_level(findings)
    title, en, hi = OVERALL[level]
    return {"level": level, "title": title, "english": en, "hinglish": hi, "rows": rows,
            "vitals": vitals_row(vitals), "steps": next_steps(rows), "not_checked": NOT_DETECTABLE}
