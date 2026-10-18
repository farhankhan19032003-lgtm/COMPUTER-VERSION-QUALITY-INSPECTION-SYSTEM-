"""Turn detected defects into practical next steps.

This is a rule-based knowledge base, not a diagnosis: it lists likely causes and sensible
actions for what the photo shows. Entries marked `safety` always outrank everything else.
Defect names beyond the built-ins (dent, bulge, liquid_damage...) come from custom-trained
YOLO classes, see ROADMAP.md.
"""
from __future__ import annotations

from typing import Any, Dict, List

URGENCY_ORDER = {"monitor": 0, "soon": 1, "urgent": 2, "safety": 3}

# key -> knowledge entry. Keys are matched against Defect.type (substring match, lowercase).
KB: Dict[str, Dict[str, Any]] = {
    "crack": {
        "title": "Cracked glass or body",
        "causes": ["Drop or impact", "Pressure on the display", "Thermal stress"],
        "actions": [
            "Back up your data now: cracks often spread and can kill the touch layer.",
            "Cover with clear tape or a screen protector so glass shards don't cut fingers.",
            "Replace the screen/back glass at an authorised service centre.",
            "Avoid water exposure: a cracked display is no longer sealed.",
        ],
        "factory": "Reject. Do not ship as new; send to rework or scrap.",
        "min_urgency": "soon",
    },
    "scratch": {
        "title": "Scratch on surface",
        "causes": ["Keys/coins in a pocket", "Dust or sand on the surface", "Handling during assembly"],
        "actions": [
            "If it is only on the surface and you can't feel it with a fingernail, it is cosmetic.",
            "Add a tempered-glass protector to prevent more scratches.",
            "A deep scratch you can catch with a nail may grow into a crack: monitor it.",
        ],
        "factory": "Grade as cosmetic B if within spec; otherwise rework (polish/replace part).",
        "min_urgency": "monitor",
    },
    "spot": {
        "title": "Spot or mark",
        "causes": ["Dirt, fingerprint or stain", "Dust under the glass", "Dead or stuck pixel (if it stays when the screen is on)"],
        "actions": [
            "Wipe with a soft microfibre cloth and re-photograph.",
            "Switch the screen on with a plain white and a plain black image: a spot that stays is a pixel fault.",
            "A fault that stays under warranty should go back to the seller or service centre.",
        ],
        "factory": "Re-clean and re-inspect once; if it stays, hold for manual inspection.",
        "min_urgency": "monitor",
    },
    "anomaly": {
        "title": "Looks different from a normal unit",
        "causes": ["Missing or misaligned part", "Colour or finish variation", "Damage the simple checks can't name", "Odd lighting or angle"],
        "actions": [
            "Re-take the photo with even light, no glare, straight on, and compare with a known-good unit.",
            "Look closely at the highlighted area: gaps, missing screws, lifted edges, discolouration.",
            "If it is still flagged, get it checked by a person.",
        ],
        "factory": "Hold for manual inspection at the flagged area.",
        "min_urgency": "soon",
    },
    # ---- classes you get after training a custom YOLO (see ROADMAP.md) -------------
    "dent": {
        "title": "Dent or deformed frame",
        "causes": ["Drop on a corner or edge"],
        "actions": [
            "A dented frame can press on the display and battery: have the inside checked.",
            "Stop using the device if the screen shows lines, lifts at an edge, or the battery gets hot.",
        ],
        "factory": "Reject or rework the housing.",
        "min_urgency": "soon",
    },
    "bulge": {
        "title": "Swollen / bulging battery or lifting back panel",
        "causes": ["Swollen lithium-ion battery (ageing, overcharging, heat, damage)"],
        "actions": [
            "SAFETY: stop charging and switch the device off now.",
            "Do not press, bend, puncture or heat it, and keep it away from flammable things.",
            "Take it to an authorised service centre for battery replacement; do not keep using it.",
            "Do not throw it in normal rubbish: use an e-waste or battery recycling point.",
        ],
        "factory": "Quarantine immediately and follow your battery-safety procedure.",
        "min_urgency": "safety",
    },
    "swell": {"alias": "bulge"},
    "liquid": {
        "title": "Signs of liquid damage or corrosion",
        "causes": ["Water or sweat ingress", "Humidity", "Corroded charging port or contacts"],
        "actions": [
            "Power off, do not charge, and do not use heat or rice. Dry gently and get it serviced quickly.",
            "Corrosion keeps spreading after the device dries, so early cleaning at a service centre matters.",
        ],
        "factory": "Reject; investigate sealing and storage conditions.",
        "min_urgency": "urgent",
    },
    "corrosion": {"alias": "liquid"},
    "missing_screw": {
        "title": "Missing or loose screw",
        "causes": ["Assembly error", "Previous repair"],
        "actions": ["Have the device re-assembled; loose parts can rattle and damage the board or battery."],
        "factory": "Rework: fit the missing fastener, re-inspect.",
        "min_urgency": "soon",
    },
    "gap": {
        "title": "Uneven gap or misaligned panel",
        "causes": ["Assembly error", "Swollen battery pushing the panel", "Previous repair"],
        "actions": [
            "Check whether the back or screen is lifting: if so treat it like a possible swollen battery.",
            "Otherwise have the housing re-seated.",
        ],
        "factory": "Rework alignment; hold if the panel is lifting.",
        "min_urgency": "soon",
    },
}


def _entry(defect_type: str) -> Dict[str, Any] | None:
    t = defect_type.lower().replace(" ", "_")
    for key, entry in KB.items():
        if key in t:
            return KB[entry["alias"]] if "alias" in entry else entry
    return None


def _urgency(entry: Dict[str, Any], severity: str) -> str:
    base = {"low": "monitor", "medium": "soon", "high": "urgent"}.get(severity, "monitor")
    floor = entry.get("min_urgency", "monitor")
    return max(base, floor, key=lambda u: URGENCY_ORDER[u])


def build_advice(defects: List[Any], verdict: str, product_type: str = "product") -> Dict[str, Any]:
    """`defects` are inspection Defect objects. Returns advice for owners and a QC disposition."""
    seen: Dict[str, Dict[str, Any]] = {}
    for d in defects:
        entry = _entry(d.type)
        if entry is None:
            continue
        urg = _urgency(entry, d.severity)
        cur = seen.get(entry["title"])
        if cur is None or URGENCY_ORDER[urg] > URGENCY_ORDER[cur["urgency"]]:
            seen[entry["title"]] = {
                "issue": entry["title"],
                "detected_as": d.type,
                "severity": d.severity,
                "urgency": urg,
                "likely_causes": entry["causes"],
                "recommended_actions": entry["actions"],
                "factory_action": entry["factory"],
                "bbox": [int(v) for v in d.bbox],
            }
    items = sorted(seen.values(), key=lambda a: URGENCY_ORDER[a["urgency"]], reverse=True)

    if any(a["urgency"] == "safety" for a in items):
        headline = "SAFETY ISSUE: stop using and charging this device and have it checked."
    elif verdict == "FAIL":
        headline = f"This {product_type} has visible damage that should be repaired or reworked."
    elif verdict == "REVIEW":
        headline = "Not clear from the photo: have a person check the flagged areas or re-take the photo."
    else:
        headline = "No visible problem found in this photo."

    disposition = {"PASS": "RELEASE", "REVIEW": "MANUAL_INSPECTION", "FAIL": "REJECT_OR_REWORK"}[verdict]
    if any(a["urgency"] == "safety" for a in items):
        disposition = "QUARANTINE"

    return {
        "headline": headline,
        "qc_disposition": disposition,
        "items": items,
        "not_visible_in_photos": [
            "Internal faults (battery capacity, board, storage, sensors): use the phone diagnostics check.",
            "Software problems and whether the product is genuine.",
        ],
    }
