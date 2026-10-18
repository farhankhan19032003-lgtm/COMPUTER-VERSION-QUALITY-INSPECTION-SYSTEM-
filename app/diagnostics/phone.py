"""Phone health check from device-reported data and user-reported symptoms.

A camera cannot see inside a phone. What CAN be known without opening it is what the
phone itself reports (battery capacity/temperature/cycles, storage, memory) plus symptoms.
This module scores that data and lists likely causes and next steps. It cannot see physical
board damage and gives hypotheses, not diagnoses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

SEV_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "safety": 4}
PENALTY = {"info": 0, "low": 5, "medium": 12, "high": 25, "safety": 60}


class Battery(BaseModel):
    level_pct: Optional[float] = Field(None, ge=0, le=100)
    health_pct: Optional[float] = Field(None, ge=0, le=130, description="full-charge capacity vs design")
    cycle_count: Optional[int] = Field(None, ge=0, le=20000)
    temperature_c: Optional[float] = Field(None, ge=-40, le=120)
    voltage_mv: Optional[float] = Field(None, ge=0, le=20000)
    android_health: Optional[Literal[
        "UNKNOWN", "GOOD", "OVERHEAT", "DEAD", "OVER_VOLTAGE", "UNSPECIFIED_FAILURE", "COLD"]] = None
    charging: Optional[bool] = None


class Storage(BaseModel):
    total_gb: float = Field(..., gt=0, le=100000)
    free_gb: float = Field(..., ge=0, le=100000)


class Memory(BaseModel):
    total_mb: float = Field(..., gt=0, le=1_000_000)
    available_mb: float = Field(..., ge=0, le=1_000_000)


SYMPTOMS = {
    "overheating": ("medium", "Phone gets too hot",
        ["Heavy app or background activity", "Worn battery", "Faulty charger/cable", "Thick case trapping heat", "Hardware fault"],
        ["Remove the case, close heavy apps, and test with the original charger.",
         "If it still overheats while idle, get the battery and board checked."]),
    "fast_drain": ("medium", "Battery drains fast",
        ["Worn battery", "Apps running in the background", "Weak signal", "High screen brightness"],
        ["Check Settings > Battery for the apps using the most power.",
         "If battery health is low, a battery replacement is the fix."]),
    "random_restarts": ("high", "Random restarts or shutdowns",
        ["Worn battery that can't supply peak power", "Software crash or bad update", "Storage full", "Board fault"],
        ["Back up your data, update the OS, and free storage.",
         "If it keeps happening, have the battery and board checked."]),
    "touch_issues": ("medium", "Touch not responding / ghost touches",
        ["Dirty screen or poor screen protector", "Software glitch", "Digitizer damaged by a drop", "Cheap charger noise"],
        ["Clean the screen, remove the protector, restart, and test with another charger.",
         "If it persists, the display assembly may need replacing."]),
    "screen_flicker": ("medium", "Screen flickers",
        ["Software or brightness setting", "Loose display connector", "Failing panel"],
        ["Restart, update, and turn off adaptive brightness to test.", "Persistent flicker usually means a display service."]),
    "screen_lines": ("high", "Lines or colour bands on screen",
        ["Damaged display panel (often after impact or pressure)", "Display connector fault"],
        ["Back up your data: panel faults tend to get worse.", "Needs display replacement."]),
    "not_charging": ("medium", "Not charging",
        ["Lint or dirt in the charging port", "Bad cable or charger", "Damaged port", "Dead battery or board fault"],
        ["Try another cable and charger, and gently clean the port with a wooden toothpick (power off first).",
         "If it still won't charge, have the port and battery checked."]),
    "slow_charging": ("low", "Charges slowly",
        ["Low-power charger or cable", "Hot phone limiting charge speed", "Worn battery"],
        ["Use the original charger and cable and charge without the case."]),
    "no_sound": ("medium", "No sound or crackling speaker",
        ["Silent/volume setting", "Speaker grille blocked", "Software", "Speaker or audio chip fault"],
        ["Check volume and Do Not Disturb, clean the speaker grille gently, restart."]),
    "camera_blur": ("low", "Blurry camera",
        ["Smudged or cracked lens", "Protective film left on", "Autofocus or stabiliser fault"],
        ["Clean the lens with microfibre and check for cracks.", "If still blurry, service the camera module."]),
    "wifi_drops": ("low", "Wi-Fi drops",
        ["Router issue", "Software bug", "Antenna fault"],
        ["Test on another network. If only this phone drops, reset network settings."]),
    "drops_signal": ("low", "Mobile signal drops",
        ["Network coverage", "SIM or SIM tray", "Antenna damage"],
        ["Try the SIM in another phone and reseat it."]),
    "apps_crash": ("low", "Apps crash",
        ["Storage full", "Outdated OS or app", "Corrupt app data"],
        ["Free storage, update the OS and apps, and reinstall the failing app."]),
    "water_exposure": ("high", "Phone was exposed to liquid",
        ["Liquid ingress causing corrosion that keeps spreading"],
        ["Power off, don't charge, and don't use heat or rice.", "Take it to a service centre soon: early cleaning matters."]),
    "bulging": ("safety", "Swollen battery or lifting back/screen",
        ["Swollen lithium-ion battery"],
        ["SAFETY: stop charging and power the phone off now.",
         "Do not press, puncture or heat it. Take it to an authorised service centre."]),
}


class PhoneDiagnostics(BaseModel):
    platform: Literal["android", "ios", "other"] = "android"
    model: Optional[str] = Field(None, max_length=80)
    os_version: Optional[str] = Field(None, max_length=40)
    age_months: Optional[int] = Field(None, ge=0, le=240)
    battery: Optional[Battery] = None
    storage: Optional[Storage] = None
    memory: Optional[Memory] = None
    symptoms: List[str] = Field(default_factory=list, max_length=15)

    @field_validator("symptoms")
    @classmethod
    def _clean(cls, v: List[str]) -> List[str]:
        return list(dict.fromkeys(s.strip().lower() for s in v if s and len(s) <= 40))


@dataclass
class Finding:
    area: str
    severity: str
    title: str
    evidence: str = ""
    causes: Optional[List[str]] = None
    actions: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "", [])}


def _battery_findings(b: Battery) -> List[Finding]:
    f: List[Finding] = []
    if b.android_health in ("OVERHEAT", "DEAD", "OVER_VOLTAGE", "UNSPECIFIED_FAILURE"):
        f.append(Finding("battery", "high", f"Battery reports a fault ({b.android_health})",
                         "Reported by Android's battery service.",
                         actions=["Stop charging, let it cool, and have the battery checked at a service centre."]))
    if b.temperature_c is not None:
        if b.temperature_c >= 55:
            f.append(Finding("battery", "safety", "Battery is dangerously hot", f"{b.temperature_c} °C",
                             actions=["Unplug, power off, remove the case and let it cool away from flammable items.",
                                      "If it gets this hot again, stop using it and get it checked."]))
        elif b.temperature_c >= 45:
            f.append(Finding("battery", "medium", "Battery running hot", f"{b.temperature_c} °C",
                             actions=["Close heavy apps, remove the case and avoid charging in the sun."]))
    if b.health_pct is not None:
        if b.health_pct < 70:
            f.append(Finding("battery", "high", "Battery is heavily worn", f"{b.health_pct}% of original capacity",
                             actions=["Replace the battery: expect short run time and possible shutdowns."]))
        elif b.health_pct < 80:
            f.append(Finding("battery", "medium", "Battery is worn", f"{b.health_pct}% of original capacity",
                             actions=["80% is the usual point where makers recommend a battery service."]))
        elif b.health_pct < 90:
            f.append(Finding("battery", "low", "Battery shows some wear", f"{b.health_pct}% of original capacity"))
    if b.cycle_count is not None:
        if b.cycle_count > 1000:
            f.append(Finding("battery", "medium", "Very high charge-cycle count", f"{b.cycle_count} cycles",
                             actions=["Most phone batteries are rated for roughly 500 to 1000 cycles."]))
        elif b.cycle_count > 500:
            f.append(Finding("battery", "low", "High charge-cycle count", f"{b.cycle_count} cycles"))
    return f


def _storage_findings(s: Storage, m: Optional[Memory]) -> List[Finding]:
    f: List[Finding] = []
    free_pct = s.free_gb / s.total_gb * 100
    if free_pct < 5 or s.free_gb < 1:
        f.append(Finding("storage", "medium", "Storage almost full", f"{s.free_gb:.1f} GB free ({free_pct:.0f}%)",
                         actions=["Free space: a full phone slows down, crashes apps and can fail OS updates."]))
    elif free_pct < 10:
        f.append(Finding("storage", "low", "Storage getting full", f"{s.free_gb:.1f} GB free ({free_pct:.0f}%)"))
    if m is not None and m.available_mb / m.total_mb < 0.08:
        f.append(Finding("memory", "low", "Very little free RAM right now",
                         f"{m.available_mb:.0f} of {m.total_mb:.0f} MB available",
                         actions=["Restart the phone. If this is always the case, uninstall heavy background apps."]))
    return f


def evaluate(d: PhoneDiagnostics) -> Dict[str, Any]:
    findings: List[Finding] = []
    if d.battery:
        findings += _battery_findings(d.battery)
    if d.storage:
        findings += _storage_findings(d.storage, d.memory)

    unknown = []
    for s in d.symptoms:
        if s not in SYMPTOMS:
            unknown.append(s)
            continue
        sev, title, causes, actions = SYMPTOMS[s]
        # Symptoms backed by matching measurements are more convincing.
        evidence = "Reported by you."
        if s in ("fast_drain", "random_restarts") and d.battery and d.battery.health_pct and d.battery.health_pct < 85:
            evidence += f" Battery health is {d.battery.health_pct}%, which fits."
        if s == "overheating" and d.battery and d.battery.temperature_c and d.battery.temperature_c >= 45:
            evidence += f" Battery is at {d.battery.temperature_c} °C, which fits."
        findings.append(Finding("symptom", sev, title, evidence, causes, actions))

    findings.sort(key=lambda x: SEV_ORDER[x.severity], reverse=True)
    expected = [d.battery, d.storage, d.memory]
    completeness = round(sum(x is not None for x in expected) / len(expected), 2)
    has_data = any(x is not None for x in expected) or bool(d.symptoms)

    score = max(0, 100 - sum(PENALTY[f.severity] for f in findings))
    if not has_data:
        verdict = "INSUFFICIENT_DATA"
    elif any(f.severity == "safety" for f in findings):
        verdict = "UNSAFE"
    elif score < 55 or any(f.severity == "high" for f in findings):
        verdict = "POOR"
    elif score < 80 or any(f.severity == "medium" for f in findings):
        verdict = "FAIR"          # a real medium problem must not be hidden by a good-looking score
    else:
        verdict = "GOOD"

    steps: List[str] = []
    for f in findings:
        for a in f.actions or []:
            if a not in steps:
                steps.append(a)

    notes = ["Based on data the phone reports and what you told us. It cannot see physical board, "
             "connector or water damage, and the causes listed are possibilities, not a diagnosis."]
    if d.platform == "ios":
        notes.append("iPhone: enter battery health and cycle count from Settings > Battery manually; "
                     "automatic collection is not possible without Apple tools.")
    if unknown:
        notes.append("Ignored unknown symptoms: " + ", ".join(unknown[:5]))
    if completeness < 0.67 and has_data:
        notes.append("Few measurements were provided, so the score is rough. Add battery and storage values for a better result.")

    return {
        "health_score": score,
        "verdict": verdict,
        "data_completeness": completeness,
        "findings": [f.to_dict() for f in findings],
        "next_steps": steps,
        "notes": notes,
        "known_symptoms": sorted(SYMPTOMS),
    }
