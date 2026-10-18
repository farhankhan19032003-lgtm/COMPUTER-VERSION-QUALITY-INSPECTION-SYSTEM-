"""Parsers for Android `adb shell` output -> the diagnostics payload the API accepts.

Pure functions (no adb needed) so they can be unit-tested with sample text.
Availability varies by phone maker and Android version: every field is optional.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

BATTERY_HEALTH = {
    1: "UNKNOWN", 2: "GOOD", 3: "OVERHEAT", 4: "DEAD",
    5: "OVER_VOLTAGE", 6: "UNSPECIFIED_FAILURE", 7: "COLD",
}


def _kv_lines(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def _num(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def parse_dumpsys_battery(text: str) -> Dict[str, Any]:
    """`adb shell dumpsys battery` -> battery dict (temperature is reported in tenths of a degree C)."""
    kv = _kv_lines(text)
    out: Dict[str, Any] = {}
    level = _num(kv.get("level"))
    if level is not None and 0 <= level <= 100:
        out["level_pct"] = level
    volt = _num(kv.get("voltage"))
    if volt:
        out["voltage_mv"] = volt
    temp = _num(kv.get("temperature"))
    if temp is not None:
        out["temperature_c"] = round(temp / 10.0, 1)
    health = _num(kv.get("health"))
    if health is not None:
        out["android_health"] = BATTERY_HEALTH.get(int(health), "UNKNOWN")
    powered = [kv.get(k, "false").lower() == "true" for k in ("ac powered", "usb powered", "wireless powered")]
    status = _num(kv.get("status"))
    if any(powered) or status == 2:
        out["charging"] = True
    elif status is not None:
        out["charging"] = False
    cyc = _num(kv.get("cycle count"))
    if cyc is not None and cyc > 0:
        out["cycle_count"] = int(cyc)
    return out


def parse_df(text: str) -> Dict[str, float]:
    """`adb shell df -k /data` -> total/free in GB."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return {}
    tokens = " ".join(lines[1:]).split()
    if len(tokens) < 6:
        return {}
    try:
        total_kb, avail_kb = float(tokens[-5]), float(tokens[-3])
    except ValueError:
        return {}
    return {"total_gb": round(total_kb / 1024 / 1024, 1), "free_gb": round(avail_kb / 1024 / 1024, 1)}


def parse_meminfo(text: str) -> Dict[str, float]:
    kv = _kv_lines(text)
    total, avail = _num(kv.get("memtotal")), _num(kv.get("memavailable"))
    if not total or avail is None:
        return {}
    return {"total_mb": round(total / 1024, 0), "available_mb": round(avail / 1024, 0)}


def capacity_health(charge_full: Optional[str], charge_full_design: Optional[str]) -> Optional[float]:
    """Battery health % = current full-charge capacity / design capacity (sysfs, in uAh)."""
    full, design = _num(charge_full), _num(charge_full_design)
    if full and design and design > 0:
        pct = full / design * 100
        if 10 <= pct <= 130:  # reject nonsense values
            return round(pct, 1)
    return None


def build_payload(
    model: str = "",
    release: str = "",
    battery_text: str = "",
    df_text: str = "",
    meminfo_text: str = "",
    charge_full: Optional[str] = None,
    charge_full_design: Optional[str] = None,
    cycle_count: Optional[str] = None,
) -> Dict[str, Any]:
    battery = parse_dumpsys_battery(battery_text)
    health = capacity_health(charge_full, charge_full_design)
    if health is not None:
        battery["health_pct"] = health
    cyc = _num(cycle_count)
    if cyc is not None and cyc > 0:
        battery["cycle_count"] = int(cyc)
    payload: Dict[str, Any] = {"platform": "android", "model": model.strip() or None, "os_version": release.strip() or None}
    if battery:
        payload["battery"] = battery
    if (s := parse_df(df_text)):
        payload["storage"] = s
    if (m := parse_meminfo(meminfo_text)):
        payload["memory"] = m
    return payload
