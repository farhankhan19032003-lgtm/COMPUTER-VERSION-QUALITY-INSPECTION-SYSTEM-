"""Tests for advice, phone diagnostics, Android parsers, rPPG vitals and dataset splitting."""
import sys
import time
from pathlib import Path

import jwt
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import MemoryStore
from app.diagnostics import android
from app.diagnostics.phone import PhoneDiagnostics, evaluate
from app.inspection.types import Defect
from app.main import create_app
from app.repair.advice import build_advice
from app.vitals.rppg import RPPGEstimator

from .synth import jpeg, make_phone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


# ---------------------------------------------------------------------------- advice
def D(t, sev="high"):
    return Defect(type=t, severity=sev, score=0.8, bbox=(1, 2, 30, 40), source="cv")


def test_advice_for_crack_and_headline():
    a = build_advice([D("crack")], "FAIL", "phone")
    assert a["qc_disposition"] == "REJECT_OR_REWORK"
    assert a["items"][0]["issue"] == "Cracked glass or body"
    assert any("Back up" in x for x in a["items"][0]["recommended_actions"])


def test_safety_items_outrank_everything_and_quarantine():
    a = build_advice([D("crack"), D("bulge", "low")], "FAIL", "phone")
    assert a["items"][0]["urgency"] == "safety"       # even though the bulge was only 'low' severity
    assert a["qc_disposition"] == "QUARANTINE"
    assert a["headline"].startswith("SAFETY")


def test_advice_pass_has_no_items():
    a = build_advice([], "PASS")
    assert a["items"] == [] and a["qc_disposition"] == "RELEASE"


def test_inspect_api_includes_advice(client):
    r = client.post("/api/v1/inspect", files={"file": ("p.jpg", jpeg(make_phone(2, crack=True)), "image/jpeg")},
                    data={"product_type": "phone"}).json()
    assert r["advice"]["items"] and r["advice"]["qc_disposition"] == "REJECT_OR_REWORK"


# -------------------------------------------------------------------- android parsers
DUMPSYS = """Current Battery Service state:
  AC powered: false
  USB powered: true
  Wireless powered: false
  Max charging current: 500000
  status: 2
  health: 3
  present: true
  level: 87
  scale: 100
  voltage: 4123
  temperature: 462
  technology: Li-ion
"""
DF = """Filesystem     1K-blocks     Used Available Use% Mounted on
/dev/block/dm-9 118525088 112000000   6000000  95% /data
"""
MEM = "MemTotal:        5700000 kB\nMemFree:          100000 kB\nMemAvailable:     400000 kB\n"


def test_android_parsers():
    b = android.parse_dumpsys_battery(DUMPSYS)
    assert b == {"level_pct": 87.0, "voltage_mv": 4123.0, "temperature_c": 46.2,
                 "android_health": "OVERHEAT", "charging": True}
    s = android.parse_df(DF)
    assert 5.5 < s["free_gb"] < 5.8 and 112 < s["total_gb"] < 114
    assert android.parse_meminfo(MEM) == {"total_mb": 5566.0, "available_mb": 391.0}
    assert android.capacity_health("3800000", "5000000") == 76.0
    assert android.capacity_health("nonsense", "5000000") is None
    assert android.capacity_health("1", "5000000") is None            # absurd values rejected


def test_android_build_payload_validates_against_schema():
    payload = android.build_payload("Pixel 7", "14", DUMPSYS, DF, MEM, "3800000", "5000000", "712")
    PhoneDiagnostics(**payload)                                        # must not raise
    assert payload["battery"]["health_pct"] == 76.0 and payload["battery"]["cycle_count"] == 712


# ---------------------------------------------------------------- phone diagnostics
def test_healthy_phone_is_good():
    r = evaluate(PhoneDiagnostics(battery={"health_pct": 96, "cycle_count": 120, "temperature_c": 31},
                                  storage={"total_gb": 128, "free_gb": 60}))
    assert r["verdict"] == "GOOD" and r["health_score"] >= 95 and r["findings"] == []


def test_worn_hot_battery_with_symptoms_is_poor_and_explained():
    r = evaluate(PhoneDiagnostics(
        battery={"health_pct": 68, "temperature_c": 47}, storage={"total_gb": 64, "free_gb": 1.5},
        symptoms=["fast_drain", "random_restarts", "Overheating", "made_up_symptom"]))
    titles = [f["title"] for f in r["findings"]]
    assert r["verdict"] == "POOR"
    assert "Battery is heavily worn" in titles and "Storage almost full" in titles
    assert any("which fits" in f.get("evidence", "") for f in r["findings"])      # symptom corroborated by data
    assert any("made_up_symptom" in n for n in r["notes"])
    assert r["next_steps"]


def test_swollen_battery_or_extreme_heat_is_unsafe():
    assert evaluate(PhoneDiagnostics(symptoms=["bulging"]))["verdict"] == "UNSAFE"
    assert evaluate(PhoneDiagnostics(battery={"temperature_c": 58}))["verdict"] == "UNSAFE"


def test_no_data_is_insufficient_not_good():
    assert evaluate(PhoneDiagnostics())["verdict"] == "INSUFFICIENT_DATA"


def test_phone_diagnostics_api_validation_and_history(client):
    ok = client.post("/api/v1/diagnostics/phone", json={"platform": "ios", "battery": {"health_pct": 79, "cycle_count": 640}})
    assert ok.status_code == 200
    body = ok.json()
    assert body["verdict"] == "FAIR" and any("iPhone" in n for n in body["notes"]) and body["id"]
    assert client.get("/api/v1/diagnostics/phone").json()["count"] == 1
    bad = client.post("/api/v1/diagnostics/phone", json={"battery": {"temperature_c": 9999}})
    assert bad.status_code == 422
    assert client.post("/api/v1/diagnostics/phone", json={"storage": {"total_gb": 0, "free_gb": 1}}).status_code == 422


# ------------------------------------------------------------------------------ rPPG
def synth_trace(bpm, seconds=15, fps=30, noise=0.003, seed=0):
    rng = np.random.default_rng(seed)
    est = RPPGEstimator()
    t = 0.0
    for _ in range(int(seconds * fps)):
        t += 1 / fps + rng.normal(0, 0.004)
        pulse = 0 if bpm is None else np.array([0.003, 0.008, 0.004]) * np.sin(2 * np.pi * bpm / 60 * t)
        rgb = np.array([185.0, 125.0, 105.0]) * (1 + 0.02 * np.sin(2 * np.pi * 0.05 * t) + pulse + rng.normal(0, noise, 3))
        est.add(t, *rgb, 0.7)
    return est


@pytest.mark.parametrize("bpm", [55, 72, 95, 130])
def test_rppg_recovers_pulse(bpm):
    r = synth_trace(bpm).estimate()
    assert r["status"] == "ok" and abs(r["bpm"] - bpm) <= 3 and r["confidence"] > 0.6


def test_rppg_refuses_to_guess_from_noise_or_short_data():
    r = synth_trace(None).estimate()
    assert r["bpm"] is None and r["status"] == "uncertain"
    assert synth_trace(72, noise=0.03).estimate()["bpm"] is None
    assert synth_trace(72, seconds=4).estimate()["status"] == "collecting"


def test_rppg_no_face():
    est = RPPGEstimator()
    for i in range(450):
        est.add(i / 30, 0, 0, 0, 0.0)
    assert est.estimate()["status"] == "no_face"


def test_vitals_websocket_and_rest(client):
    rows = []
    rng = np.random.default_rng(3)
    for i in range(450):
        t = i / 30
        pulse = np.array([0.003, 0.008, 0.004]) * np.sin(2 * np.pi * 1.2 * t)           # 72 bpm
        r, g, b = np.array([185.0, 125.0, 105.0]) * (1 + pulse + rng.normal(0, 0.003, 3))
        rows.append([t * 1000, float(r), float(g), float(b), 0.7])
    rows.append(["bad", 1, 2, 3, 4])                    # garbage is skipped, not fatal
    rows.append([1e6, 999, 0, 0, 0.5])                  # out-of-range colour is skipped

    rest = client.post("/api/v1/vitals/estimate", json={"samples": rows[:-2]}).json()
    assert rest["status"] == "ok" and abs(rest["bpm"] - 72) <= 3 and "not a medical device" in rest["disclaimer"].lower()

    with client.websocket_connect("/ws/vitals") as ws:
        for i in range(0, len(rows), 100):
            ws.send_json({"samples": rows[i:i + 100]})
            msg = ws.receive_json()
        assert msg["status"] == "ok" and abs(msg["bpm"] - 72) <= 3
        ws.send_text("not json")
        assert "error" in ws.receive_json()


def test_new_endpoints_require_auth_when_enabled(tmp_path):
    secret = "x" * 40
    s = Settings(_env_file=None, auth_required=True, supabase_jwt_secret=secret, device="cpu",
                 yolo_weights="__none__.pt", anomaly_bank_dir=str(tmp_path / "a"), screening_model_dir=str(tmp_path / "s"))
    tok = jwt.encode({"sub": "u1", "aud": "authenticated", "exp": int(time.time()) + 600}, secret, algorithm="HS256")
    with TestClient(create_app(s, store=MemoryStore())) as c:
        assert c.post("/api/v1/diagnostics/phone", json={"symptoms": ["bulging"]}).status_code == 401
        assert c.post("/api/v1/vitals/estimate", json={"samples": []}).status_code == 401
        assert c.post("/api/v1/diagnostics/phone", json={"symptoms": ["bulging"]},
                      headers={"Authorization": f"Bearer {tok}"}).status_code == 200
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws/vitals"):
                pass


# ----------------------------------------------------------- leakage-safe splitting
def test_split_groups_never_shares_a_group_and_keeps_all_classes():
    from _split import assert_no_leakage, split_groups

    groups = {f"L{i}": ("suspicious" if i % 5 == 0 else "benign") for i in range(100)}
    assign = split_groups(groups, 0.15, 0.15, seed=1)
    assert set(assign.values()) == {"train", "val", "test"}
    for lab in ("suspicious", "benign"):
        assert {assign[g] for g, l in groups.items() if l == lab} == {"train", "val", "test"}
    items = [(Path(f"{g}_{k}.jpg"), assign[g], groups[g], g) for g in groups for k in range(3)]
    assert_no_leakage(items)
    items.append((Path("x.jpg"), "test" if assign["L1"] != "test" else "train", "benign", "L1"))
    with pytest.raises(AssertionError):
        assert_no_leakage(items)


def test_quality_gate_limits_are_configurable():
    from app.inspection.cv_defects import assess_quality

    small = np.random.default_rng(0).integers(60, 200, (96, 96, 3), dtype=np.uint8)
    assert not assess_quality(small)["ok"]
    assert assess_quality(small, min_side=48)["ok"]


@pytest.mark.parametrize("bpm,zone,level", [
    (35, "danger_low", "danger"), (45, "low", "caution"), (60, "normal", "normal"), (72, "normal", "normal"),
    (100, "normal", "normal"), (101, "elevated", "caution"), (120, "elevated", "caution"),
    (135, "high", "caution"), (151, "danger_high", "danger"), (170, "danger_high", "danger")])
def test_pulse_zone_bands(bpm, zone, level):
    from app.vitals.rppg import classify, interpret

    c = classify(bpm)
    assert (c["zone"], c["level"]) == (zone, level) and c["zone_label"]
    assert interpret(bpm)


def test_rppg_result_carries_the_zone():
    r = synth_trace(72).estimate()
    assert r["status"] == "ok" and r["zone"] == "normal" and r["level"] == "normal" and r["zone_label"] == "Normal"
    assert "zone" not in synth_trace(None).estimate()          # no number, no verdict
