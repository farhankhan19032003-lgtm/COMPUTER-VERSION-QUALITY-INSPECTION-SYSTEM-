import csv
import sys
import time
import uuid
from pathlib import Path

import jwt
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import MemoryStore
from app.main import create_app

from .synth import jpeg, make_phone

SECRET = "x" * 40
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def inspect(client, crack, headers=None):
    r = client.post("/api/v1/inspect", files={"file": ("p.jpg", jpeg(make_phone(2, crack=crack)), "image/jpeg")},
                    data={"product_type": "phone"}, headers=headers or {})
    assert r.status_code == 200
    return r.json()


def test_confirm_override_and_summary(client):
    bad, good = inspect(client, True), inspect(client, False)
    assert bad["verdict"] == "FAIL" and good["verdict"] == "PASS"

    # reviewer agrees with the FAIL
    r = client.post(f"/api/v1/inspections/{bad['id']}/feedback", json={"verdict": "FAIL"})
    assert r.status_code == 200 and r.json()["model_verdict"] == "FAIL"
    # reviewer says the "good" one actually has a problem: a miss
    client.post(f"/api/v1/inspections/{good['id']}/feedback", json={"verdict": "REVIEW", "note": "  hairline scratch "})

    fb = client.get("/api/v1/feedback").json()
    assert fb["summary"] == {"reviewed": 2, "agree": 1, "overridden": 1, "agreement_rate": 0.5,
                             "false_alarms": 0, "misses": 1}
    notes = {i["inspection_id"]: i["note"] for i in fb["items"]}
    assert notes[good["id"]] == "hairline scratch"

    # history shows the reviewer's verdict next to the model's
    hist = {i["id"]: i for i in client.get("/api/v1/inspections").json()["items"]}
    assert hist[bad["id"]]["feedback"]["verdict"] == "FAIL"
    assert hist[good["id"]]["feedback"] == {"verdict": "REVIEW", "note": "hairline scratch"}


def test_resubmitting_replaces_the_review(client):
    iid = inspect(client, True)["id"]
    client.post(f"/api/v1/inspections/{iid}/feedback", json={"verdict": "PASS"})
    client.post(f"/api/v1/inspections/{iid}/feedback", json={"verdict": "FAIL"})
    fb = client.get("/api/v1/feedback").json()
    assert fb["summary"]["reviewed"] == 1 and fb["items"][0]["reviewer_verdict"] == "FAIL"


def test_false_alarm_counted(client):
    iid = inspect(client, True)["id"]
    client.post(f"/api/v1/inspections/{iid}/feedback", json={"verdict": "PASS", "note": "glare"})
    assert client.get("/api/v1/feedback").json()["summary"]["false_alarms"] == 1


def test_feedback_validation(client):
    iid = inspect(client, True)["id"]
    assert client.post(f"/api/v1/inspections/{iid}/feedback", json={"verdict": "MAYBE"}).status_code == 422
    assert client.post(f"/api/v1/inspections/{iid}/feedback", json={"verdict": "PASS", "note": "x" * 501}).status_code == 422
    assert client.post(f"/api/v1/inspections/{uuid.uuid4()}/feedback", json={"verdict": "PASS"}).status_code == 404
    assert client.post("/api/v1/inspections/not-a-uuid/feedback", json={"verdict": "PASS"}).status_code == 404


def test_empty_summary(client):
    fb = client.get("/api/v1/feedback").json()
    assert fb["summary"]["reviewed"] == 0 and fb["summary"]["agreement_rate"] is None


@pytest.fixture
def secure_client(tmp_path):
    s = Settings(_env_file=None, auth_required=True, supabase_jwt_secret=SECRET, device="cpu",
                 yolo_weights="__none__.pt", anomaly_bank_dir=str(tmp_path / "a"),
                 screening_model_dir=str(tmp_path / "s"))
    with TestClient(create_app(s, store=MemoryStore())) as c:
        yield c


def auth(sub):
    t = jwt.encode({"sub": sub, "aud": "authenticated", "exp": int(time.time()) + 600}, SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {t}"}


def test_feedback_needs_login_and_is_private(secure_client):
    a, b = auth("aaaaaaaa-0000-0000-0000-000000000000"), auth("bbbbbbbb-0000-0000-0000-000000000000")
    iid = inspect(secure_client, True, a)["id"]

    assert secure_client.post(f"/api/v1/inspections/{iid}/feedback", json={"verdict": "FAIL"}).status_code == 401
    assert secure_client.get("/api/v1/feedback").status_code == 401

    # another user cannot review, or see, A's inspection
    assert secure_client.post(f"/api/v1/inspections/{iid}/feedback", json={"verdict": "PASS"}, headers=b).status_code == 404

    assert secure_client.post(f"/api/v1/inspections/{iid}/feedback", json={"verdict": "FAIL"}, headers=a).status_code == 200
    assert secure_client.get("/api/v1/feedback", headers=a).json()["summary"]["reviewed"] == 1
    assert secure_client.get("/api/v1/feedback", headers=b).json()["summary"]["reviewed"] == 0


# ------------------------------------------------------------------ export script
def _insp(iid, verdict, product="phone", path="u/x.jpg", defects=()):
    return {"id": iid, "verdict": verdict, "product_type": product, "image_path": path, "source": "upload",
            "created_at": "2026-10-18T00:00:00Z", "defects": [{"type": t} for t in defects]}


def _fb(iid, verdict, note=None, user="u1"):
    return {"inspection_id": iid, "reviewer_verdict": verdict, "note": note, "user_id": user}


def test_export_plan_folders_conflicts_and_filters():
    import export_feedback as ex

    insp = {i["id"]: i for i in [
        _insp("a", "FAIL", defects=("crack",)), _insp("b", "FAIL"), _insp("c", "PASS"),
        _insp("d", "PASS", path=None), _insp("e", "FAIL", product="Laptop Pro!"), _insp("f", "PASS"),
    ]}
    fb = [_fb("a", "FAIL"), _fb("b", "PASS", "glare"), _fb("c", "REVIEW"), _fb("d", "FAIL"),
          _fb("e", "FAIL"), _fb("zzz", "PASS"), _fb("f", "PASS", user="u1"), _fb("f", "FAIL", user="u2")]
    items, conflicts, skipped = ex.plan_export(fb, insp)

    where = {i["inspection_id"]: (i["product"], i["folder"]) for i in items}
    assert where == {"a": ("phone", "defective"), "b": ("phone", "good"), "c": ("phone", "borderline"),
                     "e": ("laptop_pro", "defective")}
    assert [c["inspection_id"] for c in conflicts] == ["f"]
    assert conflicts[0]["reviewer_verdicts"] == "FAIL|PASS"
    assert skipped["no_image"] == 1 and skipped["no_inspection"] == 1
    rows = {i["inspection_id"]: i["row"] for i in items}
    assert rows["a"]["agrees"] is True and rows["a"]["defects"] == "crack"
    assert rows["b"]["agrees"] is False and rows["b"]["note"] == "glare"

    only, _, sk = ex.plan_export(fb, insp, only_overrides=True)
    assert {i["inspection_id"] for i in only} == {"b", "c"} and sk["agreed"] == 2
    phones, _, sk = ex.plan_export(fb, insp, product="phone")
    assert "e" not in {i["inspection_id"] for i in phones} and sk["other_product"] == 1


def test_export_writes_photos_and_is_rerunnable(tmp_path):
    import export_feedback as ex

    img = jpeg(make_phone(1))
    calls = []

    def download(path):
        calls.append(path)
        return b"junk" if path == "bad.jpg" else img

    insp = {i["id"]: i for i in [_insp("a", "FAIL", path="a.jpg"), _insp("b", "PASS", path="bad.jpg")]}
    items, conflicts, _ = ex.plan_export([_fb("a", "PASS", "glare"), _fb("b", "PASS")], insp)
    out = tmp_path / "fb"
    counts = ex.write_export(items, conflicts, out, download)
    assert counts == {"downloaded": 1, "already_there": 0, "moved": 0, "failed": 1}   # corrupt file skipped, rest kept
    assert (out / "phone" / "good" / "a.jpg").read_bytes() == img
    assert not (out / "phone" / "good" / "b.jpg").exists()
    labels = list(csv.DictReader(open(out / "labels.csv", encoding="utf-8")))
    assert [(r["file"], r["model_verdict"], r["reviewer_verdict"], r["note"]) for r in labels] == [
        ("phone/good/a.jpg", "FAIL", "PASS", "glare")]

    # second run: nothing is downloaded again
    calls.clear()
    counts = ex.write_export(items, conflicts, out, download)
    assert counts["downloaded"] == 0 and counts["already_there"] == 1 and calls == ["bad.jpg"]

    # the reviewer changes their mind: the photo moves out of good/ so it can't end up in the bank
    items, _, _ = ex.plan_export([_fb("a", "FAIL")], insp)
    counts = ex.write_export(items, [], out, download)
    assert counts["moved"] == 1 and counts["downloaded"] == 0
    assert (out / "phone" / "defective" / "a.jpg").exists() and not (out / "phone" / "good" / "a.jpg").exists()
