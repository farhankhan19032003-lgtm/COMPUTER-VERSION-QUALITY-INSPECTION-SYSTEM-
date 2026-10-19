import json
import time

import jwt
import pytest
import torch
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import MemoryStore
from app.main import create_app
from app.screening.classifier import build_arch

from .synth import jpeg, make_phone


def post_image(client, img, **data):
    return client.post("/api/v1/inspect", files={"file": ("p.jpg", jpeg(img), "image/jpeg")}, data=data)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200


def test_inspect_clean_and_cracked(client):
    ok = post_image(client, make_phone(1), product_type="phone")
    assert ok.status_code == 200
    body = ok.json()
    assert body["verdict"] == "PASS"
    assert body["annotated_image"].startswith("data:image/jpeg;base64,")
    assert "disclaimer" in body

    bad = post_image(client, make_phone(2, crack=True), product_type="phone").json()
    assert bad["verdict"] == "FAIL"
    assert any(d["type"] == "crack" for d in bad["defects"])


def test_blurry_photo_cannot_pass(client):
    import cv2

    blurry = cv2.GaussianBlur(make_phone(5), (0, 0), 14)
    body = post_image(client, blurry).json()
    assert body["verdict"] == "REVIEW"
    assert body["quality"]["ok"] is False


def test_input_validation(client):
    r = client.post("/api/v1/inspect", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400
    r = client.post("/api/v1/inspect", files={"file": ("x.jpg", b"\0" * (2 * 1024 * 1024), "image/jpeg")})
    assert r.status_code == 413
    r = post_image(client, make_phone(1), product_type="../../etc/passwd")
    assert r.status_code == 422


def test_batch_and_history_and_analytics(client):
    files = [("files", (f"{i}.jpg", jpeg(make_phone(i, crack=(i == 1))), "image/jpeg")) for i in range(3)]
    r = client.post("/api/v1/inspect/batch", files=files, data={"product_type": "phone"})   # located product: FAIL allowed
    assert r.status_code == 200
    assert r.json()["summary"]["FAIL"] >= 1

    items = client.get("/api/v1/inspections").json()["items"]
    assert len(items) == 3

    s = client.get("/api/v1/analytics/summary").json()
    assert s["history"]["total"] == 3
    assert s["runtime"]["totals"]["inspections"] == 3
    assert any(w["level"] == "WARNING" for w in s["warnings"])  # memory store warning


def test_websocket_stream(client):
    with client.websocket_connect("/ws/stream?product=phone") as ws:
        ws.send_bytes(jpeg(make_phone(2, crack=True)))
        msg = ws.receive_json()
        assert msg["verdict"] == "FAIL" and msg["smoothed_verdict"] == "FAIL"
        assert msg["annotated"]
        ws.send_bytes(b"garbage")
        assert "error" in ws.receive_json()


def test_screening_not_installed_returns_503_and_lists_catalog(client):
    r = client.post("/api/v1/screen/skin_lesion", files={"file": ("a.jpg", jpeg(make_phone(1)), "image/jpeg")})
    assert r.status_code == 503
    mods = client.get("/api/v1/screen/modules").json()
    assert any(m["name"] == "skin_lesion" and not m["installed"] for m in mods["modules"])
    assert "NOT a medical diagnosis" in mods["disclaimer"]


def test_screening_installed_module_works(client, settings):
    from pathlib import Path

    d = Path(settings.screening_model_dir)
    d.mkdir(parents=True, exist_ok=True)
    net = build_arch("resnet18", 2)
    torch.save(net.state_dict(), d / "demo.pt")
    (d / "demo.json").write_text(
        json.dumps({"title": "Demo", "labels": ["benign", "suspicious"], "concern_labels": ["suspicious"],
                    "arch": "resnet18", "input_size": 128, "min_confidence": 0.0})
    )
    r = client.post("/api/v1/screen/demo", data={"explain": "true"},
                    files={"file": ("a.jpg", jpeg(make_phone(1)), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["triage"] in {"LOW", "MODERATE", "HIGH", "INCONCLUSIVE"}
    assert body["not_a_diagnosis"] is True
    assert body["heatmap_image"].startswith("data:image/jpeg")
    assert client.post("/api/v1/screen/..%2Fsecret", files={"file": ("a.jpg", jpeg(make_phone(1)), "image/jpeg")}).status_code in (404, 503)


# ------------------------------------------------------------------------------- auth
SECRET = "x" * 40


def token(**over):
    claims = {"sub": "11111111-1111-1111-1111-111111111111", "aud": "authenticated",
              "exp": int(time.time()) + 600, "email": "a@b.co"}
    claims.update(over)
    return jwt.encode(claims, SECRET, algorithm="HS256")


@pytest.fixture
def secure_client(tmp_path):
    s = Settings(_env_file=None, auth_required=True, supabase_jwt_secret=SECRET, device="cpu",
                 yolo_weights="__none__.pt", anomaly_bank_dir=str(tmp_path / "a"),
                 screening_model_dir=str(tmp_path / "s"))
    with TestClient(create_app(s, store=MemoryStore())) as c:
        yield c


def test_auth_enforced(secure_client):
    files = {"file": ("p.jpg", jpeg(make_phone(1)), "image/jpeg")}
    assert secure_client.post("/api/v1/inspect", files=files).status_code == 401
    assert secure_client.post("/api/v1/inspect", files=files, headers={"Authorization": "Bearer junk"}).status_code == 401
    expired = token(exp=int(time.time()) - 10)
    assert secure_client.post("/api/v1/inspect", files=files, headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    wrong_aud = token(aud="other")
    assert secure_client.post("/api/v1/inspect", files=files, headers={"Authorization": f"Bearer {wrong_aud}"}).status_code == 401
    ok = secure_client.post("/api/v1/inspect", files=files, headers={"Authorization": f"Bearer {token()}"})
    assert ok.status_code == 200


def test_users_only_see_their_own_history(secure_client):
    files = lambda: {"file": ("p.jpg", jpeg(make_phone(1)), "image/jpeg")}  # noqa: E731
    a = {"Authorization": "Bearer " + token(sub="aaaaaaaa-0000-0000-0000-000000000000")}
    b = {"Authorization": "Bearer " + token(sub="bbbbbbbb-0000-0000-0000-000000000000")}
    secure_client.post("/api/v1/inspect", files=files(), headers=a)
    assert secure_client.get("/api/v1/inspections", headers=a).json()["count"] == 1
    assert secure_client.get("/api/v1/inspections", headers=b).json()["count"] == 0


def test_websocket_requires_token(secure_client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with secure_client.websocket_connect("/ws/stream"):
            pass
    with secure_client.websocket_connect(f"/ws/stream?token={token()}") as ws:
        ws.send_bytes(jpeg(make_phone(1)))
        assert "verdict" in ws.receive_json()


def test_ready_reports_what_loaded(client):
    r = client.get("/ready").json()
    assert r["status"] == "ready" and "product_classifier" in r and isinstance(r["screening_models"], list)
    assert r["torch_threads"] >= 1 and r["max_concurrent_inference"] >= 1


def test_warmup_loads_every_stage_and_reports(tmp_path):
    from app import warmup
    from app.inspection.pipeline import InspectionPipeline

    pipe = InspectionPipeline(Settings(_env_file=None, device="cpu", yolo_weights="__none__.pt", anomaly_bank_dir=str(tmp_path / "a")))
    state = {}
    warmup.run(pipe, state)
    assert state["status"] == "ok" and set(state["stages"]) == {"detect_and_checks_s", "product_classifier_s", "anomaly_s"}
    assert state["seconds"] >= 0 and state["rss_mb"] > 0
