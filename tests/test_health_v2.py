"""Tests for v2 health features: calibration, OOD guard, dataset adapters, training, patients, report."""
import csv
import json
import sys
from pathlib import Path

import cv2
import jwt
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import MemoryStore
from app.main import create_app
from app.screening import calibration as cal
from app.screening.catalog import CATALOG, NOT_DETECTABLE, module_info
from app.screening.classifier import build_arch

from .synth import jpeg, make_phone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


# ------------------------------------------------------------------------------ catalog
def test_catalog_entries_are_complete_and_honest():
    for name, c in CATALOG.items():
        for k in ("title", "body_area", "finding", "input", "device", "evidence", "suggested_labels", "suggested_concern",
                  "datasets", "referral", "red_flags"):
            assert k in c, f"{name} missing {k}"
        assert set(c["suggested_concern"]) <= set(c["suggested_labels"]), name
        assert c["evidence"] in ("device_established", "smartphone_research", "early_research")
        assert c["datasets"] and all("verified" in d for d in c["datasets"])
    assert len(CATALOG) >= 10
    # organs a camera cannot see must be called out
    assert any("cannot see inside" in x for x in NOT_DETECTABLE)
    assert module_info("nonexistent")["referral"]


# ------------------------------------------------------------------- calibration and OOD
def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    true_logit = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 1.5, n)           # honest logit
    logits = np.stack([np.zeros(n), true_logit * 3.0], 1)                        # model is 3x over-confident
    T = cal.fit_temperature(logits, y)
    assert 2.4 < T < 3.6
    p_raw = 1 / (1 + np.exp(-logits[:, 1]))
    p_cal = 1 / (1 + np.exp(-logits[:, 1] / T))
    assert cal.expected_calibration_error(p_cal, y) < cal.expected_calibration_error(p_raw, y)
    assert cal.fit_temperature(logits[:10], y[:10]) == 1.0                       # too little data: no fitting


def test_threshold_hits_target_sensitivity_and_refuses_tiny_sets():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(200), np.zeros(800)].astype(int)
    p = np.r_[rng.beta(5, 2, 200), rng.beta(2, 5, 800)]
    t = cal.threshold_for_sensitivity(y, p, 0.90)
    assert abs((p[y == 1] >= t).mean() - 0.90) < 0.02
    out = cal.choose_triage_thresholds(y, p, 0.90, 0.95)
    assert out["moderate_threshold"] < out["high_threshold"] <= 0.99
    few = slice(190, 220)                                             # 10 positives, 20 negatives
    assert cal.threshold_for_sensitivity(y[few], p[few]) is None
    assert cal.choose_triage_thresholds(y[few], p[few])["moderate_threshold"] == 0.30


def test_mahalanobis_flags_unfamiliar_features():
    rng = np.random.default_rng(2)
    d = 32
    lab = rng.integers(0, 2, 600)
    feats = rng.normal(0, 1, (600, d)) + lab[:, None] * 3.0
    val = rng.normal(0, 1, (300, d)) + rng.integers(0, 2, 300)[:, None] * 3.0
    ood = cal.MahalanobisOOD.fit(feats, lab, val)
    fresh = rng.normal(0, 1, (300, d)) + rng.integers(0, 2, 300)[:, None] * 3.0
    far = rng.normal(0, 1, (300, d)) + 9.0
    assert ood.is_ood(fresh).mean() < 0.05
    assert ood.is_ood(far).mean() > 0.95


# --------------------------------------------------------------------- dataset adapters
def _img(path: Path, v=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((40, 40, 3), v, np.uint8))


def _leak_free(out: Path):
    seen = {}
    for p in out.rglob("*.*"):
        if p.suffix in (".png", ".jpg", ".jpeg"):
            split = p.relative_to(out).parts[0]
            assert seen.setdefault(p.name, split) == split


def test_adapter_aptos(tmp_path):
    import prepare_dataset as pdset

    src = tmp_path / "aptos"
    rows = [("id%03d" % i, i % 5) for i in range(100)]
    src.mkdir()
    with open(src / "train.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id_code", "diagnosis"]); w.writerows(rows)
    for i, _ in rows:
        _img(src / "train_images" / f"{i}.png")
    out = tmp_path / "out"
    pdset.main(["aptos", "--src", str(src), "--out", str(out)])
    classes = {p.name for p in (out / "train").iterdir()}
    assert classes == {"no_dr", "mild", "moderate", "severe", "proliferative"}
    assert sum(1 for _ in out.rglob("*.png")) == 100
    _leak_free(out)


def test_adapter_ham10000_splits_by_lesion_with_subgroups(tmp_path):
    import prepare_dataset as pdset

    src, out = tmp_path / "ham", tmp_path / "out"
    src.mkdir()
    rows = []
    for l in range(60):
        for k in range(3):                                    # 3 photos per lesion
            rows.append({"lesion_id": f"HAM_{l}", "image_id": f"ISIC_{l}_{k}", "dx": "mel" if l % 4 == 0 else "nv",
                         "age": str(20 + l), "sex": "female" if l % 2 else "male", "localization": "back"})
            _img(src / "imgs" / f"ISIC_{l}_{k}.jpg")
    with open(src / "HAM10000_metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    pdset.main(["ham10000", "--src", str(src), "--out", str(out)])
    lesion_split = {}
    for p in out.rglob("*.jpg"):
        lesion = "_".join(p.stem.split("_")[:2])
        assert lesion_split.setdefault(lesion, p.relative_to(out).parts[0]) == p.relative_to(out).parts[0]
    assert {p.name for p in (out / "train").iterdir()} == {"benign", "suspicious"}
    sub = list(csv.DictReader(open(out / "subgroups.csv")))
    assert {"file", "split", "sex", "age_band", "site"} <= set(sub[0])
    assert len(sub) == 180


def test_adapter_pad_ufes_patient_level(tmp_path):
    import prepare_dataset as pdset

    src, out = tmp_path / "pad", tmp_path / "out"
    rows = []
    for pt in range(50):
        for k in range(2):
            rows.append({"patient_id": f"PAT_{pt}", "lesion_id": str(pt), "img_id": f"PAT_{pt}_{k}.png",
                         "diagnostic": ["BCC", "NEV", "ACK", "SEK", "MEL"][pt % 5], "fitspatrick": str(1 + pt % 6),
                         "age": str(30 + pt), "region": "ARM"})
            _img(src / "imgs_part_1" / f"PAT_{pt}_{k}.png")
    with open(src / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    pdset.main(["pad_ufes", "--src", str(src), "--out", str(out)])
    patient_split = {}
    for p in out.rglob("*.png"):
        pat = "_".join(p.stem.split("_")[:2])
        assert patient_split.setdefault(pat, p.relative_to(out).parts[0]) == p.relative_to(out).parts[0]
    sub = list(csv.DictReader(open(out / "subgroups.csv")))
    assert "fitzpatrick" in sub[0]


def test_adapter_malaria_groups_by_slide(tmp_path):
    import prepare_dataset as pdset

    src, out = tmp_path / "cells", tmp_path / "out"
    for s in range(30):
        for c in range(4):
            _img(src / "Parasitized" / f"C{s}P1ThinF_IMG_2015_cell_{c}.png")
            _img(src / "Uninfected" / f"C{s + 100}N1ThinF_IMG_2015_cell_{c}.png")
    pdset.main(["malaria", "--src", str(src), "--out", str(out)])
    slide_split = {}
    for p in out.rglob("*.png"):
        slide = p.stem.split("_cell_")[0]
        assert slide_split.setdefault(slide, p.relative_to(out).parts[0]) == p.relative_to(out).parts[0]


def test_adapter_chest_xray_pools_and_groups_by_person(tmp_path):
    import prepare_dataset as pdset

    src, out = tmp_path / "cxr", tmp_path / "out"
    for i in range(40):
        for k in range(2):
            _img(src / "train" / "PNEUMONIA" / f"person{i}_bacteria_{i * 10 + k}.jpeg")
        _img(src / "val" / "NORMAL" / f"IM-{i:04d}-0001.jpeg")
    pdset.main(["chest_xray", "--src", str(src), "--out", str(out)])
    person_split = {}
    for p in out.rglob("*.jpeg"):
        if p.stem.startswith("person"):
            pid = p.stem.split("_")[0]
            assert person_split.setdefault(pid, p.relative_to(out).parts[0]) == p.relative_to(out).parts[0]
    assert {p.name for p in (out / "train").iterdir()} == {"normal", "pneumonia"}


def test_adapter_odir_multilabel_keeps_both_eyes_together(tmp_path):
    import pandas as pd
    import prepare_dataset as pdset

    src, out = tmp_path / "odir", tmp_path / "out"
    rows = []
    for i in range(80):
        flags = {"N": 0, "D": 0, "G": 0, "C": 0, "A": 0, "H": 0, "M": 0, "O": 0}
        flags["N" if i % 3 == 0 else "D"] = 1
        if i % 7 == 0:
            flags["M"] = 1
        rows.append({"ID": i, "Patient Age": 40 + i % 30, "Patient Sex": "Female" if i % 2 else "Male",
                     "Left-Fundus": f"{i}_left.jpg", "Right-Fundus": f"{i}_right.jpg", **flags})
        _img(src / "imgs" / f"{i}_left.jpg"); _img(src / "imgs" / f"{i}_right.jpg")
    pd.DataFrame(rows).to_excel(src / "ODIR-5K_Training_Annotations.xlsx", index=False)
    pdset.main(["odir", "--src", str(src), "--out", str(out)])
    seen = {}
    for split in ("train", "val", "test"):
        for r in csv.DictReader(open(out / split / "labels.csv")):
            pid = r["file"].split("_")[0]
            assert seen.setdefault(pid, split) == split
    assert any("|" in r["labels"] for r in csv.DictReader(open(out / "train" / "labels.csv")))


def test_adapter_errors_are_clear_not_silent(tmp_path):
    import prepare_dataset as pdset

    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit) as e:
        pdset.main(["aptos", "--src", str(tmp_path / "empty"), "--out", str(tmp_path / "o")])
    assert "train.csv" in str(e.value)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "train.csv").write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit) as e:
        pdset.main(["aptos", "--src", str(bad), "--out", str(tmp_path / "o")])
    assert "id_code" in str(e.value)


# ----------------------------------------------------------- train -> load -> predict -> evaluate
def _toy_split(root: Path, multilabel=False):
    rng = np.random.default_rng(0)
    for split, n in (("train", 24), ("val", 12), ("test", 12)):
        rows = []
        for i in range(n):
            pos = i % 2
            img = np.clip(rng.normal(60 if not pos else 190, 12, (72, 72, 3)), 0, 255).astype(np.uint8)
            name = f"{split}_{i}.jpg"
            if multilabel:
                d = root / split
            else:
                d = root / split / ("bright" if pos else "dark")
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / name), img)
            rows.append([name, "bright" if pos else "dark"])
        if multilabel:
            with open(root / split / "labels.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["file", "labels"]); w.writerows(rows)


@pytest.mark.parametrize("multilabel", [False, True])
def test_train_calibrates_saves_ood_and_model_loads(tmp_path, monkeypatch, multilabel):
    import train_classifier

    data, out = tmp_path / "data", tmp_path / "models"
    _toy_split(data, multilabel)
    argv = ["train", "--data", str(data), "--name", "toy", "--arch", "resnet18", "--img-size", "64", "--epochs", "2",
            "--batch", "8", "--concern", "bright", "--out", str(out), "--device", "cpu", "--workers", "0",
            "--dataset-note", "toy"] + (["--multilabel"] if multilabel else [])
    monkeypatch.setattr(sys, "argv", argv)
    train_classifier.main()
    spec = json.loads((out / "toy.json").read_text())
    assert spec["temperature"] > 0 and spec["multilabel"] == multilabel and "threshold_source" in spec
    assert (out / "toy.ood.npz").exists()

    from app.screening.classifier import ScreeningRegistry

    m = ScreeningRegistry(str(out), "cpu").get("toy")
    assert m.ood is not None and m.provenance()["ood_guard"]
    img = cv2.imread(str(next((data / "val").rglob("*.jpg"))))
    r = m.predict(img, explain=True)
    assert r.ood is not None and r.heatmap_jpeg
    weird = np.random.default_rng(5).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    cv2.circle(weird, (150, 150), 80, (0, 255, 255), -1)          # nothing like the training set
    assert m.predict(weird).triage in ("INCONCLUSIVE", "LOW", "MODERATE", "HIGH")


def test_evaluate_with_subgroups_flags_a_weak_group(tmp_path, monkeypatch, capsys):
    import evaluate_classifier

    d = tmp_path / "models"
    d.mkdir()
    torch.manual_seed(0)
    net = build_arch("resnet18", 2)
    torch.save(net.state_dict(), d / "demo.pt")
    (d / "demo.json").write_text(json.dumps({"title": "Demo", "labels": ["benign", "suspicious"], "concern_labels": ["suspicious"],
                                             "arch": "resnet18", "input_size": 64, "min_confidence": 0.0,
                                             "moderate_threshold": 0.5, "high_threshold": 0.9, "min_sharpness": 0, "min_brightness": 0, "min_side": 16}))
    test = tmp_path / "test"
    rng = np.random.default_rng(3)
    subs = [["file", "split", "sex"]]
    for cls in ("benign", "suspicious"):
        for i in range(40):
            p = test / cls / f"{cls}_{i}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
            subs.append([p.name, "test", "female" if i % 2 else "male"])
    sub_csv = tmp_path / "subgroups.csv"
    with open(sub_csv, "w", newline="") as f:
        csv.writer(f).writerows(subs)
    rep = tmp_path / "rep.json"
    monkeypatch.setattr(sys, "argv", ["e", "--name", "demo", "--data", str(test), "--model-dir", str(d), "--subgroups", str(sub_csv),
                                      "--report", str(rep), "--boot", "100"])
    evaluate_classifier.main()
    out = capsys.readouterr().out
    r = json.loads(rep.read_text())
    assert "Subgroup: sex" in out and set(r["subgroups"]["sex"]) == {"female", "male"}
    assert "ece" in r and r["auc"]["ci95"]


# --------------------------------------------------------------------------- patients API
@pytest.fixture
def demo_client(tmp_path):
    s = Settings(_env_file=None, auth_required=False, allow_anonymous_patients=True, device="cpu", yolo_weights="__none__.pt",
                 anomaly_bank_dir=str(tmp_path / "a"), screening_model_dir=str(tmp_path / "s"))
    d = Path(s.screening_model_dir)
    d.mkdir(parents=True)
    torch.save(build_arch("resnet18", 2).state_dict(), d / "retina_dr.pt")
    (d / "retina_dr.json").write_text(json.dumps({
        "title": "DR", "labels": ["no_dr", "moderate"], "concern_labels": ["moderate"], "arch": "resnet18", "input_size": 64,
        "min_confidence": 0.0, "min_sharpness": 0, "min_brightness": 0, "min_side": 16, "temperature": 1.7,
        "training_data": "toy", "validation": {"n_images": 10, "accuracy": 0.9, "macro_auc": 0.95, "note": "internal"}}))
    with TestClient(create_app(s, store=MemoryStore())) as c:
        yield c


def test_patient_requires_consent_and_cascades_delete(demo_client):
    c = demo_client
    assert c.post("/api/v1/patients", json={"full_name": "A B", "consent_given": False}).status_code == 422
    assert c.post("/api/v1/patients", json={"full_name": "A B"}).status_code == 422
    p = c.post("/api/v1/patients", json={"full_name": "Asha Rao", "age_years": 52, "sex": "female", "consent_given": True})
    assert p.status_code == 201
    pid = p.json()["id"]
    r = c.post("/api/v1/screen/retina_dr", data={"patient_id": pid},
               files={"file": ("a.jpg", jpeg(make_phone(1)), "image/jpeg")})
    assert r.status_code == 200
    assert c.get(f"/api/v1/patients/{pid}/screenings").json()["count"] == 1
    assert c.post("/api/v1/screen/retina_dr", data={"patient_id": "nope"},
                  files={"file": ("a.jpg", jpeg(make_phone(1)), "image/jpeg")}).status_code == 404
    d = c.delete(f"/api/v1/patients/{pid}").json()
    assert d["deleted"] and d["screenings_deleted"] == 1
    assert c.get(f"/api/v1/patients/{pid}").status_code == 404
    log = c.get("/api/v1/patients/audit/log").json()["items"]
    actions = [a["action"] for a in log]
    assert {"patient.create", "screening.create", "patient.delete"} <= set(actions)
    assert "Asha" not in json.dumps(log)                                   # audit trail holds no personal data


def test_patients_are_not_stored_for_anonymous_by_default(client):
    r = client.post("/api/v1/patients", json={"full_name": "X", "consent_given": True})
    assert r.status_code == 403
    assert client.get("/api/v1/patients").status_code == 403


def test_patient_isolation_between_users(tmp_path):
    secret = "x" * 40
    s = Settings(_env_file=None, auth_required=True, supabase_jwt_secret=secret, device="cpu", yolo_weights="__none__.pt",
                 anomaly_bank_dir=str(tmp_path / "a"), screening_model_dir=str(tmp_path / "s"))
    import time

    tok = lambda sub: {"Authorization": "Bearer " + jwt.encode({"sub": sub, "aud": "authenticated", "exp": int(time.time()) + 600},  # noqa: E731
                                                                  secret, algorithm="HS256")}
    with TestClient(create_app(s, store=MemoryStore())) as c:
        pid = c.post("/api/v1/patients", json={"full_name": "Mine", "consent_given": True}, headers=tok("u1")).json()["id"]
        assert c.get(f"/api/v1/patients/{pid}", headers=tok("u1")).status_code == 200
        assert c.get(f"/api/v1/patients/{pid}", headers=tok("u2")).status_code == 404
        assert c.delete(f"/api/v1/patients/{pid}", headers=tok("u2")).status_code == 404
        assert c.get("/api/v1/patients", headers=tok("u2")).json()["count"] == 0
        assert c.get("/api/v1/patients").status_code == 401


# ----------------------------------------------------------------------------- report
def _text(pdf: bytes) -> str:
    import io

    from pypdf import PdfReader

    return " ".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)


def _finding(module, triage, p, heat=None):
    f = {"module": module, "triage": triage, "top_label": "x", "concern_probability": p, "probabilities": {"x": p, "y": 1 - p},
         "quality": {"ok": True, "warnings": []}}
    if heat:
        f["heatmap_image"] = heat
    return f


def test_report_pdf_content_and_levels(demo_client):
    import base64

    heat = "data:image/jpeg;base64," + base64.b64encode(jpeg(make_phone(2))).decode()
    body = {"patient": {"name": "Asha Rao", "age_years": 52, "sex": "female", "consent_given": True, "symptoms": ["blurred vision"]},
            "findings": [_finding("retina_dr", "HIGH", 0.8, heat), _finding("skin_lesion", "LOW", 0.05),
                         _finding("jaundice", "INCONCLUSIVE", 0.0)],
            "vitals": {"bpm": 74, "confidence": 0.8, "status": "ok"}}
    r = demo_client.post("/api/v1/report", json=body)
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content[:4] == b"%PDF"
    t = _text(r.content)
    for needle in ("Asha Rao", "EYE (RETINA)", "Ophthalmologist", "Professional evaluation recommended soon",
                   "not a medical diagnosis".lower(), "Could not be assessed", "Camera pulse", "did NOT check", "Clinician review"):
        assert needle.lower() in t.lower(), needle
    assert "Internal validation: 10 images" in t.replace("\n", " ")                     # provenance came from the server's model
    assert "temperature 1.70" in t.replace("\n", " ")
    low_only = demo_client.post("/api/v1/report", json={"findings": [_finding("skin_lesion", "LOW", 0.05)]})
    assert "No concerning pattern found by the checks done" in _text(low_only.content).replace("\n", " ")
    assert "NOT a clearance" in _text(low_only.content).replace("\n", " ")
    inc = demo_client.post("/api/v1/report", json={"findings": [_finding("skin_lesion", "LOW", 0.05), _finding("jaundice", "INCONCLUSIVE", 0)]})
    assert "Incomplete" in _text(inc.content)                       # a gap must never read as an all-clear


def test_report_survives_hostile_and_non_latin_input(demo_client):
    body = {"patient": {"name": "<b>x</b>&<script> अशोक 张伟", "symptoms": ["<i>"]},
            "findings": [{**_finding("retina_dr", "HIGH", 0.7), "heatmap_image": "data:image/jpeg;base64,@@@notbase64"}],
            "operator": "A" * 120}
    r = demo_client.post("/api/v1/report", json=body)
    assert r.status_code == 200 and r.content[:4] == b"%PDF"
    assert demo_client.post("/api/v1/report", json={"findings": []}).status_code == 422
    assert demo_client.post("/api/v1/report", json={"findings": [{**_finding("../x", "HIGH", 0.5)}]}).status_code == 422
    assert demo_client.post("/api/v1/report", json={"findings": [{**_finding("skin_lesion", "SCARY", 0.5)}]}).status_code == 422


def test_report_never_claims_diagnosis():
    from app.report.summary import describe_finding, summarise

    for tri in ("HIGH", "MODERATE", "LOW", "INCONCLUSIVE"):
        for m in CATALOG:
            row = describe_finding({"module": m, "triage": tri, "probabilities": {}, "quality": {"warnings": []}})
            txt = (row["english"] + row["hinglish"]).lower().replace("if you have symptoms", "")
            assert "you have" not in txt and "diagnosed with" not in txt and "confirmed" not in txt
    assert summarise([])["level"] == "INCOMPLETE"


def test_sample_report_is_clearly_synthetic(client):
    r = client.get("/api/v1/report/sample")
    assert r.status_code == 200
    t = _text(r.content)
    assert "SYNTHETIC SAMPLE" in t and "SAMPLE PATIENT" in t and "Every name, number and image" in t


def test_report_for_stored_patient(demo_client):
    pid = demo_client.post("/api/v1/patients", json={"full_name": "Stored Person", "consent_given": True, "age_years": 40}).json()["id"]
    r = demo_client.post("/api/v1/report", json={"patient_id": pid, "findings": [_finding("skin_lesion", "MODERATE", 0.4)]})
    assert r.status_code == 200 and "Stored Person" in _text(r.content)
    assert demo_client.post("/api/v1/report", json={"patient_id": "missing", "findings": [_finding("skin_lesion", "LOW", 0.1)]}).status_code == 404
    acts = [a["action"] for a in demo_client.get("/api/v1/patients/audit/log").json()["items"]]
    assert "report.create" in acts


def test_modules_doc_covers_every_catalog_module():
    doc = (Path(__file__).resolve().parent.parent / "MODULES.md").read_text()
    for name in CATALOG:
        assert f"`{name}`" in doc, f"MODULES.md is missing {name}"
