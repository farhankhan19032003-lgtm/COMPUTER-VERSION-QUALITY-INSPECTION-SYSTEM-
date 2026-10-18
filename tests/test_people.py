"""People are detected like products but are never defect-checked, stored or counted."""
import numpy as np
import pytest

from app.config import Settings
from app.inspection.detector import ProductDetector, normalize_product
from app.inspection.pipeline import InspectionPipeline
from app.inspection.types import Detection

from .synth import jpeg, make_phone

IMG = make_phone(1)
H, W = IMG.shape[:2]
PERSON = Detection("person", "person", 0.95, (10, 10, W - 10, H - 10))   # big and confident
PHONE = Detection("cell phone", "phone", 0.6, (160, 100, 480, 700))


@pytest.fixture
def pipe(tmp_path):
    return InspectionPipeline(Settings(_env_file=None, device="cpu", yolo_weights="__none__.pt",
                                       anomaly_bank_dir=str(tmp_path / "a"), require_bank_for_fail=False))


def test_person_label_is_recognised():
    assert normalize_product("person") == "person"


def test_a_person_never_outranks_a_product():
    det = ProductDetector(Settings(_env_file=None, device="cpu", yolo_weights="__none__.pt"))
    det.yolo.model = object()                                  # pretend YOLO loaded
    det.yolo.detect = lambda img: [PERSON, PHONE]              # person is larger and more confident
    assert [d.product_type for d in det.detect(IMG)] == ["phone", "person"]
    assert det.detect(IMG, max_items=1)[0].product_type == "phone"   # what crop_product() uses for banks


def test_only_a_person_is_reported_as_person_not_unknown(pipe):
    pipe.detector.detect = lambda img, hint=None, max_items=3: [PERSON]
    r = pipe.inspect(IMG)
    assert r.product_type == "person" and r.people == 1
    assert r.items[0].product_type == "person" and r.defects == []
    assert any("person was detected" in x for x in r.reasons)
    assert r.advice == {} and r.annotated
    assert r.to_dict()["people"] == 1


def test_phone_in_hand_is_still_inspected_as_a_phone(pipe):
    pipe.detector.detect = lambda img, hint=None, max_items=3: [PHONE, PERSON]
    r = pipe.inspect(make_phone(2, crack=True))
    assert r.product_type == "phone" and r.people == 1
    assert [i.product_type for i in r.items] == ["phone"]
    assert r.verdict == "FAIL"


def test_unknown_only_when_nothing_at_all_is_found(pipe):
    pipe.detector.detect = lambda img, hint=None, max_items=3: []
    assert pipe.inspect(IMG).product_type == "unknown"


def test_person_photos_are_not_stored_or_counted(client):
    client.app.state.ctx.pipeline.detector.detect = lambda img, hint=None, max_items=3: [PERSON]
    r = client.post("/api/v1/inspect", files={"file": ("p.jpg", jpeg(IMG), "image/jpeg")}, data={"product_type": ""})
    body = r.json()
    assert r.status_code == 200 and body["product_type"] == "person" and body["id"] is None
    assert client.get("/api/v1/inspections").json()["count"] == 0
    s = client.get("/api/v1/analytics/summary").json()
    assert s["history"]["total"] == 0 and s["runtime"]["totals"]["inspections"] == 0


def test_chosen_product_is_not_replaced_by_a_person(pipe):
    """A hand holding the phone is a 'person' to YOLO. If the user picked 'phone', still inspect the phone."""
    det = pipe.detector
    det.yolo.model = object()
    det.yolo.detect = lambda img: [PERSON]                 # YOLO saw only the person
    got = det.detect(make_phone(2, crack=True), hint="phone")
    assert got and got[0].product_type == "phone"          # classical finder took over, hinted as phone
    assert pipe.inspect(make_phone(2, crack=True), "phone").product_type == "phone"
    assert pipe.inspect(IMG).product_type == "person"      # without a hint it is reported as a person


def test_a_part_seen_by_yolo_is_replaced_by_the_whole_the_classifier_names():
    """YOLO sees only the keyboard of a laptop; the whole-photo classifier says laptop (even at a low confidence)."""
    det = ProductDetector(Settings(_env_file=None, device="cpu", yolo_weights="__none__.pt"))
    det.yolo.model = object()
    det.yolo.detect = lambda img: [Detection("keyboard", "keyboard", 0.9, (50, 400, 600, 700)),
                                   Detection("tv", "monitor", 0.84, (80, 50, 560, 390))]   # the screen of the same laptop
    probs = {None: None, 0.45: ("laptop", 0.5)}                      # below the normal 0.55 cut-off, above the part-of-whole one
    det.classifier.ok = True
    det.classifier.predict = lambda img, min_conf=None: probs[min_conf]
    got = det.detect(IMG)
    assert [d.product_type for d in got] == ["laptop"] and got[0].source == "classifier"   # screen + keyboard = one laptop
    assert got[0].bbox == (50, 50, 600, 700)                                                # box covers both parts
    det.classifier.predict = lambda img, min_conf=None: ("tablet", 0.5) if min_conf else None   # not a whole of a keyboard
    assert det.detect(IMG)[0].product_type == "keyboard"


def test_one_product_seen_as_several_overlapping_boxes_is_one_item():
    """YOLO can box the same phone three times (whole phone, a shifted copy, the screen): that is one product."""
    det = ProductDetector(Settings(_env_file=None, device="cpu", yolo_weights="__none__.pt"))
    det.yolo.model = object()
    big = Detection("cell phone", "phone", 0.9, (100, 100, 600, 1100))
    shifted = Detection("cell phone", "phone", 0.8, (90, 500, 620, 1200))
    screen = Detection("cell phone", "phone", 0.7, (250, 600, 520, 1000))
    det.yolo.detect = lambda img: [big, shifted, screen]
    got = det.detect(IMG)
    assert [d.product_type for d in got] == ["phone"] and got[0].bbox == big.bbox
    # separate products of the same type stay separate
    left, right = Detection("laptop", "laptop", 0.9, (0, 0, 200, 200)), Detection("laptop", "laptop", 0.8, (300, 0, 500, 200))
    det.yolo.detect = lambda img: [left, right]
    assert len(det.detect(IMG, max_items=3)) == 2
