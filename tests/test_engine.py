import numpy as np

from app.inspection.anomaly import AnomalyModel
from app.inspection.cv_defects import assess_quality, find_defects
from app.inspection.temporal import TemporalSmoother

from .synth import jitter, make_phone, missing_part


def test_clean_product_has_no_serious_defects():
    defects = find_defects(make_phone(1))
    assert not [d for d in defects if d.severity in ("medium", "high")]


def test_crack_is_found_as_high_severity():
    defects = find_defects(make_phone(2, crack=True))
    assert any(d.type == "crack" and d.severity == "high" for d in defects)


def test_scratch_is_found():
    defects = find_defects(make_phone(3, scratch=True))
    assert any(d.type == "scratch" for d in defects)


def test_quality_gate_flags_blur_and_darkness():
    sharp = make_phone(4)
    import cv2

    blurry = cv2.GaussianBlur(sharp, (0, 0), 12)
    dark = (sharp * 0.1).astype(np.uint8)
    assert assess_quality(sharp)["ok"]
    assert not assess_quality(blurry)["ok"]
    assert not assess_quality(dark)["ok"]


def test_anomaly_bank_scores_defects_higher_and_roundtrips(tmp_path):
    good = [jitter(make_phone(100 + i), i) for i in range(12)]
    model = AnomalyModel.build(good, "phone", device="cpu", pretrained=False)

    ok_ratio = model.score(jitter(make_phone(900), 5)).ratio
    bad_ratio = model.score(jitter(missing_part(make_phone(901)), 6)).ratio
    assert bad_ratio > ok_ratio

    path = tmp_path / "phone.pt"
    model.save(path)
    reloaded = AnomalyModel.load(path, "cpu")
    again = reloaded.score(jitter(missing_part(make_phone(901)), 6)).ratio
    assert abs(again - bad_ratio) < 0.05


def test_temporal_smoother_ignores_single_bad_frame():
    s = TemporalSmoother()
    out = [s.update(v) for v in ["PASS", "PASS", "PASS", "FAIL", "PASS"]]
    assert out[-1] == "PASS"
    s = TemporalSmoother()
    out = [s.update(v) for v in ["PASS", "FAIL", "FAIL", "FAIL"]]
    assert out[-1] == "FAIL"


def test_fail_needs_trusted_evidence():
    """No bank for the product, a photo nothing like the bank, or an unlocated product: show the marks, but REVIEW."""
    import cv2  # noqa: F401
    from app.config import Settings
    from app.inspection.pipeline import InspectionPipeline
    from app.inspection.types import Defect, Detection

    img = make_phone(2, crack=True)
    h, w = img.shape[:2]
    crack = Defect("crack", "high", 0.9, (10, 10, 100, 20), "cv")

    def verdict(settings, source="yolo", ratio=None, has_model=True):
        pipe = InspectionPipeline(Settings(_env_file=None, device="cpu", yolo_weights="__none__.pt", **settings))
        det = Detection("cell phone", "phone", 0.9, (0, 0, w, h), source=source)
        item = pipe._verdict(det, (0, 0, w, h), [crack], ratio, has_model)
        return item.verdict, " ".join(item.reasons)

    assert verdict({})[0] == "FAIL"                                              # located, bank, in range: trusted
    v, why = verdict({}, has_model=False)
    assert v == "REVIEW" and "no trained reference" in why                      # no bank for this product
    assert verdict({"require_bank_for_fail": False}, has_model=False)[0] == "FAIL"   # opt out (controlled rig)
    v, why = verdict({}, ratio=8.0)
    assert v == "REVIEW" and "very different" in why                            # photo unlike the bank
    assert verdict({}, ratio=2.0)[0] == "FAIL"                                  # a real same-rig defect still fails
    assert verdict({}, source="classifier")[0] == "REVIEW"                      # product never located
