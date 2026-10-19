"""Central configuration, loaded from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_key: str = ""
    supabase_jwt_secret: str = ""
    supabase_bucket: str = "inspections"

    # Auth
    auth_required: bool = True

    # Inspection engine
    device: str = "auto"
    yolo_weights: str = "yolov8n.pt"
    yolo_conf: float = 0.35
    anomaly_image_size: int = 256
    anomaly_bank_dir: str = "models/anomaly"
    anomaly_fail_ratio: float = 1.0
    anomaly_review_ratio: float = 0.8
    # A FAIL needs evidence we trust. Plain photo checks (OpenCV lines) fire on bezels, keys and backgrounds, and on
    # uncontrolled photos they measured at chance, so without a trained bank for the product the best call is REVIEW.
    # A ratio this far above the threshold means "the photo is nothing like the bank's reference photos" (real snapshots
    # scored 6 to 9 against a bank built from drawings; a same-rig defect scores a few), so it is not damage evidence.
    require_bank_for_fail: bool = True
    anomaly_trust_ratio: float = 6.5
    # CPU limits for small hosts (see app/utils.py InferenceGate). 0 threads = 2 (or fewer cores).
    torch_threads: int = 0
    max_concurrent_inference: int = 2
    # Load every model in a background thread right after start-up (the Dockerfile turns this on; tests leave it off).
    warmup: bool = False
    # Small ONNX face detector (OpenCV YuNet) so close-up faces count as a person. Optional: missing file = skipped.
    face_model: str = "models/face/face_detection_yunet_2023mar.onnx"
    # Names products YOLO's COCO classes lack (headphones, tablets, watches). Optional: no file = YOLO only.
    product_head: str = "models/product/product_head.npz"
    product_min_conf: float = 0.55

    # Screening
    screening_model_dir: str = "models/screening"
    # Stored patient records contain personal data. Off by default: without a signed-in user the
    # patient form still works (and the PDF report can be made) but nothing is saved server-side.
    allow_anonymous_patients: bool = False

    # Server
    max_upload_mb: int = 15
    cors_origins: str = "*"
    log_level: str = "INFO"

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_key)

    @property
    def cors_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
