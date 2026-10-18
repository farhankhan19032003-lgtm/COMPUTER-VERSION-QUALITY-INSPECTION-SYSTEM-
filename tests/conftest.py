import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.db import MemoryStore  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        auth_required=False,
        device="cpu",
        yolo_weights="__not_installed__.pt",       # forces the classical fallback detector
        anomaly_bank_dir=str(tmp_path / "anomaly"),
        screening_model_dir=str(tmp_path / "screening"),
        supabase_url="",
        supabase_service_key="",
        max_upload_mb=1,
        require_bank_for_fail=False,               # the synthetic phones stand for a controlled rig with a bank
    )


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient

    app = create_app(settings, store=MemoryStore())
    with TestClient(app) as c:
        yield c
