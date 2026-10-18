"""Persistence layer.

SupabaseStore  -> Postgres tables + private Storage bucket (production).
MemoryStore    -> in-process fallback when Supabase isn't configured (demo / tests).

Both expose the same interface so routes never care which one is active.
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Protocol

from .config import Settings

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store(Protocol):
    name: str

    def upload_image(self, user_id: Optional[str], data: bytes, suffix: str = "jpg") -> Optional[str]: ...
    def signed_url(self, path: Optional[str], expires_in: int = 3600) -> Optional[str]: ...
    def insert_inspection(self, row: Dict[str, Any]) -> Dict[str, Any]: ...
    def insert_screening(self, row: Dict[str, Any]) -> Dict[str, Any]: ...
    def get_inspection(self, user_id: Optional[str], inspection_id: str) -> Optional[Dict[str, Any]]: ...
    def upsert_feedback(self, user_id: Optional[str], inspection_id: str, model_verdict: str,
                        reviewer_verdict: str, note: Optional[str]) -> Dict[str, Any]: ...
    def list_feedback(self, user_id: Optional[str], limit: int = 1000) -> List[Dict[str, Any]]: ...
    def list_inspections(self, user_id: Optional[str], limit: int = 50) -> List[Dict[str, Any]]: ...
    def list_screenings(self, user_id: Optional[str], limit: int = 50) -> List[Dict[str, Any]]: ...
    def insert_diagnostic(self, row: Dict[str, Any]) -> Dict[str, Any]: ...
    def list_diagnostics(self, user_id: Optional[str], limit: int = 50) -> List[Dict[str, Any]]: ...
    def log_event(self, level: str, message: str, meta: Optional[Dict[str, Any]] = None) -> None: ...
    def list_events(self, limit: int = 20) -> List[Dict[str, Any]]: ...
    def ping(self) -> bool: ...
    # patient records (personal data: consent required, every access audited)
    def insert_patient(self, row: Dict[str, Any]) -> Dict[str, Any]: ...
    def get_patient(self, user_id: Optional[str], patient_id: str) -> Optional[Dict[str, Any]]: ...
    def list_patients(self, user_id: Optional[str], limit: int = 100) -> List[Dict[str, Any]]: ...
    def delete_patient(self, user_id: Optional[str], patient_id: str) -> bool: ...
    def list_patient_screenings(self, user_id: Optional[str], patient_id: str, limit: int = 200) -> List[Dict[str, Any]]: ...
    def audit(self, user_id: Optional[str], action: str, patient_id: Optional[str] = None,
              meta: Optional[Dict[str, Any]] = None) -> None: ...
    def list_audit(self, user_id: Optional[str], limit: int = 100) -> List[Dict[str, Any]]: ...


# --------------------------------------------------------------------------- memory
class MemoryStore:
    name = "memory"

    def __init__(self, max_rows: int = 5000) -> None:
        self._lock = threading.Lock()
        self._inspections: Deque[Dict[str, Any]] = deque(maxlen=max_rows)
        self._screenings: Deque[Dict[str, Any]] = deque(maxlen=max_rows)
        self._diagnostics: Deque[Dict[str, Any]] = deque(maxlen=max_rows)
        self._feedback: Dict[tuple, Dict[str, Any]] = {}
        self._events: Deque[Dict[str, Any]] = deque(maxlen=500)
        self._patients: Dict[str, Dict[str, Any]] = {}
        self._audit: Deque[Dict[str, Any]] = deque(maxlen=5000)
        self._files: Dict[str, bytes] = {}

    def upload_image(self, user_id, data, suffix="jpg"):
        path = f"{user_id or 'anon'}/{datetime.now(timezone.utc):%Y-%m-%d}/{uuid.uuid4()}.{suffix}"
        with self._lock:
            if len(self._files) > 200:  # keep memory bounded
                self._files.pop(next(iter(self._files)))
            self._files[path] = data
        return path

    def signed_url(self, path, expires_in=3600):
        return None  # no hosted files in memory mode

    def _insert(self, bucket: Deque[Dict[str, Any]], row: Dict[str, Any]) -> Dict[str, Any]:
        row = {**row, "id": str(uuid.uuid4()), "created_at": _now()}
        with self._lock:
            bucket.appendleft(row)
        return row

    def insert_inspection(self, row):
        return self._insert(self._inspections, row)

    def insert_screening(self, row):
        return self._insert(self._screenings, row)

    def _list(self, bucket: Deque[Dict[str, Any]], user_id: Optional[str], limit: int):
        with self._lock:
            rows = [r for r in bucket if user_id is None or r.get("user_id") == user_id]
        return rows[:limit]

    def list_inspections(self, user_id, limit=50):
        return self._list(self._inspections, user_id, limit)

    def get_inspection(self, user_id, inspection_id):
        with self._lock:
            return next((r for r in self._inspections if r["id"] == inspection_id
                         and (user_id is None or r.get("user_id") == user_id)), None)

    def upsert_feedback(self, user_id, inspection_id, model_verdict, reviewer_verdict, note):
        key = (inspection_id, user_id)
        with self._lock:
            old = self._feedback.get(key)
            row = {
                "id": old["id"] if old else str(uuid.uuid4()),
                "inspection_id": inspection_id, "user_id": user_id,
                "model_verdict": model_verdict, "reviewer_verdict": reviewer_verdict, "note": note,
                "created_at": old["created_at"] if old else _now(), "updated_at": _now(),
            }
            self._feedback[key] = row
        return row

    def list_feedback(self, user_id, limit=1000):
        with self._lock:
            rows = [r for r in self._feedback.values() if user_id is None or r.get("user_id") == user_id]
        return sorted(rows, key=lambda r: r["updated_at"], reverse=True)[:limit]

    def list_screenings(self, user_id, limit=50):
        return self._list(self._screenings, user_id, limit)

    def insert_diagnostic(self, row):
        return self._insert(self._diagnostics, row)

    def list_diagnostics(self, user_id, limit=50):
        return self._list(self._diagnostics, user_id, limit)

    def log_event(self, level, message, meta=None):
        with self._lock:
            self._events.appendleft({"level": level, "message": message, "meta": meta or {}, "created_at": _now()})

    def list_events(self, limit=20):
        with self._lock:
            return list(self._events)[:limit]

    def ping(self) -> bool:
        return True

    # patients -----------------------------------------------------------------
    def insert_patient(self, row):
        row = {**row, "id": str(uuid.uuid4()), "created_at": _now()}
        with self._lock:
            self._patients[row["id"]] = row
        return row

    def get_patient(self, user_id, patient_id):
        with self._lock:
            r = self._patients.get(patient_id)
        return r if r and (user_id is None or r.get("user_id") == user_id) else None

    def list_patients(self, user_id, limit=100):
        with self._lock:
            rows = [r for r in self._patients.values() if user_id is None or r.get("user_id") == user_id]
        return sorted(rows, key=lambda r: r["created_at"], reverse=True)[:limit]

    def delete_patient(self, user_id, patient_id):
        if self.get_patient(user_id, patient_id) is None:
            return False
        with self._lock:
            self._patients.pop(patient_id, None)
            keep = [r for r in self._screenings if r.get("patient_id") != patient_id]
            self._screenings.clear()
            self._screenings.extend(keep)
        return True

    def list_patient_screenings(self, user_id, patient_id, limit=200):
        with self._lock:
            rows = [r for r in self._screenings if r.get("patient_id") == patient_id
                    and (user_id is None or r.get("user_id") == user_id)]
        return rows[:limit]

    def audit(self, user_id, action, patient_id=None, meta=None):
        with self._lock:
            self._audit.appendleft({"id": str(uuid.uuid4()), "user_id": user_id, "action": action,
                                    "patient_id": patient_id, "meta": meta or {}, "created_at": _now()})

    def list_audit(self, user_id, limit=100):
        with self._lock:
            rows = [r for r in self._audit if user_id is None or r.get("user_id") == user_id]
        return rows[:limit]


# ------------------------------------------------------------------------- supabase
class SupabaseStore:
    name = "supabase"

    def __init__(self, settings: Settings) -> None:
        from supabase import create_client  # imported lazily so memory mode needs no network

        self._client = create_client(settings.supabase_url, settings.supabase_service_key)
        self._bucket = settings.supabase_bucket

    # storage ----------------------------------------------------------------
    def upload_image(self, user_id, data, suffix="jpg"):
        path = f"{user_id or 'anon'}/{datetime.now(timezone.utc):%Y-%m-%d}/{uuid.uuid4()}.{suffix}"
        try:
            self._client.storage.from_(self._bucket).upload(
                path, data, {"content-type": "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"}
            )
            return path
        except Exception:
            log.exception("Storage upload failed")
            return None

    def signed_url(self, path, expires_in=3600):
        if not path:
            return None
        try:
            res = self._client.storage.from_(self._bucket).create_signed_url(path, expires_in)
            return res.get("signedURL") or res.get("signedUrl")
        except Exception:
            log.exception("Could not sign URL for %s", path)
            return None

    # tables -----------------------------------------------------------------
    def _insert(self, table: str, row: Dict[str, Any]) -> Dict[str, Any]:
        res = self._client.table(table).insert(row).execute()
        return res.data[0] if res.data else row

    def insert_inspection(self, row):
        return self._insert("inspections", row)

    def insert_screening(self, row):
        return self._insert("screenings", row)

    def _list(self, table: str, user_id: Optional[str], limit: int):
        q = self._client.table(table).select("*").order("created_at", desc=True).limit(limit)
        if user_id:
            q = q.eq("user_id", user_id)
        return q.execute().data or []

    def list_inspections(self, user_id, limit=50):
        return self._list("inspections", user_id, limit)

    def get_inspection(self, user_id, inspection_id):
        q = self._client.table("inspections").select("*").eq("id", inspection_id)
        if user_id:
            q = q.eq("user_id", user_id)
        data = q.limit(1).execute().data
        return data[0] if data else None

    def upsert_feedback(self, user_id, inspection_id, model_verdict, reviewer_verdict, note):
        t = self._client.table("inspection_feedback")
        q = t.select("id").eq("inspection_id", inspection_id)
        q = q.eq("user_id", user_id) if user_id else q.is_("user_id", "null")
        old = q.limit(1).execute().data
        fields = {"model_verdict": model_verdict, "reviewer_verdict": reviewer_verdict, "note": note,
                  "updated_at": _now()}
        if old:
            res = t.update(fields).eq("id", old[0]["id"]).execute()
        else:
            res = t.insert({"inspection_id": inspection_id, "user_id": user_id, **fields}).execute()
        return res.data[0] if res.data else {"inspection_id": inspection_id, **fields}

    def list_feedback(self, user_id, limit=1000):
        q = self._client.table("inspection_feedback").select("*").order("updated_at", desc=True).limit(limit)
        if user_id:
            q = q.eq("user_id", user_id)
        return q.execute().data or []

    def list_screenings(self, user_id, limit=50):
        return self._list("screenings", user_id, limit)

    def insert_diagnostic(self, row):
        return self._insert("phone_diagnostics", row)

    def list_diagnostics(self, user_id, limit=50):
        return self._list("phone_diagnostics", user_id, limit)

    def log_event(self, level, message, meta=None):
        try:
            self._client.table("system_events").insert(
                {"level": level, "message": message, "meta": meta or {}}
            ).execute()
        except Exception:
            log.exception("Could not write system event")

    def list_events(self, limit=20):
        try:
            return (
                self._client.table("system_events").select("*").order("created_at", desc=True).limit(limit).execute().data
                or []
            )
        except Exception:
            log.exception("Could not read system events")
            return []

    # patients -----------------------------------------------------------------
    def insert_patient(self, row):
        return self._insert("patients", row)

    def get_patient(self, user_id, patient_id):
        q = self._client.table("patients").select("*").eq("id", patient_id)
        if user_id:
            q = q.eq("user_id", user_id)
        data = q.limit(1).execute().data
        return data[0] if data else None

    def list_patients(self, user_id, limit=100):
        return self._list("patients", user_id, limit)

    def delete_patient(self, user_id, patient_id):
        if self.get_patient(user_id, patient_id) is None:
            return False
        # screenings.patient_id has ON DELETE CASCADE in schema.sql, so this removes them too.
        self._client.table("patients").delete().eq("id", patient_id).execute()
        return True

    def list_patient_screenings(self, user_id, patient_id, limit=200):
        q = (self._client.table("screenings").select("*").eq("patient_id", patient_id)
             .order("created_at", desc=True).limit(limit))
        if user_id:
            q = q.eq("user_id", user_id)
        return q.execute().data or []

    def audit(self, user_id, action, patient_id=None, meta=None):
        try:
            self._client.table("audit_log").insert(
                {"user_id": user_id, "action": action, "patient_id": patient_id, "meta": meta or {}}).execute()
        except Exception:
            log.exception("Could not write audit log")

    def list_audit(self, user_id, limit=100):
        try:
            return self._list("audit_log", user_id, limit)
        except Exception:
            log.exception("Could not read audit log")
            return []

    def ping(self) -> bool:
        try:
            self._client.table("system_events").select("id").limit(1).execute()
            return True
        except Exception:
            return False


def build_store(settings: Settings) -> Store:
    if settings.supabase_enabled:
        try:
            store = SupabaseStore(settings)
            log.info("Using Supabase store")
            return store
        except Exception:
            log.exception("Supabase init failed; falling back to in-memory store")
    else:
        log.warning("Supabase not configured: using in-memory store (data is lost on restart)")
    return MemoryStore()
