"""Dashboard data: inspection metrics, daily trend, system warnings, edge stability."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request

from ..auth import User, current_user
from ..deps import AppState, get_state

router = APIRouter(prefix="/api/v1", tags=["analytics"])


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


@router.get("/analytics/summary")
async def summary(ctx: AppState = Depends(get_state), user: User = Depends(current_user)) -> Dict[str, Any]:
    rows = ctx.store.list_inspections(user.id, 1000)
    verdicts = Counter(r["verdict"] for r in rows)
    by_product: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_product[r.get("product_type", "unknown")][r["verdict"]] += 1

    today = datetime.now(timezone.utc).date()
    days = [today - timedelta(days=i) for i in range(13, -1, -1)]
    daily = {d.isoformat(): Counter() for d in days}
    for r in rows:
        try:
            key = _parse(r["created_at"]).date().isoformat()
        except Exception:
            continue
        if key in daily:
            daily[key][r["verdict"]] += 1

    snap = ctx.metrics.snapshot()
    status = ctx.pipeline.status()
    status["screening_installed"] = ctx.screening.installed()
    warnings: List[Dict[str, str]] = ctx.metrics.warnings(snap, status, ctx.store.name)

    return {
        "store": ctx.store.name,
        "history": {
            "total": len(rows),
            "verdicts": {v: verdicts.get(v, 0) for v in ("PASS", "REVIEW", "FAIL")},
            "by_product": {p: dict(c) for p, c in by_product.items()},
            "daily": [
                {"day": d, "PASS": c.get("PASS", 0), "REVIEW": c.get("REVIEW", 0), "FAIL": c.get("FAIL", 0)}
                for d, c in daily.items()
            ],
            "recent": [
                {
                    "id": r.get("id"),
                    "created_at": r.get("created_at"),
                    "product_type": r.get("product_type"),
                    "verdict": r.get("verdict"),
                    "score": r.get("score"),
                    "defects": [d.get("type") for d in (r.get("defects") or [])][:4],
                    "latency_ms": r.get("latency_ms"),
                }
                for r in rows[:12]
            ],
        },
        "runtime": snap,
        "models": status,
        "warnings": warnings,
        "events": ctx.store.list_events(10),
    }
