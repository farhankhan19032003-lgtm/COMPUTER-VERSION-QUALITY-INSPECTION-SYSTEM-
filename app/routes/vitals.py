"""Live vitals (pulse) from camera colour samples.

The browser averages the skin colour of each video frame itself and sends only numbers
(t, r, g, b, skin fraction), so no video or image leaves the device. Vitals are health
data and are NOT stored.

WebSocket /ws/vitals?token=   client -> {"samples": [[t_ms, r, g, b, skin_frac], ...]}
                              server -> estimate JSON after each message
POST /api/v1/vitals/estimate  same samples in one request, for batch/offline use
"""
from __future__ import annotations

import json
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ..auth import User, current_user, resolve_user
from ..vitals.rppg import RPPGEstimator

log = logging.getLogger(__name__)
router = APIRouter(tags=["vitals"])

MAX_SAMPLES_PER_MSG = 200
MAX_MSG_CHARS = 40_000
DISCLAIMER = ("Camera pulse estimate: not a medical device and not a diagnosis. "
              "It cannot detect internal disease. If you feel unwell, seek medical care.")


def _feed(est: RPPGEstimator, samples) -> int:
    n = 0
    for s in samples[:MAX_SAMPLES_PER_MSG]:
        try:
            t, r, g, b, skin = (float(x) for x in s[:5])
        except (TypeError, ValueError):
            continue
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255 and 0 <= skin <= 1 and t == t):
            continue
        est.add(t / 1000.0, r, g, b, skin)
        n += 1
    return n


class SamplesIn(BaseModel):
    samples: List[List[float]] = Field(..., max_length=5000, description="[t_ms, r, g, b, skin_frac] rows")


@router.post("/api/v1/vitals/estimate")
async def estimate(body: SamplesIn, user: User = Depends(current_user)):
    est = RPPGEstimator()
    for i in range(0, len(body.samples), MAX_SAMPLES_PER_MSG):
        _feed(est, body.samples[i:i + MAX_SAMPLES_PER_MSG])
    res = est.estimate()
    res["disclaimer"] = DISCLAIMER
    return res


@router.websocket("/ws/vitals")
async def vitals_ws(ws: WebSocket):
    try:
        resolve_user(ws.query_params.get("token"), ws.app.state.ctx.settings)
    except HTTPException as exc:
        await ws.close(code=4401, reason=str(exc.detail))
        return
    await ws.accept()
    est = RPPGEstimator()
    try:
        while True:
            raw = await ws.receive_text()
            if len(raw) > MAX_MSG_CHARS:
                await ws.send_text(json.dumps({"error": "Message too large"}))
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"error": "Invalid JSON"}))
                continue
            if msg.get("reset"):
                est.reset()
            _feed(est, msg.get("samples") or [])
            res = est.estimate()
            res["disclaimer"] = DISCLAIMER
            await ws.send_text(json.dumps(res))
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("Vitals socket error")
        try:
            await ws.close(code=1011)
        except Exception:
            pass
