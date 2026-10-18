"""Real-time inspection over WebSocket.

Protocol (ack-based, so slow inference never builds a frame backlog):
  client -> server : binary JPEG frame
  server -> client : JSON {verdict, smoothed_verdict, score, defects, latency_ms, annotated?}
The client sends the next frame after it receives the result for the previous one.

Query params: token (Supabase access token), product (hint), annotate (0/1).
"""
from __future__ import annotations

import base64
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from ..auth import resolve_user
from ..inspection.temporal import TemporalSmoother
from ..utils import decode_image
from ..deps import clean_product
from .inspect import persist_inspection, record
from fastapi import HTTPException

log = logging.getLogger(__name__)
router = APIRouter(tags=["stream"])

SAVE_INTERVAL_S = 2.0   # persist at most one non-PASS frame per interval per connection
MAX_FRAME_BYTES = 4 * 1024 * 1024


@router.websocket("/ws/stream")
async def stream(ws: WebSocket):
    ctx = ws.app.state.ctx
    try:
        user = resolve_user(ws.query_params.get("token"), ctx.settings)
        hint = clean_product(ws.query_params.get("product"))
    except HTTPException as exc:
        await ws.close(code=4401, reason=str(exc.detail))
        return
    annotate = ws.query_params.get("annotate", "1") != "0"

    await ws.accept()
    smoother = TemporalSmoother()
    last_saved = 0.0
    try:
        while True:
            data = await ws.receive_bytes()
            if len(data) > MAX_FRAME_BYTES:
                await ws.send_text(json.dumps({"error": "Frame too large"}))
                continue
            try:
                img = decode_image(data, max_side=1280)
                result = await run_in_threadpool(ctx.pipeline.inspect, img, hint, annotate)
            except ValueError as exc:
                await ws.send_text(json.dumps({"error": str(exc)}))
                continue
            except Exception:
                ctx.metrics.record_error()
                log.exception("Stream frame failed")
                await ws.send_text(json.dumps({"error": "Inspection failed"}))
                continue

            record(ctx, result, "stream")
            smoothed = smoother.update(result.verdict)

            saved = {}
            now = time.monotonic()
            if smoothed != "PASS" and now - last_saved >= SAVE_INTERVAL_S:
                last_saved = now
                saved = await run_in_threadpool(persist_inspection, ctx, user, result, data, "stream")

            body = result.to_dict()
            body["smoothed_verdict"] = smoothed
            body["id"] = saved.get("id")
            if annotate and result.annotated:
                body["annotated"] = base64.b64encode(result.annotated).decode("ascii")
            await ws.send_text(json.dumps(body))
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("Stream closed with error")
        try:
            await ws.close(code=1011)
        except Exception:
            pass
