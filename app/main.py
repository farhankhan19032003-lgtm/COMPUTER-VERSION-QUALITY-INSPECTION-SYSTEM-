"""FastAPI application factory."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import Settings, get_settings
from .db import Store, build_store
from .deps import AppState
from .inspection.pipeline import InspectionPipeline
from .metrics import Metrics
from .routes import analytics, diagnostics, feedback, inspect, patients, report, screening, stream, vitals
from .screening.classifier import ScreeningRegistry
from .utils import configure_runtime, resolve_device

STATIC = Path(__file__).parent / "static"


def create_app(settings: Optional[Settings] = None, store: Optional[Store] = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log = logging.getLogger("visioninspect")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        device = resolve_device(settings.device)
        runtime = configure_runtime(settings.torch_threads, settings.max_concurrent_inference)
        st = store or build_store(settings)
        pipeline = InspectionPipeline(settings)
        app.state.ctx = AppState(
            settings=settings,
            store=st,
            pipeline=pipeline,
            screening=ScreeningRegistry(settings.screening_model_dir, settings.device),
            metrics=Metrics(device),
        )
        app.state.runtime = runtime
        st.log_event("INFO", "Service started", {"device": device, **runtime, **pipeline.status()})
        log.info("VisionInspect ready (device=%s, store=%s)", device, st.name)
        yield
        st.log_event("INFO", "Service stopped")

    app = FastAPI(
        title="VisionInspect",
        version="1.0.0",
        description="Computer-vision quality inspection and image-based health screening.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    app.include_router(inspect.router)
    app.include_router(feedback.router)
    app.include_router(screening.router)
    app.include_router(analytics.router)
    app.include_router(stream.router)
    app.include_router(diagnostics.router)
    app.include_router(vitals.router)
    app.include_router(patients.router)
    app.include_router(report.router)

    @app.get("/health", tags=["system"])
    async def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["system"])
    async def ready(request: Request):
        ctx: AppState = request.app.state.ctx
        ok = ctx.store.ping()
        body = {
            "status": "ready" if ok else "degraded", "store": ctx.store.name, **ctx.pipeline.status(),
            # what actually loaded, so a deploy can be checked without signing in
            "product_classifier": bool(ctx.pipeline.detector.classifier.ok),
            "screening_models": ctx.screening.installed(),
            **getattr(request.app.state, "runtime", {}),
        }
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/api/v1/config", tags=["system"])
    async def public_config(request: Request):
        """Public values the dashboard needs to sign users in. Never includes secrets."""
        s = request.app.state.ctx.settings
        return {
            "auth_required": s.auth_required,
            "supabase_url": s.supabase_url or None,
            "supabase_anon_key": s.supabase_anon_key or None,
            "max_upload_mb": s.max_upload_mb,
        }

    @app.get("/", include_in_schema=False)
    async def dashboard():
        return FileResponse(STATIC / "dashboard.html")

    return app


app = create_app()
