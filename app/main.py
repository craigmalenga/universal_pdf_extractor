"""
FastAPI application factory.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.api.router import api_router
from app.models.database import close_db
from app.observability.logging import setup_logging

# Startup print - visible in Railway logs immediately
print(f"[STARTUP] Statement Extraction Platform v{settings.APP_VERSION}", flush=True)
print(f"[STARTUP] PORT={os.environ.get('PORT', 'NOT SET')}", flush=True)
print(f"[STARTUP] DATABASE_URL={'SET' if settings.DATABASE_URL else 'NOT SET'}", flush=True)
print(f"[STARTUP] Python {sys.version}", flush=True)

# Templates
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    # Startup
    setup_logging()

    # Sentry init if configured
    if settings.SENTRY_DSN:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            integrations=[FastApiIntegration()],
            traces_sample_rate=0.1,
        )

    yield

    # Shutdown
    await close_db()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="UK Statement Extraction Platform",
        description="Automated extraction of transactions from UK bank statements and motor finance documents.",
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics endpoint
    if settings.PROMETHEUS_ENABLED:
        from prometheus_client import make_asgi_app
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

    # Include all API routes
    app.include_router(api_router)

    # Dashboard at /
    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "version": settings.APP_VERSION,
        })

    return app


# Application instance
app = create_app()