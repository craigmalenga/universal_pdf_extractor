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

    # Direct process endpoint — bypasses router import chain issues
    @app.post("/process/{doc_id}")
    async def process_document_direct(doc_id: str):
        """Process a single document inline. Bypasses worker."""
        try:
            from app.pipeline.orchestrator import DocumentPipeline
        except Exception as e:
            return {"error": f"Import failed: {e}"}
        try:
            pipeline = DocumentPipeline()
            result = await pipeline.process(doc_id)
            return result
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}

    @app.post("/process-all")
    async def process_all_direct():
        """Process all QUEUED documents inline."""
        from app.models.database import async_session_factory
        from sqlalchemy import select as sa_select
        from app.models.tables import Document as DocModel
        try:
            from app.pipeline.orchestrator import DocumentPipeline
        except Exception as e:
            return {"error": f"Import failed: {e}"}

        async with async_session_factory() as session:
            result = await session.execute(
                sa_select(DocModel).where(DocModel.status == "QUEUED")
            )
            docs = result.scalars().all()

        if not docs:
            return {"message": "No queued documents", "processed": 0}

        pipeline = DocumentPipeline()
        results = []
        for doc in docs:
            try:
                output = await pipeline.process(str(doc.doc_id))
                results.append({"doc_id": str(doc.doc_id), "status": output.get("status")})
            except Exception as e:
                results.append({"doc_id": str(doc.doc_id), "error": str(e)})
        return {"processed": len(results), "results": results}

    return app


# Application instance
app = create_app()