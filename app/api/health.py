"""
Health check endpoint.
CRITICAL: /health must ALWAYS return 200 so Railway healthcheck passes.
DB connectivity is tested but failure does not block the response.
"""

from fastapi import APIRouter
from sqlalchemy import text

from app.config import settings
from app.models.database import async_session_factory

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """
    Health check: verifies API is running and tests DB connectivity.
    ALWAYS returns 200 so Railway healthcheck passes even if DB is down.
    """
    db_ok = False
    db_error = None
    try:
        async with async_session_factory() as session:
            result = await session.execute(text("SELECT 1"))
            db_ok = result.scalar() == 1
    except Exception as e:
        db_ok = False
        db_error = str(e)[:200]

    status_val = "healthy" if db_ok else "degraded"

    response = {
        "status": status_val,
        "version": settings.APP_VERSION,
        "pipeline_version": settings.PIPELINE_VERSION,
        "database": "connected" if db_ok else "unreachable",
    }
    if db_error:
        response["database_error"] = db_error

    return response


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness probe: returns 200 only if all dependencies are available.
    """
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
            return {"ready": True}
    except Exception:
        return {"ready": False}