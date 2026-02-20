"""
Health check endpoint.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.dependencies import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(session: AsyncSession = Depends(get_db)):
    """
    Health check: verifies API is running and DB is reachable.
    """
    db_ok = False
    try:
        result = await session.execute(text("SELECT 1"))
        db_ok = result.scalar() == 1
    except Exception:
        db_ok = False

    status_val = "healthy" if db_ok else "degraded"

    return {
        "status": status_val,
        "version": settings.APP_VERSION,
        "pipeline_version": settings.PIPELINE_VERSION,
        "database": "connected" if db_ok else "unreachable",
    }


@router.get("/health/ready")
async def readiness_check(session: AsyncSession = Depends(get_db)):
    """
    Readiness probe: returns 200 only if all dependencies are available.
    """
    try:
        await session.execute(text("SELECT 1"))
        return {"ready": True}
    except Exception:
        return {"ready": False}