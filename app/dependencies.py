"""
FastAPI dependency injection.
Provides DB sessions, artifact store, and API key validation.
"""

from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.database import get_session
from app.storage.artifact_store import ArtifactStore


# ── Singleton instances ──────────────────────────────────────
_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Get or create the artifact store singleton."""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore()
    return _artifact_store


async def get_db() -> AsyncSession:
    """Yield an async DB session."""
    async for session in get_session():
        yield session


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[str]:
    """
    Verify API key if configured.
    If API_KEY is not set, all requests are allowed (dev mode).
    """
    if settings.API_KEY is None:
        return None

    if x_api_key is None or x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key