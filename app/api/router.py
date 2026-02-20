"""
Top-level API router.
Combines all sub-routers into a single router.
"""

from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.documents import router as documents_router
from app.api.jobs import router as jobs_router
from app.api.fingerprints import router as fingerprints_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(documents_router)
api_router.include_router(jobs_router)
api_router.include_router(fingerprints_router)