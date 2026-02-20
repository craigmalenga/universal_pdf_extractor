"""
Pydantic request/response schemas for the /api/v1/jobs endpoints.
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class JobStatus(BaseModel):
    """Status of a processing job."""
    job_id: str
    doc_id: str
    status: str  # queued, started, finished, failed, deferred
    enqueued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[dict] = None


class QueueStats(BaseModel):
    """Queue statistics."""
    queue_name: str
    queued: int
    started: int
    finished: int
    failed: int
    deferred: int
    workers: int