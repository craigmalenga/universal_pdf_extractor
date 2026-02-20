"""
/api/v1/jobs endpoints.
Queue management and job status.
"""

from fastapi import APIRouter, Depends, HTTPException
from redis import Redis

from app.config import settings
from app.dependencies import verify_api_key
from app.schemas.jobs import JobStatus, QueueStats

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"], dependencies=[Depends(verify_api_key)])


def _get_redis() -> Redis:
    """Get a Redis connection."""
    return Redis.from_url(settings.REDIS_URL)


@router.get("/queue/stats", response_model=QueueStats)
async def queue_stats():
    """Get current queue statistics."""
    try:
        from rq import Queue
        from rq.worker import Worker

        conn = _get_redis()
        q = Queue(settings.QUEUE_NAME, connection=conn)
        workers = Worker.all(connection=conn)

        return QueueStats(
            queue_name=settings.QUEUE_NAME,
            queued=len(q),
            started=q.started_job_registry.count,
            finished=q.finished_job_registry.count,
            failed=q.failed_job_registry.count,
            deferred=q.deferred_job_registry.count,
            workers=len(workers),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {str(e)}")


@router.get("/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a specific processing job."""
    try:
        from rq.job import Job

        conn = _get_redis()
        job = Job.fetch(job_id, connection=conn)

        return JobStatus(
            job_id=job_id,
            doc_id=job.args[0] if job.args else "",
            status=job.get_status(),
            enqueued_at=job.enqueued_at,
            started_at=job.started_at,
            ended_at=job.ended_at,
            error_message=str(job.exc_info) if job.exc_info else None,
            result=job.result if job.is_finished else None,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job not found: {str(e)}")
    
    