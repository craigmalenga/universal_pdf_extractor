"""
RQ job functions for document extraction pipeline.
These are the entry points that the worker calls.
"""

import structlog
from redis import Redis
from rq import Queue

from app.config import settings

logger = structlog.get_logger(__name__)


def get_queue() -> Queue:
    """Get the extraction job queue."""
    conn = Redis.from_url(settings.REDIS_URL)
    return Queue(settings.QUEUE_NAME, connection=conn)


def enqueue_extraction(doc_id: str, priority: int = 5) -> str:
    """
    Enqueue a document for extraction processing.
    Returns the job ID.
    """
    q = get_queue()
    job = q.enqueue(
        process_document_job,
        doc_id,
        job_timeout=settings.JOB_TIMEOUT_SECONDS,
        result_ttl=86400,  # Keep results for 24 hours
        failure_ttl=604800,  # Keep failures for 7 days
    )
    logger.info("job_enqueued", doc_id=doc_id, job_id=job.id, priority=priority)
    return job.id


def process_document_job(doc_id: str) -> dict:
    """
    Main job function: process a document through the extraction pipeline.
    This runs inside the RQ worker process.
    """
    import asyncio

    logger.info("job_started", doc_id=doc_id)

    try:
        result = asyncio.run(_process_document_async(doc_id))
        logger.info("job_completed", doc_id=doc_id, status=result.get("status"))
        return result
    except Exception as e:
        logger.error("job_failed", doc_id=doc_id, error=str(e))
        raise


async def _process_document_async(doc_id: str) -> dict:
    """
    Async wrapper for document processing.
    Runs the pipeline orchestrator end-to-end.
    """
    from app.pipeline.orchestrator import DocumentPipeline

    pipeline = DocumentPipeline()
    result = await pipeline.process(doc_id)
    return result