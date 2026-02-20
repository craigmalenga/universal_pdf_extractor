"""
Review queue management.
Spec reference: BATON_PASS_BUILD_SPEC Part 27.
"""

import uuid
from typing import Optional

import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tables import ReviewQueueItem
from app.models.enums import ReviewStatus

logger = structlog.get_logger(__name__)


async def route_to_review(
    session: AsyncSession,
    run_id: str,
    doc_id: str,
    reason: str,
    reason_details: Optional[str] = None,
    segment_id: Optional[str] = None,
    priority: int = 5,
) -> str:
    """
    Add a document/run to the review queue.
    Returns the review_id.
    """
    review_item = ReviewQueueItem(
        run_id=uuid.UUID(run_id),
        doc_id=uuid.UUID(doc_id),
        segment_id=uuid.UUID(segment_id) if segment_id else None,
        reason=reason,
        reason_details=reason_details,
        priority=priority,
        status=ReviewStatus.PENDING.value,
    )
    session.add(review_item)
    await session.flush()

    logger.info(
        "routed_to_review",
        review_id=str(review_item.review_id),
        doc_id=doc_id,
        reason=reason,
        priority=priority,
    )

    return str(review_item.review_id)


async def get_pending_reviews(
    session: AsyncSession,
    limit: int = 50,
    offset: int = 0,
) -> list[ReviewQueueItem]:
    """Get pending review items ordered by priority."""
    result = await session.execute(
        select(ReviewQueueItem)
        .where(ReviewQueueItem.status == ReviewStatus.PENDING.value)
        .order_by(ReviewQueueItem.priority, ReviewQueueItem.created_at)
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_review_queue_stats(session: AsyncSession) -> dict:
    """Get review queue statistics."""
    result = await session.execute(
        select(
            ReviewQueueItem.status,
            func.count(ReviewQueueItem.review_id),
        ).group_by(ReviewQueueItem.status)
    )
    stats = {row[0]: row[1] for row in result.all()}
    return {
        "pending": stats.get(ReviewStatus.PENDING.value, 0),
        "in_review": stats.get(ReviewStatus.IN_REVIEW.value, 0),
        "resolved": stats.get(ReviewStatus.RESOLVED.value, 0),
        "skipped": stats.get(ReviewStatus.SKIPPED.value, 0),
        "total": sum(stats.values()),
    }