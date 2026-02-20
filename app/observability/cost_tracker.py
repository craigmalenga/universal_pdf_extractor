"""
Per-request cost instrumentation.
Decision D-011: Instrument, don't estimate. Measure reality.
"""

import time
from typing import Optional
from decimal import Decimal

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tables import CostEvent
from app.observability.metrics import external_api_cost_usd, external_api_latency_seconds

logger = structlog.get_logger(__name__)


class CostTracker:
    """Track costs of external API calls per document/run."""

    def __init__(self, session: AsyncSession, run_id: Optional[str] = None, doc_id: Optional[str] = None):
        self.session = session
        self.run_id = run_id
        self.doc_id = doc_id
        self._events: list[dict] = []

    async def record(
        self,
        engine_name: str,
        operation: str,
        page_count: int = 0,
        cost_usd: float = 0.0,
        cost_gbp: float = 0.0,
        latency_ms: int = 0,
    ) -> None:
        """Record a cost event to DB and Prometheus."""
        event = CostEvent(
            run_id=self.run_id,
            doc_id=self.doc_id,
            engine_name=engine_name,
            operation=operation,
            page_count=page_count,
            cost_usd=Decimal(str(cost_usd)),
            cost_gbp=Decimal(str(cost_gbp)),
            latency_ms=latency_ms,
        )
        self.session.add(event)

        # Prometheus metrics
        external_api_cost_usd.labels(
            engine_name=engine_name,
            operation=operation,
        ).inc(cost_usd)

        external_api_latency_seconds.labels(
            engine_name=engine_name,
            operation=operation,
        ).observe(latency_ms / 1000.0)

        self._events.append({
            "engine_name": engine_name,
            "operation": operation,
            "page_count": page_count,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
        })

        logger.info(
            "cost_event_recorded",
            engine_name=engine_name,
            operation=operation,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )

    def summary(self) -> dict:
        """Return summary of all cost events for this run."""
        total_cost_usd = sum(e["cost_usd"] for e in self._events)
        total_pages = sum(e["page_count"] for e in self._events)
        return {
            "total_cost_usd": round(total_cost_usd, 6),
            "total_pages": total_pages,
            "event_count": len(self._events),
            "events": self._events,
        }