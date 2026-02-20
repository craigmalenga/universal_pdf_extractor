"""
Worker entry point.
Run with: python -m app.worker.runner
"""

import sys

from redis import Redis
from rq import Worker

from app.config import settings
from app.observability.logging import setup_logging


def main():
    """Start the RQ worker."""
    setup_logging()

    conn = Redis.from_url(settings.REDIS_URL)
    worker = Worker(
        queues=[settings.QUEUE_NAME],
        connection=conn,
        name=f"extraction-worker-{settings.APP_VERSION}",
    )

    print(f"Starting worker on queue '{settings.QUEUE_NAME}'...")
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()