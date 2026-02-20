"""
Prometheus metrics for the statement extraction platform.
"""

from prometheus_client import Counter, Histogram, Gauge


# ── Document Processing ─────────────────────────────────────
documents_uploaded_total = Counter(
    "documents_uploaded_total",
    "Total documents uploaded",
    ["doc_family"],
)

documents_processed_total = Counter(
    "documents_processed_total",
    "Total documents processed to completion",
    ["doc_family", "validation_status"],
)

documents_failed_total = Counter(
    "documents_failed_total",
    "Total documents that failed processing",
    ["error_code"],
)

document_processing_duration_seconds = Histogram(
    "document_processing_duration_seconds",
    "Time to process a document end-to-end",
    ["doc_family"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

# ── Pipeline Stages ──────────────────────────────────────────
pipeline_stage_duration_seconds = Histogram(
    "pipeline_stage_duration_seconds",
    "Time per pipeline stage",
    ["stage"],
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60],
)

# ── Extraction ───────────────────────────────────────────────
pages_extracted_total = Counter(
    "pages_extracted_total",
    "Total pages extracted",
    ["extraction_path", "engine_name"],
)

transactions_extracted_total = Counter(
    "transactions_extracted_total",
    "Total transactions extracted",
    ["direction"],
)

confidence_scores = Histogram(
    "confidence_scores",
    "Distribution of document confidence scores",
    ["doc_family"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

# ── Review Queue ─────────────────────────────────────────────
review_queue_depth = Gauge(
    "review_queue_depth",
    "Current number of items in review queue",
    ["status"],
)

# ── External API Costs ───────────────────────────────────────
external_api_cost_usd = Counter(
    "external_api_cost_usd_total",
    "Cumulative cost of external API calls in USD",
    ["engine_name", "operation"],
)

external_api_latency_seconds = Histogram(
    "external_api_latency_seconds",
    "Latency of external API calls",
    ["engine_name", "operation"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
)

# ── Worker ───────────────────────────────────────────────────
worker_jobs_active = Gauge(
    "worker_jobs_active",
    "Number of currently active worker jobs",
)

worker_queue_depth = Gauge(
    "worker_queue_depth",
    "Number of jobs waiting in queue",
)