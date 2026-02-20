"""
Application configuration using pydantic-settings.
All settings read from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Central configuration for the statement extraction platform."""

    # ── Application ──────────────────────────────────────────
    APP_NAME: str = "statement-extraction"
    APP_VERSION: str = "0.1.0"
    PIPELINE_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Database ─────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/statements"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 5
    DB_ECHO: bool = False

    # ── Redis ────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    QUEUE_NAME: str = "extraction"
    JOB_TIMEOUT_SECONDS: int = 600

    # ── Storage ──────────────────────────────────────────────
    ARTIFACT_ROOT: str = "/data/artifacts"
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_MIME_TYPES: str = "application/pdf"

    # ── PDF Rendering ────────────────────────────────────────
    RENDER_DPI: int = 300
    POPPLER_PATH: Optional[str] = None

    # ── Extraction Engines ───────────────────────────────────
    # Feature flag: enable Google Document AI
    ENABLE_DOCAI: bool = False
    DOCAI_PROJECT_ID: Optional[str] = None
    DOCAI_LOCATION: str = "eu"
    DOCAI_PROCESSOR_ID: Optional[str] = None

    # Feature flag: enable Tesseract fallback
    ENABLE_TESSERACT: bool = True
    TESSERACT_CMD: str = "tesseract"

    # ── Confidence Thresholds ────────────────────────────────
    CONFIDENCE_PASS_THRESHOLD: float = 0.85
    CONFIDENCE_WARN_THRESHOLD: float = 0.70
    CONFIDENCE_FAIL_THRESHOLD: float = 0.50

    # ── Balance Solver ───────────────────────────────────────
    BALANCE_TOLERANCE_EXACT: float = 0.00
    BALANCE_TOLERANCE_PENNY: float = 0.01
    BALANCE_TOLERANCE_ROUNDING: float = 0.05

    # ── Webhooks ─────────────────────────────────────────────
    WEBHOOK_TIMEOUT_SECONDS: int = 30
    WEBHOOK_MAX_RETRIES: int = 3

    # ── Observability ────────────────────────────────────────
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_ENABLED: bool = True

    # ── Security ─────────────────────────────────────────────
    API_KEY: Optional[str] = None
    CORS_ORIGINS: str = "*"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Singleton instance
settings = Settings()