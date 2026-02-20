"""
Abstract base class for all extraction engines.
Every engine must produce NormalizedPageExtraction.
"""

from abc import ABC, abstractmethod
from typing import Optional

from app.schemas.contracts import BBox, NormalizedPageExtraction


class ExtractionEngine(ABC):
    """
    Abstract base class for all extraction engines.

    Every engine must:
    1. Accept a document/page reference
    2. Return NormalizedPageExtraction
    3. Report its name and version
    4. Handle errors gracefully (raise EngineError, never crash)
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Unique identifier: 'pdfplumber', 'google_docai', 'tesseract'"""
        ...

    @property
    @abstractmethod
    def engine_version(self) -> str:
        """Semver or API version string."""
        ...

    @property
    @abstractmethod
    def supports_tables(self) -> bool:
        """Whether this engine returns structured tables."""
        ...

    @property
    @abstractmethod
    def supports_key_values(self) -> bool:
        """Whether this engine returns key-value pairs."""
        ...

    @abstractmethod
    async def extract_page(
        self,
        doc_id: str,
        page_index: int,
        pdf_path: str,
        image_path: Optional[str] = None,
        region_bbox: Optional[BBox] = None,
    ) -> NormalizedPageExtraction:
        """
        Extract text, lines, tables, and key-values from a single page.

        Must return NormalizedPageExtraction with all invariants satisfied.
        Must raise EngineError on failure (never return partial/corrupt data).
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify engine is available and responding."""
        ...


class EngineError(Exception):
    """Raised when an extraction engine fails."""

    def __init__(self, engine_name: str, error_code: str, message: str):
        self.engine_name = engine_name
        self.error_code = error_code
        self.message = message
        super().__init__(f"[{engine_name}] {error_code}: {message}")