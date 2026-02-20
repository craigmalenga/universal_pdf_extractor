"""
Abstract base class for all extraction engines.
Every engine must produce NormalizedPageExtraction.
"""

from abc import ABC, abstractmethod
from typing import Optional

from app.schemas.contracts import BBox, NormalizedPageExtraction


class EngineError(Exception):
    """Raised when an extraction engine fails."""
    def __init__(self, message: str, engine_name: str = "", error_code: str = "ERR_ENGINE"):
        self.engine_name = engine_name
        self.error_code = error_code
        self.message = message
        super().__init__(message)


class ExtractionEngine(ABC):
    """
    Abstract base class for all extraction engines.
    Every engine must return NormalizedPageExtraction.
    """

    engine_name: str = "base"
    engine_version: str = "0.0.0"
    supports_tables: bool = False
    supports_key_values: bool = False

    @abstractmethod
    async def extract_page(
        self,
        doc_id: str,
        page_index: int,
        pdf_path: str,
        image_path: Optional[str] = None,
        region_bbox: Optional[BBox] = None,
    ) -> NormalizedPageExtraction:
        """Extract content from a single page."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify engine is available."""
        ...