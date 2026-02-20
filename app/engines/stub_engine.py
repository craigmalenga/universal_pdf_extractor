"""
Stub extraction engine for testing pipeline plumbing.
Returns empty NormalizedPageExtraction - validates end-to-end flow
without requiring real PDF processing or OCR.
"""

from typing import Optional

from app.engines.base import ExtractionEngine
from app.schemas.contracts import (
    BBox,
    NormalizedPageExtraction,
    PageMetrics,
)


class StubEngine(ExtractionEngine):
    """Fake adapter that returns empty but valid extractions."""

    @property
    def engine_name(self) -> str:
        return "stub"

    @property
    def engine_version(self) -> str:
        return "0.1.0"

    @property
    def supports_tables(self) -> bool:
        return False

    @property
    def supports_key_values(self) -> bool:
        return False

    async def extract_page(
        self,
        doc_id: str,
        page_index: int,
        pdf_path: str,
        image_path: Optional[str] = None,
        region_bbox: Optional[BBox] = None,
    ) -> NormalizedPageExtraction:
        """Return a valid but empty extraction."""
        return NormalizedPageExtraction(
            doc_id=doc_id,
            page_index=page_index,
            tokens=[],
            lines=[],
            tables=[],
            key_values=[],
            metrics=PageMetrics(
                dpi=300,
                extraction_path="STUB",
                engine_name=self.engine_name,
                engine_version=self.engine_version,
                processing_time_ms=0,
            ),
            raw_text="",
        )

    async def health_check(self) -> bool:
        """Stub is always healthy."""
        return True