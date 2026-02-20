"""
pdfplumber extraction engine.
Primary path for PDFs with embedded text layers.
Produces NormalizedPageExtraction from raw PDF text + coordinates.
"""

import pdfplumber
import structlog
from typing import Optional

from app.engines.base import ExtractionEngine, EngineError
from app.schemas.contracts import (
    NormalizedPageExtraction, ExtractedToken, ExtractedLine, BBox, PageMetrics,
)

logger = structlog.get_logger(__name__)


def _build_lines_from_tokens(tokens: list[ExtractedToken], y_tolerance: float = 0.005) -> list[ExtractedLine]:
    """
    Cluster tokens into lines by y-overlap.
    Tokens within y_tolerance of each other are on the same line.
    """
    if not tokens:
        return []

    # Sort by y0 then x0
    sorted_tokens = sorted(tokens, key=lambda t: (t.bbox.y0, t.bbox.x0))

    lines = []
    current_line_tokens = [sorted_tokens[0]]
    current_y = sorted_tokens[0].bbox.y0

    for token in sorted_tokens[1:]:
        if abs(token.bbox.y0 - current_y) <= y_tolerance:
            current_line_tokens.append(token)
        else:
            # Finalize current line
            lines.append(_make_line(current_line_tokens, len(lines)))
            current_line_tokens = [token]
            current_y = token.bbox.y0

    if current_line_tokens:
        lines.append(_make_line(current_line_tokens, len(lines)))

    return lines


def _make_line(tokens: list[ExtractedToken], line_index: int) -> ExtractedLine:
    """Create an ExtractedLine from a list of tokens on the same line."""
    tokens_sorted = sorted(tokens, key=lambda t: t.bbox.x0)
    text = " ".join(t.text for t in tokens_sorted)
    x0 = min(t.bbox.x0 for t in tokens_sorted)
    y0 = min(t.bbox.y0 for t in tokens_sorted)
    x1 = max(t.bbox.x1 for t in tokens_sorted)
    y1 = max(t.bbox.y1 for t in tokens_sorted)
    mean_conf = sum(t.confidence for t in tokens_sorted) / len(tokens_sorted) if tokens_sorted else 0.0

    return ExtractedLine(
        text=text,
        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
        tokens=tokens_sorted,
        line_index=line_index,
        confidence=mean_conf,
    )


class PdfPlumberEngine(ExtractionEngine):
    """
    Extraction engine using pdfplumber for PDFs with embedded text.
    Extracts words with coordinates, normalises to 0-1 page space.
    """

    engine_name = "pdfplumber"
    engine_version = "0.11"
    supports_tables = False   # We do our own table detection downstream
    supports_key_values = False

    async def extract_page(
        self,
        doc_id: str,
        page_index: int,
        pdf_path: str,
        image_path: Optional[str] = None,
        region_bbox: Optional[BBox] = None,
    ) -> NormalizedPageExtraction:
        """Extract text tokens from a single PDF page using pdfplumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_index >= len(pdf.pages):
                    raise EngineError(f"Page {page_index} out of range (PDF has {len(pdf.pages)} pages)")

                page = pdf.pages[page_index]
                page_width = float(page.width)
                page_height = float(page.height)

                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=False,
                    extra_attrs=["fontname", "size"],
                )

        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"pdfplumber failed on page {page_index}: {e}") from e

        # Convert to normalised tokens (0-1 coordinate space)
        tokens = []
        for w in words:
            text = w.get("text", "").strip()
            if not text:
                continue

            tokens.append(ExtractedToken(
                text=text,
                bbox=BBox(
                    x0=round(w["x0"] / page_width, 6),
                    y0=round(w["top"] / page_height, 6),
                    x1=round(w["x1"] / page_width, 6),
                    y1=round(w["bottom"] / page_height, 6),
                ),
                confidence=0.95,  # PDF text assumed high confidence
                source="pdfplumber",
            ))

        # Build lines from tokens
        lines = _build_lines_from_tokens(tokens)

        # Raw text
        raw_text = "\n".join(line.text for line in lines)

        logger.debug(
            "pdfplumber_extraction_complete",
            doc_id=doc_id,
            page_index=page_index,
            token_count=len(tokens),
            line_count=len(lines),
        )

        return NormalizedPageExtraction(
            doc_id=doc_id,
            page_index=page_index,
            tokens=tokens,
            lines=lines,
            tables=[],
            key_values=[],
            metrics=PageMetrics(
                extraction_path="PDF_TEXT",
                engine_name=self.engine_name,
                engine_version=self.engine_version,
                mean_token_confidence=0.95 if tokens else 0.0,
                token_count=len(tokens),
            ),
            raw_text=raw_text,
        )

    async def health_check(self) -> bool:
        """pdfplumber is always available (pure Python)."""
        return True

    def get_page_count(self, pdf_path: str) -> int:
        """Get total page count of a PDF."""
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)

    def has_text_layer(self, pdf_path: str, page_index: int = 0) -> bool:
        """Check if a page has an embedded text layer worth using."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_index >= len(pdf.pages):
                    return False
                page = pdf.pages[page_index]
                words = page.extract_words(x_tolerance=3, y_tolerance=3)
                if len(words) < 10:
                    return False
                # Check for gibberish
                texts = [w.get("text", "") for w in words[:50]]
                alpha_count = sum(1 for t in texts if any(c.isalpha() for c in t))
                alpha_ratio = alpha_count / len(texts) if texts else 0
                return alpha_ratio > 0.3
        except Exception:
            return False