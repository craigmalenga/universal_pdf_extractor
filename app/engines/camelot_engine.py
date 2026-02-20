"""
Camelot table extraction engine.
Supports both STREAM (no gridlines) and LATTICE (gridlines) modes.
Camelot excels at structured table detection directly from PDF.
"""

import os
import structlog
from typing import Optional

from app.engines.base import ExtractionEngine, EngineError
from app.schemas.contracts import (
    NormalizedPageExtraction, ExtractedToken, ExtractedLine, ExtractedTable,
    ExtractedCell, BBox, PageMetrics,
)

logger = structlog.get_logger(__name__)


def _camelot_available() -> bool:
    """Check if camelot is installed."""
    try:
        import camelot
        return True
    except ImportError:
        return False


class CamelotEngine(ExtractionEngine):
    """
    Camelot table extraction engine.
    Extracts structured tables directly from PDF with cell-level coordinates.

    Two modes:
    - LATTICE: For PDFs with visible gridlines/borders
    - STREAM: For PDFs without gridlines (uses whitespace analysis)
    """

    engine_name = "camelot"
    engine_version = "0.11"
    supports_tables = True
    supports_key_values = False

    def __init__(self, flavor: str = "stream"):
        """
        Args:
            flavor: 'stream' (no gridlines) or 'lattice' (gridlines)
        """
        if flavor not in ("stream", "lattice"):
            raise ValueError(f"Invalid flavor: {flavor}. Must be 'stream' or 'lattice'.")
        self.flavor = flavor

    async def extract_page(
        self,
        doc_id: str,
        page_index: int,
        pdf_path: str,
        image_path: Optional[str] = None,
        region_bbox: Optional[BBox] = None,
    ) -> NormalizedPageExtraction:
        """
        Extract tables from a PDF page using Camelot.
        Returns NormalizedPageExtraction with tables populated.
        """
        if not _camelot_available():
            raise EngineError("Camelot is not installed")

        import camelot

        try:
            # Camelot uses 1-based page numbers
            page_str = str(page_index + 1)

            tables = camelot.read_pdf(
                pdf_path,
                pages=page_str,
                flavor=self.flavor,
                suppress_stdout=True,
            )
        except Exception as e:
            raise EngineError(f"Camelot {self.flavor} failed on page {page_index}: {e}") from e

        extracted_tables = []
        all_tokens = []
        all_lines = []

        for table_idx, table in enumerate(tables):
            df = table.df
            n_rows, n_cols = df.shape

            # Get table bounding box (camelot provides it)
            # table._bbox is (x0, y0, x1, y1) in PDF coordinates
            # We need page dimensions to normalize
            try:
                pdf_bbox = table._bbox
                # Get page dimensions from camelot's table object
                page_dims = _get_page_dimensions(pdf_path, page_index)
                pw, ph = page_dims
                t_bbox = BBox(
                    x0=max(0.0, pdf_bbox[0] / pw),
                    y0=max(0.0, 1.0 - pdf_bbox[3] / ph),  # PDF coords are bottom-up
                    x1=min(1.0, pdf_bbox[2] / pw),
                    y1=min(1.0, 1.0 - pdf_bbox[1] / ph),
                )
            except Exception:
                t_bbox = BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)

            cells = []
            for ri in range(n_rows):
                for ci in range(n_cols):
                    cell_text = str(df.iloc[ri, ci]).strip()
                    if cell_text == "nan":
                        cell_text = ""

                    # Approximate cell bbox within table bbox
                    col_width = (t_bbox.x1 - t_bbox.x0) / max(n_cols, 1)
                    row_height = (t_bbox.y1 - t_bbox.y0) / max(n_rows, 1)

                    cell_bbox = BBox(
                        x0=t_bbox.x0 + ci * col_width,
                        y0=t_bbox.y0 + ri * row_height,
                        x1=t_bbox.x0 + (ci + 1) * col_width,
                        y1=t_bbox.y0 + (ri + 1) * row_height,
                    )

                    cells.append(ExtractedCell(
                        text=cell_text,
                        bbox=cell_bbox,
                        row_index=ri,
                        col_index=ci,
                        confidence=table.accuracy / 100.0 if hasattr(table, 'accuracy') else 0.8,
                        is_header=(ri == 0),
                    ))

                    # Also create tokens from cell text
                    if cell_text:
                        for word in cell_text.split():
                            all_tokens.append(ExtractedToken(
                                text=word,
                                bbox=cell_bbox,
                                confidence=table.accuracy / 100.0 if hasattr(table, 'accuracy') else 0.8,
                                source=f"camelot_{self.flavor}",
                            ))

            accuracy = table.accuracy if hasattr(table, 'accuracy') else 80.0

            extracted_tables.append(ExtractedTable(
                cells=cells,
                bbox=t_bbox,
                row_count=n_rows,
                col_count=n_cols,
                has_grid_lines=(self.flavor == "lattice"),
                header_row_index=0,
                confidence=accuracy / 100.0,
            ))

            logger.debug("camelot_table_extracted",
                         table_idx=table_idx,
                         rows=n_rows,
                         cols=n_cols,
                         accuracy=round(accuracy, 1),
                         flavor=self.flavor)

        # Build lines from tokens
        all_lines = _build_lines(all_tokens)
        raw_text = "\n".join(line.text for line in all_lines)

        mean_conf = 0.0
        if all_tokens:
            mean_conf = sum(t.confidence for t in all_tokens) / len(all_tokens)

        return NormalizedPageExtraction(
            doc_id=doc_id,
            page_index=page_index,
            tokens=all_tokens,
            lines=all_lines,
            tables=extracted_tables,
            key_values=[],
            metrics=PageMetrics(
                extraction_path="PDF_TEXT",
                engine_name=f"camelot_{self.flavor}",
                engine_version=self.engine_version,
                mean_token_confidence=round(mean_conf, 4),
                token_count=len(all_tokens),
            ),
            raw_text=raw_text,
        )

    async def health_check(self) -> bool:
        return _camelot_available()


class CamelotLatticeEngine(CamelotEngine):
    """Camelot in lattice mode — for PDFs with visible table borders."""
    engine_name = "camelot_lattice"

    def __init__(self):
        super().__init__(flavor="lattice")


class CamelotStreamEngine(CamelotEngine):
    """Camelot in stream mode — for PDFs without table borders."""
    engine_name = "camelot_stream"

    def __init__(self):
        super().__init__(flavor="stream")


# ─── Helpers ──────────────────────────────────────────────────

def _get_page_dimensions(pdf_path: str, page_index: int) -> tuple[float, float]:
    """Get page dimensions from PDF."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_index]
            return float(page.width), float(page.height)
    except Exception:
        return 612.0, 792.0  # Default US Letter


def _build_lines(tokens: list[ExtractedToken], y_tolerance: float = 0.008) -> list[ExtractedLine]:
    """Cluster tokens into lines by y-proximity."""
    if not tokens:
        return []

    sorted_tokens = sorted(tokens, key=lambda t: (t.bbox.y0, t.bbox.x0))
    lines = []
    current = [sorted_tokens[0]]
    current_y = sorted_tokens[0].bbox.y0

    for token in sorted_tokens[1:]:
        if abs(token.bbox.y0 - current_y) <= y_tolerance:
            current.append(token)
        else:
            ts = sorted(current, key=lambda t: t.bbox.x0)
            lines.append(ExtractedLine(
                text=" ".join(t.text for t in ts),
                bbox=BBox(
                    x0=min(t.bbox.x0 for t in ts),
                    y0=min(t.bbox.y0 for t in ts),
                    x1=max(t.bbox.x1 for t in ts),
                    y1=max(t.bbox.y1 for t in ts),
                ),
                tokens=ts,
                line_index=len(lines),
                confidence=sum(t.confidence for t in ts) / len(ts),
            ))
            current = [token]
            current_y = token.bbox.y0

    if current:
        ts = sorted(current, key=lambda t: t.bbox.x0)
        lines.append(ExtractedLine(
            text=" ".join(t.text for t in ts),
            bbox=BBox(
                x0=min(t.bbox.x0 for t in ts),
                y0=min(t.bbox.y0 for t in ts),
                x1=max(t.bbox.x1 for t in ts),
                y1=max(t.bbox.y1 for t in ts),
            ),
            tokens=ts,
            line_index=len(lines),
            confidence=sum(t.confidence for t in ts) / len(ts),
        ))

    return lines


def has_gridlines(pdf_path: str, page_index: int) -> bool:
    """
    Quick heuristic: check if a page likely has table gridlines.
    Uses camelot lattice accuracy vs stream to decide.
    """
    if not _camelot_available():
        return False

    import camelot
    try:
        page_str = str(page_index + 1)
        lattice = camelot.read_pdf(pdf_path, pages=page_str, flavor="lattice", suppress_stdout=True)
        if len(lattice) > 0 and hasattr(lattice[0], 'accuracy') and lattice[0].accuracy > 60:
            return True
        return False
    except Exception:
        return False