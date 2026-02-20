"""
Tesseract OCR extraction engine.
Fallback path for scanned/image PDFs without embedded text.
Uses pytesseract to extract text with word-level bounding boxes.
"""

import pytesseract
from PIL import Image
import structlog
from typing import Optional

from app.engines.base import ExtractionEngine, EngineError
from app.schemas.contracts import (
    NormalizedPageExtraction, ExtractedToken, ExtractedLine, BBox, PageMetrics,
)

logger = structlog.get_logger(__name__)


def _build_lines_from_tokens(tokens: list[ExtractedToken], y_tolerance: float = 0.008) -> list[ExtractedLine]:
    """Cluster tokens into lines by y-proximity."""
    if not tokens:
        return []

    sorted_tokens = sorted(tokens, key=lambda t: (t.bbox.y0, t.bbox.x0))
    lines = []
    current_tokens = [sorted_tokens[0]]
    current_y = sorted_tokens[0].bbox.y0

    for token in sorted_tokens[1:]:
        if abs(token.bbox.y0 - current_y) <= y_tolerance:
            current_tokens.append(token)
        else:
            lines.append(_make_line(current_tokens, len(lines)))
            current_tokens = [token]
            current_y = token.bbox.y0

    if current_tokens:
        lines.append(_make_line(current_tokens, len(lines)))

    return lines


def _make_line(tokens: list[ExtractedToken], line_index: int) -> ExtractedLine:
    tokens_sorted = sorted(tokens, key=lambda t: t.bbox.x0)
    text = " ".join(t.text for t in tokens_sorted)
    mean_conf = sum(t.confidence for t in tokens_sorted) / len(tokens_sorted) if tokens_sorted else 0.0
    return ExtractedLine(
        text=text,
        bbox=BBox(
            x0=min(t.bbox.x0 for t in tokens_sorted),
            y0=min(t.bbox.y0 for t in tokens_sorted),
            x1=max(t.bbox.x1 for t in tokens_sorted),
            y1=max(t.bbox.y1 for t in tokens_sorted),
        ),
        tokens=tokens_sorted,
        line_index=line_index,
        confidence=mean_conf,
    )


class TesseractEngine(ExtractionEngine):
    """
    Tesseract OCR engine for scanned/image PDFs.
    Requires page images (rendered via pdf2image).
    """

    engine_name = "tesseract"
    engine_version = "5.x"
    supports_tables = False
    supports_key_values = False

    def __init__(self, lang: str = "eng", psm: int = 6):
        """
        Args:
            lang: Tesseract language code
            psm: Page segmentation mode (6 = uniform block of text)
        """
        self.lang = lang
        self.psm = psm

    async def extract_page(
        self,
        doc_id: str,
        page_index: int,
        pdf_path: str,
        image_path: Optional[str] = None,
        region_bbox: Optional[BBox] = None,
    ) -> NormalizedPageExtraction:
        """Extract text from a page image using Tesseract OCR."""
        if not image_path:
            raise EngineError("TesseractEngine requires an image_path (rendered page image)")

        try:
            img = Image.open(image_path)
            img_width, img_height = img.size

            # Get word-level data with confidence scores
            data = pytesseract.image_to_data(
                img,
                lang=self.lang,
                config=f"--psm {self.psm}",
                output_type=pytesseract.Output.DICT,
            )
        except Exception as e:
            raise EngineError(f"Tesseract failed on page {page_index}: {e}") from e

        tokens = []
        confidences = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])

            # Skip empty tokens and very low confidence
            if not text or conf < 10:
                continue

            conf_norm = conf / 100.0
            confidences.append(conf_norm)

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            tokens.append(ExtractedToken(
                text=text,
                bbox=BBox(
                    x0=round(x / img_width, 6),
                    y0=round(y / img_height, 6),
                    x1=round((x + w) / img_width, 6),
                    y1=round((y + h) / img_height, 6),
                ),
                confidence=conf_norm,
                source="tesseract",
            ))

        # Build lines
        lines = _build_lines_from_tokens(tokens)
        raw_text = "\n".join(line.text for line in lines)
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

        logger.debug(
            "tesseract_extraction_complete",
            doc_id=doc_id,
            page_index=page_index,
            token_count=len(tokens),
            line_count=len(lines),
            mean_confidence=round(mean_conf, 3),
        )

        return NormalizedPageExtraction(
            doc_id=doc_id,
            page_index=page_index,
            tokens=tokens,
            lines=lines,
            tables=[],
            key_values=[],
            metrics=PageMetrics(
                extraction_path="OCR",
                engine_name=self.engine_name,
                engine_version=self.engine_version,
                mean_token_confidence=round(mean_conf, 4),
                token_count=len(tokens),
            ),
            raw_text=raw_text,
        )

    async def health_check(self) -> bool:
        """Check if Tesseract is installed and accessible."""
        try:
            version = pytesseract.get_tesseract_version()
            self.engine_version = str(version)
            return True
        except Exception:
            return False

    def detect_orientation(self, image_path: str) -> dict:
        """Detect page orientation using Tesseract OSD."""
        try:
            img = Image.open(image_path)
            osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
            return {
                "orientation": osd.get("orientation", 0),
                "rotation": osd.get("rotate", 0),
                "confidence": osd.get("orientation_conf", 0) / 100.0,
                "script": osd.get("script", "Latin"),
            }
        except Exception as e:
            logger.warning("tesseract_osd_failed", error=str(e))
            return {"orientation": 0, "rotation": 0, "confidence": 0.0, "script": "Unknown"}

    def quick_confidence_sample(self, image_path: str, max_tokens: int = 50) -> float:
        """Run quick OCR and return mean confidence for first N tokens."""
        try:
            img = Image.open(image_path)
            data = pytesseract.image_to_data(
                img, lang=self.lang, config=f"--psm {self.psm}",
                output_type=pytesseract.Output.DICT,
            )
            confs = []
            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])
                if text and conf > 0:
                    confs.append(conf / 100.0)
                    if len(confs) >= max_tokens:
                        break
            return sum(confs) / len(confs) if confs else 0.0
        except Exception:
            return 0.0