"""
PDF rendering and image preprocessing.
Parts 8.1-8.4 of the spec.

Renders PDF pages to images, detects/corrects orientation and skew,
applies enhancement profiles for optimal OCR.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RenderedPage:
    page_index: int
    image_path: str
    width: int
    height: int
    dpi: int = 300
    orientation_detected: int = 0
    skew_degrees: float = 0.0
    preprocessing_profile: str = "none"


# ─── PDF Rendering ────────────────────────────────────────────

def render_pdf_pages(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
) -> list[RenderedPage]:
    """
    Render all pages of a PDF to PNG images.
    Returns list of RenderedPage with file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            fmt="png",
            thread_count=2,
        )
    except Exception as e:
        logger.error("pdf_render_failed", pdf_path=pdf_path, error=str(e))
        raise RuntimeError(f"Failed to render PDF: {e}") from e

    rendered = []
    for i, img in enumerate(images):
        page_path = os.path.join(output_dir, f"page_{i:04d}.png")
        img.save(page_path, "PNG")
        rendered.append(RenderedPage(
            page_index=i,
            image_path=page_path,
            width=img.width,
            height=img.height,
            dpi=dpi,
        ))

    logger.info("pdf_rendered", page_count=len(rendered), dpi=dpi)
    return rendered


# ─── Orientation Detection ────────────────────────────────────

def detect_and_fix_orientation(page: RenderedPage) -> RenderedPage:
    """
    Detect page orientation using Tesseract OSD and correct if needed.
    Modifies image file in-place and updates RenderedPage.
    """
    try:
        import pytesseract
        img = Image.open(page.image_path)
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        rotation = osd.get("rotate", 0)
        conf = osd.get("orientation_conf", 0)

        if rotation != 0 and conf > 0.5:
            # Rotate the image
            rotated = img.rotate(-rotation, expand=True)
            rotated.save(page.image_path, "PNG")
            page.orientation_detected = rotation
            page.width, page.height = rotated.size
            logger.debug("orientation_corrected", page=page.page_index, rotation=rotation, conf=conf)
    except Exception as e:
        logger.debug("orientation_detection_skipped", page=page.page_index, reason=str(e))

    return page


# ─── Skew Detection & Correction ─────────────────────────────

def detect_and_fix_skew(page: RenderedPage) -> RenderedPage:
    """
    Detect and correct skew using Hough line transform.
    Only corrects small angles (0.5° - 15°).
    """
    try:
        img = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return page

        # Edge detection
        edges = cv2.Canny(img, 50, 150, apertureSize=3)

        # Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=img.shape[1] * 0.2, maxLineGap=10)

        if lines is None or len(lines) < 3:
            return page

        # Compute angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines
                if abs(angle) < 20:
                    angles.append(angle)

        if not angles:
            return page

        median_angle = np.median(angles)

        # Only correct if angle is meaningful but not too large
        if 0.5 < abs(median_angle) < 15:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            corrected = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            # Save back — reload as color if original was color
            color_img = cv2.imread(page.image_path)
            if color_img is not None and len(color_img.shape) == 3:
                corrected_color = cv2.warpAffine(color_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                cv2.imwrite(page.image_path, corrected_color)
            else:
                cv2.imwrite(page.image_path, corrected)

            page.skew_degrees = round(median_angle, 3)
            logger.debug("skew_corrected", page=page.page_index, angle=round(median_angle, 3))

    except Exception as e:
        logger.debug("skew_detection_skipped", page=page.page_index, reason=str(e))

    return page


# ─── Enhancement Profiles ─────────────────────────────────────

def apply_enhancement(page: RenderedPage, mean_confidence: float) -> RenderedPage:
    """
    Apply image enhancement profile based on OCR confidence.
    Profile selection per spec Part 8.4.
    """
    try:
        img = cv2.imread(page.image_path)
        if img is None:
            return page

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        if mean_confidence >= 0.85:
            # Profile A: Mild contrast normalisation
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            page.preprocessing_profile = "A_mild_contrast"

        elif mean_confidence >= 0.70:
            # Profile B: Adaptive threshold
            enhanced = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            page.preprocessing_profile = "B_adaptive_threshold"

        elif mean_confidence >= 0.50:
            # Profile C: Denoise + sharpen
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            enhanced = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            page.preprocessing_profile = "C_denoise_sharpen"

        else:
            # Profile D: High contrast for faint scans
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            contrast = clahe.apply(gray)
            kernel = np.ones((2, 2), np.uint8)
            opened = cv2.morphologyEx(contrast, cv2.MORPH_OPEN, kernel)
            _, enhanced = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            page.preprocessing_profile = "D_high_contrast"

        cv2.imwrite(page.image_path, enhanced)
        logger.debug("enhancement_applied", page=page.page_index, profile=page.preprocessing_profile)

    except Exception as e:
        logger.debug("enhancement_failed", page=page.page_index, reason=str(e))

    return page


# ─── Full Preprocessing Pipeline ──────────────────────────────

def preprocess_page(page: RenderedPage, ocr_sample_fn=None) -> RenderedPage:
    """
    Full preprocessing: orientation → skew → enhancement.
    If ocr_sample_fn is provided, uses it to measure confidence for profile selection.
    """
    # Step 1: Orientation
    page = detect_and_fix_orientation(page)

    # Step 2: Skew
    page = detect_and_fix_skew(page)

    # Step 3: Enhancement (based on OCR confidence)
    mean_conf = 0.85  # Default: assume good quality (PDF text path)
    if ocr_sample_fn:
        try:
            mean_conf = ocr_sample_fn(page.image_path)
        except Exception:
            pass

    if mean_conf < 0.85:
        page = apply_enhancement(page, mean_conf)

    return page