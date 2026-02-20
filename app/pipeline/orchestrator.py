"""
Pipeline orchestrator: coordinates 8-stage document processing.
Part 6 of the spec.

Stages: LOAD → RENDER → PREPROCESS → EXTRACT → SEGMENT → ANALYSE → VALIDATE → PERSIST
"""

import os
import time
import uuid
import tempfile
import traceback
from datetime import datetime, date as date_type
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.database import async_session_factory
from app.models.tables import (
    Document, DocumentSegment, ExtractionRun, Page, Transaction, DetectedTable,
)
from app.storage.artifact_store import ArtifactStore
from app.storage.paths import raw_pdf_path, rendered_page_path

# Engines
from app.engines.pdfplumber_engine import PdfPlumberEngine
from app.engines.tesseract_engine import TesseractEngine

# Pipeline stages
from app.pipeline.renderer import render_pdf_pages, preprocess_page, RenderedPage
from app.pipeline.segmenter import detect_segment_boundaries, build_segments
from app.pipeline.doc_classifier import classify_document
from app.pipeline.provider_detector import detect_provider
from app.pipeline.table_extractor import (
    detect_columns, reconstruct_rows, detect_header_line,
    extract_header_texts, ColumnRegion, ExtractedRow,
)
from app.pipeline.semantic_mapper import assign_column_roles, ColumnRole
from app.pipeline.date_parser import parse_date_uk
from app.pipeline.amount_parser import parse_amount_uk
from app.pipeline.balance_solver import solve_directions
from app.pipeline.confidence_scorer import score_document

from app.schemas.contracts import NormalizedPageExtraction

logger = structlog.get_logger(__name__)


class PipelineError(Exception):
    """Fatal pipeline error."""
    def __init__(self, message: str, error_code: str = "ERR_PIPELINE"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class DocumentPipeline:
    """
    Main document processing pipeline.
    Processes a single document through all 8 stages.
    """

    def __init__(self):
        self.pdfplumber = PdfPlumberEngine()
        self.tesseract = TesseractEngine()
        self.store = ArtifactStore(root_dir=settings.ARTIFACT_ROOT)

    async def process(self, doc_id: str) -> dict:
        """
        Main entry point: process a document end-to-end.
        Returns summary dict with status, transaction count, confidence.
        """
        started_at = time.time()
        run_id = str(uuid.uuid4())

        logger.info("pipeline_started", doc_id=doc_id, run_id=run_id)

        async with async_session_factory() as session:
            try:
                # ── Stage 1: LOAD ──
                await self._update_status(session, doc_id, "RENDERING")
                doc = await self._load_document(session, doc_id)
                pdf_path = self._get_pdf_path(doc)

                # Create extraction run
                run = ExtractionRun(
                    run_id=uuid.UUID(run_id),
                    doc_id=uuid.UUID(doc_id),
                    pipeline_version=settings.PIPELINE_VERSION,
                    engine_versions_json={
                        "pdfplumber": self.pdfplumber.engine_version,
                        "tesseract": self.tesseract.engine_version,
                    },
                    status="RUNNING",
                    is_latest=True,
                )
                session.add(run)
                await session.flush()

                # ── Stage 2: RENDER ──
                render_dir = os.path.join(settings.ARTIFACT_ROOT, doc_id, "rendered")
                rendered_pages = render_pdf_pages(pdf_path, render_dir, dpi=300)
                page_count = len(rendered_pages)

                # Update page count
                doc.page_count = page_count
                await session.flush()

                # Store page records
                for rp in rendered_pages:
                    page_rec = Page(
                        doc_id=uuid.UUID(doc_id),
                        page_index=rp.page_index,
                        width_px=rp.width,
                        height_px=rp.height,
                        dpi=rp.dpi,
                        raw_render_uri=rp.image_path,
                    )
                    session.add(page_rec)
                await session.flush()

                # ── Stage 3: PREPROCESS ──
                await self._update_status(session, doc_id, "PREPROCESSING")
                for rp in rendered_pages:
                    ocr_fn = self.tesseract.quick_confidence_sample
                    preprocess_page(rp, ocr_sample_fn=ocr_fn)

                # ── Stage 4: EXTRACT ──
                await self._update_status(session, doc_id, "EXTRACTING")
                extractions: list[NormalizedPageExtraction] = []

                for rp in rendered_pages:
                    # Choose extraction path
                    has_text = self.pdfplumber.has_text_layer(pdf_path, rp.page_index)

                    if has_text:
                        extraction = await self.pdfplumber.extract_page(
                            doc_id=doc_id,
                            page_index=rp.page_index,
                            pdf_path=pdf_path,
                        )
                    else:
                        extraction = await self.tesseract.extract_page(
                            doc_id=doc_id,
                            page_index=rp.page_index,
                            pdf_path=pdf_path,
                            image_path=rp.image_path,
                        )

                    extractions.append(extraction)
                    logger.debug("page_extracted",
                                 page=rp.page_index,
                                 path=extraction.metrics.extraction_path,
                                 tokens=extraction.metrics.token_count)

                # ── Stage 5: SEGMENT ──
                await self._update_status(session, doc_id, "SEGMENTING")
                boundaries = detect_segment_boundaries(extractions)
                segments_info = build_segments(boundaries, page_count)

                # Store segments
                segment_records = []
                for seg in segments_info:
                    seg_rec = DocumentSegment(
                        doc_id=uuid.UUID(doc_id),
                        segment_index=seg["segment_index"],
                        start_page=seg["start_page"],
                        end_page=seg["end_page"],
                    )
                    session.add(seg_rec)
                    await session.flush()
                    segment_records.append(seg_rec)

                # If no segments detected, treat whole doc as one segment
                if not segment_records:
                    seg_rec = DocumentSegment(
                        doc_id=uuid.UUID(doc_id),
                        segment_index=0,
                        start_page=0,
                        end_page=page_count - 1,
                    )
                    session.add(seg_rec)
                    await session.flush()
                    segment_records = [seg_rec]

                # ── Stage 6: ANALYSE ──
                await self._update_status(session, doc_id, "EXTRACTING")

                # Classify document
                all_text = "\n".join(e.raw_text for e in extractions if e.raw_text)
                doc_family, family_conf = classify_document(all_text)
                doc.doc_family = doc_family
                doc.doc_family_confidence = Decimal(str(round(family_conf, 4)))

                # Detect provider
                provider, prov_conf = detect_provider(all_text)
                if provider:
                    doc.provider_guess = provider
                    doc.provider_confidence = Decimal(str(round(prov_conf, 4)))

                await session.flush()

                # Process each segment
                all_transactions = []
                total_balance_confirmed = 0

                for seg_rec in segment_records:
                    seg_extractions = [
                        e for e in extractions
                        if seg_rec.start_page <= e.page_index <= seg_rec.end_page
                    ]

                    seg_txns, seg_balance_confirmed = await self._analyse_segment(
                        session=session,
                        doc_id=doc_id,
                        run_id=run_id,
                        segment=seg_rec,
                        extractions=seg_extractions,
                    )
                    all_transactions.extend(seg_txns)
                    total_balance_confirmed += seg_balance_confirmed

                # ── Stage 7: VALIDATE ──
                await self._update_status(session, doc_id, "VALIDATING")
                recon_rate = total_balance_confirmed / len(all_transactions) if all_transactions else 0.0
                confidence = score_document(
                    transactions=all_transactions,
                    reconciliation_rate=recon_rate,
                )

                # Determine final status
                if confidence >= settings.CONFIDENCE_PASS_THRESHOLD:
                    validation_status = "PASS"
                    final_status = "COMPLETED"
                elif confidence >= settings.CONFIDENCE_WARN_THRESHOLD:
                    validation_status = "PASS_WITH_WARNINGS"
                    final_status = "COMPLETED"
                elif confidence >= settings.CONFIDENCE_FAIL_THRESHOLD:
                    validation_status = "NEEDS_REVIEW"
                    final_status = "NEEDS_REVIEW"
                else:
                    validation_status = "FAIL"
                    final_status = "NEEDS_REVIEW"

                # ── Stage 8: PERSIST ──
                duration_ms = int((time.time() - started_at) * 1000)

                run.status = "COMPLETED"
                run.validation_status = validation_status
                run.confidence_document = Decimal(str(round(confidence, 4)))
                run.reconciliation_rate = Decimal(str(round(recon_rate, 4)))
                run.row_count = len(all_transactions)
                run.rows_with_balance_confirmation = total_balance_confirmed
                run.ended_at = datetime.utcnow()
                run.duration_ms = duration_ms

                doc.status = final_status
                await session.commit()

                logger.info(
                    "pipeline_completed",
                    doc_id=doc_id,
                    run_id=run_id,
                    transactions=len(all_transactions),
                    confidence=round(confidence, 4),
                    validation=validation_status,
                    duration_ms=duration_ms,
                )

                return {
                    "doc_id": doc_id,
                    "run_id": run_id,
                    "status": final_status,
                    "transactions": len(all_transactions),
                    "confidence": round(confidence, 4),
                    "duration_ms": duration_ms,
                }

            except PipelineError as e:
                await self._fail_document(session, doc_id, run_id, e.error_code, str(e))
                raise
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error("pipeline_failed", doc_id=doc_id, error=error_msg,
                             traceback=traceback.format_exc())
                await self._fail_document(session, doc_id, run_id, "ERR_PIPELINE", error_msg)
                raise PipelineError(error_msg) from e

    # ─── Stage Helpers ────────────────────────────────────────

    async def _load_document(self, session: AsyncSession, doc_id: str) -> Document:
        """Load document from DB."""
        result = await session.execute(
            select(Document).where(Document.doc_id == uuid.UUID(doc_id))
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise PipelineError(f"Document {doc_id} not found", "ERR_DOC_NOT_FOUND")
        return doc

    def _get_pdf_path(self, doc: Document) -> str:
        """Get the local PDF file path."""
        path = os.path.join(settings.ARTIFACT_ROOT, doc.raw_file_uri)
        if not os.path.exists(path):
            raise PipelineError(f"PDF not found at {path}", "ERR_FILE_NOT_FOUND")
        return path

    async def _update_status(self, session: AsyncSession, doc_id: str, status: str):
        """Update document status."""
        await session.execute(
            update(Document).where(Document.doc_id == uuid.UUID(doc_id)).values(status=status)
        )
        await session.flush()

    async def _fail_document(self, session: AsyncSession, doc_id: str, run_id: str,
                             error_code: str, error_message: str):
        """Mark document and run as failed."""
        try:
            await session.execute(
                update(Document).where(Document.doc_id == uuid.UUID(doc_id)).values(status="FAILED")
            )
            if run_id:
                await session.execute(
                    update(ExtractionRun)
                    .where(ExtractionRun.run_id == uuid.UUID(run_id))
                    .values(
                        status="FAILED",
                        error_code=error_code,
                        error_message=error_message[:500],
                        ended_at=datetime.utcnow(),
                    )
                )
            await session.commit()
        except Exception:
            logger.error("failed_to_mark_failure", doc_id=doc_id)

    # ─── Segment Analysis ─────────────────────────────────────

    async def _analyse_segment(
        self,
        session: AsyncSession,
        doc_id: str,
        run_id: str,
        segment: DocumentSegment,
        extractions: list[NormalizedPageExtraction],
    ) -> tuple[list[Transaction], int]:
        """
        Analyse a single segment: detect tables, extract transactions.
        Returns (list of Transaction ORM objects, balance_confirmed_count).
        """
        # Collect all lines across pages in this segment
        all_lines = []
        for ext in extractions:
            all_lines.extend(ext.lines)

        if not all_lines:
            logger.warning("empty_segment", segment=segment.segment_index)
            return [], 0

        # Detect columns from token positions
        columns = detect_columns(all_lines)
        if not columns:
            logger.warning("no_columns_detected", segment=segment.segment_index)
            return [], 0

        # Detect header
        header_idx = detect_header_line(all_lines)
        header_texts = None
        if header_idx is not None:
            header_texts = extract_header_texts(all_lines[header_idx], columns)
            # Remove header and anything above from transaction lines
            all_lines = all_lines[header_idx + 1:]

        # Assign column roles
        # First do a preliminary row reconstruction to get sample data
        preliminary_rows = reconstruct_rows(
            all_lines, columns,
            date_column_index=0,
            amount_column_indices=[c.column_index for c in columns if c.column_index > 0],
        )

        roles = assign_column_roles(columns, header_texts, preliminary_rows)

        # Find key column indices
        date_col = next((i for i, r in roles.items() if r == ColumnRole.DATE), 0)
        amount_cols = [i for i, r in roles.items()
                       if r in (ColumnRole.DEBIT, ColumnRole.CREDIT, ColumnRole.SINGLE_AMOUNT, ColumnRole.BALANCE)]

        # Reconstruct rows with proper column knowledge
        rows = reconstruct_rows(all_lines, columns, date_col, amount_cols)
        transaction_rows = [r for r in rows if not r.is_balance_marker]

        if not transaction_rows:
            logger.warning("no_transaction_rows", segment=segment.segment_index)
            return [], 0

        # Extract field values from rows
        raw_transactions = []
        for row_idx, row in enumerate(transaction_rows):
            tx_data = self._extract_fields_from_row(row, roles, columns)
            tx_data["row_index"] = row_idx
            raw_transactions.append(tx_data)

        # Run balance solver for direction inference
        solved = solve_directions(raw_transactions)

        # Persist transactions
        tx_records = []
        balance_confirmed = 0

        for tx in solved:
            posted_date = None
            if tx.get("parsed_date"):
                try:
                    d = tx["parsed_date"]
                    if isinstance(d, date_type):
                        posted_date = d
                except Exception:
                    pass

            amount = None
            if tx.get("parsed_amount") is not None:
                try:
                    amount = Decimal(str(round(tx["parsed_amount"], 2)))
                except Exception:
                    pass

            balance = None
            if tx.get("parsed_balance") is not None:
                try:
                    balance = Decimal(str(round(tx["parsed_balance"], 2)))
                except Exception:
                    pass

            direction = tx.get("direction", "UNKNOWN")
            bc = tx.get("balance_confirmed", False)
            if bc:
                balance_confirmed += 1

            tx_rec = Transaction(
                run_id=uuid.UUID(run_id),
                doc_id=uuid.UUID(doc_id),
                segment_id=segment.segment_id,
                page_index=tx.get("page_index", 0),
                row_index=tx.get("row_index", 0),
                posted_date=posted_date,
                description_raw=tx.get("description", ""),
                description_clean=tx.get("description", "").strip(),
                amount=amount,
                direction=direction,
                running_balance=balance,
                balance_confirmed=bc,
                direction_source=tx.get("direction_source", ""),
                confidence_overall=Decimal("0.80"),
                confidence_amount=Decimal(str(round(tx.get("amount_confidence", 0.8), 4))),
                confidence_date=Decimal(str(round(tx.get("date_confidence", 0.8), 4))),
                confidence_direction=Decimal(str(round(tx.get("direction_confidence", 0.5), 4))),
            )
            session.add(tx_rec)
            tx_records.append(tx_rec)

        await session.flush()

        logger.info("segment_analysed",
                     segment=segment.segment_index,
                     transactions=len(tx_records),
                     balance_confirmed=balance_confirmed)

        return tx_records, balance_confirmed

    def _extract_fields_from_row(
        self,
        row: ExtractedRow,
        roles: dict[int, ColumnRole],
        columns: list[ColumnRegion],
    ) -> dict:
        """Extract date, description, amounts, balance from a row using column roles."""
        result = {
            "description": "",
            "raw_date": "",
            "raw_debit": "",
            "raw_credit": "",
            "raw_amount": "",
            "raw_balance": "",
            "parsed_date": None,
            "parsed_amount": None,
            "parsed_balance": None,
            "direction": "UNKNOWN",
            "direction_source": "",
            "amount_confidence": 0.8,
            "date_confidence": 0.8,
            "direction_confidence": 0.5,
            "page_index": 0,
        }

        # Deduplicate cells by column (take first occurrence)
        seen_cols = set()
        unique_cells = []
        for cell in row.cells:
            if cell.column_index not in seen_cols:
                seen_cols.add(cell.column_index)
                unique_cells.append(cell)

        for cell in unique_cells:
            role = roles.get(cell.column_index, ColumnRole.UNKNOWN)
            text = cell.text.strip()

            if role == ColumnRole.DATE:
                result["raw_date"] = text
                parsed = parse_date_uk(text)
                if parsed:
                    result["parsed_date"] = parsed
                    result["date_confidence"] = 0.9

            elif role == ColumnRole.DESCRIPTION:
                result["description"] = (result["description"] + " " + text).strip()

            elif role == ColumnRole.DEBIT:
                if text:
                    result["raw_debit"] = text
                    amt = parse_amount_uk(text)
                    if amt is not None:
                        result["parsed_amount"] = abs(amt)
                        result["direction"] = "DEBIT"
                        result["direction_source"] = "column_debit"
                        result["direction_confidence"] = 0.95
                        result["amount_confidence"] = 0.9

            elif role == ColumnRole.CREDIT:
                if text:
                    result["raw_credit"] = text
                    amt = parse_amount_uk(text)
                    if amt is not None:
                        result["parsed_amount"] = abs(amt)
                        result["direction"] = "CREDIT"
                        result["direction_source"] = "column_credit"
                        result["direction_confidence"] = 0.95
                        result["amount_confidence"] = 0.9

            elif role == ColumnRole.SINGLE_AMOUNT:
                if text:
                    result["raw_amount"] = text
                    amt = parse_amount_uk(text)
                    if amt is not None:
                        result["parsed_amount"] = abs(amt)
                        # Direction unknown from single amount — balance solver decides
                        result["direction"] = "UNKNOWN"
                        result["direction_source"] = "single_amount"
                        result["amount_confidence"] = 0.9

            elif role == ColumnRole.BALANCE:
                if text:
                    result["raw_balance"] = text
                    bal = parse_amount_uk(text)
                    if bal is not None:
                        result["parsed_balance"] = bal

        return result