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
        self.store = ArtifactStore(root=settings.ARTIFACT_ROOT)

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
                # ── Cleanup old attempts ──
                from sqlalchemy import text
                doc_uuid = uuid.UUID(doc_id)
                doc_str = str(doc_uuid)
                # Delete in FK order using raw SQL to avoid session issues
                await session.execute(text(
                    "DELETE FROM transaction_evidence WHERE transaction_id IN "
                    "(SELECT transaction_id FROM transactions WHERE doc_id = :did)"
                ), {"did": doc_str})
                await session.execute(text(
                    "DELETE FROM transactions WHERE doc_id = :did"
                ), {"did": doc_str})
                await session.execute(text(
                    "DELETE FROM detected_tables WHERE doc_id = :did"
                ), {"did": doc_str})
                await session.execute(text(
                    "DELETE FROM pages WHERE doc_id = :did"
                ), {"did": doc_str})
                await session.execute(text(
                    "DELETE FROM document_segments WHERE doc_id = :did"
                ), {"did": doc_str})
                await session.execute(text(
                    "DELETE FROM extraction_runs WHERE doc_id = :did"
                ), {"did": doc_str})
                await session.commit()
                logger.info("cleanup_complete", doc_id=doc_id)

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
                classification = classify_document([all_text])
                doc.doc_family = classification.doc_family
                doc.doc_family_confidence = Decimal(str(round(classification.confidence, 4)))

                # Detect provider
                provider_result = detect_provider([all_text])
                if provider_result.provider_name:
                    doc.provider_guess = provider_result.provider_name
                    doc.provider_confidence = Decimal(str(round(provider_result.confidence, 4)))

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

                # Convert ORM Transaction objects to dicts for scoring
                tx_dicts = []
                for tx_obj in all_transactions:
                    tx_dicts.append({
                        "confidence_amount": float(tx_obj.confidence_amount) if tx_obj.confidence_amount else 0.0,
                        "confidence_direction": float(tx_obj.confidence_direction) if tx_obj.confidence_direction else 0.0,
                        "confidence_date": float(tx_obj.confidence_date) if tx_obj.confidence_date else 0.0,
                        "confidence_balance": 0.8 if tx_obj.balance_confirmed else 0.0,
                        "balance_confirmed": tx_obj.balance_confirmed or False,
                    })

                score_result = score_document(transactions=tx_dicts)
                confidence = score_result.document_confidence

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

        if not amount_cols:
            logger.warning("no_amount_columns", segment=segment.segment_index,
                           columns=len(columns), roles={i: r.value for i, r in roles.items()})
            # ── FALLBACK: Try pdfplumber native table extraction ──
            logger.info("trying_pdfplumber_fallback", segment=segment.segment_index)
            fallback_txns = await self._analyse_segment_pdfplumber_fallback(
                session, doc_id, run_id, segment, extractions
            )
            if fallback_txns:
                logger.info("pdfplumber_fallback_success",
                            segment=segment.segment_index, transactions=len(fallback_txns))
                return fallback_txns, 0
            return [], 0

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
        # Build column_roles dict for solver
        role_map = {i: r.value for i, r in roles.items()}

        # Find opening/closing balance from balance markers
        opening_balance = None
        closing_balance = None
        balance_markers = [r for r in rows if r.is_balance_marker]
        for bm in balance_markers:
            for cell in bm.cells:
                role = roles.get(cell.column_index)
                if role == ColumnRole.BALANCE and cell.text.strip():
                    bal_result = parse_amount_uk(cell.text.strip())
                    if bal_result.amount is not None:
                        # First marker is opening, last is closing
                        if opening_balance is None:
                            opening_balance = bal_result.amount
                        closing_balance = bal_result.amount

        solver_results = solve_directions(raw_transactions, opening_balance, closing_balance, role_map)

        # Merge solver results back into raw_transactions
        for i, sr in enumerate(solver_results):
            if i < len(raw_transactions):
                tx = raw_transactions[i]
                if tx["direction"] == "UNKNOWN" and sr.direction != "UNKNOWN":
                    tx["direction"] = sr.direction
                    tx["direction_source"] = sr.direction_source
                    tx["direction_confidence"] = sr.confidence
                tx["balance_confirmed"] = sr.balance_confirmed

        solved = raw_transactions

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
                    amt_result = parse_amount_uk(text)
                    if amt_result.amount is not None:
                        result["parsed_amount"] = abs(amt_result.amount)
                        result["direction"] = "DEBIT"
                        result["direction_source"] = "column_debit"
                        result["direction_confidence"] = 0.95
                        result["amount_confidence"] = amt_result.confidence

            elif role == ColumnRole.CREDIT:
                if text:
                    result["raw_credit"] = text
                    amt_result = parse_amount_uk(text)
                    if amt_result.amount is not None:
                        result["parsed_amount"] = abs(amt_result.amount)
                        result["direction"] = "CREDIT"
                        result["direction_source"] = "column_credit"
                        result["direction_confidence"] = 0.95
                        result["amount_confidence"] = amt_result.confidence

            elif role == ColumnRole.SINGLE_AMOUNT:
                if text:
                    result["raw_amount"] = text
                    amt_result = parse_amount_uk(text)
                    if amt_result.amount is not None:
                        result["parsed_amount"] = abs(amt_result.amount)
                        result["direction"] = "UNKNOWN"
                        result["direction_source"] = "single_amount"
                        result["amount_confidence"] = amt_result.confidence

            elif role == ColumnRole.BALANCE:
                if text:
                    result["raw_balance"] = text
                    bal_result = parse_amount_uk(text)
                    if bal_result.amount is not None:
                        result["parsed_balance"] = bal_result.amount

        return result

    # ─── Fallback: pdfplumber native table extraction ─────────

    async def _analyse_segment_pdfplumber_fallback(
        self,
        session: AsyncSession,
        doc_id: str,
        run_id: str,
        segment: DocumentSegment,
        extractions: list[NormalizedPageExtraction],
    ) -> list[Transaction]:
        """
        Fallback: use pdfplumber's built-in extract_tables() when our
        custom column detection fails.
        Parses into plain dicts first, then persists — keeping session clean.
        """
        import pdfplumber

        doc = await self._load_document(session, doc_id)
        pdf_path = self._get_pdf_path(doc)

        parsed_rows = []  # Collect as dicts, persist later

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx in range(segment.start_page, segment.end_page + 1):
                    if page_idx >= len(pdf.pages):
                        continue
                    page = pdf.pages[page_idx]

                    tables = page.extract_tables({
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 5,
                        "join_tolerance": 5,
                        "min_words_vertical": 2,
                        "min_words_horizontal": 1,
                    })

                    if not tables:
                        tables = page.extract_tables({
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                        })

                    for table in tables:
                        if not table or len(table) < 2:
                            continue

                        header = [str(c).lower().strip() if c else "" for c in table[0]]
                        col_map = self._map_table_columns(header)

                        if not col_map.get("amount_cols"):
                            if len(table) > 2:
                                header = [str(c).lower().strip() if c else "" for c in table[1]]
                                col_map = self._map_table_columns(header)
                                data_rows = table[2:]
                            else:
                                continue
                        else:
                            data_rows = table[1:]

                        last_date = None
                        for row in data_rows:
                            row_strs = [str(c).strip() if c else "" for c in row]

                            date_val = None
                            if col_map.get("date_col") is not None:
                                raw_date = row_strs[col_map["date_col"]]
                                if raw_date:
                                    date_val = parse_date_uk(raw_date)
                                    if date_val:
                                        last_date = date_val
                                elif last_date:
                                    date_val = last_date

                            desc = ""
                            if col_map.get("desc_col") is not None:
                                desc = row_strs[col_map["desc_col"]]

                            amount = None
                            direction = "UNKNOWN"
                            balance = None

                            for ac in col_map.get("amount_cols", []):
                                idx, role = ac["index"], ac["role"]
                                if idx < len(row_strs) and row_strs[idx]:
                                    amt_result = parse_amount_uk(row_strs[idx])
                                    if amt_result.amount is not None:
                                        if role == "paid_in":
                                            amount = abs(amt_result.amount)
                                            direction = "CREDIT"
                                        elif role == "withdrawn":
                                            amount = abs(amt_result.amount)
                                            direction = "DEBIT"
                                        elif role == "balance":
                                            balance = amt_result.amount
                                        elif role == "amount" and amount is None:
                                            amount = abs(amt_result.amount)

                            if amount is None and balance is None:
                                continue

                            from app.pipeline.table_extractor import is_balance_marker
                            if is_balance_marker(desc):
                                continue

                            parsed_rows.append({
                                "date_val": date_val,
                                "desc": desc[:500] if desc else "",
                                "amount": amount,
                                "direction": direction,
                                "balance": balance,
                                "page_idx": page_idx,
                            })

        except Exception as e:
            logger.error("pdfplumber_fallback_parse_failed",
                         segment=segment.segment_index, error=str(e))

        if not parsed_rows:
            # Try Tabula next
            logger.info("trying_tabula_fallback", segment=segment.segment_index)
            return await self._analyse_segment_tabula_fallback(
                session, doc_id, run_id, segment
            )

        # Persist parsed rows using savepoint so failure doesn't dirty session
        tx_records = []
        try:
            async with session.begin_nested():
                for idx, pr in enumerate(parsed_rows):
                    tx_rec = Transaction(
                        doc_id=uuid.UUID(doc_id),
                        run_id=uuid.UUID(run_id),
                        segment_id=segment.segment_id,
                        row_index=idx,
                        posted_date=pr["date_val"],
                        description_raw=pr["desc"],
                        description_clean=pr["desc"].strip(),
                        amount=pr["amount"],
                        direction=pr["direction"],
                        direction_source="pdfplumber_table",
                        running_balance=pr["balance"],
                        balance_confirmed=False,
                        page_index=pr["page_idx"],
                        confidence_overall=Decimal("0.80"),
                        confidence_amount=Decimal("0.80"),
                        confidence_date=Decimal("0.80") if pr["date_val"] else Decimal("0.30"),
                        confidence_direction=Decimal("0.90") if pr["direction"] != "UNKNOWN" else Decimal("0.40"),
                    )
                    session.add(tx_rec)
                    tx_records.append(tx_rec)
            # Savepoint committed successfully
            logger.info("pdfplumber_fallback_persisted", transactions=len(tx_records))

        except Exception as e:
            logger.error("pdfplumber_fallback_persist_failed",
                         segment=segment.segment_index, error=str(e))
            tx_records = []
            # Savepoint rolled back automatically, session is clean
            logger.info("trying_tabula_fallback_after_persist_fail", segment=segment.segment_index)
            return await self._analyse_segment_tabula_fallback(
                session, doc_id, run_id, segment
            )

        return tx_records

    # ─── Fallback: Tabula table extraction ────────────────────

    async def _analyse_segment_tabula_fallback(
        self,
        session: AsyncSession,
        doc_id: str,
        run_id: str,
        segment: DocumentSegment,
    ) -> list[Transaction]:
        """
        Tabula fallback: uses Java-based PDF table extraction.
        Often better than Camelot for borderless bank statement tables.
        Falls through to Camelot if it fails.
        """
        try:
            import tabula
        except ImportError:
            logger.warning("tabula_not_installed")
            return await self._analyse_segment_camelot_fallback(
                session, doc_id, run_id, segment
            )

        doc = await self._load_document(session, doc_id)
        pdf_path = self._get_pdf_path(doc)
        pages = f"{segment.start_page + 1}-{segment.end_page + 1}"

        parsed_rows = []

        for method in ["stream", "lattice"]:
            try:
                dfs = tabula.read_pdf(
                    pdf_path,
                    pages=pages,
                    multiple_tables=True,
                    stream=(method == "stream"),
                    lattice=(method == "lattice"),
                    silent=True,
                )

                if not dfs:
                    continue

                col_map = {}
                header_found = False

                for df in dfs:
                    if df.empty or df.shape[1] < 2:
                        continue

                    # Check if column names are the header
                    header = [str(c).lower().strip() for c in df.columns]
                    col_map = self._map_table_columns(header)
                    if col_map.get("amount_cols"):
                        header_found = True
                        logger.info("tabula_header_found", method=method, col_map=col_map)
                    else:
                        # Try first row as header
                        if df.shape[0] > 1:
                            first_row = [str(df.iloc[0, ci]).lower().strip() for ci in range(df.shape[1])]
                            col_map = self._map_table_columns(first_row)
                            if col_map.get("amount_cols"):
                                header_found = True
                                df = df.iloc[1:]  # Skip header row
                                logger.info("tabula_header_found_row1", method=method, col_map=col_map)

                    if not header_found or not col_map.get("amount_cols"):
                        continue

                    last_date = None
                    for ri in range(df.shape[0]):
                        cells = [str(df.iloc[ri, ci]).strip() if str(df.iloc[ri, ci]) != "nan" else ""
                                 for ci in range(df.shape[1])]
                        row_lower = " ".join(cells).lower()

                        if any(kw in row_lower for kw in ["brought forward", "carried forward", "b/f", "c/f"]):
                            continue

                        date_val = None
                        if col_map.get("date_col") is not None and col_map["date_col"] < len(cells):
                            raw_date = cells[col_map["date_col"]]
                            if raw_date:
                                date_val = parse_date_uk(raw_date)
                                if date_val:
                                    last_date = date_val
                            if not date_val and last_date:
                                date_val = last_date

                        desc = ""
                        if col_map.get("desc_col") is not None and col_map["desc_col"] < len(cells):
                            desc = cells[col_map["desc_col"]]

                        amount = None
                        direction = "UNKNOWN"
                        balance = None

                        for ac in col_map.get("amount_cols", []):
                            idx, role = ac["index"], ac["role"]
                            if idx < len(cells) and cells[idx]:
                                amt_result = parse_amount_uk(cells[idx])
                                if amt_result.amount is not None:
                                    if role == "paid_in":
                                        amount = abs(amt_result.amount)
                                        direction = "CREDIT"
                                    elif role == "withdrawn":
                                        amount = abs(amt_result.amount)
                                        direction = "DEBIT"
                                    elif role == "balance":
                                        balance = amt_result.amount
                                    elif role == "amount" and amount is None:
                                        amount = abs(amt_result.amount)

                        if amount is None and balance is None:
                            continue

                        from app.pipeline.table_extractor import is_balance_marker
                        if is_balance_marker(desc):
                            continue

                        parsed_rows.append({
                            "date_val": date_val,
                            "desc": desc[:500].replace("\n", " ") if desc else "",
                            "amount": amount,
                            "direction": direction,
                            "balance": balance,
                            "method": method,
                        })

                if parsed_rows:
                    logger.info("tabula_parse_success", method=method, rows=len(parsed_rows))
                    break

            except Exception as e:
                logger.warning("tabula_fallback_error", method=method, error=str(e))
                parsed_rows = []
                continue

        if not parsed_rows:
            # Fall through to Camelot
            logger.info("trying_camelot_fallback", segment=segment.segment_index)
            return await self._analyse_segment_camelot_fallback(
                session, doc_id, run_id, segment
            )

        # Persist using savepoint
        tx_records = []
        try:
            async with session.begin_nested():
                for idx, pr in enumerate(parsed_rows):
                    tx_rec = Transaction(
                        doc_id=uuid.UUID(doc_id),
                        run_id=uuid.UUID(run_id),
                        segment_id=segment.segment_id,
                        row_index=idx,
                        posted_date=pr["date_val"],
                        description_raw=pr["desc"],
                        description_clean=pr["desc"].strip(),
                        amount=pr["amount"],
                        direction=pr["direction"],
                        direction_source=f"tabula_{pr['method']}",
                        running_balance=pr["balance"],
                        balance_confirmed=False,
                        page_index=segment.start_page,
                        confidence_overall=Decimal("0.82"),
                        confidence_amount=Decimal("0.82"),
                        confidence_date=Decimal("0.82") if pr["date_val"] else Decimal("0.30"),
                        confidence_direction=Decimal("0.90") if pr["direction"] != "UNKNOWN" else Decimal("0.40"),
                    )
                    session.add(tx_rec)
                    tx_records.append(tx_rec)
            logger.info("tabula_fallback_persisted", transactions=len(tx_records))

        except Exception as e:
            logger.error("tabula_fallback_persist_failed", error=str(e))
            tx_records = []
            # Fall through to Camelot
            return await self._analyse_segment_camelot_fallback(
                session, doc_id, run_id, segment
            )

        return tx_records

    # ─── Fallback: Camelot table extraction ───────────────────

    async def _analyse_segment_camelot_fallback(
        self,
        session: AsyncSession,
        doc_id: str,
        run_id: str,
        segment: DocumentSegment,
    ) -> list[Transaction]:
        """Final fallback: Camelot stream then lattice."""
        try:
            import camelot
        except ImportError:
            logger.warning("camelot_not_installed")
            return []

        doc = await self._load_document(session, doc_id)
        pdf_path = self._get_pdf_path(doc)
        page_range = f"{segment.start_page + 1}-{segment.end_page + 1}"

        tx_records = []

        for flavor in ["stream", "lattice"]:
            parsed_rows = []
            try:
                tables = camelot.read_pdf(
                    pdf_path, pages=page_range, flavor=flavor, suppress_stdout=True
                )
                if not tables:
                    continue

                col_map = {}
                header_found = False

                for table in tables:
                    df = table.df
                    for ri in range(df.shape[0]):
                        cells = [str(df.iloc[ri, ci]).strip() for ci in range(df.shape[1])]
                        row_lower = " ".join(cells).lower()

                        # Detect header
                        if not header_found and any(kw in row_lower for kw in
                                ["date", "description", "paid in", "withdrawn",
                                 "balance", "money in", "money out", "debit", "credit"]):
                            header = [c.lower() for c in cells]
                            col_map = self._map_table_columns(header)
                            if col_map.get("amount_cols"):
                                header_found = True
                                logger.info("camelot_header_found", flavor=flavor, col_map=col_map)
                            continue

                        if not header_found or not col_map.get("amount_cols"):
                            continue

                        if any(kw in row_lower for kw in ["brought forward", "carried forward", "b/f", "c/f"]):
                            continue

                        # Extract fields using existing column map
                        row_strs = cells
                        date_val = None
                        if col_map.get("date_col") is not None and col_map["date_col"] < len(row_strs):
                            raw_date = row_strs[col_map["date_col"]]
                            if raw_date:
                                date_val = parse_date_uk(raw_date)

                        desc = ""
                        if col_map.get("desc_col") is not None and col_map["desc_col"] < len(row_strs):
                            desc = row_strs[col_map["desc_col"]]

                        amount = None
                        direction = "UNKNOWN"
                        balance = None

                        for ac in col_map.get("amount_cols", []):
                            idx, role = ac["index"], ac["role"]
                            if idx < len(row_strs) and row_strs[idx]:
                                amt_result = parse_amount_uk(row_strs[idx])
                                if amt_result.amount is not None:
                                    if role == "paid_in":
                                        amount = abs(amt_result.amount)
                                        direction = "CREDIT"
                                    elif role == "withdrawn":
                                        amount = abs(amt_result.amount)
                                        direction = "DEBIT"
                                    elif role == "balance":
                                        balance = amt_result.amount
                                    elif role == "amount" and amount is None:
                                        amount = abs(amt_result.amount)

                        if amount is None and balance is None:
                            continue

                        from app.pipeline.table_extractor import is_balance_marker
                        if is_balance_marker(desc):
                            continue

                        parsed_rows.append({
                            "date_val": date_val,
                            "desc": desc[:500].replace("\n", " ") if desc else "",
                            "amount": amount,
                            "direction": direction,
                            "balance": balance,
                            "flavor": flavor,
                        })

                if parsed_rows:
                    logger.info("camelot_fallback_success", flavor=flavor, rows=len(parsed_rows))
                    break  # Use first successful flavor

            except Exception as e:
                logger.warning("camelot_fallback_error", flavor=flavor, error=str(e))
                parsed_rows = []
                continue

        if not parsed_rows:
            return tx_records

        # Persist using savepoint so failure doesn't dirty session
        try:
            async with session.begin_nested():
                for idx, pr in enumerate(parsed_rows):
                    tx_rec = Transaction(
                        doc_id=uuid.UUID(doc_id),
                        run_id=uuid.UUID(run_id),
                        segment_id=segment.segment_id,
                        row_index=idx,
                        posted_date=pr["date_val"],
                        description_raw=pr["desc"],
                        description_clean=pr["desc"].strip(),
                        amount=pr["amount"],
                        direction=pr["direction"],
                        direction_source=f"camelot_{pr['flavor']}",
                        running_balance=pr["balance"],
                        balance_confirmed=False,
                        page_index=segment.start_page,
                        confidence_overall=Decimal("0.75"),
                        confidence_amount=Decimal("0.75"),
                        confidence_date=Decimal("0.75") if pr["date_val"] else Decimal("0.30"),
                        confidence_direction=Decimal("0.85") if pr["direction"] != "UNKNOWN" else Decimal("0.40"),
                    )
                    session.add(tx_rec)
                    tx_records.append(tx_rec)
            logger.info("camelot_fallback_persisted", transactions=len(tx_records))
        except Exception as e:
            logger.error("camelot_fallback_persist_failed", error=str(e))
            tx_records = []

        return tx_records

    def _map_table_columns(self, header: list[str]) -> dict:
        """Map header row to column roles for pdfplumber fallback."""
        result = {"date_col": None, "desc_col": None, "amount_cols": []}

        date_kw = ["date"]
        desc_kw = ["description", "details", "particulars", "narrative", "transaction"]
        paid_in_kw = ["paid in", "credit", "money in", "deposit", "in"]
        withdrawn_kw = ["withdrawn", "debit", "money out", "paid out", "withdrawal", "out"]
        balance_kw = ["balance"]
        amount_kw = ["amount"]

        for i, h in enumerate(header):
            h_lower = h.lower().strip()
            if not h_lower:
                continue

            if any(kw in h_lower for kw in date_kw) and result["date_col"] is None:
                result["date_col"] = i

            elif any(kw in h_lower for kw in desc_kw) and result["desc_col"] is None:
                result["desc_col"] = i

            elif any(kw in h_lower for kw in paid_in_kw):
                result["amount_cols"].append({"index": i, "role": "paid_in"})

            elif any(kw in h_lower for kw in withdrawn_kw):
                result["amount_cols"].append({"index": i, "role": "withdrawn"})

            elif any(kw in h_lower for kw in balance_kw):
                result["amount_cols"].append({"index": i, "role": "balance"})

            elif any(kw in h_lower for kw in amount_kw):
                result["amount_cols"].append({"index": i, "role": "amount"})

        return result