"""
/api/v1/admin/fingerprints endpoints.
Manages fingerprint templates for bank statement format recognition.
"""

import uuid
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, verify_api_key
from app.models.tables import Template, TemplateVersion

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/admin/fingerprints",
    tags=["fingerprints"],
    dependencies=[Depends(verify_api_key)],
)


# ---- Pydantic schemas for fingerprint API ----

class FingerprintInput(BaseModel):
    """Single fingerprint to load."""
    bank_name: str
    format_id: str
    account_type: str = "PERSONAL_CURRENT"
    layout_pattern: str = "UNKNOWN"
    fingerprint_json: dict
    column_mapping: Optional[dict] = None
    quirks: Optional[list] = None


class FingerprintBatchInput(BaseModel):
    """Batch of fingerprints to load."""
    fingerprints: list[FingerprintInput]


class TemplateResponse(BaseModel):
    """Template summary in API response."""
    template_id: str
    template_name: str
    provider_name: Optional[str] = None
    doc_family: str = "BANK_STATEMENT"
    is_active: bool = True
    version_count: int = 0
    latest_version: Optional[int] = None
    sample_count: int = 0

    model_config = {"from_attributes": True}


class TemplateDetailResponse(BaseModel):
    """Full template detail with version info."""
    template_id: str
    template_name: str
    provider_name: Optional[str] = None
    doc_family: str = "BANK_STATEMENT"
    is_active: bool = True
    versions: list[dict] = []

    model_config = {"from_attributes": True}


class TemplateListResponse(BaseModel):
    """List of templates."""
    templates: list[TemplateResponse]
    total: int


class LoadResultResponse(BaseModel):
    """Result of fingerprint loading."""
    loaded: int
    updated: int
    errors: list[str] = []


# ---- Endpoints ----

@router.post("", response_model=LoadResultResponse, status_code=status.HTTP_201_CREATED)
async def load_fingerprints(
    batch: FingerprintBatchInput,
    session: AsyncSession = Depends(get_db),
):
    """
    Load fingerprint JSON array into templates + template_versions tables.
    Creates new templates or adds versions to existing ones.
    """
    loaded = 0
    updated = 0
    errors = []

    for fp in batch.fingerprints:
        try:
            # Check if template exists for this bank + format
            result = await session.execute(
                select(Template).where(
                    Template.template_name == fp.format_id,
                    Template.provider_name == fp.bank_name,
                )
            )
            template = result.scalar_one_or_none()

            if not template:
                template = Template(
                    template_name=fp.format_id,
                    provider_name=fp.bank_name,
                    doc_family="BANK_STATEMENT",
                    is_active=True,
                )
                session.add(template)
                await session.flush()
                loaded += 1
            else:
                updated += 1

            # Determine next version number
            ver_result = await session.execute(
                select(func.max(TemplateVersion.version_number)).where(
                    TemplateVersion.template_id == template.template_id
                )
            )
            max_ver = ver_result.scalar() or 0

            # Build column mapping from fingerprint
            col_mapping = fp.column_mapping
            if col_mapping is None and "columns" in fp.fingerprint_json:
                col_mapping = {
                    "columns": fp.fingerprint_json.get("columns", []),
                    "layout_pattern": fp.layout_pattern,
                }

            quirks_data = None
            if fp.quirks is not None:
                quirks_data = {"quirks": fp.quirks}
            elif "quirks" in fp.fingerprint_json:
                quirks_data = {"quirks": fp.fingerprint_json.get("quirks", [])}

            version = TemplateVersion(
                template_id=template.template_id,
                version_number=max_ver + 1,
                fingerprint_json=fp.fingerprint_json,
                column_mapping_json=col_mapping,
                quirks_json=quirks_data,
                sample_count=len(fp.fingerprint_json.get("sample_files", [])),
            )
            session.add(version)

            logger.info(
                "fingerprint_loaded",
                bank=fp.bank_name,
                format_id=fp.format_id,
                version=max_ver + 1,
            )

        except Exception as e:
            errors.append(f"{fp.format_id}: {str(e)}")
            logger.error("fingerprint_load_error", format_id=fp.format_id, error=str(e))

    await session.flush()

    return LoadResultResponse(loaded=loaded, updated=updated, errors=errors)


@router.get("", response_model=TemplateListResponse)
async def list_fingerprints(
    active_only: bool = Query(True),
    provider: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_db),
):
    """List all fingerprint templates."""
    query = select(Template)

    if active_only:
        query = query.where(Template.is_active == True)
    if provider:
        query = query.where(Template.provider_name.ilike(f"%{provider}%"))

    query = query.order_by(Template.provider_name, Template.template_name)
    result = await session.execute(query)
    templates = result.scalars().all()

    template_responses = []
    for t in templates:
        # Get version info
        ver_result = await session.execute(
            select(
                func.count(TemplateVersion.version_id),
                func.max(TemplateVersion.version_number),
                func.sum(TemplateVersion.sample_count),
            ).where(TemplateVersion.template_id == t.template_id)
        )
        ver_row = ver_result.one()

        template_responses.append(TemplateResponse(
            template_id=str(t.template_id),
            template_name=t.template_name,
            provider_name=t.provider_name,
            doc_family=t.doc_family,
            is_active=t.is_active,
            version_count=ver_row[0] or 0,
            latest_version=ver_row[1],
            sample_count=ver_row[2] or 0,
        ))

    return TemplateListResponse(templates=template_responses, total=len(template_responses))


@router.get("/{template_id}", response_model=TemplateDetailResponse)
async def get_fingerprint(
    template_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Get full detail for a specific fingerprint template."""
    try:
        tid = uuid.UUID(template_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid template_id format")

    result = await session.execute(
        select(Template).where(Template.template_id == tid)
    )
    template = result.scalar_one_or_none()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Get all versions
    ver_result = await session.execute(
        select(TemplateVersion)
        .where(TemplateVersion.template_id == tid)
        .order_by(TemplateVersion.version_number.desc())
    )
    versions = ver_result.scalars().all()

    return TemplateDetailResponse(
        template_id=str(template.template_id),
        template_name=template.template_name,
        provider_name=template.provider_name,
        doc_family=template.doc_family,
        is_active=template.is_active,
        versions=[
            {
                "version_id": str(v.version_id),
                "version_number": v.version_number,
                "fingerprint_json": v.fingerprint_json,
                "column_mapping_json": v.column_mapping_json,
                "quirks_json": v.quirks_json,
                "sample_count": v.sample_count,
                "accuracy_score": float(v.accuracy_score) if v.accuracy_score else None,
                "created_at": v.created_at.isoformat() if v.created_at else None,
            }
            for v in versions
        ],
    )


@router.delete("/{template_id}", status_code=status.HTTP_200_OK)
async def deactivate_fingerprint(
    template_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Deactivate a fingerprint template (soft delete)."""
    try:
        tid = uuid.UUID(template_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid template_id format")

    result = await session.execute(
        select(Template).where(Template.template_id == tid)
    )
    template = result.scalar_one_or_none()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    template.is_active = False
    await session.flush()

    return {"message": f"Template {template.template_name} deactivated", "template_id": template_id}


@router.post("/match")
async def match_document_to_template(
    provider_name: str = Query(..., description="Detected provider name"),
    header_tokens: Optional[str] = Query(None, description="Comma-separated header tokens"),
    session: AsyncSession = Depends(get_db),
):
    """
    Match a document to the best fingerprint template.
    Used by the pipeline to get column mapping hints.
    """
    # Find active templates for this provider
    result = await session.execute(
        select(Template).where(
            Template.provider_name.ilike(f"%{provider_name}%"),
            Template.is_active == True,
        )
    )
    templates = result.scalars().all()

    if not templates:
        return {"match": None, "message": f"No templates found for provider: {provider_name}"}

    best_match = None
    best_score = 0.0

    for template in templates:
        # Get latest version
        ver_result = await session.execute(
            select(TemplateVersion)
            .where(TemplateVersion.template_id == template.template_id)
            .order_by(TemplateVersion.version_number.desc())
            .limit(1)
        )
        version = ver_result.scalar_one_or_none()
        if not version:
            continue

        score = 0.5  # Base score for provider match

        # If header tokens provided, compare against fingerprint
        if header_tokens and version.column_mapping_json:
            input_tokens = set(t.strip().lower() for t in header_tokens.split(",") if t.strip())
            fp_columns = version.column_mapping_json.get("columns", [])
            fp_headers = set()
            for col in fp_columns:
                if isinstance(col, dict) and "header_text" in col:
                    fp_headers.add(col["header_text"].lower().strip())

            if fp_headers and input_tokens:
                # Jaccard similarity
                intersection = input_tokens & fp_headers
                union = input_tokens | fp_headers
                jaccard = len(intersection) / len(union) if union else 0
                score = 0.3 + (0.7 * jaccard)

        if score > best_score:
            best_score = score
            best_match = {
                "template_id": str(template.template_id),
                "template_name": template.template_name,
                "provider_name": template.provider_name,
                "match_score": round(score, 4),
                "column_mapping": version.column_mapping_json,
                "quirks": version.quirks_json,
                "layout_pattern": version.column_mapping_json.get("layout_pattern") if version.column_mapping_json else None,
            }

    if best_match and best_score >= 0.5:
        return {"match": best_match}
    else:
        return {"match": None, "message": "No confident match found", "best_score": round(best_score, 4)}