"""
Content-addressable path generation for artifact storage.
All paths are relative to ARTIFACT_ROOT.
"""

import hashlib
from pathlib import Path


def doc_hash(file_bytes: bytes) -> str:
    """SHA-256 hash of file content."""
    return hashlib.sha256(file_bytes).hexdigest()


def raw_pdf_path(doc_id: str, file_name: str) -> str:
    """Path for the original uploaded PDF."""
    return f"{doc_id}/raw/{file_name}"


def rendered_page_path(doc_id: str, page_index: int, dpi: int = 300) -> str:
    """Path for a rendered page image."""
    return f"{doc_id}/rendered/page_{page_index:04d}_{dpi}dpi.png"


def normalized_page_path(doc_id: str, page_index: int) -> str:
    """Path for a preprocessed/normalized page image."""
    return f"{doc_id}/normalized/page_{page_index:04d}.png"


def extraction_artifact_path(doc_id: str, run_id: str, page_index: int) -> str:
    """Path for a page extraction JSON artifact."""
    return f"{doc_id}/extractions/{run_id}/page_{page_index:04d}.json"


def document_extraction_path(doc_id: str, run_id: str) -> str:
    """Path for the full document extraction JSON."""
    return f"{doc_id}/extractions/{run_id}/document.json"


def canonical_output_path(doc_id: str, run_id: str) -> str:
    """Path for the canonical output JSON."""
    return f"{doc_id}/output/{run_id}/canonical.json"


def ensure_parent_dirs(artifact_root: str, relative_path: str) -> Path:
    """Create parent directories and return the full absolute path."""
    full_path = Path(artifact_root) / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    return full_path