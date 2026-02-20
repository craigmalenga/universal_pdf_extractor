"""
Artifact store for saving and loading pipeline artifacts.
Phase 1: Local filesystem (Railway volume mount).
Migration path to S3-compatible storage designed in.
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import structlog

from app.config import settings
from app.storage.paths import ensure_parent_dirs

logger = structlog.get_logger(__name__)


class ArtifactStore:
    """
    Save and load pipeline artifacts to/from storage.
    All paths are relative to ARTIFACT_ROOT.
    """

    def __init__(self, root: Optional[str] = None):
        self.root = Path(root or settings.ARTIFACT_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, relative_path: str, data: bytes) -> str:
        """Save raw bytes (PDF, image). Returns the relative path."""
        full_path = ensure_parent_dirs(str(self.root), relative_path)
        full_path.write_bytes(data)
        logger.info("artifact_saved", path=relative_path, size_bytes=len(data))
        return relative_path

    def save_json(self, relative_path: str, data: dict) -> str:
        """Save a JSON artifact. Returns the relative path."""
        full_path = ensure_parent_dirs(str(self.root), relative_path)
        full_path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
        logger.info("artifact_saved_json", path=relative_path)
        return relative_path

    def save_text(self, relative_path: str, text: str) -> str:
        """Save a text artifact. Returns the relative path."""
        full_path = ensure_parent_dirs(str(self.root), relative_path)
        full_path.write_text(text, encoding="utf-8")
        logger.info("artifact_saved_text", path=relative_path)
        return relative_path

    def load_bytes(self, relative_path: str) -> bytes:
        """Load raw bytes from storage."""
        full_path = self.root / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"Artifact not found: {relative_path}")
        return full_path.read_bytes()

    def load_json(self, relative_path: str) -> dict:
        """Load a JSON artifact from storage."""
        full_path = self.root / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"Artifact not found: {relative_path}")
        return json.loads(full_path.read_text(encoding="utf-8"))

    def load_text(self, relative_path: str) -> str:
        """Load a text artifact from storage."""
        full_path = self.root / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"Artifact not found: {relative_path}")
        return full_path.read_text(encoding="utf-8")

    def exists(self, relative_path: str) -> bool:
        """Check if an artifact exists."""
        return (self.root / relative_path).exists()

    def delete(self, relative_path: str) -> bool:
        """Delete an artifact. Returns True if it existed."""
        full_path = self.root / relative_path
        if full_path.exists():
            full_path.unlink()
            logger.info("artifact_deleted", path=relative_path)
            return True
        return False

    def delete_doc_artifacts(self, doc_id: str) -> int:
        """Delete all artifacts for a document. Returns count deleted."""
        doc_dir = self.root / doc_id
        if not doc_dir.exists():
            return 0
        count = sum(1 for _ in doc_dir.rglob("*") if _.is_file())
        shutil.rmtree(doc_dir)
        logger.info("doc_artifacts_deleted", doc_id=doc_id, count=count)
        return count

    def full_path(self, relative_path: str) -> Path:
        """Get the absolute filesystem path for an artifact."""
        return self.root / relative_path

    def list_artifacts(self, doc_id: str) -> list[str]:
        """List all artifact paths for a document."""
        doc_dir = self.root / doc_id
        if not doc_dir.exists():
            return []
        return [
            str(p.relative_to(self.root))
            for p in doc_dir.rglob("*")
            if p.is_file()
        ]