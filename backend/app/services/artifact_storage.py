"""Artifact storage abstraction for optimization candidates and experiment results (Step 27).

Provides safe, atomic, filesystem-backed storage for candidate snapshots, trial traces,
and evaluation payloads under a configured artifact root:
1. Prevents path traversal and arbitrary filesystem escape.
2. Resolves artifact locations from trusted server-generated identifiers.
3. Performs atomic file writes using temp files and atomic rename.
4. Computes cryptographic SHA-256 content checksums.
5. Employs restricted serialization formats (strict JSON / text; no arbitrary pickle).
6. Returns safe, portable relative URNs (artifact://...) instead of machine-specific paths.
"""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import uuid

from app.core.config import Settings, get_settings
from app.core.exceptions import ArtifactStorageError
from app.core.logging import get_logger

logger = get_logger("artifact_storage")


class ArtifactStorageService:
    """Safe, atomic artifact persistence under an isolated root directory."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None, settings: Optional[Settings] = None) -> None:
        cfg = settings or get_settings()
        raw_base = base_dir or cfg.ARTIFACT_STORAGE_DIR
        self.base_dir = Path(raw_base).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, relative_key_or_urn: str) -> Path:
        """Resolve a relative key or artifact URN to an absolute Path inside base_dir.

        Raises ArtifactStorageError if path traversal is detected.
        """
        # Strip URN scheme if present
        clean_key = relative_key_or_urn
        if clean_key.startswith("artifact://"):
            clean_key = clean_key[len("artifact://"):]
        elif clean_key.startswith("artifacts://"):
            clean_key = clean_key[len("artifacts://"):]

        clean_key = clean_key.lstrip("/\\")

        # Normalize and resolve
        resolved = (self.base_dir / clean_key).resolve()

        # Enforce that resolved path stays within base_dir
        try:
            resolved.relative_to(self.base_dir)
        except ValueError as exc:
            logger.error("Path traversal blocked for artifact key '%s'", relative_key_or_urn)
            raise ArtifactStorageError(
                f"Invalid artifact path: access outside artifact root is forbidden.",
                original_error=exc,
            )

        return resolved

    def write_artifact(
        self,
        relative_key: str,
        data: Union[str, bytes, Dict[str, Any]],
        overwrite: bool = True,
    ) -> str:
        """Atomically persist an artifact and return its safe relative URN.

        Parameters:
            relative_key: Server-controlled relative path (e.g. 'jobs/{job_id}/candidate_{id}.json').
            data: Text, JSON dictionary, or raw bytes.
            overwrite: Whether overwriting an existing artifact at the exact key is permitted.

        Returns:
            Uniform relative artifact reference (e.g. 'artifact://jobs/{job_id}/...').
        """
        target_path = self._resolve_safe_path(relative_key)

        if not overwrite and target_path.exists():
            raise ArtifactStorageError(f"Artifact at '{relative_key}' already exists and overwrite is False.")

        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize payload
        if isinstance(data, dict) or isinstance(data, list):
            content_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        elif isinstance(data, str):
            content_bytes = data.encode("utf-8")
        elif isinstance(data, bytes):
            content_bytes = data
        else:
            raise ArtifactStorageError(
                f"Unsupported artifact data type '{type(data).__name__}'. Only JSON, str, and bytes are allowed."
            )

        # Atomic write via temporary file
        temp_filename = f".tmp_{uuid.uuid4().hex}_{target_path.name}"
        temp_path = target_path.parent / temp_filename

        try:
            with open(temp_path, "wb") as f:
                f.write(content_bytes)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_path, target_path)
        except Exception as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            logger.error("Failed to write artifact '%s': %s", relative_key, exc)
            raise ArtifactStorageError(f"Failed to persist artifact '{relative_key}'", original_error=exc)

        # Normalize relative key for URN
        rel_path = target_path.relative_to(self.base_dir).as_posix()
        urn = f"artifact://{rel_path}"
        logger.debug("Successfully persisted artifact '%s' (%d bytes)", urn, len(content_bytes))
        return urn

    def read_artifact(self, relative_key_or_urn: str) -> str:
        """Read artifact content as a UTF-8 string."""
        target_path = self._resolve_safe_path(relative_key_or_urn)
        if not target_path.is_file():
            raise ArtifactStorageError(f"Artifact '{relative_key_or_urn}' not found.")

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as exc:
            logger.error("Failed reading artifact '%s': %s", relative_key_or_urn, exc)
            raise ArtifactStorageError(f"Failed to read artifact '{relative_key_or_urn}'", original_error=exc)

    def read_json_artifact(self, relative_key_or_urn: str) -> Dict[str, Any]:
        """Read and deserialize a JSON artifact."""
        raw_text = self.read_artifact(relative_key_or_urn)
        try:
            return json.loads(raw_text)
        except Exception as exc:
            logger.error("Corrupted JSON artifact '%s': %s", relative_key_or_urn, exc)
            raise ArtifactStorageError(f"Artifact '{relative_key_or_urn}' contains invalid JSON.", original_error=exc)

    def artifact_exists(self, relative_key_or_urn: str) -> bool:
        """Check whether an artifact exists on disk without raising."""
        try:
            target_path = self._resolve_safe_path(relative_key_or_urn)
            return target_path.is_file()
        except ArtifactStorageError:
            return False

    def compute_checksum(self, relative_key_or_urn: str) -> str:
        """Compute SHA-256 hex digest of an existing artifact."""
        target_path = self._resolve_safe_path(relative_key_or_urn)
        if not target_path.is_file():
            raise ArtifactStorageError(f"Artifact '{relative_key_or_urn}' not found.")

        sha = hashlib.sha256()
        with open(target_path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()


@lru_cache(maxsize=1)
def get_artifact_storage_service() -> ArtifactStorageService:
    """Provide a singleton instance of the ArtifactStorageService."""
    return ArtifactStorageService()
