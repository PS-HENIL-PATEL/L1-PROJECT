"""
Enterprise RAG OS — Local Directory Loader
============================================

Purpose:
    Loads documents from a local filesystem directory.
    Supports reading text files as strings and binary files (like PDF) as bytes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.loaders.base import BaseLoader, LoadedDocument
from app.logging.logger import get_logger

logger = get_logger(__name__)


class LocalDirectoryLoader(BaseLoader):
    """
    Loader for local filesystem directories.

    Scans a directory recursively and loads files that match
    the supported formats.
    """

    # Formats that should be read as text (UTF-8)
    TEXT_FORMATS = {".txt", ".md", ".csv", ".json"}  # noqa: RUF012

    # Formats that should be read as binary
    BINARY_FORMATS = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp"}  # noqa: RUF012

    def __init__(self, allowed_extensions: set[str] | None = None) -> None:
        """
        Initialize the loader.

        Args:
            allowed_extensions: Optional subset of extensions to load (e.g., {".pdf"}).
                              If None, loads all supported extensions.
        """
        self.allowed_extensions = (
            allowed_extensions
            if allowed_extensions is not None
            else self.TEXT_FORMATS | self.BINARY_FORMATS
        )

    async def load(
        self,
        source: str | Path,
        recursive: bool = True,
        **kwargs: Any,  # noqa: ARG002
    ) -> list[LoadedDocument]:
        """
        Load documents from the specified directory.

        Args:
            source: Directory path to scan.
            recursive: Whether to scan subdirectories.

        Returns:
            List of LoadedDocument objects.
        """
        dir_path = Path(source)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error("Directory not found or is not a directory", path=str(dir_path))
            return []

        documents: list[LoadedDocument] = []
        logger.info(
            "Starting directory load",
            path=str(dir_path),
            recursive=recursive,
            extensions=list(self.allowed_extensions),
        )

        # Walk directory
        for root, _, files in os.walk(dir_path):
            if not recursive and root != str(dir_path):
                continue

            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                if ext not in self.allowed_extensions:
                    continue

                try:
                    doc = self._load_single_file(file_path, ext)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.error(
                        "Failed to load file",
                        file=str(file_path),
                        error=str(e),
                    )

        logger.info("Finished directory load", loaded_count=len(documents))
        return documents

    def _load_single_file(self, file_path: Path, ext: str) -> LoadedDocument | None:
        """Load a single file from disk."""
        stat = file_path.stat()
        size_bytes = stat.st_size

        # Determine read mode based on extension type
        if ext in self.TEXT_FORMATS:
            with open(file_path, encoding="utf-8") as f:
                content: str | bytes = f.read()
        elif ext in self.BINARY_FORMATS:
            with open(file_path, "rb") as f:
                content = f.read()
        else:
            return None

        # Build metadata
        metadata = {
            "file_name": file_path.name,
            "absolute_path": str(file_path.absolute()),
            "last_modified": stat.st_mtime,
        }

        return LoadedDocument(
            content=content,
            metadata=metadata,
            source=str(file_path),
            format=ext.lstrip("."),
            size_bytes=size_bytes,
        )

    def supported_formats(self) -> list[str]:
        """Return list of supported file extensions."""
        return list(self.TEXT_FORMATS | self.BINARY_FORMATS)

    @property
    def name(self) -> str:
        return "local_directory_loader"
