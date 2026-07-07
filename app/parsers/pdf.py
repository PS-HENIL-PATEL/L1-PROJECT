"""
Enterprise RAG OS — PDF Parser
=================================

Purpose:
    Parses PDF documents into structured text using PyMuPDF (fitz).
    PyMuPDF is extremely fast and provides good text extraction capabilities.
"""

from __future__ import annotations

from typing import Any

import fitz  # PyMuPDF

from app.core.exceptions import DocumentParsingError, UnsupportedFormatError
from app.logging.logger import get_logger
from app.parsers.base import BaseParser, ParsedContent

logger = get_logger(__name__)


class PyMuPDFParser(BaseParser):
    """
    Parser for PDF documents using PyMuPDF.

    Extracts text page by page. Supports basic metadata extraction.
    """

    async def parse(
        self,
        content: str | bytes,
        format: str,
        **kwargs: Any,
    ) -> ParsedContent:
        """
        Parse PDF bytes into structured text.

        Args:
            content: Raw PDF bytes.
            format: Must be 'pdf'.

        Returns:
            ParsedContent containing extracted text and pages.

        Raises:
            DocumentParsingError: If content is not bytes or PDF is corrupt.
        """
        if format.lower() != "pdf":
            raise UnsupportedFormatError(
                detail=f"PyMuPDFParser only supports PDF format, got {format}"
            )

        if not isinstance(content, bytes):
            raise DocumentParsingError(
                detail="PyMuPDFParser requires binary content (bytes)."
            )

        pages_text: list[str] = []
        pdf_metadata: dict[str, Any] = {}

        try:
            # Open the PDF from memory
            doc = fitz.open(stream=content, filetype="pdf")

            # Extract basic metadata
            pdf_metadata.update(doc.metadata)
            pdf_metadata["page_count"] = doc.page_count

            # Extract text from each page
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text()
                pages_text.append(text)

            doc.close()

        except Exception as e:
            logger.error("PyMuPDF parsing failed", error=str(e))
            raise DocumentParsingError(
                detail=f"Failed to parse PDF document: {e}",
                context={"exception": str(e)}
            ) from e

        # Combine all pages with explicit page breaks for context
        full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text)

        return ParsedContent(
            text=full_text.strip(),
            metadata=pdf_metadata,
            pages=pages_text,
        )

    def supported_formats(self) -> list[str]:
        return ["pdf"]

    @property
    def name(self) -> str:
        return "pymupdf_parser"
