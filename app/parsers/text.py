"""
Enterprise RAG OS — Text Parser
=================================

Purpose:
    Parses raw text and markdown documents into structured text.
    Mainly acts as a pass-through but standardizes the output format.
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import DocumentParsingError, UnsupportedFormatError
from app.logging.logger import get_logger
from app.parsers.base import BaseParser, ParsedContent

logger = get_logger(__name__)


class TextParser(BaseParser):
    """
    Parser for plain text and markdown documents.
    """

    async def parse(
        self,
        content: str | bytes,
        format: str,
        **kwargs: Any,  # noqa: ARG002
    ) -> ParsedContent:
        """
        Parse text/markdown into structured text.

        Args:
            content: Raw text string or bytes.
            format: Must be 'txt', 'md', 'csv', or 'json'.

        Returns:
            ParsedContent containing the text.
        """
        if format.lower() not in self.supported_formats():
            raise UnsupportedFormatError(
                detail=f"TextParser does not support format: {format}"
            )

        if isinstance(content, bytes):
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as e:
                logger.error("Failed to decode text as UTF-8", error=str(e))
                raise DocumentParsingError(
                    detail="Content is not valid UTF-8 text.",
                ) from e
        elif isinstance(content, str):
            text = content
        else:
            raise DocumentParsingError(
                detail=f"Expected str or bytes, got {type(content).__name__}",
            )

        return ParsedContent(
            text=text.strip(),
            metadata={"parsed_format": format},
            pages=None,  # Plain text doesn't usually have explicit pages
        )

    def supported_formats(self) -> list[str]:
        return ["txt", "md", "csv", "json"]

    @property
    def name(self) -> str:
        return "text_parser"
