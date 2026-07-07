"""
Enterprise RAG OS — Vision Image Parser
=========================================

Extracts text from images using an OpenAI-compatible Vision API (e.g. Groq llama-3.2-90b-vision-preview).
"""

import base64
from typing import Any
from openai import AsyncOpenAI
from app.parsers.base import BaseParser, ParsedContent
from app.config.settings import get_settings
from app.logging.logger import get_logger

logger = get_logger(__name__)


def encode_image(content: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(content).decode("utf-8")


class VisionImageParser(BaseParser):
    """
    Parser that uses an LLM Vision API to extract text from images.
    """
    
    def __init__(self) -> None:
        self.settings = get_settings()
        base_url = self.settings.llm.openai_base_url
        if self.settings.llm.default_provider == "ollama":
            base_url = self.settings.llm.ollama_base_url + "/v1"
            
        self.client = AsyncOpenAI(
            api_key=self.settings.llm.openai_api_key or "no-key",
            base_url=base_url,
        )
        self.model = "qwen/qwen3.6-27b"

    @property
    def name(self) -> str:
        return "vision_image_parser"

    def supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "webp"]

    async def parse(
        self,
        content: str | bytes,
        format: str,
        **kwargs: Any,
    ) -> ParsedContent:
        """
        Extract text from image using Vision API.
        """
        logger.info(f"Parsing image format {format} with Vision LLM")
        
        try:
            if isinstance(content, str):
                content = content.encode("utf-8")
                
            base64_image = encode_image(content)
            
            mime_type = "image/jpeg"
            if format.lower() == "png":
                mime_type = "image/png"
            elif format.lower() == "webp":
                mime_type = "image/webp"

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all readable text, data, and descriptions from this image. DO NOT TRUNCATE. If the image contains a table, you MUST transcribe EVERY SINGLE ROW exactly as it appears. Do not skip any rows or data. Format it cleanly as a Markdown table. CRITICAL: Ensure every row of the table is on a NEW LINE (\\n). Do NOT put the entire table on a single line. Provide a structured, accurate, and fully complete extraction."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=4096,
                temperature=0.1
            )
            
            extracted = response.choices[0].message.content
            if extracted is None:
                extracted = ""
                
            return ParsedContent(text=extracted.strip(), metadata={"parser": "vision_llm", "format": format})
            
        except Exception as e:
            logger.error(f"Vision parsing failed: {e}")
            raise RuntimeError(f"Vision parsing failed: {e}") from e
