"""
Enterprise RAG OS — Documents API
=================================

Handles document ingestion and management.
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from typing import Any

import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.config.settings import Settings  # noqa: TC001
from app.core.dependencies import get_current_settings, get_ingestion_pipeline, get_vector_store
from app.logging.logger import get_logger
from app.pipelines.ingestion import DocumentIngestionPipeline  # noqa: TC001

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


class URLRequest(BaseModel):
    url: str


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    pipeline: DocumentIngestionPipeline = Depends(get_ingestion_pipeline),
    settings: Settings = Depends(get_current_settings),
) -> dict[str, Any]:
    """Upload and ingest a document into the knowledge base."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    upload_dir = settings.project_root / "data" / "uploaded"
    upload_dir.mkdir(parents=True, exist_ok=True)
    unique_dir = upload_dir / str(uuid.uuid4())
    unique_dir.mkdir(parents=True, exist_ok=True)

    temp_file_path = unique_dir / file.filename
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Processing uploaded file: {file.filename}")
        stats = await pipeline.run(input=str(unique_dir), recursive=False)

        final_path = upload_dir / file.filename
        if final_path.exists():
            final_path.unlink(missing_ok=True)

        # Retry logic for Windows transient file locks (WinError 5)
        for attempt in range(3):
            try:
                shutil.move(str(temp_file_path), str(final_path))
                break
            except OSError as e:
                if attempt == 2:
                    raise e
                await asyncio.sleep(0.2)

        # Clean up unique dir
        try:  # noqa: SIM105
            unique_dir.rmdir()
        except OSError:
            pass

        return {
            "status": "success",
            "message": f"Successfully ingested {file.filename}",
            "stats": stats,
            "filename": file.filename,
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        try:
            if temp_file_path.exists():
                temp_file_path.unlink()
        except OSError:
            pass

        try:
            if unique_dir.exists():
                unique_dir.rmdir()
        except OSError:
            pass

        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e


@router.post("/url", status_code=status.HTTP_202_ACCEPTED)
async def ingest_url(
    req: URLRequest,
    pipeline: DocumentIngestionPipeline = Depends(get_ingestion_pipeline),
    settings: Settings = Depends(get_current_settings),
) -> dict[str, Any]:
    """Scrape and ingest a URL into the knowledge base."""
    if not req.url.startswith("http"):
        req.url = "https://" + req.url

    try:
        logger.info(f"Fetching URL: {req.url}")

        is_youtube = "youtube.com" in req.url or "youtu.be" in req.url
        text = ""

        # 1. Always fetch the raw HTML to get the Title and Description
        res = requests.get(req.url, timeout=10)
        res.raise_for_status()

        soup = BeautifulSoup(res.text, "html.parser")

        if is_youtube:
            # 2. Extract Video Title
            video_title = soup.title.string.strip() if soup.title and soup.title.string else "Unknown YouTube Video"  # noqa: E501

            # Extract video ID
            video_id = None
            if "youtu.be" in req.url:
                video_id = req.url.split("/")[-1].split("?")[0]
            elif "youtube.com" in req.url:  # noqa: SIM102
                if "v=" in req.url:
                    video_id = req.url.split("v=")[1].split("&")[0]

            transcript_text = ""
            if video_id:
                try:
                    from youtube_transcript_api import YouTubeTranscriptApi
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    transcript_text = " ".join([t['text'] for t in transcript])
                    logger.info(f"Successfully fetched transcript for video {video_id}")
                except Exception as e:
                    logger.warning(f"Failed to fetch YouTube transcript: {e}")

            if transcript_text:
                text = f"Video Title: {video_title}\n\nTranscript:\n{transcript_text}"
            else:
                # If no transcript, just use whatever we can scrape from HTML
                for script in soup(["script", "style", "nav", "footer"]):
                    script.extract()
                text = f"Video Title: {video_title}\n\n{soup.get_text(separator='\n', strip=True)}"

        else:
            # 3. Standard HTML Scraper
            for script in soup(["script", "style", "nav", "footer"]):
                script.extract()
            text = soup.get_text(separator="\n", strip=True)

        upload_dir = settings.project_root / "data" / "uploaded"
        upload_dir.mkdir(parents=True, exist_ok=True)
        unique_dir = upload_dir / str(uuid.uuid4())
        unique_dir.mkdir(parents=True, exist_ok=True)

        import re
        filename = re.sub(r'[<>:"/\\|?*&=#+%]', '_', req.url.split("//")[-1]) + ".txt"
        temp_file_path = unique_dir / filename

        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(f"Source URL: {req.url}\n\n{text}")

        stats = await pipeline.run(input=str(unique_dir), recursive=False)

        if temp_file_path.exists():
            temp_file_path.unlink()
        unique_dir.rmdir()

        return {
            "status": "success",
            "message": f"Successfully ingested {req.url}",
            "stats": stats,
            "filename": req.url,
        }
    except Exception as e:
        logger.error(f"URL Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"URL Ingestion failed: {e}") from e


@router.post("/clear", status_code=status.HTTP_200_OK)
async def clear_database(
    vectorstore: Any = Depends(get_vector_store)
) -> dict[str, Any]:
    """Wipes the vector database."""
    try:
        if hasattr(vectorstore, "client"):
            client = vectorstore.client
            await client.delete_collection(vectorstore.collection_name)
            await vectorstore.setup_collection()
        return {"status": "success", "message": "Knowledge base cleared successfully."}
    except Exception as e:
        logger.error(f"Failed to clear database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {e}") from e

@router.get("/list", status_code=status.HTTP_200_OK)
async def list_documents(
    vectorstore: Any = Depends(get_vector_store)
) -> dict[str, Any]:
    """List all unique ingested sources."""
    try:
        if hasattr(vectorstore, "list_sources"):
            sources = await vectorstore.list_sources()
            return {"status": "success", "sources": sources}
        return {"status": "success", "sources": []}
    except Exception as e:
        logger.error(f"Failed to list sources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sources: {e}") from e
