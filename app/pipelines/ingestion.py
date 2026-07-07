"""
Enterprise RAG OS — Document Ingestion Pipeline
===================================================

Purpose:
    Orchestrates the entire document ingestion process:
    Load -> Parse -> Chunk -> Embed -> Store
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.exceptions import PipelineError
from app.logging.logger import get_logger
from app.pipelines.base import BasePipeline, PipelineContext
from app.utils.ids import generate_id
from app.utils.timing import Timer

if TYPE_CHECKING:
    from app.chunking.base import BaseChunker
    from app.embeddings.base import BaseEmbeddingProvider
    from app.loaders.base import BaseLoader
    from app.parsers.base import BaseParser
    from app.vectorstores.base import BaseVectorStore

logger = get_logger(__name__)


class DocumentIngestionPipeline(BasePipeline[str | Path, dict[str, Any]]):
    """
    Pipeline for ingesting documents into the vector store.
    """

    def __init__(
        self,
        loader: BaseLoader,
        parsers: dict[str, BaseParser],
        chunker: BaseChunker,
        embedder: BaseEmbeddingProvider,
        vectorstore: BaseVectorStore,
    ) -> None:
        """
        Initialize the ingestion pipeline.

        Args:
            loader: Loader to fetch raw documents.
            parsers: Mapping of formats (e.g., "pdf") to parsers.
            chunker: Chunker to split parsed text.
            embedder: Embedding provider.
            vectorstore: Vector store to save embeddings and chunks.
        """
        self.loader = loader
        self.parsers = parsers
        self.chunker = chunker
        self.embedder = embedder
        self.vectorstore = vectorstore

    async def run(
        self,
        input: str | Path,
        context: PipelineContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run the ingestion pipeline.

        Args:
            input: Path to the directory to ingest.
            context: Optional pipeline context.
            **kwargs: Additional args passed to components.

        Returns:
            Dictionary with ingestion stats.
        """
        ctx = context or PipelineContext()
        stats = {
            "files_found": 0,
            "files_parsed": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "errors": 0,
        }

        try:
            # 1. Load Documents
            with Timer("pipeline_load") as t:
                loaded_docs = await self.loader.load(input, **kwargs)
                stats["files_found"] = len(loaded_docs)
            ctx.timings["load_ms"] = t.elapsed_ms

            if not loaded_docs:
                logger.warning("No documents found to ingest", source=str(input))
                return stats

            # Ensure collection exists if it's a Qdrant store
            if hasattr(self.vectorstore, "setup_collection"):
                await self.vectorstore.setup_collection()

            all_ids = []
            all_embeddings = []
            all_texts = []
            all_metadatas = []

            # 2. Process each document (Parse -> Chunk -> Embed)
            for doc in loaded_docs:
                try:
                    # Get appropriate parser
                    parser = self.parsers.get(doc.format.lower())
                    if not parser:
                        logger.warning(
                            "No parser configured for format",
                            format=doc.format,
                            file=doc.source,
                        )
                        ctx.errors.append(f"No parser for {doc.format}")
                        stats["errors"] += 1
                        continue

                    # Parse
                    parsed = await parser.parse(doc.content, doc.format)
                    stats["files_parsed"] += 1

                    # Combine document metadata with parsed metadata
                    combined_meta = {**doc.metadata, **parsed.metadata}
                    combined_meta["source_type"] = self.loader.name
                    combined_meta["document_id"] = generate_id()

                    # Chunk
                    chunks = self.chunker.chunk(parsed.text, metadata=combined_meta)
                    stats["chunks_created"] += len(chunks)

                    if not chunks:
                        continue

                    # Embed chunks in batches to avoid OOM
                    batch_size = 100
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i : i + batch_size]
                        texts_to_embed = [c.content for c in batch]

                        embed_result = await self.embedder.embed(texts_to_embed)

                        # Prepare for vector store
                        for chunk, emb in zip(batch, embed_result.embeddings, strict=False):
                            chunk_id = generate_id()
                            chunk_meta = chunk.metadata.copy()
                            chunk_meta.update({
                                "chunk_index": chunk.chunk_index,
                                "start_char": chunk.start_char,
                                "end_char": chunk.end_char,
                            })

                            all_ids.append(chunk_id)
                            all_embeddings.append(emb)
                            all_texts.append(chunk.content)
                            all_metadatas.append(chunk_meta)

                except Exception as e:
                    logger.error("Failed to process document", file=doc.source, error=str(e))
                    ctx.errors.append(f"Failed {doc.source}: {e}")
                    stats["errors"] += 1

            # 3. Store in Vector Database
            if all_ids:
                with Timer("pipeline_store") as t:
                    await self.vectorstore.add(
                        ids=all_ids,
                        embeddings=all_embeddings,
                        documents=all_texts,
                        metadatas=all_metadatas,
                    )
                ctx.timings["store_ms"] = t.elapsed_ms
                stats["chunks_stored"] = len(all_ids)

            logger.info("Ingestion pipeline complete", stats=stats)
            return stats

        except Exception as e:
            logger.error("Ingestion pipeline failed", error=str(e))
            raise PipelineError(f"Ingestion pipeline failed: {e}") from e

    async def health_check(self) -> bool:
        """Check health of all components."""
        embed_health = await self.embedder.health_check()
        vs_health = await self.vectorstore.health_check()
        return embed_health and vs_health

    @property
    def name(self) -> str:
        return "document_ingestion_pipeline"

    @property
    def stages(self) -> list[str]:
        return ["load", "parse", "chunk", "embed", "store"]
