"""
Enterprise RAG OS — Qdrant Vector Store
==========================================

Purpose:
    Implementation of the BaseVectorStore using Qdrant.
    Provides fast, scalable similarity search and filtering.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.core.exceptions import VectorStoreError
from app.logging.logger import get_logger
from app.vectorstores.base import BaseVectorStore, VectorSearchResult

logger = get_logger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector database backend.

    Expects Qdrant to be running locally via Docker (or remotely).
    Uses the asynchronous QdrantClient.
    """

    def __init__(
        self,
        collection_name: str,
        dimension: int,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        distance_metric: Distance = Distance.COSINE,
    ) -> None:
        """
        Initialize the Qdrant connection.

        Args:
            collection_name: Name of the collection to use.
            dimension: Vector dimensionality (must match embedding model).
            host: Qdrant server host.
            port: REST API port.
            grpc_port: gRPC port (for faster binary transfer).
            distance_metric: Similarity metric (Cosine, Dot, or Euclid).
        """
        self.collection_name = collection_name
        self.dimension = dimension
        self.distance_metric = distance_metric

        try:
            self.client = AsyncQdrantClient(
                host=host,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=True,
            )
            logger.info(
                "Initialized Qdrant client",
                host=host,
                collection=collection_name,
                dimension=dimension,
            )
        except Exception as e:
            logger.error("Failed to initialize Qdrant client", error=str(e))
            raise VectorStoreError(f"Qdrant connection failed: {e}") from e

    async def setup_collection(self) -> None:
        """
        Create the collection if it doesn't exist.
        Must be called before adding documents if the collection is new.
        """
        try:
            exists = await self.client.collection_exists(self.collection_name)
            if not exists:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=self.distance_metric,
                    ),
                )
        except Exception as e:
            logger.error("Failed to setup Qdrant collection", error=str(e))
            raise VectorStoreError(f"Collection setup failed: {e}") from e

    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Add documents and embeddings to Qdrant."""
        if not ids:
            return

        if len(ids) != len(embeddings) or len(ids) != len(documents):
            raise VectorStoreError(
                "Lists for ids, embeddings, and documents must have the same length."
            )

        metadatas = metadatas or [{} for _ in ids]

        # Qdrant payloads combine metadata and the raw text
        points = []
        for _i, (doc_id, vector, text, meta) in enumerate(
            zip(ids, embeddings, documents, metadatas, strict=False)
        ):
            payload = meta.copy()
            payload["text"] = text
            points.append(
                PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload=payload,
                )
            )

        try:
            # Upsert in batches to avoid overwhelming the gRPC connection
            batch_size = kwargs.get("batch_size", 100)
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True,  # Wait for indexing to complete
                )
            logger.info("Successfully added points to Qdrant", count=len(points))
        except Exception as e:
            logger.error("Failed to add points to Qdrant", error=str(e))
            raise VectorStoreError(f"Failed to upsert points: {e}") from e

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> list[VectorSearchResult]:
        """Search for similar documents in Qdrant."""
        try:
            # Basic mapping of dictionary filters to Qdrant FieldConditions
            # Note: For production, a more complex filter builder is needed to support
            # nested logical conditions (AND/OR).
            qdrant_filter = None
            if filters:
                from qdrant_client.http.models import FieldCondition, Filter, MatchValue
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                qdrant_filter = Filter(must=conditions)

            search_results = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False,
            )

            results = []
            for hit in search_results.points:
                payload = hit.payload or {}
                text = payload.pop("text", "")
                results.append(
                    VectorSearchResult(
                        id=str(hit.id),
                        score=hit.score,
                        content=text,
                        metadata=payload,
                    )
                )

            return results

        except UnexpectedResponse as e:
            if e.status_code == 404:
                logger.warning(
                    "Collection not found during search", collection=self.collection_name
                )
                return []
            logger.error("Qdrant search failed", error=str(e))
            raise VectorStoreError(f"Search failed: {e}") from e
        except Exception as e:
            logger.error("Qdrant search failed", error=str(e))
            raise VectorStoreError(f"Search failed: {e}") from e

    async def delete(
        self,
        ids: list[str],
        **_kwargs: Any,
    ) -> None:
        """Delete points by ID."""
        if not ids:
            return

        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids,
                wait=True,
            )
            logger.info("Successfully deleted points from Qdrant", count=len(ids))
        except Exception as e:
            logger.error("Failed to delete points from Qdrant", error=str(e))
            raise VectorStoreError(f"Failed to delete points: {e}") from e

    @property
    def name(self) -> str:
        return "qdrant"

    async def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            # This is a fast, lightweight call
            await self.client.get_collections()
            return True
        except Exception as e:
            logger.error("Qdrant health check failed", error=str(e))
            return False

    async def count(self) -> int:
        """Return the total number of documents in the store."""
        try:
            result = await self.client.count(collection_name=self.collection_name)
            return result.count
        except UnexpectedResponse as e:
            if e.status_code == 404:
                return 0
            raise VectorStoreError(f"Count failed: {e}") from e
        except Exception as e:
            logger.error("Qdrant count failed", error=str(e))
            raise VectorStoreError(f"Count failed: {e}") from e

    async def list_sources(self) -> list[dict[str, Any]]:
        """Return a list of unique document sources and metadata."""
        try:
            sources = {}
            offset = None
            
            while True:
                results, next_offset = await self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                for point in results:
                    payload = point.payload or {}
                    source = payload.get("source") or payload.get("absolute_path") or payload.get("url") or payload.get("file_name")
                    if source and source not in sources:
                        sources[source] = {
                            "source": source,
                            "type": payload.get("source_type", "unknown"),
                            "format": payload.get("format", "unknown"),
                            "file_name": payload.get("file_name", source)
                        }
                        
                offset = next_offset
                if offset is None:
                    break
                    
            return list(sources.values())
        except Exception as e:
            logger.error("Failed to list sources", error=str(e))
            return []
