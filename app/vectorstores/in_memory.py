"""
Enterprise RAG OS — In Memory Vector Store
==========================================

Purpose:
    A lightweight, numpy-based in-memory vector store that requires zero
    external services or Docker containers. Ideal for local development.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from app.core.exceptions import VectorStoreError
from app.logging.logger import get_logger
from app.vectorstores.base import BaseVectorStore, VectorSearchResult

logger = get_logger(__name__)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

class InMemoryVectorStore(BaseVectorStore):
    """
    In-memory vector database backend.
    """

    def __init__(self, collection_name: str, dimension: int) -> None:
        self.collection_name = collection_name
        self.dimension = dimension
        
        # In-memory storage
        self.ids: list[str] = []
        self.embeddings: list[np.ndarray] = []
        self.documents: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        
        logger.info(f"Initialized InMemoryVectorStore for collection '{collection_name}'")

    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Add documents and embeddings to memory."""
        if not ids:
            return

        if len(ids) != len(embeddings) or len(ids) != len(documents):
            raise VectorStoreError(
                "Lists for ids, embeddings, and documents must have the same length."
            )

        metadatas = metadatas or [{} for _ in ids]

        for i, doc_id in enumerate(ids):
            # If doc_id already exists, remove it first to "upsert"
            if doc_id in self.ids:
                idx = self.ids.index(doc_id)
                self.ids.pop(idx)
                self.embeddings.pop(idx)
                self.documents.pop(idx)
                self.metadatas.pop(idx)
                
            self.ids.append(doc_id)
            self.embeddings.append(np.array(embeddings[i]))
            self.documents.append(documents[i])
            self.metadatas.append(metadatas[i])
            
        logger.info(f"Added {len(ids)} points to InMemoryVectorStore")

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        """Search for similar documents using cosine similarity."""
        if not self.embeddings:
            return []

        q_vec = np.array(query_embedding)
        results = []
        
        for idx in range(len(self.ids)):
            # Basic exact match filtering
            if filters:
                skip = False
                for k, v in filters.items():
                    if self.metadatas[idx].get(k) != v:
                        skip = True
                        break
                if skip:
                    continue
            
            score = cosine_similarity(q_vec, self.embeddings[idx])
            results.append((score, idx))
            
        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        
        top_results = []
        for score, idx in results[:top_k]:
            top_results.append(
                VectorSearchResult(
                    id=self.ids[idx],
                    content=self.documents[idx],
                    score=score,
                    metadata=self.metadatas[idx]
                )
            )
            
        return top_results

    async def delete(self, ids: list[str], **kwargs: Any) -> None:
        """Delete points by ID."""
        for doc_id in ids:
            if doc_id in self.ids:
                idx = self.ids.index(doc_id)
                self.ids.pop(idx)
                self.embeddings.pop(idx)
                self.documents.pop(idx)
                self.metadatas.pop(idx)
        logger.info(f"Deleted points from InMemoryVectorStore")

    @property
    def name(self) -> str:
        return "in_memory"

    async def health_check(self) -> bool:
        """Always healthy."""
        return True

    async def count(self) -> int:
        """Return the total number of documents in the store."""
        return len(self.ids)
