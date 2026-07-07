"""
Verification script for Document Ingestion Pipeline.

Usage:
    # First, start Qdrant:
    docker-compose up -d qdrant
    
    # Run the ingestion script:
    python scripts/verify_ingestion.py
"""

import asyncio
from pathlib import Path

from app.chunking.recursive import RecursiveCharacterChunker
from app.embeddings.sentence_transformers import SentenceTransformerEmbedding
from app.loaders.local import LocalDirectoryLoader
from app.parsers.pdf import PyMuPDFParser
from app.parsers.text import TextParser
from app.pipelines.ingestion import DocumentIngestionPipeline
from app.vectorstores.qdrant import QdrantVectorStore


async def main() -> None:
    # 1. Initialize components
    print("Initializing components...")
    loader = LocalDirectoryLoader()

    parsers = {
        "txt": TextParser(),
        "md": TextParser(),
        "pdf": PyMuPDFParser(),
    }

    chunker = RecursiveCharacterChunker(chunk_size=512, chunk_overlap=50)
    embedder = SentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")

    vectorstore = QdrantVectorStore(
        collection_name="enterprise_test_collection",
        dimension=embedder.dimension,
        host="localhost",
        port=6333,
        grpc_port=6334,
    )

    # Wire the pipeline
    pipeline = DocumentIngestionPipeline(
        loader=loader,
        parsers=parsers,
        chunker=chunker,
        embedder=embedder,
        vectorstore=vectorstore,
    )

    # Create test data
    test_dir = Path("data/test_docs")
    test_dir.mkdir(parents=True, exist_ok=True)

    test_file = test_dir / "sample.txt"
    test_file.write_text(
        "Retrieval-Augmented Generation (RAG) is an AI framework for improving the quality "
        "of LLM-generated responses by grounding the model on external sources of knowledge. "
        "It combines an information retrieval component with a text generator model.",
        encoding="utf-8"
    )

    print("\nStarting ingestion pipeline...")
    stats = await pipeline.run(test_dir)
    print("\nIngestion Complete!")
    print("Stats:", stats)

    # Test Retrieval
    query = "What is RAG?"
    print(f"\nSearching for: '{query}'")

    query_vector = await embedder.embed_query(query)
    results = await vectorstore.search(query_vector, top_k=2)

    print(f"Found {len(results)} results:")
    for i, res in enumerate(results, 1):
        print(f"\nResult {i} (Score: {res.score:.4f}):")
        print(f"Text: {res.text}")
        print(f"Metadata: {res.metadata}")


if __name__ == "__main__":
    asyncio.run(main())
