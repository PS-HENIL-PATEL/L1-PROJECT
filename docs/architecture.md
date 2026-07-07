# Enterprise RAG OS — System Architecture

## Overview

Enterprise RAG OS is a production-grade Retrieval-Augmented Generation platform
designed for enterprise environments. It follows Clean Architecture principles
with a clear separation between API, service, domain, and infrastructure layers.

## High-Level Architecture

```mermaid
graph TB
    subgraph "API Layer"
        A[FastAPI Routes] --> B[Middleware Stack]
        B --> C[Health Endpoints]
    end

    subgraph "Service Layer"
        D[RAG Pipeline Service]
        E[Document Service]
        F[Query Service]
    end

    subgraph "Domain Layer"
        G[Schemas / DTOs]
        H[Base Interfaces]
        I[Domain Models]
    end

    subgraph "Infrastructure Layer"
        J[Vector Stores]
        K[Embedding Providers]
        L[LLM Providers]
        M[Document Loaders]
    end

    A --> D
    A --> E
    A --> F
    D --> H
    E --> H
    F --> H
    H --> J
    H --> K
    H --> L
    H --> M
```

## RAG Pipeline Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as API
    participant QU as Query Understanding
    participant R as Retriever
    participant RR as Reranker
    participant CO as Context Optimizer
    participant LLM as LLM Provider
    participant E as Evaluator

    U->>A: POST /api/v1/query
    A->>QU: Analyze & rewrite query
    QU->>R: Enhanced query
    R->>R: Dense + BM25 search
    R->>RR: Top-20 candidates
    RR->>CO: Top-5 reranked
    CO->>LLM: Optimized context + query
    LLM->>E: Answer + sources
    E->>A: Evaluated response
    A->>U: Answer + citations + explainability
```

## Component Interaction

```mermaid
graph LR
    subgraph "Document Ingestion"
        L[Loader] --> P[Parser]
        P --> C[Chunker]
        C --> E[Embedder]
        E --> VS[Vector Store]
    end

    subgraph "Query Processing"
        QU[Query Understanding] --> RET[Retriever]
        RET --> RR[Reranker]
        RR --> CTX[Context Builder]
        CTX --> LLM[LLM]
    end

    VS -.-> RET
```

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Web Framework | FastAPI | Async-native, auto-docs, Pydantic integration |
| Configuration | Pydantic Settings | Type-safe, validated, .env support |
| Logging | structlog | Structured JSON logs, correlation IDs |
| Testing | pytest + httpx | Async support, coverage, fixtures |
| Linting | ruff | Fast, comprehensive, Python-native |
| Type Checking | mypy | Strict type safety |
| Containerization | Docker | Reproducible builds, CI/CD |

## Security Architecture

- JWT-based authentication (Phase 6)
- Role-based access control (Phase 6)
- Input validation via Pydantic (all phases)
- Rate limiting (Phase 6)
- Prompt injection protection (Phase 6)
- Non-root Docker user (Phase 1)
- Secrets via environment variables (Phase 1)
