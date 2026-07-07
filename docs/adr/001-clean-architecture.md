# Architecture Decision Record: Clean Architecture

## Status
**Accepted** — Phase 1

## Context
Enterprise RAG OS needs an architecture that supports:
- Multiple interchangeable components (vector stores, LLMs, embedders)
- Independent testing of each layer
- Long-term maintainability as the system grows
- Clear dependency direction (no circular dependencies)

## Decision
We adopt **Clean Architecture** with four layers:

1. **API Layer** — HTTP routes, request/response handling, middleware
2. **Service Layer** — Business logic, pipeline orchestration, agent coordination
3. **Domain Layer** — Schemas, models, abstract interfaces (base classes)
4. **Infrastructure Layer** — Concrete implementations (ChromaDB, OpenAI, etc.)

**Dependency Rule**: Inner layers never depend on outer layers. The Domain layer
has zero external dependencies. The Infrastructure layer depends on Domain
(implements its interfaces) but never on API or Service directly.

## Alternatives Considered

### MVC (Model-View-Controller)
- **Pro**: Well-known, simple for CRUD apps
- **Con**: No "view" in an API-only system; couples controllers to models
- **Rejected**: Too simplistic for a multi-component AI pipeline

### Hexagonal Architecture (Ports & Adapters)
- **Pro**: Conceptually identical to our approach
- **Con**: Less widely understood terminology
- **Decision**: We use Clean Architecture terminology but the result is equivalent

### No formal architecture
- **Pro**: Faster initial development
- **Con**: Becomes unmaintainable past ~10K lines; coupling prevents testing
- **Rejected**: Unacceptable for an enterprise system

## Consequences
- **Positive**: Each component is independently testable and replaceable
- **Positive**: New developers understand the codebase quickly
- **Positive**: Adding new implementations (e.g., new vector store) requires no existing code changes
- **Negative**: More files and boilerplate (base classes, interfaces)
- **Negative**: Simple operations require more indirection
- **Mitigation**: The boilerplate cost is amortized over the system's lifetime
