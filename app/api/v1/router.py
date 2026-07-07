"""
Enterprise RAG OS — API v1 Router
====================================

Purpose:
    Version 1 API router. Groups all v1 endpoints under /api/v1/.
    In Phase 1, this is a placeholder. Phase 2+ will add document,
    query, and admin endpoints here.

Why API versioning?
    Breaking changes are inevitable. Versioning the API (/api/v1/, /api/v2/)
    allows existing clients to continue working while new clients use
    the latest version. This is standard practice in enterprise APIs.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.evaluate import router as evaluate_router
from app.api.v1.query import router as query_router
from app.api.v1.search import router as search_router
from app.api.v1.documents import router as documents_router

router = APIRouter(prefix="/api/v1", tags=["v1"])

router.include_router(search_router)
router.include_router(query_router)
router.include_router(evaluate_router)
router.include_router(documents_router)


@router.get(
    "/",
    summary="API v1 Root",
    description="API version 1 information and available endpoints.",
)
async def api_v1_root() -> dict[str, str | list[str]]:
    """Return API v1 information."""
    return {
        "version": "v1",
        "status": "active",
        "description": "Enterprise RAG OS API v1",
        "endpoints": [
            "/api/v1/search",
            "/api/v1/query",
            "/api/v1/evaluate",
            "/api/v1/documents (Phase 5)",
            "/api/v1/admin (Phase 6)",
        ],
    }
