"""
Enterprise RAG OS — Root API Router
======================================

Purpose:
    Aggregates all API routers (health, v1, etc.) into a single router
    that the FastAPI app includes. This is the single import point
    for all routes.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.v1.router import router as v1_router

# Root router aggregates all sub-routers
root_router = APIRouter()

# Health endpoints (no prefix — directly at /health)
root_router.include_router(health_router)

# API v1 endpoints (prefixed at /api/v1)
root_router.include_router(v1_router)
