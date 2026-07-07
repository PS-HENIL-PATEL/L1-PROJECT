# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                Enterprise RAG OS — Production Dockerfile                     ║
# ║                                                                              ║
# ║  Multi-stage build for minimal production image.                             ║
# ║  Stage 1: Install dependencies                                               ║
# ║  Stage 2: Copy only what's needed into slim runtime image                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --prefix=/install .

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /workspace

# Security: Run as non-root user
RUN groupadd -r raguser && useradd -r -g raguser raguser

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/

# Create runtime directories
RUN mkdir -p data vector_store cache logs \
    && chown -R raguser:raguser /workspace

# Switch to non-root user
USER raguser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
