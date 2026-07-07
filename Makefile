.PHONY: help install dev test lint format type-check clean docker docker-up docker-down run health

# ── Variables ─────────────────────────────────────────────────────────────────
PYTHON := python
PIP := pip
APP := app.main:app
HOST := 0.0.0.0
PORT := 8000

# ── Help ──────────────────────────────────────────────────────────────────────
help: ## Show this help message
	@echo "Enterprise RAG OS - Development Commands"
	@echo "========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────────────────────
install: ## Install production dependencies
	$(PIP) install -e .

dev: ## Install all dependencies (production + dev)
	$(PIP) install -e ".[dev]"
	@echo "\n✅ Development environment ready."

setup: dev ## Full first-time setup
	copy .env.example .env 2>NUL || cp .env.example .env
	@echo "\n✅ Setup complete. Edit .env with your configuration."

# ── Development ───────────────────────────────────────────────────────────────
run: ## Start development server with auto-reload
	uvicorn $(APP) --host $(HOST) --port $(PORT) --reload --log-level debug

run-prod: ## Start production server
	uvicorn $(APP) --host $(HOST) --port $(PORT) --workers 4 --log-level info

# ── Quality ───────────────────────────────────────────────────────────────────
test: ## Run all tests with coverage
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit -v

test-integration: ## Run integration tests only
	pytest tests/integration -v -m integration

test-fast: ## Run tests without coverage (faster)
	pytest --no-cov -q

lint: ## Run linter (ruff)
	ruff check app/ tests/

format: ## Format code (ruff)
	ruff format app/ tests/
	ruff check --fix app/ tests/

type-check: ## Run type checker (mypy)
	mypy app/

check: lint type-check test ## Run all quality checks (lint + types + tests)

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build: ## Build Docker image
	docker build -f docker/Dockerfile -t enterprise-rag-os .

docker-up: ## Start all services with Docker Compose
	docker compose up -d

docker-down: ## Stop all Docker Compose services
	docker compose down

docker-logs: ## View Docker Compose logs
	docker compose logs -f

# ── Utilities ─────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	@if exist __pycache__ rd /s /q __pycache__
	@if exist .pytest_cache rd /s /q .pytest_cache
	@if exist .mypy_cache rd /s /q .mypy_cache
	@if exist .ruff_cache rd /s /q .ruff_cache
	@if exist htmlcov rd /s /q htmlcov
	@if exist reports rd /s /q reports
	@if exist dist rd /s /q dist
	@if exist build rd /s /q build
	@if exist *.egg-info rd /s /q *.egg-info
	@echo "✅ Cleaned."

health: ## Check application health
	curl -s http://localhost:$(PORT)/health | python -m json.tool
