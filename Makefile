.PHONY: help install dev-install format lint type-check check test clean

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install project dependencies
	uv sync

dev-install:  ## Install project with development dependencies
	uv sync --extra dev

format:  ## Format code with black and isort
	uv run black src/
	uv run isort src/

format-check:  ## Check code formatting without making changes
	uv run black --check src/
	uv run isort --check-only src/

lint:  ## Run flake8 linting
	uv run flake8 src/

type-check:  ## Run mypy type checking
	MYPYPATH=src uv run mypy --namespace-packages src/

check: format-check lint type-check  ## Run all checks (format, lint, type)

fix: format  ## Auto-fix formatting issues

test:  ## Run tests (placeholder - add your test framework)
	@echo "No tests configured yet. Add pytest or your preferred test framework."

clean:  ## Clean up cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

api:  ## Run the FastAPI application
	uv run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
