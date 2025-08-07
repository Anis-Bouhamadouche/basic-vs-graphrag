# Development Setup Guide

This project now uses modern Python tooling for dependency management and code quality.

## Prerequisites

- Python 3.11+
- uv (installed automatically via setup script)

## Quick Setup

1. **Install uv and setup the project:**

   ```bash
   # Run the setup script to install uv and configure PATH
   ./setup.sh

   # Or install uv manually:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.local/bin/env  # or restart your shell
   ```

2. **Install dependencies:**

   ```bash
   # Install project dependencies
   uv sync

   # Install with development dependencies
   uv sync --extra dev
   ```

## Development Commands

### Using Make (Recommended)

```bash
# Show all available commands
make help

# Format code (auto-fix)
make format

# Check formatting without changes
make format-check

# Run linting
make lint

# Run type checking
make type-check

# Run all checks
make check

# Run API server
make api

# Clean cache files
make clean
```

### Using UV Directly

```bash
# Format code
uv run black src/
uv run isort src/

# Lint code
uv run flake8 src/

# Type check
uv run mypy src/

# Run API server
uv run uvicorn src.api.app:app --reload
```

## Pre-commit Hooks (Optional)

Install pre-commit hooks to automatically run checks before each commit:

```bash
uv run pre-commit install
```

## Code Quality Tools

- **uv**: Fast Python package manager and project manager
- **black**: Code formatter
- **isort**: Import sorter
- **flake8**: Linter with docstring and type checking plugins
- **mypy**: Static type checker
- **pre-commit**: Git hooks for automated checks

## Configuration Files

- `pyproject.toml`: Project metadata and tool configuration
- `.flake8`: Flake8 linting configuration
- `.pre-commit-config.yaml`: Pre-commit hooks configuration
- `Makefile`: Development commands
- `uv.lock`: Locked dependency versions

## Migration from requirements.txt

Your original `requirements.txt` dependencies have been migrated to `pyproject.toml`. The file is kept for compatibility but `uv` manages dependencies through the `pyproject.toml` file.

## Development Workflow

1. Make your changes
2. Run `make format` to auto-fix formatting
3. Run `make check` to validate all quality checks
4. Commit your changes

The tools are configured to work together harmoniously and follow Python best practices.
