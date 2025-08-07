#!/bin/bash

# Add uv to PATH if not already present
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
    echo "Added uv to PATH in ~/.zshrc"
    echo "Please restart your shell or run: source ~/.zshrc"
fi

echo "uv is installed and ready to use!"
echo ""
echo "Available commands:"
echo "  make help          - Show all available commands"
echo "  make format        - Format code with black and isort"
echo "  make lint          - Run flake8 linting"
echo "  make type-check    - Run mypy type checking"
echo "  make check         - Run all checks (format, lint, type)"
echo "  make api           - Run the FastAPI application"
echo ""
echo "To use uv directly:"
echo "  ~/.local/bin/uv --help"
