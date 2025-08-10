#!/bin/bash

# Start API script for basic-vs-graphrag
# Launches the FastAPI application using uvicorn

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Activated virtual environment (.venv)"
else
    echo "⚠️  No .venv directory found. Using system Python."
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Loaded environment variables from .env"
else
    echo "⚠️  No .env file found. Using system environment variables."
fi

# Start the API server
echo "🚀 Starting FastAPI server..."
cd src && uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
