FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from project root
COPY ../../pyproject.toml ./

# Install dependencies directly with pip (simpler for containers)
RUN pip install fastapi uvicorn[standard] python-dotenv \
    neo4j-graphrag[openai] langchain-openai qdrant-client \
    tenacity langgraph langchain-community langchain pypdf \
    langchain-qdrant

# Copy the entire src directory to maintain import structure
COPY ../../src/ ./

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
