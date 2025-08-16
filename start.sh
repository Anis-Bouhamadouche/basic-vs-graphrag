#!/bin/bash

# Build and start the RAG Chat application with containers

echo "🚀 Building RAG Chat Application..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your actual API keys and configuration"
    echo "💡 You'll need to set your OPENAI_API_KEY at minimum"
fi

# Build and start all services
echo "🏗️  Building and starting services..."
docker compose up --build

echo "✅ Application started!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔗 Backend API: http://localhost:8000"
echo "🔍 API Docs: http://localhost:8000/docs"
echo "📊 Neo4j Browser: http://localhost:7474"
