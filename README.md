# Basic vs GraphRAG Comparison

A comprehensive comparison between conventional RAG (Retrieval-Augmented Generation) and GraphRAG approaches, featuring both implementations with a FastAPI interface for performance evaluation.

## 🎯 Overview

This project demonstrates the differences in performance, accuracy, and implementation complexity between:

- **Basic RAG**: Traditional vector similarity search using Qdrant vector database
- **GraphRAG**: Graph-based retrieval using Neo4j graph database with enhanced relationship understanding

Both implementations use the EU AI Act document as a knowledge base for compliance-related questions.

## 🏗️ Architecture

### Basic RAG Implementation
- **Vector Store**: Qdrant for efficient similarity search
- **Embeddings**: OpenAI text-embedding-3-large
- **Framework**: LangGraph for state management
- **LLM**: gpt-4.1-mini for response generation

### GraphRAG Implementation  
- **Graph Database**: Neo4j for relationship-aware storage
- **Embeddings**: OpenAI embeddings with graph enhancement
- **Framework**: Neo4j GraphRAG with NLP pipeline
- **LLM**: gpt-4.1-mini with graph-augmented context

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

### 1. Clone and Setup
```bash
git clone https://github.com/Anis-Bouhamadouche/basic-vs-graphrag.git
cd basic-vs-graphrag

# Run setup script
chmod +x setup.sh
./setup.sh

# Install dependencies
make install
```

### 2. Environment Configuration
Create a `.env` file in the project root:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ENDPOINT=https://api.openai.com/v1
OPENAI_DEPLOYMENT_EMBEDDING=text-embedding-3-large

# Neo4j Configuration  
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=abcd1234
INDEX_NAME=eu_ai_act_index

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=optional_api_key
```

### 3. Start Services
```bash
# Start Neo4j and Qdrant
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 4. Run the Application
```bash
# Start the FastAPI server
make api
# or
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

## 📊 API Endpoints

### Health & Status
- `GET /` - Root endpoint
- `GET /health` - Health check

### Basic RAG
- `POST /basic-rag/ingest-pdf` - Ingest PDF document into vector store
- `POST /basic-rag/chat` - Chat with basic RAG system

### GraphRAG
- `POST /graph-rag/ingest-pdf` - Ingest PDF into graph database
- `GET /graph-rag/stats` - Get graph statistics
- `GET /graph-rag/schema` - View graph schema
- `PUT /graph-rag/schema` - Update graph schema
- `POST /graph-rag/create-index` - Create vector index
- `DELETE /graph-rag/clear` - Clear graph data

### Example Chat Request
```bash
curl -X POST "http://localhost:8000/basic-rag/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the definition of risk according to the EU AI Act?",
    "temperature": 0.0,
    "max_tokens": 500
  }'
```

## 🛠️ Development

### Code Quality
```bash
# Format code
make format

# Run linting
make lint  

# Type checking
make type-check

# Run all checks
make check
```

### Project Structure
```
src/
├── api/                 # FastAPI application
│   ├── app.py          # Main application
│   └── models.py       # Pydantic models
├── basic_rag/          # Basic RAG implementation
│   ├── rag.py          # LangGraph RAG system
│   ├── vector_db.py    # Qdrant integration
│   ├── loader.py       # Document processing
│   └── utils.py        # Utilities
└── graph_rag/          # GraphRAG implementation
    └── graphdb.py      # Neo4j integration
```

## 🔧 Configuration

### RAG Configuration
```python
from basic_rag.rag import RAGConfig

config = RAGConfig(
    collection_name="eu_ai_act",
    embedding_model="text-embedding-3-large", 
    top_k=10,
    temperature=0.0,
    chat_model_name="gpt-4.1-mini"
)
```

### Advanced Usage
```python
from basic_rag.rag import BasicRAGChat

# Initialize RAG system
rag_chat = BasicRAGChat(config)

# Simple question
answer = rag_chat.get_answer("What is AI risk?")

# Get answer with context documents
answer, context = rag_chat.get_answer("What is AI risk?", include_context=True)

# Stream responses
for update in rag_chat.stream({"question": "What is AI risk?"}):
    print(update)
```

## 📈 Performance Comparison

The project enables direct comparison between Basic RAG and GraphRAG approaches across multiple dimensions:

- **Accuracy**: Precision of answers to domain-specific questions
- **Context Relevance**: Quality of retrieved supporting information  
- **Response Time**: Speed of query processing and response generation
- **Scalability**: Performance with larger document collections
- **Relationship Understanding**: Ability to connect related concepts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code quality (`make check`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Neo4j GraphRAG](https://neo4j.com/labs/graphrag/)
- [Qdrant Vector Database](https://qdrant.tech/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 📞 Support

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainers.
