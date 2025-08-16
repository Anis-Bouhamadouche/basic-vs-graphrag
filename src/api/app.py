"""FastAPI application for basic vs graph RAG comparison."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Literal, Optional, cast

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_openai import OpenAIEmbeddings

from api.models import (
    ChatRequest,
    ChatResponse,
    CreateIndexRequest,
    IngestPDFGraphRequest,
    IngestPDFGraphResponse,
    IngestPDFRequest,
    IngestPDFResponse,
    SchemaUpdateRequest,
)
from basic_rag.loader import DocumentLoader
from basic_rag.rag import BasicRAGChat, RAGConfig
from graph_rag.graphdb import Neo4jConnection

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
INDEX_NAME = os.getenv("INDEX_NAME")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_DEPLOYMENT_EMBEDDING = os.getenv("OPENAI_DEPLOYMENT_EMBEDDING")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

SCHEMA: Optional[Dict[str, Any]] = None

# Global Neo4j connection
neo4j_conn = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Handle application startup and shutdown."""
    global neo4j_conn

    # Startup
    try:
        # Validate environment variables
        if not NEO4J_URI:
            raise ValueError("NEO4J_URI environment variable is required")
        if not NEO4J_USERNAME:
            raise ValueError("NEO4J_USERNAME environment variable is required")
        if not NEO4J_PASSWORD:
            raise ValueError("NEO4J_PASSWORD environment variable is required")

        neo4j_conn = Neo4jConnection(
            uri=NEO4J_URI, user=NEO4J_USERNAME, password=NEO4J_PASSWORD
        )
        neo4j_conn.__enter__()
        logger.info("Connected to Neo4j database")
        yield
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        raise
    finally:
        # Shutdown
        if neo4j_conn:
            neo4j_conn.__exit__(None, None, None)
            logger.info("Disconnected from Neo4j database")


app = FastAPI(
    title="Neo4j GraphRAG API",
    description="A simple API for Neo4j Graph RAG operations",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],  # Frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    logger.error(f"Validation error for {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid input data",
            "errors": exc.errors(),
            "status": "validation_error",
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle ValueError exceptions."""
    logger.error(f"Value error for {request.url}: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "status": "value_error"},
    )


@app.exception_handler(ConnectionError)
async def connection_error_handler(
    request: Request, exc: ConnectionError
) -> JSONResponse:
    """Handle connection errors."""
    logger.error(f"Connection error for {request.url}: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Service temporarily unavailable",
            "status": "connection_error",
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other unhandled exceptions."""
    logger.error(f"Unhandled exception for {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status": "server_error"},
    )


@app.get("/")
async def hello_world() -> dict[str, Any]:
    """Hello world endpoint that tests database connectivity."""
    try:
        if not neo4j_conn:
            raise HTTPException(
                status_code=500, detail="Database connection not available"
            )

        # Test database connectivity by getting server info
        with neo4j_conn.driver.session() as session:
            result = session.run("RETURN 'Hello from Neo4j!' as message")
            record = result.single()
            message = record["message"] if record else "No response"

        return {
            "message": "Hello World from FastAPI!",
            "database_message": message,
            "status": "connected",
        }

    except HTTPException:
        raise
    except ConnectionError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in hello_world: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    try:
        if not neo4j_conn:
            return {"status": "unhealthy", "database": "disconnected"}

        # Simple database ping
        with neo4j_conn.driver.session() as session:
            session.run("RETURN 1")

        return {"status": "healthy", "database": "connected"}

    except ConnectionError as e:
        logger.error(f"Database connection failed during health check: {e}")
        return {"status": "unhealthy", "database": "connection_failed", "error": str(e)}
    except Exception as e:
        logger.error(f"Health check failed with unexpected error: {e}")
        return {"status": "unhealthy", "database": "error", "error": str(e)}


@app.post("/graph-rag/ingest-pdf", response_model=IngestPDFGraphResponse)
async def ingest_pdf_to_kg(request: IngestPDFGraphRequest) -> IngestPDFGraphResponse:
    """Ingest a PDF file into the knowledge graph.

    This endpoint processes a PDF file and populates the knowledge graph
    using schema and OpenAI services.
    """
    try:
        if not neo4j_conn:
            raise HTTPException(
                status_code=500, detail="Database connection not available"
            )

        # Check if PDF file exists (additional check beyond Pydantic validation)
        if not os.path.exists(request.pdf_path):
            raise HTTPException(
                status_code=404, detail=f"PDF file not found: {request.pdf_path}"
            )

        # Validate OpenAI configuration
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

        logger.info(f"Starting PDF ingestion to knowledge graph: {request.pdf_path}")

        # Test file readability
        try:
            with open(request.pdf_path, "rb") as f:
                f.read(1)  # Try to read first byte
        except PermissionError:
            raise HTTPException(
                status_code=403, detail="Permission denied accessing the PDF file"
            )
        except Exception as e:
            logger.error(f"Error accessing PDF file: {e}")
            raise HTTPException(
                status_code=400, detail=f"Cannot access PDF file: {str(e)}"
            )

        # Validate required parameters before attempting population
        if SCHEMA is None:
            raise HTTPException(
                status_code=400,
                detail="Schema must be created first. Call /create-schema endpoint.",
            )
        if not OPENAI_ENDPOINT:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_ENDPOINT environment variable is required",
            )
        if not OPENAI_DEPLOYMENT_EMBEDDING:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_DEPLOYMENT_EMBEDDING environment variable is required",
            )

        # Populate the knowledge graph
        try:
            await neo4j_conn.populate_kg_from_pdf(
                pdf_path=request.pdf_path,
                schema=SCHEMA,
                api_key=OPENAI_API_KEY,
                endpoint=OPENAI_ENDPOINT,
                deployment=OPENAI_DEPLOYMENT_EMBEDDING,
                llm_model_name=request.llm_model_name,
                clear_existing=request.clear_existing,
                run_entity_resolution=request.run_entity_resolution,
            )
        except ConnectionError as e:
            logger.error(f"Database connection error during ingestion: {e}")
            raise HTTPException(status_code=503, detail="Database service unavailable")
        except ValueError as e:
            logger.error(f"Invalid input data for knowledge graph ingestion: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
        except Exception as e:
            logger.error(f"Error during knowledge graph population: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal server error during knowledge graph processing",
            )

        # Get knowledge graph statistics
        try:
            kg_stats = neo4j_conn.get_kg_stats()
        except Exception as e:
            logger.warning(f"Failed to get KG stats after ingestion: {e}")
            kg_stats = {"total_nodes": 0, "total_relationships": 0, "error_count": 1}

        logger.info("PDF ingestion to knowledge graph completed successfully")

        return IngestPDFGraphResponse(
            message="PDF successfully ingested into knowledge graph",
            status="success",
            kg_stats=kg_stats,
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"PDF file not found: {e}")
        raise HTTPException(status_code=404, detail=f"PDF file not found: {str(e)}")
    except PermissionError as e:
        logger.error(f"Permission denied accessing PDF: {e}")
        raise HTTPException(
            status_code=403, detail="Permission denied accessing the PDF file"
        )
    except ConnectionError as e:
        logger.error(f"Database connection error during ingestion: {e}")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    except ValueError as e:
        logger.error(f"Invalid input data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during PDF ingestion: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during PDF processing"
        )


@app.get("/graph-rag/stats")
async def get_kg_stats() -> dict[str, Any]:
    """Get knowledge graph statistics."""
    try:
        if not neo4j_conn:
            raise HTTPException(
                status_code=500, detail="Database connection not available"
            )

        stats = neo4j_conn.get_kg_stats()
        return {"status": "success", "kg_stats": stats}

    except HTTPException:
        raise
    except ConnectionError as e:
        logger.error(f"Database connection error getting stats: {e}")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error getting KG stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/graph-rag/clear")
async def clear_knowledge_graph() -> dict[str, str]:
    """Clear all data from the knowledge graph."""
    try:
        if not neo4j_conn:
            raise HTTPException(
                status_code=500, detail="Database connection not available"
            )

        neo4j_conn.clear_database()

        return {"message": "Knowledge graph cleared successfully", "status": "success"}

    except HTTPException:
        raise
    except ConnectionError as e:
        logger.error(f"Database connection error during clear: {e}")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error clearing KG: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/graph-rag/schema")
async def get_schema() -> dict[str, Any]:
    """Get the current knowledge graph schema configuration."""
    return {
        "status": "success",
        "schema": SCHEMA,
        "description": "Knowledge graph schema",
    }


@app.post("/graph-rag/create-index")
async def create_vector_index_endpoint(request: CreateIndexRequest) -> dict[str, Any]:
    """Create a vector index for similarity search.

    This endpoint creates a vector index on a specific node label
    to enable similarity search functionality.
    """
    try:
        if not neo4j_conn:
            raise HTTPException(
                status_code=500, detail="Database connection not available"
            )

        logger.info(
            f"Creating vector index '{request.index_name}' "
            f"for label '{request.label}'"
        )

        # Test database connectivity before creating index
        try:
            with neo4j_conn.driver.session() as session:
                session.run("RETURN 1")
        except Exception as e:
            logger.error(f"Database connectivity test failed: {e}")
            raise HTTPException(status_code=503, detail="Database service unavailable")

        # Create the vector index
        try:
            # Validate similarity function
            if request.similarity_fn not in ["cosine", "euclidean"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid similarity_fn: {request.similarity_fn}. Must be 'cosine' or 'euclidean'",
                )

            neo4j_conn.create_vector_index(
                index_name=request.index_name,
                label=request.label,
                embedding_property=request.embedding_property,
                dimensions=request.dimensions,
                similarity_fn=cast(
                    "Literal['cosine', 'euclidean']", request.similarity_fn
                ),
            )
        except ValueError as e:
            logger.error(f"Invalid index parameters: {e}")
            raise HTTPException(
                status_code=400, detail=f"Invalid index configuration: {str(e)}"
            )
        except ConnectionError as e:
            logger.error(f"Database connection error creating index: {e}")
            raise HTTPException(status_code=503, detail="Database service unavailable")
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            # Check if it's an index already exists error
            if "already exists" in str(e).lower():
                raise HTTPException(
                    status_code=409,
                    detail=f"Index '{request.index_name}' already exists",
                )
            raise HTTPException(
                status_code=500, detail="Internal server error creating vector index"
            )

        return {
            "message": f"Vector index '{request.index_name}' created successfully",
            "status": "success",
            "index_details": {
                "name": request.index_name,
                "label": request.label,
                "embedding_property": request.embedding_property,
                "dimensions": request.dimensions,
                "similarity_function": request.similarity_fn,
            },
        }

    except HTTPException:
        raise
    except ConnectionError as e:
        logger.error(f"Database connection error creating index: {e}")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    except ValueError as e:
        logger.error(f"Invalid index parameters: {e}")
        raise HTTPException(
            status_code=400, detail=f"Invalid index configuration: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error creating vector index: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.put("/graph-rag/schema")
async def update_schema(request: SchemaUpdateRequest) -> dict[str, Any]:
    """Update the knowledge graph schema configuration."""
    global SCHEMA

    try:
        # Create the schema structure (Pydantic handles validation)
        new_schema = {
            "node_types": request.node_types,
            "relationship_types": request.relationship_types,
            "patterns": request.patterns,
        }

        # Update the global schema
        SCHEMA = new_schema

        logger.info(
            f"Schema updated successfully with {len(request.node_types)} node types and {len(request.relationship_types)} relationship types"
        )

        return {
            "message": "Schema updated successfully",
            "status": "success",
            "schema": SCHEMA,
        }

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Invalid schema data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid schema format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error updating schema: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/basic-rag/ingest-pdf", response_model=IngestPDFResponse)
async def ingest_pdf(request: IngestPDFRequest) -> IngestPDFResponse:
    """Process a PDF file and store its content in a vector store."""
    try:
        # Check if PDF file exists (additional check beyond Pydantic validation)
        if not os.path.exists(request.pdf_path):
            raise HTTPException(
                status_code=404, detail=f"PDF file not found: {request.pdf_path}"
            )

        logger.info(f"Starting PDF ingestion: {request.pdf_path}")

        # Create a DocumentLoader instance
        try:
            loader = DocumentLoader(file_path=request.pdf_path)
        except Exception as e:
            logger.error(f"Failed to create DocumentLoader: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize PDF loader: {str(e)}"
            )

        # Create embeddings instance
        try:
            embeddings = OpenAIEmbeddings(model=request.embedding_model_name)
        except Exception as e:
            logger.error(f"Failed to create OpenAI embeddings: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize embeddings model: {str(e)}",
            )

        # Load the PDF into the vector store
        try:
            loader.load_to_vector_store(
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                embeddings=embeddings,
                collection_name=request.collection_name,
                clear_existing=request.clear_existing,
            )
        except FileNotFoundError as e:
            logger.error(f"PDF file not found during processing: {e}")
            raise HTTPException(status_code=404, detail=f"PDF file not found: {str(e)}")
        except PermissionError as e:
            logger.error(f"Permission denied accessing PDF: {e}")
            raise HTTPException(
                status_code=403, detail="Permission denied accessing the PDF file"
            )
        except ValueError as e:
            logger.error(f"Invalid parameters for PDF processing: {e}")
            raise HTTPException(
                status_code=400, detail=f"Invalid processing parameters: {str(e)}"
            )
        except ConnectionError as e:
            logger.error(f"Vector store connection error: {e}")
            raise HTTPException(
                status_code=503, detail="Vector store service unavailable"
            )
        except Exception as e:
            logger.error(f"Unexpected error during PDF processing: {e}")
            raise HTTPException(
                status_code=500, detail="Internal server error during PDF processing"
            )

        logger.info("PDF ingestion completed successfully")

        return IngestPDFResponse(
            message="PDF successfully ingested into vector store",
            status="success",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ingest_pdf endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/basic-rag/chat", response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest) -> ChatResponse:
    """Chat with the RAG system using the basic RAG implementation."""
    try:
        logger.info(f"Starting chat request: {request.question[:100]}...")

        # Create RAG configuration from request parameters
        try:
            rag_config = RAGConfig(
                collection_name=request.collection_name,
                embedding_model=request.embedding_model,
                top_k=request.top_k,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                chat_model_name=request.chat_model_name,
            )
        except Exception as e:
            logger.error(f"Failed to create RAG configuration: {e}")
            raise HTTPException(
                status_code=400, detail=f"Invalid RAG configuration: {str(e)}"
            )

        # Initialize the RAG chat system
        try:
            rag_chat = BasicRAGChat(config=rag_config)
        except Exception as e:
            logger.error(f"Failed to initialize RAG chat system: {e}")
            raise HTTPException(
                status_code=503, detail=f"Failed to initialize RAG system: {str(e)}"
            )

        # Get the answer from the RAG system
        try:
            result = rag_chat.get_answer(request.question, include_context=True)
            answer, context_documents = result
        except ValueError as e:
            logger.error(f"Invalid question or parameters: {e}")
            raise HTTPException(
                status_code=400, detail=f"Invalid question or parameters: {str(e)}"
            )
        except ConnectionError as e:
            logger.error(f"Connection error with vector store or LLM: {e}")
            raise HTTPException(status_code=503, detail="External service unavailable")
        except Exception as e:
            logger.error(f"Unexpected error during RAG processing: {e}")
            raise HTTPException(
                status_code=500, detail="Internal server error during RAG processing"
            )

        # Prepare metadata
        metadata = {
            "model": request.chat_model_name,
            "embedding_model": request.embedding_model,
            "collection": request.collection_name,
            "top_k": request.top_k,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        logger.info("Chat request completed successfully")

        return ChatResponse(
            answer=answer,
            context_documents=context_documents,
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
