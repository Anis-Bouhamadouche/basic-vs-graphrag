import os
import logging
import tempfile
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from graph_rag.graphdb import Neo4jConnection
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from api.models import (
    IngestPDFRequest,
    IngestPDFResponse,
    CreateIndexRequest,
    SchemaUpdateRequest,
)


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
SCHEMA = None

# Global Neo4j connection
neo4j_conn = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    global neo4j_conn

    # Startup
    try:
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


@app.get("/")
async def hello_world():
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
async def health_check():
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


@app.post("/graph-rag/ingest-pdf", response_model=IngestPDFResponse)
async def ingest_pdf_to_kg(request: IngestPDFRequest):
    """
    Ingest a PDF file into the knowledge graph.

    This endpoint processes a PDF file and populates the knowledge graph
    using the EU AI Act schema and OpenAI services.
    """
    try:
        if not neo4j_conn:
            raise HTTPException(
                status_code=500, detail="Database connection not available"
            )

        # Check if PDF file exists
        if not os.path.exists(request.pdf_path):
            raise HTTPException(
                status_code=404, detail=f"PDF file not found: {request.pdf_path}"
            )

        logger.info(f"Starting PDF ingestion: {request.pdf_path}")

        # Populate the knowledge graph
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

        # Get knowledge graph statistics
        kg_stats = neo4j_conn.get_kg_stats()

        logger.info("PDF ingestion completed successfully")

        return IngestPDFResponse(
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
async def get_kg_stats():
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
async def clear_knowledge_graph():
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
async def get_schema():
    """Get the current knowledge graph schema configuration."""
    return {
        "status": "success",
        "schema": SCHEMA,
        "description": "EU AI Act knowledge graph schema",
    }


@app.post("/graph-rag/create-index")
async def create_vector_index_endpoint(request: CreateIndexRequest):
    """
    Create a vector index for similarity search.

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

        # Create the vector index
        neo4j_conn.create_vector_index(
            index_name=request.index_name,
            label=request.label,
            embedding_property=request.embedding_property,
            dimensions=request.dimensions,
            similarity_fn=request.similarity_fn,
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
async def update_schema(request: SchemaUpdateRequest):
    """Update the knowledge graph schema configuration."""
    global SCHEMA

    try:
        # Validate the schema structure
        new_schema = {
            "node_types": request.node_types,
            "relationship_types": request.relationship_types,
            "patterns": request.patterns,
        }

        # Update the global schema
        SCHEMA = new_schema

        return {
            "message": "Schema updated successfully",
            "status": "success",
            "schema": SCHEMA,
        }

    except ValueError as e:
        logger.error(f"Invalid schema data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid schema format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error updating schema: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
