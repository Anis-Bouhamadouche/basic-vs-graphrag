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
    GraphResolutionRequest,
    GraphResolutionResponse,
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

SCHEMA: Optional[Dict[str, Any]] = {
    "node_types": [
        "Act",
        "Chapter",
        "Section",
        "Article",
        "Annex",
        "Recital",
        "Term",
        "Definition",
        "RiskLevel",
        "ProhibitedPractice",
        "HighRiskCategory",
        "LimitedRiskMeasure",
        "MinimalRisk",
        "GPAI",
        "GeneralPurposeModel",
        "Role",
        "Actor",
        "Provider",
        "Deployer",
        "Importer",
        "Distributor",
        "AuthorizedRepresentative",
        "NotifiedBody",
        "MarketSurveillanceAuthority",
        "Standard",
        "TechnicalSpecification",
        "Obligation",
        "Requirement",
        "Provision",
        "Procedure",
        "ConformityAssessment",
        "PostMarketMonitoring",
        "RiskManagementSystem",
        "DataGovernance",
        "TransparencyMeasure",
        "HumanOversight",
        "AccuracyRobustnessCybersecurity",
        "Documentation",
        "Registration",
        "IncidentReporting",
        "Enforcement",
        "Sanction",
        "Fine",
        "Exemption",
        "Derogation",
        "UseCase",
        "System",
        "Modality",
        "Sector",
        "Purpose",
        "Capability",
        "Harm",
        "Caveat",
        "Citation",
    ],
    "relationship_types": [
        "has_chapter",
        "has_section",
        "has_article",
        "has_annex",
        "has_recital",
        "defines_term",
        "has_definition",
        "defined_in_article",
        "defined_in_annex",
        "mentions_term",
        "scoped_by_article",
        "grounded_in_article",
        "grounded_in_section",
        "grounded_in_annex",
        "cross_references_article",
        "has_risk_level",
        "classifies_as_high_risk",
        "classifies_as_prohibited",
        "classifies_as_limited_risk",
        "classifies_as_minimal_risk",
        "details_provision_for",
        "requires",
        "entails",
        "prohibits",
        "allows_with_measures",
        "exempts",
        "derogates",
        "applicable_to_role",
        "obligation_for_role",
        "obligation_at_risk_level",
        "provision_at_risk_level",
        "uses_standard",
        "conforms_to_technical_spec",
        "requires_conformity_assessment",
        "performed_by_notified_body",
        "monitored_by_authority",
        "requires_registration",
        "requires_documentation",
        "requires_data_governance",
        "requires_human_oversight",
        "requires_transparency",
        "requires_accuracy_robustness_cybersecurity",
        "requires_post_market_monitoring",
        "requires_incident_reporting",
        "subject_to_enforcement",
        "subject_to_sanction",
        "penalized_by_fine",
        "has_exception_caveat",
        "belongs_to_sector",
        "has_purpose",
        "has_capability",
        "poses_harm",
        "is_gpai",
        "is_general_purpose_model",
        "has_citation",
        "is_a",
        "maps_to",
    ],
    "patterns": [
        ["Act", "has_article", "Article"],
        ["Act", "has_annex", "Annex"],
        ["Act", "has_chapter", "Chapter"],
        ["Chapter", "has_section", "Section"],
        ["Section", "has_article", "Article"],
        ["Article", "has_definition", "Definition"],
        ["Definition", "defines_term", "Term"],
        ["Definition", "grounded_in_article", "Article"],
        ["Term", "defined_in_article", "Article"],
        ["Annex", "has_definition", "Definition"],
        ["Article", "cross_references_article", "Article"],
        ["Article", "details_provision_for", "Provision"],
        ["Provision", "grounded_in_article", "Article"],
        ["Term", "mentions_term", "RiskLevel"],
        ["RiskLevel", "provision_at_risk_level", "Provision"],
        ["RiskLevel", "obligation_at_risk_level", "Obligation"],
        ["RiskLevel", "grounded_in_article", "Article"],
        ["GPAI", "is_general_purpose_model", "GeneralPurposeModel"],
        ["GPAI", "obligation_at_risk_level", "Obligation"],
        ["GPAI", "grounded_in_article", "Article"],
        ["Role", "obligation_for_role", "Obligation"],
        ["Obligation", "grounded_in_article", "Article"],
        ["Obligation", "requires", "Requirement"],
        ["Requirement", "grounded_in_article", "Article"],
        ["HighRiskCategory", "grounded_in_annex", "Annex"],
        ["HighRiskCategory", "provision_at_risk_level", "Provision"],
        ["HighRiskCategory", "requires_conformity_assessment", "ConformityAssessment"],
        ["ConformityAssessment", "performed_by_notified_body", "NotifiedBody"],
        ["ConformityAssessment", "grounded_in_article", "Article"],
        ["ProhibitedPractice", "prohibits", "System"],
        ["ProhibitedPractice", "grounded_in_article", "Article"],
        ["LimitedRiskMeasure", "requires_transparency", "TransparencyMeasure"],
        ["LimitedRiskMeasure", "grounded_in_article", "Article"],
        ["MinimalRisk", "allows_with_measures", "Provision"],
        ["MinimalRisk", "grounded_in_article", "Article"],
        ["Requirement", "uses_standard", "Standard"],
        ["Requirement", "conforms_to_technical_spec", "TechnicalSpecification"],
        ["Standard", "grounded_in_article", "Article"],
        ["TechnicalSpecification", "grounded_in_article", "Article"],
        ["UseCase", "has_purpose", "Purpose"],
        ["UseCase", "belongs_to_sector", "Sector"],
        ["UseCase", "has_capability", "Capability"],
        ["UseCase", "poses_harm", "Harm"],
        ["UseCase", "maps_to", "System"],
        ["System", "has_risk_level", "RiskLevel"],
        ["System", "classifies_as_high_risk", "HighRiskCategory"],
        ["System", "classifies_as_prohibited", "ProhibitedPractice"],
        ["System", "classifies_as_limited_risk", "LimitedRiskMeasure"],
        ["System", "classifies_as_minimal_risk", "MinimalRisk"],
        ["System", "is_gpai", "GPAI"],
        ["System", "grounded_in_article", "Article"],
        ["Actor", "is_a", "Role"],
        ["Actor", "applicable_to_role", "Role"],
        ["Actor", "has_risk_level", "RiskLevel"],
        ["Role", "requires_registration", "Registration"],
        ["Role", "requires_documentation", "Documentation"],
        ["Role", "requires_data_governance", "DataGovernance"],
        ["Role", "requires_human_oversight", "HumanOversight"],
        [
            "Role",
            "requires_accuracy_robustness_cybersecurity",
            "AccuracyRobustnessCybersecurity",
        ],
        ["Role", "requires_post_market_monitoring", "PostMarketMonitoring"],
        ["Role", "requires_incident_reporting", "IncidentReporting"],
        ["Enforcement", "subject_to_sanction", "Sanction"],
        ["Sanction", "penalized_by_fine", "Fine"],
        ["Sanction", "grounded_in_article", "Article"],
        ["Provision", "has_exception_caveat", "Caveat"],
        ["Caveat", "grounded_in_article", "Article"],
        ["Term", "has_citation", "Citation"],
        ["Provision", "has_citation", "Citation"],
        ["Obligation", "has_citation", "Citation"],
    ],
}


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


@app.post("/graph-rag/resolve-entities", response_model=GraphResolutionResponse)
async def resolve_graph_entities(
    request: GraphResolutionRequest,
) -> GraphResolutionResponse:
    """Resolve entities in the knowledge graph using SpaCy semantic matching.

    This endpoint runs entity resolution to merge similar entities based on
    semantic similarity of their properties.
    """
    import time

    try:
        if not neo4j_conn:
            raise HTTPException(
                status_code=500, detail="Database connection not available"
            )

        logger.info("Starting entity resolution process...")
        start_time = time.time()

        # Get initial entity count for comparison
        initial_stats = neo4j_conn.get_kg_stats()
        initial_entities = initial_stats.get("total_nodes", 0)

        # Create entity resolver with provided configuration
        resolver = neo4j_conn.create_entity_resolver(
            resolve_properties=request.resolve_properties,
            similarity_threshold=request.similarity_threshold,
        )

        # Run entity resolution
        await resolver.run()

        # Get final entity count
        final_stats = neo4j_conn.get_kg_stats()
        final_entities = final_stats.get("total_nodes", 0)

        # Calculate resolved entities (entities that were merged/removed)
        resolved_entities = max(0, initial_entities - final_entities)

        execution_time = time.time() - start_time

        logger.info(
            f"Entity resolution completed. Resolved {resolved_entities} entities in {execution_time:.2f}s"
        )

        return GraphResolutionResponse(
            message=f"Entity resolution completed successfully. Resolved {resolved_entities} duplicate entities.",
            status="success",
            resolved_entities=resolved_entities,
            execution_time=execution_time,
        )

    except HTTPException:
        raise
    except ConnectionError as e:
        logger.error(f"Database connection error during entity resolution: {e}")
        raise HTTPException(status_code=503, detail="Database service unavailable")
    except ImportError as e:
        logger.error(f"SpaCy dependency error: {e}")
        raise HTTPException(
            status_code=500,
            detail="SpaCy language model not available. Please ensure 'en_core_web_sm' model is installed in the container.",
        )
    except Exception as e:
        logger.error(f"Error during entity resolution: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during entity resolution"
        )


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


@app.post("/graph-rag/chat", response_model=ChatResponse)
async def chat_with_graph_rag(request: ChatRequest) -> ChatResponse:
    """Chat with the GraphRAG system using the Neo4j implementation."""
    try:
        logger.info(f"Starting graph RAG chat request: {request.question[:100]}...")

        # Validate Neo4j connection
        if not neo4j_conn:
            raise HTTPException(
                status_code=500, detail="Database connection not available"
            )

        # Extract allowed labels and relationships from schema if available
        allowed_labels = None
        allowed_relationships = None

        if SCHEMA is not None:
            allowed_labels = SCHEMA.get("node_types")
            allowed_relationships = SCHEMA.get("relationship_types")
            logger.info(
                f"Using schema with {len(allowed_labels) if allowed_labels else 0} node types and {len(allowed_relationships) if allowed_relationships else 0} relationship types"
            )
        else:
            logger.warning(
                "No schema available, using default labels and relationships"
            )

        # Get the answer from the GraphRAG system
        # Use collection_name as the index_name for GraphRAG
        try:
            result = neo4j_conn.get_answer(
                question=request.question,
                index_name=request.collection_name,  # Use collection_name as index_name
                top_k=request.top_k,
                temperature=request.temperature,
                embedding_model=request.embedding_model,
                chat_model=request.chat_model_name,  # Map chat_model_name to chat_model
                include_context=True,
                allowed_labels=allowed_labels,
                allowed_relationships=allowed_relationships,
            )
            answer, context_documents = result
        except ValueError as e:
            logger.error(f"Invalid question or parameters: {e}")
            raise HTTPException(
                status_code=400, detail=f"Invalid question or parameters: {str(e)}"
            )
        except ConnectionError as e:
            logger.error(f"Connection error with Neo4j or LLM: {e}")
            raise HTTPException(status_code=503, detail="External service unavailable")
        except Exception as e:
            logger.error(f"Unexpected error during GraphRAG processing: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal server error during GraphRAG processing",
            )

        # Prepare metadata
        metadata = {
            "model": request.chat_model_name,
            "embedding_model": request.embedding_model,
            "index_name": request.collection_name,  # Show the index name used
            "top_k": request.top_k,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "rag_type": "graph",
            "schema_used": SCHEMA is not None,
            "allowed_labels_count": len(allowed_labels) if allowed_labels else 0,
            "allowed_relationships_count": (
                len(allowed_relationships) if allowed_relationships else 0
            ),
        }

        logger.info("Graph RAG chat request completed successfully")

        return ChatResponse(
            answer=answer,
            context_documents=context_documents,
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in graph RAG chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
