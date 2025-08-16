"""Pydantic models for API request and response validation."""

from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class BaseIngestPDFRequest(BaseModel):
    """Base model for PDF ingestion requests."""

    pdf_path: str = Field(..., description="Path to the PDF file", min_length=1)
    clear_existing: bool = Field(False, description="Whether to clear existing data")

    @field_validator("pdf_path")
    @classmethod
    def validate_pdf_path(cls, v: str) -> str:
        """Validate PDF path field."""
        if not v.strip():
            raise ValueError("pdf_path cannot be empty or whitespace only")
        if not v.lower().endswith(".pdf"):
            raise ValueError("pdf_path must end with .pdf extension")
        return v.strip()


class IngestPDFRequest(BaseIngestPDFRequest):
    """Model for PDF ingestion request with vector store configuration."""

    chunk_size: int = Field(1000, description="Size of text chunks", gt=0, le=10000)
    chunk_overlap: int = Field(200, description="Overlap between chunks", ge=0)
    collection_name: str = Field(
        ..., description="Name of the vector store collection", min_length=1
    )
    embedding_model_name: str = Field(
        "text-embedding-3-large", description="OpenAI embedding model name"
    )

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "IngestPDFRequest":
        """Validate chunk overlap is less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        """Validate collection name field."""
        if not v.strip():
            raise ValueError("collection_name cannot be empty or whitespace only")
        return v.strip()


class IngestPDFGraphRequest(IngestPDFRequest):
    """Model for PDF ingestion request for graph RAG."""

    llm_model_name: str = Field("gpt-4.1-mini", description="LLM model for graph processing")
    run_entity_resolution: bool = Field(
        True, description="Whether to run entity resolution"
    )


class IngestPDFResponse(BaseModel):
    """Base response model for PDF ingestion."""

    message: str
    status: str


class IngestPDFGraphResponse(IngestPDFResponse):
    """Response model for graph RAG PDF ingestion."""

    kg_stats: Optional[dict[str, Any]] = None


class SchemaUpdateRequest(BaseModel):
    """Model for schema update requests."""

    node_types: List[str] = Field(..., description="List of node types")
    relationship_types: List[str] = Field(..., description="List of relationship types")
    patterns: List[List[str]] = Field(..., description="List of patterns as [source_node, relationship, target_node] triplets")

    @field_validator("node_types")
    @classmethod
    def validate_node_types(cls, v: List[str]) -> List[str]:
        """Validate node types field."""
        if not v:
            raise ValueError("node_types cannot be empty")
        for i, node_type in enumerate(v):
            if not isinstance(node_type, str) or not node_type.strip():
                raise ValueError(f"node_types[{i}] must be a non-empty string")
        return [node_type.strip() for node_type in v]

    @field_validator("relationship_types")
    @classmethod
    def validate_relationship_types(cls, v: List[str]) -> List[str]:
        """Validate relationship types field."""
        if not v:
            raise ValueError("relationship_types cannot be empty")
        for i, rel_type in enumerate(v):
            if not isinstance(rel_type, str) or not rel_type.strip():
                raise ValueError(f"relationship_types[{i}] must be a non-empty string")
        return [rel_type.strip() for rel_type in v]

    @field_validator("patterns")
    @classmethod
    def validate_patterns(cls, v: List[List[str]]) -> List[List[str]]:
        """Validate patterns field."""
        if not v:
            raise ValueError("patterns cannot be empty")
        for i, pattern in enumerate(v):
            if not isinstance(pattern, list) or len(pattern) != 3:
                raise ValueError(f"patterns[{i}] must be a list with exactly 3 elements [source_node, relationship, target_node]")
            for j, element in enumerate(pattern):
                if not isinstance(element, str) or not element.strip():
                    raise ValueError(f"patterns[{i}][{j}] must be a non-empty string")
        return [[element.strip() for element in pattern] for pattern in v]


class CreateIndexRequest(BaseModel):
    """Model for vector index creation requests."""

    index_name: str = Field(..., description="Name of the vector index", min_length=1)
    label: str = Field(..., description="Neo4j node label", min_length=1)
    embedding_property: str = Field(
        "embedding", description="Property name for embeddings", min_length=1
    )
    dimensions: int = Field(3072, description="Vector dimensions", gt=0, le=10000)
    similarity_fn: str = Field("cosine", description="Similarity function")

    @field_validator("index_name")
    @classmethod
    def validate_index_name(cls, v: str) -> str:
        """Validate index name field."""
        if not v.strip():
            raise ValueError("index_name cannot be empty or whitespace only")
        return v.strip()

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Validate label field."""
        if not v.strip():
            raise ValueError("label cannot be empty or whitespace only")
        return v.strip()

    @field_validator("embedding_property")
    @classmethod
    def validate_embedding_property(cls, v: str) -> str:
        """Validate embedding property field."""
        if not v.strip():
            raise ValueError("embedding_property cannot be empty or whitespace only")
        return v.strip()

    @field_validator("similarity_fn")
    @classmethod
    def validate_similarity_fn(cls, v: str) -> str:
        """Validate similarity function field."""
        valid_functions = ["cosine", "euclidean", "dot"]
        if v not in valid_functions:
            raise ValueError(
                f'similarity_fn must be one of: {", ".join(valid_functions)}'
            )
        return v


class ChatRequest(BaseModel):
    """Model for chat request."""

    question: str = Field(..., description="User question", min_length=1)
    collection_name: str = Field(
        "eu_ai_act", description="Name of the vector store collection", min_length=1
    )
    top_k: int = Field(
        3, description="Number of top documents to retrieve", gt=0, le=20
    )
    temperature: float = Field(0.1, description="LLM temperature", ge=0.0, le=2.0)
    max_tokens: int = Field(
        1000, description="Maximum tokens in response", gt=0, le=4000
    )
    chat_model_name: str = Field("gpt-4.1-mini", description="OpenAI chat model name")
    embedding_model: str = Field(
        "text-embedding-3-large", description="OpenAI embedding model name"
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate question field."""
        if not v.strip():
            raise ValueError("question cannot be empty or whitespace only")
        return v.strip()

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        """Validate collection name field."""
        if not v.strip():
            raise ValueError("collection_name cannot be empty or whitespace only")
        return v.strip()


class ChatResponse(BaseModel):
    """Model for chat response."""

    answer: str = Field(..., description="AI-generated answer")
    context_documents: List[str] = Field(
        ..., description="Context documents used for generation"
    )
    metadata: dict = Field(..., description="Additional metadata about the response")


class GraphResolutionRequest(BaseModel):
    """Model for graph entity resolution request."""

    resolve_properties: List[str] = Field(
        ["name"], description="Properties to compare for similarity"
    )
    similarity_threshold: float = Field(
        0.5, description="Threshold for entity matching", ge=0.0, le=1.0
    )
    
    @field_validator("resolve_properties")
    @classmethod
    def validate_resolve_properties(cls, v: List[str]) -> List[str]:
        """Validate resolve properties field."""
        if not v:
            raise ValueError("resolve_properties cannot be empty")
        for i, prop in enumerate(v):
            if not isinstance(prop, str) or not prop.strip():
                raise ValueError(f"resolve_properties[{i}] must be a non-empty string")
        return [prop.strip() for prop in v]


class GraphResolutionResponse(BaseModel):
    """Model for graph entity resolution response."""

    message: str = Field(..., description="Response message")
    status: str = Field(..., description="Operation status")
    resolved_entities: int = Field(..., description="Number of entities resolved")
    execution_time: float = Field(..., description="Execution time in seconds")
