"""Pydantic models for API request and response validation."""
from typing import Any, Optional, List

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
    llm_model_name: str = Field("gpt-4o", description="LLM model for graph processing")
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
    relationship_types: List[str] = Field(
        ..., description="List of relationship types"
    )
    patterns: List[str] = Field(..., description="List of patterns")

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
