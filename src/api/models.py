from typing import Optional
from pydantic import BaseModel


class IngestPDFRequest(BaseModel):
    pdf_path: str
    clear_existing: bool = False
    run_entity_resolution: bool = True
    llm_model_name: str = "gpt-4o"


class IngestPDFResponse(BaseModel):
    message: str
    status: str
    kg_stats: Optional[dict] = None


class SchemaUpdateRequest(BaseModel):
    node_types: list
    relationship_types: list
    patterns: list


class CreateIndexRequest(BaseModel):
    index_name: str
    label: str
    embedding_property: str = "embedding"
    dimensions: int = 3072
    similarity_fn: str = "cosine"
