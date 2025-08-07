import os
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from basic_rag.base import BaseVectorDB


class QdrantVectorDB(BaseVectorDB):
    """Qdrant vector database implementation."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        default_collection_name: str = None,
        distance: Distance = Distance.COSINE,
        embedding_size: int = 3072,
    ):
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")

        if api_key is None:
            self.api_key = os.getenv("QDRANT_API_KEY", None)
        else:
            self.api_key = api_key

        if not self.api_key:
            raise ValueError("QDRANT_API_KEY environment variable is not set.")

        self.client = QdrantClient(url=url, api_key=api_key)
        self.default_collection_name = default_collection_name

        if self.default_collection_name:
            if not distance or not embedding_size:
                raise ValueError(
                    "When default_collection_name is provided, "
                    "distance and embedding_size must also be specified."
                )
            self.distance = distance
            self.embedding_size = embedding_size
            self.create_collection(
                self.default_collection_name, self.embedding_size, self.distance
            )

    def create_collection(
        self,
        collection_name: str,
        embedding_size: Optional[int] = None,
        distance: Distance = Distance.COSINE,
    ):
        """Create a collection in the Qdrant vector database."""
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_size,
                    distance=distance,
                ),
            )

    def clear_collection(
        self,
        collection_name: str,
        embedding_size: int = 3072,
        distance: Distance = Distance.COSINE,
    ):
        """Clear the specified collection in the Qdrant vector database."""
        if self.client.collection_exists(collection_name):
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_size,
                    distance=distance,
                ),
            )
