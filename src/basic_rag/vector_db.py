import os
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from base import BaseVectorDB


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
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY", None)

        if not api_key:
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
            self._create_collection(
                self.default_collection_name, self.embedding_size, self.distance
            )

    def _create_collection(
        self,
        collection_name: Optional[str] = None,
        embedding_size: Optional[int] = None,
        distance: Optional[Distance] = None,
    ):
        """Create a collection in the Qdrant vector database."""
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_size,
                distance=distance,
            ),
        )
