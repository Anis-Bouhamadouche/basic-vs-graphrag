"""Base classes for document loaders and vector databases."""
from abc import ABC, abstractmethod
from typing import List, Any


class BaseLoader:
    """Base class for document loaders."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path."""
        self.file_path = file_path

    @abstractmethod
    def load(self) -> List[str]:
        """Load the document and return its content as a list of strings.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BaseVectorDB(ABC):
    """Base class for vector databases."""

    def __init__(self) -> None:
        """Initialize the vector database."""
        raise NotImplementedError("This class should not be instantiated directly.")

    @abstractmethod
    def create_collection(self, collection_name: str, embedding_size: int, distance: Any = None) -> None:
        """Create a collection in the vector database."""
        pass
