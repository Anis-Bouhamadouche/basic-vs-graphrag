from typing import List
from abc import ABC, abstractmethod


class BaseLoader:
    """Base class for document loaders."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def load(self) -> List[str]:
        """
        Load the document and return its content as a list of strings.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BaseVectorDB(ABC):
    """Base class for vector databases."""

    def __init__(self):
        raise NotImplementedError("This class should not be instantiated directly.")

    @abstractmethod
    def create_collection(self):
        """
        Create a collection in the vector database.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
