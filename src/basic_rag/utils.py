from typing import List
from uuid import uuid4


def create_document_ids(documents: List[str]) -> List[str]:
    """
    Create unique IDs for a list of documents.
    """
    return [str(uuid4()) for _ in range(len(documents))]
