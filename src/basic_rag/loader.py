"""Document loading and text processing utilities."""
import logging
from typing import List, Optional, Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader

from basic_rag.base import BaseLoader
from basic_rag.vector_db import QdrantVectorDB

# Configure logging
logger = logging.getLogger(__name__)


load_dotenv()  # Load environment variables from .env file


class DocumentLoader(BaseLoader):
    """Document loader class for processing PDF files."""

    def __init__(self, file_path: str) -> None:
        """Initialize the document loader with a file path."""
        super().__init__(file_path)
        logger.info(f"Initialized DocumentLoader with file: {file_path}")

    def load(self) -> List[str]:
        """Load the document and return its content as a list of strings."""
        return self._extract_text()

    def _extract_text(self) -> List[str]:
        """Load text from a PDF file."""
        logger.info(f"Starting text extraction from PDF: {self.file_path}")
        try:
            pdf_reader = PdfReader(self.file_path)
            text = []
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text.append(page_text)
                logger.debug(
                    f"Extracted text from page {page_num + 1}, length: {len(page_text)} characters"
                )

            logger.info(
                f"Successfully extracted text from {len(text)} pages, total pages: {len(pdf_reader.pages)}"
            )
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {self.file_path}: {str(e)}")
            raise

    def _chunk(
        self, text: List[str], chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[Document]:
        """Chunk the text into smaller pieces and return as Document objects."""
        logger.info(
            f"Starting text chunking with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )
        try:
            # Join all page texts into a single string
            full_text = "\n".join(text)
            logger.debug(f"Combined text length: {len(full_text)} characters")

            # Debug: Check if text is long enough to be chunked
            if len(full_text) <= chunk_size:
                logger.warning(
                    f"Text length ({len(full_text)}) is smaller than or equal to chunk_size ({chunk_size})"
                )
                logger.warning("This will result in only one chunk")

            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n",  # Explicitly set separator
                length_function=len,
            )
            text_chunks = text_splitter.split_text(full_text)

            logger.debug(f"Text splitter created {len(text_chunks)} raw chunks")
            if text_chunks:
                logger.debug(f"First raw chunk length: {len(text_chunks[0])}")
                logger.debug(f"Last raw chunk length: {len(text_chunks[-1])}")

            # Convert text chunks to Document objects
            document_chunks = []
            for i, chunk_text in enumerate(text_chunks):
                document = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": self.file_path,
                        "chunk_id": i,
                        "chunk_size": len(chunk_text),
                    },
                )
                document_chunks.append(document)

            logger.info(
                f"Successfully created {len(document_chunks)} document chunks from text"
            )
            for i, doc in enumerate(
                document_chunks[:3]
            ):  # Log first 3 chunks for debugging
                logger.debug(
                    f"Document chunk {i + 1} length: {len(doc.page_content)} characters"
                )

            return document_chunks
        except Exception as e:
            logger.error(f"Failed to chunk text: {str(e)}")
            raise

    def _embed_documents(
        self,
        chunks: List[Document],
        embeddings: Optional[OpenAIEmbeddings] = None,
    ) -> List[List[float]]:
        """Embed the document chunks using OpenAI embeddings."""
        logger.info(f"Starting embedding generation for {len(chunks)} chunks")
        try:
            if embeddings is None:
                logger.debug("Creating new OpenAIEmbeddings instance")
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

            logger.debug("Calling OpenAI API to generate embeddings")
            # Extract text content from Document objects
            chunk_texts = [doc.page_content for doc in chunks]
            embedded_chunks = embeddings.embed_documents(chunk_texts)

            logger.info(
                f"Successfully generated embeddings for {len(embedded_chunks)} chunks"
            )
            if embedded_chunks:
                logger.debug(f"First embedding dimension: {len(embedded_chunks[0])}")

            return embedded_chunks
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

    def load_to_vector_store(
        self,
        collection_name: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embeddings: Optional[OpenAIEmbeddings] = None,
        vector_store: Any = None,
        clear_existing: bool = False,
    ) -> List[str]:
        """Load the PDF file, chunk it, and embed the chunks into a vector store."""
        logger.info(f"Starting complete document loading pipeline for {self.file_path}")
        try:
            # Extract text from PDF
            text = self._extract_text()

            # Chunk the text
            chunks = self._chunk(text, chunk_size, chunk_overlap)

            # Initialize vector store
            logger.info("Initializing Qdrant vector store")
            qdrant_vector_db = QdrantVectorDB(default_collection_name=collection_name)

            if clear_existing:
                logger.info(
                    f"Clearing existing collection: {collection_name} in Qdrant"
                )
                qdrant_vector_db.clear_collection(collection_name)
            client = qdrant_vector_db.client

            if embeddings is None:
                logger.debug("Creating new OpenAIEmbeddings instance")
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="eu_ai_act",
                embedding=embeddings,
            )

            # Create document IDs
            logger.debug("Creating document IDs for chunks")
            document_ids = [str(uuid4()) for _ in range(len(chunks))]
            logger.debug(f"Generated {len(document_ids)} document IDs")

            # Debug: Log chunk information
            logger.debug(
                f"First chunk preview (100 chars): {chunks[0].page_content[:100]}..."
            )
            logger.debug(
                f"Last chunk preview (100 chars): {chunks[-1].page_content[:100]}..."
            )

            # Additional debug: Check if we actually have multiple chunks
            if len(chunks) == 1:
                logger.warning(
                    f"Only 1 chunk created from document. Chunk size: {len(chunks[0].page_content)} characters"
                )
                logger.warning("This might indicate chunking is not working properly")

            # Add to vector store
            logger.info("Adding documents to vector store")
            vector_store.add_documents(
                documents=chunks,
                ids=document_ids,
            )

            logger.info(
                f"Successfully completed document loading pipeline. Added {len(chunks)} chunks to vector store"
            )
            return document_ids

        except Exception as e:
            logger.error(f"Failed to complete document loading pipeline: {str(e)}")
            raise
