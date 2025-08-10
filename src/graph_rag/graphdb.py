import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Literal

from neo4j import GraphDatabase, Driver
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.resolver import SpaCySemanticMatchResolver
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm import OpenAILLM


class Neo4jConnection:
    """Simple Neo4j connection class that manages the driver and provides basic functionality."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j URI (e.g., "bolt://localhost:7687")
            user: Username for authentication
            password: Password for authentication
            database: Database name (default: "neo4j")
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: Optional[Driver] = None

    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )

            self._driver.verify_connectivity()
            logging.info(f"Successfully connected to Neo4j at {self.uri}")
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            raise

    @property
    def driver(self) -> Driver:
        """Get the Neo4j driver instance."""
        if self._driver is None:
            self.connect()
        if self._driver is None:
            raise RuntimeError("Failed to establish connection")
        return self._driver

    def close(self) -> None:
        """Close the connection to Neo4j."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logging.info("Neo4j connection closed")

    def execute_read(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a read query."""
        with self.driver.session(database=self.database) as session:
            return session.run(query, parameters or {})

    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a write query."""
        with self.driver.session(database=self.database) as session:
            return session.run(query, parameters or {})

    def clear_database(self) -> None:
        """Clear all nodes and relationships from the database."""
        logging.warning("Clearing entire database...")
        self.execute_write("MATCH (n) DETACH DELETE n")
        logging.info("Database cleared")

    def create_database(self, database_name: str) -> None:
        """Create a new Neo4j database."""
        try:
            # Check if multi-database is supported
            if not self._supports_multi_database():
                logging.warning(
                    "Multi-database feature not supported on this Neo4j instance"
                )
                return

            # Use system database for admin operations
            with self.driver.session(database="system") as session:
                session.write_transaction(
                    lambda tx: tx.run(f"CREATE DATABASE `{database_name}`")
                )
            logging.info(f"Database '{database_name}' created successfully")

        except Exception as e:
            if "already exists" in str(e):
                logging.warning(f"Database '{database_name}' already exists")
            elif "UnsupportedAdministrationCommand" in str(e):
                logging.warning(
                    "CREATE DATABASE not supported - "
                    "you may be using Neo4j Community Edition or older version"
                )
            else:
                logging.error(f"Error creating database: {e}")
                raise

    def delete_database(self, database_name: str) -> None:
        """Delete a Neo4j database."""
        try:
            # Check if multi-database is supported
            if not self._supports_multi_database():
                logging.warning(
                    "Multi-database feature not supported on this Neo4j instance"
                )
                return

            # Use system database for admin operations
            with self.driver.session(database="system") as session:
                session.write_transaction(
                    lambda tx: tx.run(f"DROP DATABASE `{database_name}`")
                )
            logging.info(f"Database '{database_name}' deleted successfully")

        except Exception as e:
            if "does not exist" in str(e):
                logging.warning(f"Database '{database_name}' does not exist")
            elif "UnsupportedAdministrationCommand" in str(e):
                logging.warning(
                    "DROP DATABASE not supported - "
                    "you may be using Neo4j Community Edition or older version"
                )
            else:
                logging.error(f"Error deleting database: {e}")
                raise

    def _supports_multi_database(self) -> bool:
        """Check if the Neo4j instance supports multi-database operations."""
        try:
            with self.driver.session(database="system") as session:
                session.read_transaction(
                    lambda tx: tx.run("SHOW DATABASES YIELD name LIMIT 1")
                )
            return True
        except Exception:
            return False

    def list_databases(self) -> List[Dict[str, Any]]:
        """List all databases."""
        try:
            # Check if multi-database is supported
            if not self._supports_multi_database():
                logging.warning(
                    "Multi-database feature not supported - "
                    "using default database only"
                )
                return [
                    {"name": "neo4j", "status": "online", "default": True, "home": True}
                ]

            # Use system database for admin operations
            with self.driver.session(database="system") as session:

                def _get_databases(tx: Any) -> List[Dict[str, Any]]:
                    result = tx.run("SHOW DATABASES")
                    databases = []
                    for record in result:
                        databases.append(
                            {
                                "name": record["name"],
                                "status": record.get("currentStatus", "unknown"),
                                "default": record.get("default", False),
                                "home": record.get("home", False),
                            }
                        )
                    return databases

                return session.read_transaction(_get_databases)

        except Exception as e:
            logging.error(f"Error listing databases: {e}")
            # Return default database as fallback
            return [
                {"name": "neo4j", "status": "unknown", "default": True, "home": True}
            ]

    def create_kg_pipeline(
        self,
        schema: Dict[str, Any],
        api_key: str,
        endpoint: str,
        embedding_model_name: str,
        chat_model_name: str = "gpt-4o",
    ) -> SimpleKGPipeline:
        """
        Create a SimpleKGPipeline for building knowledge graphs.

        Args:
            schema: Schema dictionary with node_types, relationship_types, and patterns
            api_key: OpenAI API key
            endpoint: OpenAI endpoint URL
            embedding_model_name: OpenAI deployment name for embeddings
            chat_model_name: Chat model name (default: "gpt-4o")

        Returns:
            Configured SimpleKGPipeline instance
        """
        # Create embedder
        embedder = OpenAIEmbeddings(
            api_key=api_key,
            model=embedding_model_name,
        )

        # Create LLM
        llm = OpenAILLM(
            model_name=chat_model_name,
            model_params={
                "response_format": {"type": "json_object"},
                "temperature": 0,
            },
            api_key=api_key,
        )

        # Create and return pipeline
        return SimpleKGPipeline(
            llm=llm,
            driver=self.driver,
            embedder=embedder,
            schema=schema,
            on_error="IGNORE",
            from_pdf=True,
            perform_entity_resolution=True,
        )

    def create_entity_resolver(
        self, resolve_properties: Optional[List[str]] = None, similarity_threshold: float = 0.5
    ) -> SpaCySemanticMatchResolver:
        """
        Create a SpaCy entity resolver for post-processing.

        Args:
            resolve_properties: Properties to compare for similarity (default: ["name"])
            similarity_threshold: Threshold for entity matching (default: 0.5)

        Returns:
            Configured SpaCySemanticMatchResolver instance
        """
        resolve_properties = resolve_properties or ["name"]
        return SpaCySemanticMatchResolver(
            self.driver,
            resolve_properties=resolve_properties,
            similarity_threshold=similarity_threshold,
        )

    async def populate_kg_from_pdf(
        self,
        pdf_path: str,
        schema: Dict[str, Any],
        api_key: str,
        endpoint: str,
        deployment: str,
        llm_model_name: str = "gpt-4o",
        clear_existing: bool = False,
        run_entity_resolution: bool = True,
    ) -> None:
        """
        Populate the knowledge graph from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            schema: Schema dictionary with node_types, relationship_types, and patterns
            api_key: OpenAI API key
            endpoint: OpenAI endpoint URL
            deployment: OpenAI deployment name for embeddings
            llm_model_name: LLM model name (default: "gpt-4o")
            clear_existing: Whether to clear existing data (default: False)
            run_entity_resolution: Whether to run entity resolution (default: True)
        """
        if clear_existing:
            self.clear_database()

        # Create KG pipeline
        kg_builder = self.create_kg_pipeline(
            schema=schema,
            api_key=api_key,
            endpoint=endpoint,
            embedding_model_name=deployment,
            chat_model_name=llm_model_name,
        )

        logging.info(f"Starting KG population from PDF: {pdf_path}")

        # Run the pipeline
        await kg_builder.run_async(file_path=pdf_path)

        # Run entity resolution if requested
        if run_entity_resolution:
            logging.info("Running entity resolution...")
            resolver = self.create_entity_resolver()
            await resolver.run()

        logging.info("KG population completed successfully")

    def get_kg_stats(self) -> Dict[str, int]:
        """
        Get basic statistics about the knowledge graph.

        Returns:
            Dictionary with node and relationship counts
        """
        stats = {}

        # Get total node count
        result = self.execute_read("MATCH (n) RETURN count(n) as node_count")
        stats["total_nodes"] = result.single()["node_count"]

        # Get total relationship count
        result = self.execute_read("MATCH ()-[r]->() RETURN count(r) as rel_count")
        stats["total_relationships"] = result.single()["rel_count"]

        # Get node counts by label
        result = self.execute_read(
            """
            MATCH (n) 
            RETURN labels(n)[0] as label, count(*) as count 
            ORDER BY count DESC
        """
        )
        stats["nodes_by_label"] = {
            record["label"]: record["count"] for record in result
        }

        # Get relationship counts by type
        result = self.execute_read(
            """
            MATCH ()-[r]->() 
            RETURN type(r) as type, count(*) as count 
            ORDER BY count DESC
        """
        )
        stats["relationships_by_type"] = {
            record["type"]: record["count"] for record in result
        }

        return stats

    def create_vector_index(
        self,
        index_name: str,
        label: str,
        embedding_property: str = "embedding",
        dimensions: int = 3072,
        similarity_fn: Literal["cosine", "euclidean"] = "cosine",
    ) -> None:
        """
        Create a vector index for similarity search.

        Args:
            index_name: Name of the index
            label: Node label to index
            embedding_property: Property containing embeddings (default: "embedding")
            dimensions: Embedding dimensions (default: 3072 for text-embedding-3-large)
            similarity_fn: Similarity function (default: "cosine")
        """
        create_vector_index(
            driver=self.driver,
            name=index_name,
            label=label,
            embedding_property=embedding_property,
            dimensions=dimensions,
            similarity_fn=similarity_fn,
        )
        logging.info(f"Created vector index '{index_name}' for label '{label}'")

    def __enter__(self) -> "Neo4jConnection":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
