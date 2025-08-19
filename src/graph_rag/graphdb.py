"""Neo4j graph database connection and operations module."""

import logging
from typing import Any, Dict, List, Literal, Optional

from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.resolver import SpaCySemanticMatchResolver
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever


class Neo4jConnection:
    """Simple Neo4j connection class that manages the driver and provides basic functionality."""

    def __init__(
        self, uri: str, user: str, password: str, database: str = "neo4j"
    ) -> None:
        """Initialize Neo4j connection.

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

    def execute_read(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute a read query and return consumed result."""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return list(result)  # Consume the result before session closes

    def execute_write(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute a write query and return consumed result."""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return list(result)  # Consume the result before session closes

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
        chat_model_name: str = "gpt-4.1-mini",
    ) -> SimpleKGPipeline:
        """Create a SimpleKGPipeline for building knowledge graphs.

        Args:
            schema: Schema dictionary with node_types, relationship_types, and patterns
            api_key: OpenAI API key
            endpoint: OpenAI endpoint URL
            embedding_model_name: OpenAI deployment name for embeddings
            chat_model_name: Chat model name (default: "gpt-4.1-mini")

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
        self,
        resolve_properties: Optional[List[str]] = None,
        similarity_threshold: float = 0.5,
    ) -> SpaCySemanticMatchResolver:
        """Create a SpaCy entity resolver for post-processing.

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
        llm_model_name: str = "gpt-4.1-mini",
        clear_existing: bool = False,
        run_entity_resolution: bool = True,
    ) -> None:
        """Populate the knowledge graph from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            schema: Schema dictionary with node_types, relationship_types, and patterns
            api_key: OpenAI API key
            endpoint: OpenAI endpoint URL
            deployment: OpenAI deployment name for embeddings
            llm_model_name: LLM model name (default: "gpt-4.1-mini")
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
        """Get basic statistics about the knowledge graph.

        Returns:
            Dictionary with node and relationship counts
        """
        stats = {}

        # Get total node count
        result = self.execute_read("MATCH (n) RETURN count(n) as node_count")
        stats["total_nodes"] = result[0]["node_count"] if result else 0

        # Get total relationship count
        result = self.execute_read("MATCH ()-[r]->() RETURN count(r) as rel_count")
        stats["total_relationships"] = result[0]["rel_count"] if result else 0

        # Get node counts by label
        result = self.execute_read(
            """
            MATCH (n)
            RETURN labels(n)[0] as label, count(*) as count
            ORDER BY count DESC
        """
        )
        stats["nodes_by_label"] = {
            record["label"]: record["count"]
            for record in result
            if record["label"] is not None
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
        """Create a vector index for similarity search.

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

    def _format_graph_context(self, content: Any) -> str:
        """Format Neo4j Record content into a readable context format.
        
        Args:
            content: The raw Neo4j Record content
            
        Returns:
            Formatted string with main text content and related entities
        """
        try:
            import ast
            import re
            
            # Convert the content to string and parse if it looks like a Record
            content_str = str(content)
            
            # Check if this is a Record object string
            if content_str.startswith("<Record "):
                # Extract the content within the Record
                match = re.search(r"<Record (.+)>", content_str, re.DOTALL)
                if match:
                    record_content = match.group(1)
                    
                    # Parse the chunk data
                    try:
                        # Try to extract the chunk dict and neighbors
                        chunk_match = re.search(r"chunk=({.+?}) score=([0-9.]+) neighbors=(\[.+\])", record_content, re.DOTALL)
                        if chunk_match:
                            chunk_str = chunk_match.group(1)
                            score = float(chunk_match.group(2))
                            neighbors_str = chunk_match.group(3)
                            
                            # Parse chunk data
                            chunk_data = ast.literal_eval(chunk_str)
                            neighbors_data = ast.literal_eval(neighbors_str)
                            
                            # Format the main content
                            main_text = chunk_data.get('text', '')
                            chunk_id = chunk_data.get('id', '')
                            
                            # Build formatted context
                            formatted_parts = []
                            
                            # Add main content section
                            if main_text:
                                formatted_parts.append("📄 **Document Content:**")
                                # Clean up the text (remove extra whitespace)
                                clean_text = re.sub(r'\s+', ' ', main_text.strip())
                                formatted_parts.append(clean_text)
                                formatted_parts.append("")  # Empty line
                            
                            # Add score
                            formatted_parts.append(f"📊 **Relevance Score:** {score:.3f}")
                            formatted_parts.append("")
                            
                            # Add related entities from neighbors
                            if neighbors_data:
                                formatted_parts.append("🔗 **Related Knowledge Graph Entities:**")
                                
                                for neighbor in neighbors_data:
                                    # Filter out technical labels
                                    labels = [label for label in neighbor.get('labels', [])
                                              if label not in ['__KGBuilder__', '__Entity__']]
                                    
                                    if labels:
                                        entity_type = labels[0]  # Primary entity type
                                        props = neighbor.get('props', {})
                                        preview = neighbor.get('preview', '')
                                        
                                        # Get name or description
                                        name = props.get('name', preview)
                                        description = props.get('description', '')
                                        
                                        entity_line = f"  • **{entity_type}**: {name}"
                                        if description and description != name:
                                            entity_line += f" - {description}"
                                        
                                        formatted_parts.append(entity_line)
                                
                                formatted_parts.append("")
                            
                            # Add document ID at the end
                            if chunk_id:
                                formatted_parts.append(f"🆔 **Source ID:** {chunk_id}")
                            
                            return "\n".join(formatted_parts)
                            
                    except (ValueError, SyntaxError) as e:
                        logging.warning(f"Failed to parse Record content, using fallback: {e}")
            
            # Fallback: just clean up the raw content
            content_str = str(content)
            # Remove embedding arrays to keep content clean
            content_str = re.sub(r"'embedding':\s*\[[^\]]*\]", "'embedding': [...]", content_str)
            content_str = re.sub(r'"embedding":\s*\[[^\]]*\]', '"embedding": [...]', content_str)
            return content_str
            
        except Exception as e:
            logging.error(f"Error formatting graph context: {e}")
            return str(content)

    def get_answer(
        self,
        question: str,
        index_name: str,
        top_k: int = 5,
        temperature: float = 0,
        embedding_model: str = "text-embedding-3-large",
        chat_model: str = "gpt-4.1-mini",
        include_context: bool = False,
        allowed_labels: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None,
    ) -> Any:
        """Get an answer to a question from the knowledge graph.

        Args:
            question: The question to answer
            index_name: The name of the vector index to use
            top_k: Number of top chunks to retrieve
            temperature: LLM temperature for response generation
            embedding_model: OpenAI embedding model to use
            chat_model: OpenAI chat model to use
            include_context: If True, returns tuple of (answer, context_documents)
                           If False, returns just the answer string
            allowed_labels: List of allowed node labels from schema (optional)
            allowed_relationships: List of allowed relationship types from schema (optional)

        Returns:
            str: The answer as a string (if include_context=False)
            tuple[str, list[str]]: Tuple of (answer, context_documents) (if include_context=True)
        """
        embedder = OpenAIEmbeddings(model=embedding_model)

        # Use provided labels and relationships, or fall back to defaults
        default_labels = [
            "Article",
            "Role",
            "RiskLevel",
            "Term",
            "Provision",
            "Definition",
            "HighRiskCategory",
            "ProhibitedPractice",
            "Annex",
        ]
        default_relationships = [
            "GROUNDED_IN",
            "REFERS_TO",
            "MENTIONS",
            "APPLIES_TO",
            "CLASSIFIES_AS",
            "CITES",
        ]

        labels_to_use = allowed_labels if allowed_labels else default_labels
        relationships_to_use = (
            allowed_relationships if allowed_relationships else default_relationships
        )

        # Define the retrieval Cypher query that runs after vector kNN
        retrieval_query = """
        // Seed `node` and `score` come from the vector kNN step
        
        // Simple neighbor collection
        OPTIONAL MATCH (node)-[r]-(neighbor)
        WHERE neighbor IS NOT NULL AND neighbor <> node
        
        // Collect neighbors with simple limit
        WITH node, score, collect(DISTINCT neighbor)[..8] AS neighbors_list
        
        // Return clean data structure
        RETURN
        node {
            id: coalesce(node.id, elementId(node)),
            text: coalesce(node.text, node.name, node.description, '')
        } AS chunk,
        score,
        [n IN neighbors_list | {
            id: coalesce(n.id, elementId(n)),
            labels: labels(n),
            props: n { .* },
            preview: coalesce(n.text, n.name, n.description, '')
        }] AS neighbors
        """

        # Create VectorCypherRetriever instead of VectorRetriever
        retriever = VectorCypherRetriever(
            driver=self._driver,
            index_name=index_name,
            retrieval_query=retrieval_query,
            embedder=embedder,
        )

        llm = OpenAILLM(
            model_name=chat_model, model_params={"temperature": temperature}
        )

        # Define query parameters once to avoid duplication
        query_params = {
            "maxHops": 2,
            "allowRels": relationships_to_use,
            "allowLabels": labels_to_use,
            "maxNeighborsPerSeed": 40,
        }

        # Create GraphRAG once
        rag = GraphRAG(retriever=retriever, llm=llm)

        if include_context:
            # Get retrieval result first for context documents - this gives us the raw Neo4j records
            retrieval_result = retriever.search(
                query_text=question, top_k=top_k, query_params=query_params
            )

            # Extract context documents from the retrieval result
            context_documents = []
            logging.info(
                f"Retrieved {len(retrieval_result.items)} items from vector search"
            )

            for i, item in enumerate(retrieval_result.items):
                try:
                    if hasattr(item, "content") and item.content:
                        content = item.content
                        
                        # Parse the Neo4j Record to extract structured information
                        formatted_context = self._format_graph_context(content)
                        context_documents.append(formatted_context)
                        
                        logging.info(
                            f"Item {i}: Added formatted context ({len(formatted_context)} chars)"
                        )

                except Exception as e:
                    logging.error(f"Error processing item {i}: {e}")
                    continue

            logging.info(
                f"Successfully extracted {len(context_documents)} context documents with graph metadata"
            )

            # Get the answer using GraphRAG
            response = rag.search(
                query_text=question,
                retriever_config={"top_k": top_k, "query_params": query_params},
            )

            # Return tuple matching basic_rag schema: (answer, context_documents)
            answer = (
                response.answer
                if response.answer is not None
                else "No answer available"
            )
            # return answer, truncated_context
            return answer, context_documents
        else:
            # Original behavior - just return the answer string
            response = rag.search(
                query_text=question,
                retriever_config={"top_k": top_k, "query_params": query_params},
            )
            answer = (
                response.answer
                if response.answer is not None
                else "No answer available"
            )
            return answer

    def __enter__(self) -> "Neo4jConnection":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
