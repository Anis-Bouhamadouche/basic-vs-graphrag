"""RAG implementation module."""

from dataclasses import dataclass
from typing import Annotated, Any, Iterator, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, START, StateGraph, add_messages

from basic_rag.vector_db import QdrantVectorDB


class RAGState(TypedDict):
    """State for the RAG system."""

    messages: Annotated[list[BaseMessage], add_messages]
    context: list[str]
    query: str


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    collection_name: str = "eu_ai_act"
    embedding_model: str = "text-embedding-3-large"
    top_k: int = 10
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    chat_model_name: str = "gpt-4.1-mini"
    system_message: str = (
        "You are a EU AI Act compliance assistant. The users will ask you questions about EU AI Act compliance. "
        "Strictly reply from the context provided. If the context is not sufficient to answer the question, "
        "simply mention to the user. Do not make up answers and do not answer questions outside of the context."
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")


class BasicRAGChat:
    """Basic RAG chat implementation using LangGraph."""

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        """Initialize the BasicRAGChat instance."""
        self.config = config or RAGConfig()
        # Initialize chat model - ChatOpenAI doesn't support max_tokens directly
        self.chat_model = ChatOpenAI(
            model=self.config.chat_model_name,
            temperature=self.config.temperature,
        )
        self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        self.vectordb_client = QdrantVectorDB()
        self.vector_store = QdrantVectorStore(
            client=self.vectordb_client.client,
            collection_name=self.config.collection_name,
            embedding=self.embeddings,
        )

        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Build the state graph for the RAG system."""
        graph_builder = StateGraph(RAGState)

        # Add nodes
        graph_builder.add_node("retrieve", self._retrieve)
        graph_builder.add_node("generate", self._generate)

        # Add edges
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", END)

        return graph_builder.compile()

    def _retrieve(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents based on the query."""
        # Get the latest user message
        last_message = state["messages"][-1]
        # Ensure we extract the string content properly
        if hasattr(last_message, "content"):
            query = str(last_message.content)
        else:
            query = str(last_message)

        # Perform retrieval
        results = self.vector_store.similarity_search(query, k=self.config.top_k)
        context = [doc.page_content for doc in results]

        # Update state
        return {
            **state,
            "context": context,
            "query": query,
        }

    def _generate(self, state: RAGState) -> RAGState:
        """Generate a response based on the query and context."""
        query = state["query"]
        context = state["context"]

        # Create context-aware prompt
        context_text = "\n\n".join(context)
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        # Generate response
        messages = [
            SystemMessage(content=self.config.system_message),
            HumanMessage(content=prompt),
        ]

        response = self.chat_model.invoke(messages)

        return {
            **state,
            "messages": [response],  # add_messages annotation handles appending
        }

    def invoke(self, input_data: dict[str, Any]) -> Any:
        """Invoke the RAG system following LangGraph standards.

        Args:
            input_data: Dictionary containing 'messages' or 'question'

        Returns:
            Dictionary with the complete state including messages and context
        """
        # Handle different input formats for convenience
        if "question" in input_data:
            # Convert question to proper message format
            from langchain_core.messages import HumanMessage

            messages = [HumanMessage(content=input_data["question"])]
        elif "messages" in input_data:
            messages = input_data["messages"]
        else:
            raise ValueError("Input must contain either 'question' or 'messages'")

        # Create initial state following LangGraph patterns
        initial_state = {
            "messages": messages,
            "context": [],
            "query": "",  # Will be populated by retrieve node
        }

        # Invoke the graph with the state
        return self.graph.invoke(initial_state)

    def stream(self, input_data: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Stream the RAG system execution following LangGraph standards.

        Args:
            input_data: Dictionary containing 'messages' or 'question'

        Yields:
            State updates as the graph executes
        """
        # Handle different input formats for convenience
        if "question" in input_data:
            from langchain_core.messages import HumanMessage

            messages = [HumanMessage(content=input_data["question"])]
        elif "messages" in input_data:
            messages = input_data["messages"]
        else:
            raise ValueError("Input must contain either 'question' or 'messages'")

        # Create initial state
        initial_state = {
            "messages": messages,
            "context": [],
            "query": "",
        }

        # Stream the graph execution
        for state_update in self.graph.stream(initial_state):
            yield state_update

    def get_answer(self, question: str, include_context: bool = False) -> Any:
        """Convenience method to get the answer text and optionally context documents.

        Args:
            question: The question to ask
            include_context: If True, returns tuple of (answer, context_documents)
                           If False, returns just the answer string

        Returns:
            str: The answer as a string (if include_context=False)
            tuple[str, list[str]]: Tuple of (answer, context_documents) (if include_context=True)
        """
        result = self.invoke({"question": question})
        final_message = result["messages"][-1]
        # Ensure we get the string content properly
        if hasattr(final_message, "content"):
            answer = str(final_message.content)
        else:
            answer = str(final_message)

        if include_context:
            context_documents = result.get("context", [])
            return answer, context_documents
        else:
            return answer
