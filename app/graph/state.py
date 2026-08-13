from typing import TypedDict


class GraphState(TypedDict):
    """State shared between LangGraph nodes."""

    # Original user question
    question: str

    # Current query used for retrieval
    current_query: str

    # ChromaDB retrieved documents
    documents: list[dict]

    # Documents that passed Gemini relevance grading
    relevant_documents: list[dict]

    # Tavily web search results
    web_results: list[dict]

    # Number of retrieval/rewrite cycles
    retry_count: int

    # Maximum number of query rewrites
    max_retries: int

    # Final generated answer
    answer: str

    # Sources used to produce the answer
    sources: list[dict]

    # "local" or "web"
    source_type: str

    # Whether relevant local documents were found
    documents_relevant: bool

    # Whether Tavily was used
    web_search_used: bool