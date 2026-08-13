from typing import TypedDict


class GraphState(TypedDict):
    """State shared between LangGraph nodes."""

    question: str

    # The current query being used for retrieval.
    current_query: str

    # Retrieved documents from ChromaDB.
    documents: list[dict]

    # Documents that passed Gemini relevance grading.
    relevant_documents: list[dict]

    # Number of retrieval/rewrite cycles completed.
    retry_count: int

    # Maximum number of retries allowed.
    max_retries: int

    # Final generated answer.
    answer: str

    # Whether relevant documents were found.
    documents_relevant: bool