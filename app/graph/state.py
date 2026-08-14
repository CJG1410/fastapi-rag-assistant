from typing import TypedDict


class GraphState(TypedDict):

    # -----------------------------------------------------
    # User question
    # -----------------------------------------------------

    question: str

    # -----------------------------------------------------
    # Current retrieval query
    # -----------------------------------------------------

    current_query: str

    # -----------------------------------------------------
    # Local retrieval
    # -----------------------------------------------------

    documents: list[dict]

    relevant_documents: list[dict]

    # -----------------------------------------------------
    # Web search
    # -----------------------------------------------------

    web_results: list[dict]

    # -----------------------------------------------------
    # Retrieval retry information
    # -----------------------------------------------------

    retry_count: int

    max_retries: int

    # -----------------------------------------------------
    # Final answer
    # -----------------------------------------------------

    answer: str

    # -----------------------------------------------------
    # Sources
    # -----------------------------------------------------

    sources: list[dict]

    source_type: str

    # -----------------------------------------------------
    # Retrieval state
    # -----------------------------------------------------

    documents_relevant: bool

    web_search_used: bool

    # -----------------------------------------------------
    # Hallucination verification
    # -----------------------------------------------------

    hallucination_checked: bool

    answer_supported: bool

    verification_reason: str

    # -----------------------------------------------------
    # Generation retry
    # -----------------------------------------------------

    generation_retry_count: int

    max_generation_retries: int