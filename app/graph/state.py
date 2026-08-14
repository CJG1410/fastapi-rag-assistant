from typing import TypedDict


class GraphState(TypedDict):

    # =====================================================
    # USER QUESTION
    # =====================================================

    question: str

    # Current query used for retrieval.
    # This can change when query rewriting occurs.
    current_query: str

    # =====================================================
    # LOCAL RETRIEVAL
    # =====================================================

    documents: list[dict]

    relevant_documents: list[dict]

    # =====================================================
    # WEB SEARCH
    # =====================================================

    web_results: list[dict]

    # =====================================================
    # RETRIEVAL RETRIES
    # =====================================================

    retry_count: int

    max_retries: int

    # =====================================================
    # GENERATED ANSWER
    # =====================================================

    answer: str

    # =====================================================
    # SOURCES
    # =====================================================

    sources: list[dict]

    source_type: str

    # =====================================================
    # RETRIEVAL STATUS
    # =====================================================

    documents_relevant: bool

    web_search_used: bool

    # =====================================================
    # HALLUCINATION / GROUNDING VERIFICATION
    # =====================================================

    hallucination_checked: bool

    answer_supported: bool

    verification_reason: str

    # =====================================================
    # ANSWER REGENERATION
    # =====================================================

    generation_retry_count: int

    max_generation_retries: int

    # =====================================================
    # CONVERSATION MEMORY
    # =====================================================

    conversation_history: list[dict]