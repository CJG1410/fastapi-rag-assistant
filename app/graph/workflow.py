from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.graph.state import GraphState

from app.graph.nodes import (
    prepare_query,
    retrieve_documents,
    grade_documents,
    rewrite_query,
    web_search,
    generate_answer,
    verify_answer,
    prepare_regeneration,
)


# =========================================================
# ROUTING AFTER DOCUMENT GRADING
# =========================================================

def route_after_grading(state: GraphState) -> str:
    """
    Decide whether to generate an answer, rewrite the query,
    or fall back to web search.
    """

    if state["documents_relevant"]:

        print(
            "\nROUTER: Relevant documents found."
        )

        print(
            "ROUTER: Proceeding to generation."
        )

        return "generate"

    retry_count = state["retry_count"]
    max_retries = state["max_retries"]

    if retry_count < max_retries:

        print(
            "\nROUTER: No relevant documents."
        )

        print(
            f"ROUTER: Retry {retry_count + 1}/"
            f"{max_retries}."
        )

        print(
            "ROUTER: Rewriting query."
        )

        return "rewrite"

    print(
        "\nROUTER: No relevant documents."
    )

    print(
        "ROUTER: Retry limit reached."
    )

    print(
        "ROUTER: Falling back to Tavily."
    )

    return "web_search"


# =========================================================
# ROUTING AFTER ANSWER VERIFICATION
# =========================================================

def route_after_verification(state: GraphState) -> str:
    """
    Decide whether the generated answer is sufficiently
    supported or needs regeneration.
    """

    if state["answer_supported"]:

        print(
            "\nROUTER: Answer is supported."
        )

        print(
            "ROUTER: Returning final answer."
        )

        return "end"

    generation_retry_count = (
        state["generation_retry_count"]
    )

    max_generation_retries = (
        state["max_generation_retries"]
    )

    if generation_retry_count < max_generation_retries:

        print(
            "\nROUTER: Answer is not sufficiently "
            "supported."
        )

        print(
            f"ROUTER: Regeneration attempt "
            f"{generation_retry_count + 1}/"
            f"{max_generation_retries}."
        )

        return "regenerate"

    print(
        "\nROUTER: Maximum answer regeneration "
        "attempts reached."
    )

    print(
        "ROUTER: Returning latest answer."
    )

    return "end"


# =========================================================
# BUILD LANGGRAPH WORKFLOW
# =========================================================

def build_graph():
    """
    Build and compile the complete self-corrective RAG
    LangGraph workflow.
    """

    workflow = StateGraph(GraphState)

    # =====================================================
    # ADD NODES
    # =====================================================

    # Conversation-aware query preparation
    workflow.add_node(
        "prepare_query",
        prepare_query,
    )

    # ChromaDB retrieval
    workflow.add_node(
        "retrieve",
        retrieve_documents,
    )

    # Gemini document grading
    workflow.add_node(
        "grade",
        grade_documents,
    )

    # Query rewriting after failed retrieval
    workflow.add_node(
        "rewrite",
        rewrite_query,
    )

    # Tavily fallback
    workflow.add_node(
        "web_search",
        web_search,
    )

    # Gemini answer generation
    workflow.add_node(
        "generate",
        generate_answer,
    )

    # Hallucination / grounding verification
    workflow.add_node(
        "verify",
        verify_answer,
    )

    # Answer regeneration counter
    workflow.add_node(
        "prepare_regeneration",
        prepare_regeneration,
    )

    # =====================================================
    # START → PREPARE QUERY
    # =====================================================

    workflow.add_edge(
        START,
        "prepare_query",
    )

    # =====================================================
    # PREPARE QUERY → RETRIEVE
    # =====================================================

    workflow.add_edge(
        "prepare_query",
        "retrieve",
    )

    # =====================================================
    # RETRIEVE → GRADE
    # =====================================================

    workflow.add_edge(
        "retrieve",
        "grade",
    )

    # =====================================================
    # GRADE → GENERATE / REWRITE / WEB SEARCH
    # =====================================================

    workflow.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "web_search": "web_search",
        },
    )

    # =====================================================
    # REWRITE → RETRIEVE
    # =====================================================

    workflow.add_edge(
        "rewrite",
        "retrieve",
    )

    # =====================================================
    # WEB SEARCH → GENERATE
    # =====================================================

    workflow.add_edge(
        "web_search",
        "generate",
    )

    # =====================================================
    # GENERATE → VERIFY
    # =====================================================

    workflow.add_edge(
        "generate",
        "verify",
    )

    # =====================================================
    # VERIFY → END / REGENERATE
    # =====================================================

    workflow.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "end": END,
            "regenerate": "prepare_regeneration",
        },
    )

    # =====================================================
    # PREPARE REGENERATION → GENERATE
    # =====================================================

    workflow.add_edge(
        "prepare_regeneration",
        "generate",
    )

    # =====================================================
    # COMPILE GRAPH
    # =====================================================

    return workflow.compile()
    