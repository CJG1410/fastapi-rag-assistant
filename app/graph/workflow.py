from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    generate_answer,
    grade_documents,
    retrieve_documents,
    rewrite_query,
    web_search,
)
from app.graph.routing import route_after_grading
from app.graph.state import GraphState


def build_graph():
    """Build the self-corrective RAG LangGraph."""

    workflow = StateGraph(GraphState)

    # --------------------------------------------------
    # Nodes
    # --------------------------------------------------

    workflow.add_node(
        "retrieve",
        retrieve_documents,
    )

    workflow.add_node(
        "grade",
        grade_documents,
    )

    workflow.add_node(
        "rewrite",
        rewrite_query,
    )

    workflow.add_node(
        "web_search",
        web_search,
    )

    workflow.add_node(
        "generate",
        generate_answer,
    )

    # --------------------------------------------------
    # START → RETRIEVE
    # --------------------------------------------------

    workflow.add_edge(
        START,
        "retrieve",
    )

    # --------------------------------------------------
    # RETRIEVE → GRADE
    # --------------------------------------------------

    workflow.add_edge(
        "retrieve",
        "grade",
    )

    # --------------------------------------------------
    # GRADE → Conditional routing
    # --------------------------------------------------

    workflow.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite",
            "web_search": "web_search",
        },
    )

    # --------------------------------------------------
    # REWRITE → RETRIEVE
    # --------------------------------------------------

    workflow.add_edge(
        "rewrite",
        "retrieve",
    )

    # --------------------------------------------------
    # WEB SEARCH → GENERATE
    # --------------------------------------------------

    workflow.add_edge(
        "web_search",
        "generate",
    )

    # --------------------------------------------------
    # GENERATE → END
    # --------------------------------------------------

    workflow.add_edge(
        "generate",
        END,
    )

    return workflow.compile()