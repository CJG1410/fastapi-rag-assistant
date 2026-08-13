from app.graph.state import GraphState


def route_after_grading(
    state: GraphState,
) -> str:
    """
    Decide what happens after document grading.

    Returns:
        "generate"      → relevant documents found
        "rewrite_query" → no relevant documents and
                          retries remain
        "end"           → no relevant documents and
                          retry limit reached
    """

    if state["documents_relevant"]:
        return "generate"

    if state["retry_count"] < state["max_retries"]:
        return "rewrite_query"

    return "end"